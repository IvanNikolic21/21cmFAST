r"""Compute the patchy kinetic Sunyaev-Zel'dovich (kSZ) signal from a lightcone.

The kSZ temperature anisotropy is obtained by projecting the free-electron momentum
field of a rectilinear lightcone along the line of sight,

.. math::

    \frac{\Delta T}{T_{\rm CMB}}(\hat{n}) = -\sigma_T \int dz \,
    \frac{n_{e,0} (1+z)\, v}{H(z)} (1+\delta)(1-x_{\rm HI})\, e^{-\tau(z)},

where :math:`v` is the comoving line-of-sight velocity of the 21cmFAST lightcone
(positive away from the observer), :math:`n_{e,0}` the comoving electron density of
fully ionized gas (hydrogen and singly ionized helium, or doubly ionized helium below
``z_re_HeII``, as in :func:`~py21cmfast.compute_tau`) and :math:`\tau` the Thomson
optical depth along each line of sight.
"""

from __future__ import annotations

import attrs
import numpy as np
from astropy import constants, units
from scipy.integrate import cumulative_trapezoid, trapezoid

from .drivers.lightcone import AngularLightcone, LightCone
from .wrapper.cfuncs import compute_tau
from .wrapper.inputs import CosmoParams

# Lightcone quantities that can be used as the line-of-sight velocity, in order
# of preference.
_VELOCITY_QUANTITIES = ("los_velocity", "velocity_z")

_CM_PER_MPC = units.Mpc.to(units.cm)
_SIGMA_T = constants.sigma_T.cgs.value
_C_CGS = constants.c.cgs.value


@attrs.define(frozen=True, kw_only=True, eq=False)
class KSZResult:
    r"""The kSZ signal computed from a lightcone.

    Parameters
    ----------
    temperature
        2D map of the kSZ temperature anisotropy (mean subtracted), in Kelvin. The
        map covers the angular extent of the near face of the lightcone, i.e. each
        pixel subtends ``cell_size / lightcone_distances[0]`` radians.
    tau_e
        Mean Thomson optical depth integrated from z=0 to the far end of the
        lightcone.
    ell
        Multipoles of the angular power spectrum bins (mean multipole in each bin).
    power
        Angular power spectrum :math:`D_\ell = \ell(\ell+1)C_\ell/2\pi`, in
        :math:`\mu K^2`.
    power_err
        Standard error of the mean of :math:`D_\ell` in each bin, in
        :math:`\mu K^2`.
    """

    temperature: np.ndarray
    tau_e: float
    ell: np.ndarray
    power: np.ndarray
    power_err: np.ndarray


def compute_ksz(
    lightcone: LightCone,
    *,
    parallel_approx: bool = False,
    rotation: bool = False,
    random_seed: int | None = None,
    n_bins: int = 30,
    z_re_HeII: float = 3.0,
) -> KSZResult:
    r"""Compute the patchy kSZ signal of a rectilinear lightcone.

    The universe is assumed to be fully ionized between z=0 and the lowest redshift
    of the lightcone; the optical depth from that range is computed with
    :func:`~py21cmfast.compute_tau`.

    Parameters
    ----------
    lightcone
        A rectilinear lightcone containing the ``density`` and ``neutral_fraction``
        quantities, and either ``los_velocity`` or ``velocity_z``.
    parallel_approx
        If False (default), slices are re-sampled so that each map pixel subtends
        the same angle at all distances (ray tracing through the periodic box). If
        True, use the parallel-ray approximation, i.e. every slice is projected with
        the same transverse comoving size. Since the lightcone is built from a
        periodic box, parallel rays pass through the same structures every
        ``BOX_LEN`` along the line of sight, and their velocities add coherently.
        This strongly overestimates the power when the ionized part of the
        lightcone is longer than ``BOX_LEN`` (by a factor of ~2.6 at
        :math:`\ell=3000` for a 500 Mpc box covering z=5-15). Ray tracing removes
        most of this repetition. The parallel approximation is only safe when the
        box is larger than the ionized part of the lightcone.
    rotation
        If True, randomly shift the transverse position of the box every ``HII_DIM``
        slices, to avoid repeating the same structures along the line of sight.
        This introduces discontinuities in the lightcone and is not recommended.
    random_seed
        Seed for the random shifts used when ``rotation`` is True. Defaults to the
        random seed of the lightcone.
    n_bins
        Number of logarithmic multipole bins of the angular power spectrum.
    z_re_HeII
        Redshift of helium double reionization, used for the optical depth between
        z=0 and the start of the lightcone.

    Returns
    -------
    KSZResult
        The kSZ temperature map, mean optical depth and angular power spectrum.

    Raises
    ------
    NotImplementedError
        If ``lightcone`` is an angular lightcone.
    ValueError
        If a required quantity is missing from the lightcone.
    """
    if isinstance(lightcone, AngularLightcone):
        raise NotImplementedError("compute_ksz only supports rectilinear lightcones.")

    density, neutral_fraction, velocity = _get_lightcone_fields(lightcone)

    redshifts = np.asarray(lightcone.lightcone_redshifts, dtype=float)
    distances = lightcone.lightcone_distances.to_value(units.Mpc)
    cosmo_params = lightcone.cosmo_params
    if random_seed is None:
        random_seed = lightcone.random_seed

    tau_start = compute_tau(
        redshifts=[redshifts[0]],
        global_xHI=[1.0],
        inputs=lightcone.inputs,
        z_re_HeII=z_re_HeII,
    )

    hubble = cosmo_params.cosmo.H(redshifts).to_value(1 / units.s)
    electrons = (
        _electron_density_today(cosmo_params, redshifts, z_re_HeII)
        * (1.0 + density)
        * (1.0 - neutral_fraction)
    )

    # Optical depth (dtau/dz) and kSZ (d(dT/T)/dz) contributions of each cell.
    dtau_dz = _SIGMA_T * _C_CGS * electrons * ((1 + redshifts) ** 2 / hubble)
    dtemp_dz = (
        -_SIGMA_T * electrons * velocity * _CM_PER_MPC * ((1 + redshifts) / hubble)
    )
    del electrons

    tau = cumulative_trapezoid(dtau_dz, x=redshifts, axis=-1, initial=0.0) + tau_start
    tau_e = float(np.mean(tau[..., -1]))
    del dtau_dz

    if rotation or not parallel_approx:
        _remap_slices(
            dtemp_dz,
            scale=distances / distances[0],
            ray_trace=not parallel_approx,
            rotation=rotation,
            rng=np.random.default_rng(random_seed),
        )

    delta_t = trapezoid(dtemp_dz * np.exp(-tau), x=redshifts, axis=-1)
    delta_t -= delta_t.mean()
    temperature = delta_t * cosmo_params.cosmo.Tcmb0.to_value(units.K)

    k, power, variance = _angular_power_spectrum(
        temperature * 1e6 / np.sqrt(2 * np.pi),
        box_len=lightcone.simulation_options.BOX_LEN,
        n_bins=n_bins,
    )
    keep = ~np.isnan(k)
    k = k[keep]
    return KSZResult(
        temperature=temperature,
        tau_e=tau_e,
        ell=k * distances[0],
        power=power[keep] * k**2,
        power_err=np.sqrt(variance[keep]) * k**2,
    )


def _get_lightcone_fields(
    lightcone: LightCone,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the density, neutral fraction and line-of-sight velocity lightcones."""
    lcs = lightcone.lightcones
    missing = [q for q in ("density", "neutral_fraction") if q not in lcs]
    velocity_key = next((q for q in _VELOCITY_QUANTITIES if q in lcs), None)
    if velocity_key is None:
        missing.append(" or ".join(_VELOCITY_QUANTITIES))
    if missing:
        raise ValueError(
            f"The lightcone is missing quantities required for kSZ: {missing}. "
            "Add them to the quantities of the lightconer."
        )
    return lcs["density"], lcs["neutral_fraction"], lcs[velocity_key]


def _electron_density_today(
    cosmo_params: CosmoParams, redshifts: np.ndarray, z_re_HeII: float
) -> np.ndarray:
    """Comoving electron density (cm^-3) of fully ionized gas at each redshift.

    This matches the C code used by :func:`~py21cmfast.compute_tau`: hydrogen plus
    singly ionized helium (``N_b0``), with helium doubly ionized below ``z_re_HeII``.
    """
    hubble0 = cosmo_params.hlittle * 3.2407e-18  # s^-1
    baryons = (
        3.0
        * hubble0**2
        / (8.0 * np.pi * constants.G.cgs.value)
        * cosmo_params.OMb
        / constants.m_p.cgs.value
    )
    n_hydrogen = baryons * (1 - cosmo_params.Y_He)
    n_helium = baryons * cosmo_params.Y_He / 4.0
    return n_hydrogen + n_helium * np.where(redshifts < z_re_HeII, 2.0, 1.0)


def _remap_slices(
    field: np.ndarray,
    *,
    scale: np.ndarray,
    ray_trace: bool,
    rotation: bool,
    rng: np.random.Generator,
) -> None:
    """Remap the transverse pixels of each lightcone slice, in place.

    With ``ray_trace``, slice ``k`` is re-sampled (periodically) so that the map has
    the angular size of the near face of the lightcone; ``scale[k]`` is the ratio of
    the comoving distance of slice ``k`` to that of the first slice. With
    ``rotation``, slices are shifted by a random transverse offset that changes
    every ``HII_DIM`` slices.
    """
    n = field.shape[0]
    centred = np.arange(-n / 2, n / 2)
    shift = (0, 0)
    for k in range(field.shape[-1]):
        if ray_trace:
            idx = np.round(centred * scale[k] + 3 * n / 2).astype(int)
            field[..., k] = (
                field[..., k]
                .take(idx, axis=0, mode="wrap")
                .take(idx, axis=1, mode="wrap")
            )
        if rotation:
            if k % n == 0:
                shift = tuple(-rng.integers(0, n, size=2))
            field[..., k] = np.roll(field[..., k], shift, axis=(0, 1))


def _angular_power_spectrum(
    field: np.ndarray, *, box_len: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the azimuthally averaged power spectrum of a square 2D field.

    This follows the conventions of :func:`powerbox.get_power` with logarithmic bins:
    the Fourier transform is normalised as a continuous transform, the power is
    normalised by the area, and the variance is the standard error of the mean
    power in each bin.

    Returns
    -------
    k
        Mean wavenumber (radians per unit length) of each bin. NaN for empty bins.
    power
        Mean power in each bin, in units of ``field**2 * length**2``.
    variance
        Variance of the mean power in each bin.
    """
    n = field.shape[0]
    dx = box_len / n
    power = np.abs(np.fft.fft2(field) * dx**2) ** 2 / box_len**2

    k1d = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    kmag = np.sqrt(k1d[:, None] ** 2 + k1d[None, :] ** 2).ravel()
    power = power.ravel()

    edges = np.logspace(
        np.log10(kmag[kmag > 0].min()), np.log10(kmag.max()), n_bins + 1
    )
    idx = np.digitize(kmag, edges)

    def _binsum(weights=None):
        return np.bincount(idx, weights=weights, minlength=n_bins + 2)[1:-1]

    counts = _binsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        k_mean = _binsum(kmag) / counts
        p_mean = _binsum(power) / counts
        resid = (power - np.concatenate(([0], p_mean, [0]))[idx]) ** 2
        variance = _binsum(resid) / (counts - 1) / counts

    return k_mean, p_mean, variance

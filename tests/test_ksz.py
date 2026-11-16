"""Tests of the kSZ module."""

import numpy as np
import pytest
from astropy import units

from py21cmfast import InputParameters, compute_ksz, compute_tau
from py21cmfast.drivers.lightcone import AngularLightcone, LightCone
from py21cmfast.ksz import _angular_power_spectrum

N_SLICES = 80
RANDOM_ZMIN = 0.5


@pytest.fixture(scope="module")
def inputs() -> InputParameters:
    return InputParameters.from_template(["simple", "tiny"], random_seed=1)


def _make_lightcone(
    inputs: InputParameters,
    *,
    density: np.ndarray | float = 0.0,
    neutral_fraction: np.ndarray | float = 0.0,
    velocity: np.ndarray | float = 0.0,
    velocity_key: str = "los_velocity",
    zmin: float = 7.0,
    cls: type[LightCone] = LightCone,
) -> LightCone:
    """Build a synthetic rectilinear lightcone starting at ``zmin``."""
    n = inputs.simulation_options.HII_DIM
    shape = (n, n, N_SLICES)
    cell = inputs.simulation_options.cell_size.to_value(units.Mpc)
    d0 = inputs.cosmo_params.cosmo.comoving_distance(zmin).to_value(units.Mpc)
    return cls(
        lightcone_distances=(d0 + cell * np.arange(N_SLICES)) * units.Mpc,
        inputs=inputs,
        lightcones={
            "density": np.broadcast_to(density, shape).astype(np.float32),
            "neutral_fraction": np.broadcast_to(neutral_fraction, shape).astype(
                np.float32
            ),
            velocity_key: np.broadcast_to(velocity, shape).astype(np.float32),
        },
    )


@pytest.fixture(scope="module")
def random_lightcone(inputs) -> LightCone:
    rng = np.random.default_rng(42)
    n = inputs.simulation_options.HII_DIM
    shape = (n, n, N_SLICES)
    return _make_lightcone(
        inputs,
        density=rng.normal(scale=0.3, size=shape),
        neutral_fraction=rng.uniform(size=shape),
        velocity=rng.normal(scale=1e-15, size=shape),  # ~ 30 km/s in Mpc/s
        # Low enough that the lightcone spans a large fraction of its distance,
        # so ray tracing actually re-samples the slices.
        zmin=RANDOM_ZMIN,
    )


def test_angular_power_spectrum_matches_powerbox():
    powerbox = pytest.importorskip("powerbox")
    rng = np.random.default_rng(1)
    field = rng.normal(size=(32, 32))

    k, power, variance = _angular_power_spectrum(field, box_len=50.0, n_bins=12)
    expected = powerbox.get_power(
        field,
        50.0,
        bins=12,
        log_bins=True,
        get_variance=True,
        bins_upto_boxlen=False,
    )

    np.testing.assert_allclose(k, expected.bin_avg, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(power, expected.power, rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(variance, expected.variance, rtol=1e-10, equal_nan=True)


def test_neutral_lightcone_has_no_signal(inputs):
    lc = _make_lightcone(inputs, neutral_fraction=1.0, velocity=1e-15)
    result = compute_ksz(lc)

    np.testing.assert_allclose(result.temperature, 0.0, atol=1e-20)
    np.testing.assert_allclose(result.power, 0.0, atol=1e-20)

    # With no free electrons in the lightcone, tau comes only from z < zmin.
    tau_start = compute_tau(
        redshifts=[lc.lightcone_redshifts[0]], global_xHI=[1.0], inputs=inputs
    )
    assert result.tau_e == pytest.approx(tau_start, rel=1e-6)


def test_tau_matches_compute_tau(inputs):
    """A fully ionized, uniform lightcone gives the same tau as the C integrator."""
    lc = _make_lightcone(inputs, neutral_fraction=0.0)
    result = compute_ksz(lc)

    tau_full = compute_tau(
        redshifts=[lc.lightcone_redshifts[-1]], global_xHI=[1.0], inputs=inputs
    )
    assert result.tau_e == pytest.approx(tau_full, rel=5e-3)


def test_defaults_are_parallel_without_rotation(random_lightcone):
    default = compute_ksz(random_lightcone)
    explicit = compute_ksz(random_lightcone, parallel_approx=True, rotation=False)
    np.testing.assert_array_equal(default.temperature, explicit.temperature)


def test_ionized_lightcone_increases_tau(inputs):
    neutral = compute_ksz(_make_lightcone(inputs, neutral_fraction=1.0))
    ionized = compute_ksz(_make_lightcone(inputs, neutral_fraction=0.0))
    assert ionized.tau_e > neutral.tau_e


def test_parallel_map_traces_transverse_velocity(inputs):
    """With uniform gas and a LoS-constant velocity, the map follows the velocity."""
    n = inputs.simulation_options.HII_DIM
    rng = np.random.default_rng(3)
    v_transverse = rng.normal(scale=1e-15, size=(n, n, 1))

    result = compute_ksz(
        _make_lightcone(inputs, velocity=v_transverse),
        parallel_approx=True,
        rotation=False,
    )

    v32 = v_transverse[..., 0].astype(np.float32).astype(float)
    expected = v32 - v32.mean()
    coeff = np.sum(result.temperature * expected) / np.sum(expected**2)
    np.testing.assert_allclose(
        result.temperature,
        coeff * expected,
        atol=1e-5 * np.abs(result.temperature).max(),
    )
    # Gas moving away from the observer gives a temperature decrement.
    assert coeff < 0


def test_signal_is_linear_in_velocity(inputs, random_lightcone):
    doubled = _make_lightcone(
        inputs,
        density=random_lightcone.lightcones["density"],
        neutral_fraction=random_lightcone.lightcones["neutral_fraction"],
        velocity=2 * random_lightcone.lightcones["los_velocity"],
        zmin=RANDOM_ZMIN,
    )
    base = compute_ksz(random_lightcone)
    double = compute_ksz(doubled)

    np.testing.assert_allclose(double.temperature, 2 * base.temperature, rtol=1e-5)
    np.testing.assert_allclose(double.power, 4 * base.power, rtol=1e-5)
    assert double.tau_e == pytest.approx(base.tau_e, rel=1e-10)


@pytest.mark.parametrize(
    ("parallel_approx", "rotation"),
    [(False, True), (False, False), (True, True), (True, False)],
)
def test_output_shapes(random_lightcone, parallel_approx, rotation):
    result = compute_ksz(
        random_lightcone, parallel_approx=parallel_approx, rotation=rotation
    )
    n = random_lightcone.simulation_options.HII_DIM

    assert result.temperature.shape == (n, n)
    assert result.temperature.mean() == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.isfinite(result.ell))
    assert np.all(np.diff(result.ell) > 0)
    assert result.ell.shape == result.power.shape == result.power_err.shape
    assert np.all(result.power >= 0)


def test_ray_tracing_and_rotation_change_the_map(random_lightcone):
    plain = compute_ksz(random_lightcone, parallel_approx=True, rotation=False)
    traced = compute_ksz(random_lightcone, parallel_approx=False, rotation=False)
    rotated = compute_ksz(random_lightcone, parallel_approx=True, rotation=True)

    assert not np.allclose(plain.temperature, traced.temperature)
    assert not np.allclose(plain.temperature, rotated.temperature)
    # Neither changes the optical depth.
    assert traced.tau_e == pytest.approx(plain.tau_e, rel=1e-10)
    assert rotated.tau_e == pytest.approx(plain.tau_e, rel=1e-10)


def test_rotation_is_reproducible(random_lightcone):
    first = compute_ksz(random_lightcone, rotation=True, random_seed=5)
    second = compute_ksz(random_lightcone, rotation=True, random_seed=5)
    other = compute_ksz(random_lightcone, rotation=True, random_seed=6)

    np.testing.assert_array_equal(first.temperature, second.temperature)
    assert not np.allclose(first.temperature, other.temperature)


def test_velocity_z_fallback(inputs, random_lightcone):
    lc = _make_lightcone(
        inputs,
        density=random_lightcone.lightcones["density"],
        neutral_fraction=random_lightcone.lightcones["neutral_fraction"],
        velocity=random_lightcone.lightcones["los_velocity"],
        velocity_key="velocity_z",
        zmin=RANDOM_ZMIN,
    )
    np.testing.assert_array_equal(
        compute_ksz(lc).temperature, compute_ksz(random_lightcone).temperature
    )


def test_missing_quantity_raises(inputs):
    lc = _make_lightcone(inputs, velocity_key="brightness_temp")
    with pytest.raises(ValueError, match="los_velocity or velocity_z"):
        compute_ksz(lc)


def test_angular_lightcone_not_supported(inputs):
    lc = _make_lightcone(inputs, cls=AngularLightcone)
    with pytest.raises(NotImplementedError, match="rectilinear"):
        compute_ksz(lc)

"""
A module containing (base) classes for computing 21cmFAST likelihoods under the context of CosmoHammer.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from .._21cmfast import wrapper as lib
from . import core
from io import IOBase

from os import path, rename
from powerbox.tools import get_power

np.seterr(invalid='ignore', divide='ignore')

import logging
logger = logging.getLogger("21CMMC")

def ensure_iter(a):
    try:
        iter(a)
        return a
    except TypeError:
        return [a]


def listify(l):
    if type(l) == list:
        return l
    else:
        return [l]


class LikelihoodBase:
    required_cores = []

    def computeLikelihood(self, model):
        raise NotImplementedError("The Base likelihood should never be used directly!")

    def simulate(self, ctx):
        raise NotImplementedError("The Base likelihood should never be used directly!")

    def _expose_core_parameters(self):
        # Try to get the params out of the core module
        for m in self.LikelihoodComputationChain.getCoreModules():
            for k in ['user_params', 'flag_options', 'cosmo_params', 'astro_params']:
                if hasattr(m, k):
                    if hasattr(self, k) and getattr(self, k) != getattr(m, k):
                        raise ValueError(
                            f"Setup has detected incompatible input parameter dicts in specified cores: {k}")
                    else:
                        setattr(self, k, getattr(m, k))

    def _check_required_cores(self):
        for rc in self.required_cores:
            if not any([isinstance(m, rc) for m in self.LikelihoodComputationChain.getCoreModules()]):
                raise ValueError("%s needs the %s to be loaded." % (self.__class__.__name__, rc.__class__.__name__))

    def setup(self):
        logger.info("Running setup")

        # Expose user, flag, cosmo, astro params to this likelihood if available.
        self._expose_core_parameters()

        # Ensure that any required cores are actually loaded.
        self._check_required_cores()

    @property
    def default_ctx(self):
        try:
            chain = self.LikelihoodComputationChain
        except AttributeError:
            raise core.NotSetupError

        return chain.core_context()

    @property
    def default_simulated_ctx(self):
        try:
            chain = self.LikelihoodComputationChain
        except AttributeError:
            raise core.NotSetupError

        return chain.core_simulated_context()

    @property
    def _core(self):
        "The *primary* core module (i.e. the first one that is a required core)."
        for rc in self.required_cores:
            for m in self.LikelihoodComputationChain.getCoreModules():
                if isinstance(m, rc):
                    return m


class LikelihoodBaseFile(LikelihoodBase):
    def __init__(self, datafile=None, noisefile=None, simulate=False, use_data=True):
        self.datafile = datafile
        self.noisefile = noisefile
        self._use_data = use_data

        # We *always* make the datafile and noisefile a list
        if isinstance(self.datafile, str) or isinstance(self.datafile, IOBase):
            self.datafile = [self.datafile]
        if isinstance(self.noisefile, str) or isinstance(self.noisefile, IOBase):
            self.noisefile = [self.noisefile]

        self._simulate = simulate

        self.data = None
        self.noise=None
    def setup(self):
        super().setup()

        if self._use_data:
            if not self._simulate and not self.datafile:
                raise ValueError("Either an existing datafile has to be specified, or simulate set to True.")

            if self._simulate:
                simctx = self.default_simulated_ctx

            # Read in or simulate the data and noise.
            self.data = self.simulate(simctx) if self._simulate else self._read_data()

            # If we can't/won't simulate noise, and no noisefile is provided, assume no noise is necessary.
            if (hasattr(self, "define_noise") or self._simulate) or self.noisefile:
                self.noise = self.define_noise(simctx, self.data) if (hasattr(self, "define_noise") and self._simulate) else self._read_noise()

            # Now, if data has been simulated, and a file is provided, write to the file.
            if self.datafile and self._simulate:
                self._write_data()

            if self.noisefile and self._simulate and hasattr(self, "define_noise"):
                self._write_noise()

        logger.info("Finished base setup")

    def _read_data(self):
        data = []
        for fl in self.datafile:
            if not path.exists(fl):
                raise FileNotFoundError(f"Could not find datafile: {fl}. If you meant to simulate data, set simulate=True.")
            else:
                data.append(dict(**np.load(fl)))

        return data

    def _read_noise(self):
        if self.noisefile:
            noise = []
            for fl in self.noisefile:
                if not path.exists(fl):
                    msg = ""
                    if hasattr(self, "define_noise"):
                        msg = "If you meant to simulate noise, set simulate=True."

                    raise FileNotFoundError(
                        f"Could not find noisefile: {fl}. {msg}")

                else:
                    noise.append(dict(**np.load(fl)))

    def _write_data(self):
        for fl, d in zip(self.datafile, self.data):
            if path.exists(fl):
                logger.warn(f"File {fl} already exists. Moving previous version to {fl}.bk")
                rename(fl, fl+".bk")

            np.savez(fl, **d)
            logger.info(f"Saving data file: {fl}")

    def _write_noise(self):
        for fl, d in zip(self.noisefile, self.noise):
            if path.exists(fl):
                logger.warn(f"File {fl} already exists. Moving previous version to {fl}.bk")
                rename(fl, fl + ".bk")

            np.savez(fl, **d)
            logger.info(f"Saved noise file: {fl}")

    def _check_data_format(self):
        pass

    def _check_noise_format(self):
        pass


class Likelihood1DPowerCoeval(LikelihoodBaseFile):
    """
    A likelihood which assumes that the spherically-averaged power spectrum is iid Gaussian in each bin.

    Requires the CoreCoevalModule to be loaded to work, and inherently deals with the multiple-redshift cubes
    which that module produces.

    If a `datafile` is provided and the datafile exists, then the data will be read from that file. Otherwise,
    theoretical data will be automatically simulated to match current parameters. This will be written to
    `datafile` if provided.
    """
    required_cores = [core.CoreCoevalModule]

    def __init__(self,n_psbins=None, min_k=0.1, max_k = 1.0, logk=True, model_uncertainty=0.15,
                 error_on_model=True, ignore_kperp_zero=True, ignore_kpar_zero=False, ignore_k_zero=False,
                 *args, **kwargs):
        """
        Initialize the likelihood.

        Parameters
        ----------
        datafile : str, optional
            The file(s) from which to read the data. Alternatively, the file to which to write the data (see class
            docstring for how this works). See notes below for details.
        noisefile : str, optional
            The file(s) from which to read the noise profile. If not given, no thermal noise or cosmic variance is
            used in the fit. The noisefile should be an .npz file with the arrays "k" and "errs" in it. This
            is *almost* the default output format of 21cmSense. See notes below on how to extend this behaviour.
        n_psbins : int, optional
            The number of bins for the spherically averaged power spectrum. By default automatically
            calculated from the number of cells.
        min_k : float, optional
            The minimum k value at which to compare model and data (units 1/Mpc).
        max_k : float, optional
            The maximum k value at which to compare model and data (units 1/Mpc).
        logk : bool, optional
            Whether the power spectrum bins should be regular in logspace or linear space.
        model_uncertainty : float, optional
            The amount of uncertainty in the modelling, per power spectral bin (as fraction of the amplitude).
        error_on_model : bool, optional
            Whether the `model_uncertainty` is applied to the model, or the data.
        ignore_kperp_zero : bool, optional
            Whether to ignore the kperp=0 when generating the power spectrum.
        ignore_kpar_zero : bool, optional
            Whether to ignore the kpar=0 when generating the power spectrum.
        ignore_k_zero : bool, optional
            Whether to ignore the |k| = 0 mode when generating the power spectrum.

        Notes
        -----
        The datafile and noisefile have specific formatting required. Both should be .npz files. The datafile should
        have 'k' and 'delta' arrays in it (k-modes in 1/Mpc and power spectrum respectively) and the noisefile should
        have 'k' and 'errs' arrays in it (k-modes and their standard deviations respectively). Note that the latter is
        *almost* the default output of 21cmSense, except that 21cmSense has k in units of h/Mpc, whereas 21cmFAST/21CMMC
        use units of 1/Mpc.

        .. warning:: Please ensure that the data/noise is in the correct units for 21CMMC, as this class
                     does not automatically convert units!

        To make this more flexible, simply subclass this class, and overwrite the :meth:`_read_data` or
        :meth:`_read_noise` methods, then use that likelihood instead of this in your likelihood chain. Both of these
        functions should return dictionaries in which the above entries exist. For example::

        >>> class MyCoevalLikelihood(Likelihood1DPowerCoeval):
        >>>    def _read_data(self):
        >>>        data = np.genfromtxt(self.datafile)
        >>>        return {"k": data[:, 0], "p": data[:, 1]}

        Also note that an extra method, `define_noise` may be used to define the noise properties dynamically (i.e.
        without reading it). This method will be called if available and simulate=True. It should have the
        signature ``define_noise(self, ctx, model)``, where ``ctx`` is the context with all cores having added their
        data, and ``model`` is the output of the :method:`simulate` method.
        """
        super().__init__(*args, **kwargs)

        if self.noisefile and self.datafile and len(self.datafile) != len(self.noisefile):
            raise ValueError("If noisefile or datafile are provided, they should have the same number of files (one for each coeval box)")

        self.n_psbins = n_psbins
        self.min_k = min_k
        self.max_k = max_k
        self.logk = logk
        self.error_on_model = error_on_model
        self.model_uncertainty = model_uncertainty
        self.ignore_k_zero = ignore_k_zero
        self.ignore_kperp_zero = ignore_kperp_zero
        self.ignore_kpar_zero = ignore_kpar_zero

    def _check_data_format(self):
        for i, d in enumerate(self.data):
            if "k" not in d or "delta" not in d:
                raise ValueError(f"datafile #{i+1} of {len(self.datafile)} has the wrong format")

    def _check_noise_format(self):
        for i, n in enumerate(self.noise):
            if "ks" not in n or "errs" not in n:
                raise ValueError(f"noisefile #{i+1} of {len(self.noise)} has the wrong format")

    def setup(self):
        super().setup()

        # Ensure that there is one dataset and noiseset per redshift.
        if len(self.data) != len(self.redshift):
            raise ValueError("There needs to be one dataset (datafile) for each redshift!")

        if self.noise and len(self.noise) != len(self.redshift):
            raise ValueError("There needs to be one dataset (noisefile) for each redshift!")

        # Check if all data is formatted correctly.
        self._check_data_format()
        if self.noise: self._check_noise_format()

        # This needs to be first-order (linear) so that symmetry between data and model can be preserved.
        self.data_spline = [InterpolatedUnivariateSpline(d['k'], d['delta'], k=1) for d in self.data]

        if self.noise:
            self.noise_spline = [InterpolatedUnivariateSpline(n['k'], n['errs'], k=1) for n in self.noise]
        else:
            self.noise_spline = None

    @staticmethod
    def compute_power(brightness_temp, L, n_psbins, log_bins=True, ignore_kperp_zero=True, ignore_kpar_zero=False,
                      ignore_k_zero=False):
        # Determine the weighting function required from ignoring k's.
        k_weights = np.ones(brightness_temp.brightness_temp.shape, dtype=np.int)
        n = k_weights.shape[0]

        if ignore_kperp_zero:
            k_weights[n//2, n//2, :] = 0
        if ignore_kpar_zero:
            k_weights[:,:, n//2] = 0
        if ignore_k_zero:
            k_weights[n//2, n//2, n//2] = 0

        res = get_power(
            brightness_temp.brightness_temp,
            boxlength = L,
            bins=n_psbins, bin_ave=False, get_variance=False, log_bins=log_bins, k_weights = k_weights
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    @property
    def core_module(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, self.required_cores[0]):
                return m

    @property
    def redshift(self):
        return self.core_module.redshift

    def computeLikelihood(self, model):
        """
        Compute the likelihood

        Parameters
        ----------
        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """

        lnl = 0
        noise = 0
        for i, (m, pd) in enumerate(zip(model, self.data_spline)):
            mask = np.logical_and(m['k'] <= self.max_k, m['k'] >= self.min_k)

            moduncert = self.model_uncertainty*pd(m['k'][mask]) if not self.error_on_model else self.model_uncertainty*m['delta'][mask]

            if self.noise_spline:
                noise = self.noise_spline[i](m['k'][mask])

            # TODO: if moduncert depends on model, not data, then it should appear as -0.5 log(sigma^2) term below.
            lnl += -0.5 * np.sum((m['delta'][mask] - pd(m['k'][mask])) ** 2 / (moduncert**2 + noise**2))
        return lnl

    def simulate(self, ctx):
        brightness_temp = ctx.get("brightness_temp")
        data = []

        for bt in brightness_temp:
            power, k = self.compute_power(
                bt, self.user_params.BOX_LEN, self.n_psbins, log_bins=self.logk,
                ignore_k_zero=self.ignore_k_zero, ignore_kpar_zero=self.ignore_kpar_zero,
                ignore_kperp_zero=self.ignore_kperp_zero
            )
            data.append({"k":k, "delta":power * k**3 / (2*np.pi**2)})

        return data

    def store(self, model, storage):
        # add the power to the written data
        for i, m in enumerate(model):
            storage.update({k+"_z%s"%self.redshift[i]:v for k,v in m.items()})


class Likelihood1DPowerLightcone(Likelihood1DPowerCoeval):
    """
    A likelihood very similar to :class:`Likelihood1DPowerCoeval`, except for a lightcone.

    Since most of the functionality is the same, please see the other documentation for details.
    """
    required_cores = [core.CoreLightConeModule]

    def __init__(self, *args, nchunks=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.nchunks = nchunks

    def setup(self):
        LikelihoodBaseFile.setup(self)

        # Ensure that there is one dataset and noiseset per redshift.
        if len(self.data) != self.nchunks:
            raise ValueError("There needs to be one dataset (datafile) for each chunk!!")

        if self.noise and len(self.noise) != self.nchunks:
            raise ValueError("There needs to be one dataset (noisefile) for each chunk!")

        # Check if all data is formatted correctly.
        self._check_data_format()
        if self.noise: self._check_noise_format()

        # This needs to be first-order (linear) so that symmetry between data and model can be preserved.
        self.data_spline = [InterpolatedUnivariateSpline(d['k'], d['delta'], k=1) for d in self.data]

        if self.noise:
            self.noise_spline = [InterpolatedUnivariateSpline(n['ks'], n['errs'], k=1) for n in self.noise]
        else:
            self.noise_spline = None

    @staticmethod
    def compute_power(box, length, n_psbins, log_bins=True, ignore_kperp_zero=True, ignore_kpar_zero=False,
                      ignore_k_zero=False):
        # Determine the weighting function required from ignoring k's.
        k_weights = np.ones(box.shape, dtype=np.int)
        n0 = k_weights.shape[0]
        n1 = k_weights.shape[-1]

        if ignore_kperp_zero:
            k_weights[n0//2, n0//2, :] = 0
        if ignore_kpar_zero:
            k_weights[:,:, n1//2] = 0
        if ignore_k_zero:
            k_weights[n0//2, n0//2, n1//2] = 0

        res = get_power(
            box,
            boxlength = length,
            bins=n_psbins, bin_ave=False, get_variance=False, log_bins=log_bins, k_weights=k_weights
        )

        res = list(res)
        k = res[1]
        if log_bins:
            k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
        else:
            k = (k[1:] + k[:-1]) / 2

        res[1] = k
        return res

    def simulate(self, ctx):
        brightness_temp = ctx.get("lightcone")
        data = []
        chunk_indices = list(range(0, brightness_temp.n_slices, round(brightness_temp.n_slices/self.nchunks)))

        if len(chunk_indices) > self.nchunks:
            chunk_indices = chunk_indices[:-1]

        chunk_indices.append(brightness_temp.n_slices)

        for i in range(self.nchunks):
            start = chunk_indices[i]
            end = chunk_indices[i+1]
            chunklen = (end-start) * brightness_temp.cell_size

            power, k = self.compute_power(
                brightness_temp.brightness_temp[:,:,start:end],
                (self.user_params.BOX_LEN, self.user_params.BOX_LEN, chunklen),
                self.n_psbins, log_bins=self.logk,
                ignore_kperp_zero=self.ignore_kperp_zero, ignore_kpar_zero=self.ignore_kpar_zero,
                ignore_k_zero=self.ignore_k_zero
            )
            data.append({"k":k, "delta":power * k**3 / (2*np.pi**2)})

        return data

    def store(self, model, storage):
        # add the power to the written data
        for i, m in enumerate(model):
            storage.update({k + "_%s" %i : v for k, v in m.items()})


class LikelihoodPlanck(LikelihoodBase):
    """
    A likelihood which utilises Planck optical depth data.

    In practice, any optical depth measurement (or mock measurement) may be used, by defining the class variables
    `tau_mean` and `tau_sigma`.
    """
    # Mean and one sigma errors for the Planck constraints
    # The Planck prior is modelled as a Gaussian: tau = 0.058 \pm 0.012 (https://arxiv.org/abs/1605.03507)
    tau_mean = 0.058
    tau_sigma = 0.012

    # Simple linear extrapolation of the redshift range provided by the user, to be able to estimate the optical depth
    n_z_interp = 15

    # Minimum of extrapolation is chosen to 5.9, to correspond to the McGreer et al. prior on the IGM neutral fraction.
    # The maximum is chosed to be z = 18., which is arbitrary.
    z_extrap_min = 5.9
    z_extrap_max = 20.0

    def computeLikelihood(self, model):
        """
        Compute the likelihood.

        This is the likelihood arising from Planck (2016) (https://arxiv.org/abs/1605.03507).

        Parameters
        ----------
        model : list of dicts
            Exactly the output of :meth:`simulate`.

        Returns
        -------
        lnl : float
            The log-likelihood for the given model.
        """
        return ((self.tau_mean - model['tau']) / self.tau_sigma)**2

    @property
    def _core(self):
        "The core module used for the xHI global value"

        if not hasattr(self, "LikelihoodComputationChain"):
            raise AttributeError("redshifts are not available until chain has been setup")

        # Try using a lightcone
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, core.CoreLightConeModule):
                return m

        # Otherwise try using a Coeval
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, core.CoreCoevalModule):
                return m

        # Otherwise, give an error
        raise AttributeError("The Planck Likelihood requires either a LightCone (preferred) or Coeval core module")

    @property
    def _is_lightcone(self):
        return isinstance(self._core, core.CoreLightConeModule)

    def simulate(self, ctx):
        # Extract relevant info from the context.

        if self._is_lightcone:

            lc = ctx.get("lightcone")

            redshifts = lc.node_redshifts
            xHI = lc.global_xHI

        else:
            redshifts = self._core.redshift
            xHI = [np.mean(x.xH_box) for x in ctx.get("xHI")]

        if len(redshifts) < 3:
            raise ValueError("You cannot use the Planck prior likelihood with less than 3 redshifts")

        # Order the redshifts in increasing order
        redshifts, xHI = np.sort(np.array([redshifts, xHI]))

        # The linear interpolation/extrapolation function, taking as input the redshift supplied by the user and
        # the corresponding neutral fractions recovered for the specific EoR parameter set
        neutral_frac_func = InterpolatedUnivariateSpline(redshifts, xHI, k=1)

        # Perform extrapolation
        z_extrap = np.linspace(self.z_extrap_min, self.z_extrap_max, self.n_z_interp)
        xHI = neutral_frac_func(z_extrap)

        # Ensure that the neutral fraction does not exceed unity, or go negative
        np.clip(xHI, 0, 1, xHI)

        # Set up the arguments for calculating the estimate of the optical depth. Once again, performed using command
        # line code.
        # TODO: not sure if this works.
        tau_value = lib.compute_tau(z_extrap, xHI, ctx.get('cosmo_params'))

        return dict(tau=tau_value)


class LikelihoodNeutralFraction(LikelihoodBase):
    """
    A likelihood based on the measured neutral fraction at a range of redshifts.

    The log-likelihood statistic is a simple chi^2 if the model has xHI > threshold, and 0 otherwise.
    """
    threshold = 0.06

    def __init__(self, redshift=5.9, xHI=0.06, xHI_sigma=0.05):
        """
        Neutral fraction likelihood/prior.

        Note that the default parameters are based on McGreer et al. constraints
        Modelled as a flat, unity prior at x_HI <= 0.06, and a one sided Gaussian at x_HI > 0.06
        ( Gaussian of mean 0.06 and one sigma of 0.05 ).

        Limit on the IGM neutral fraction at z = 5.9, from dark pixels by I. McGreer et al.
        (2015) (http://adsabs.harvard.edu/abs/2015MNRAS.447..499M)

        Parameters
        ----------
        redshift : float or list of floats
            Redshift(s) at which the neutral fraction has been measured.
        xHI : float or list of floats
            Measured values of the neutral fraction, corresponding to `redshift`.
        xHI_sigma : float or list of floats
            One-sided uncertainty of measurements.
        """
        self.redshift = ensure_iter(redshift)
        self.xHI = ensure_iter(xHI)
        self.xHI_sigma = ensure_iter(xHI_sigma)

        self._require_spline = False

    def setup(self):
        self.lightcone_modules = [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, core.CoreLightConeModule)]
        self.coeval_modules = [m for m in self.LikelihoodComputationChain.getCoreModules() if isinstance(m, core.CoreCoevalModule)]

        if not self.lightcone_modules + self.coeval_modules:
            raise ValueError("LikelihoodNeutralFraction needs the CoreLightConeModule *or* CoreCoevalModule to be loaded.")

        if not self.lightcone_modules:
            # Get all unique redshifts from all coeval boxes in cores.
            self.redshifts = list(set(sum([x.redshift for x in self.coeval_modules], [])))

            for z in self.redshift:
                if z not in self.redshifts and len(self.redshifts) < 3:
                    raise ValueError("To use LikelihoodNeutralFraction, the core must be a lightcone, or coeval with >=3 redshifts, or containing the desired redshift")
                elif z not in self.redshifts:
                    self._require_spline = True

            self._use_coeval = True

        else:
            self._use_coeval = False
            self._require_spline = True

    def simulate(self, ctx):
        if self._use_coeval:
            xHI = np.array([np.mean(x) for x in ctx.get('xHI')])
            redshifts = self.redshifts
        else:
            xHI = ctx.get("lightcone").global_xHI
            redshifts = ctx.get("lightcone").node_redshifts

        redshifts, xHI = np.sort([redshifts, xHI])
        return dict(xHI=xHI, redshifts=redshifts)

    def computeLikelihood(self, model):
        lnprob = 0

        if self._require_spline:
            model_spline = InterpolatedUnivariateSpline(model['redshifts'], model['xHI'], k=1)

        for z, data, sigma in zip(self.redshift, self.xHI, self.xHI_sigma):
            if z in model['redshifts']:
                lnprob += self.lnprob(model['xHI'][self.redshifts.index(z)], data, sigma)
            else:
                lnprob += self.lnprob(model_spline(z), data, sigma)

        return lnprob

    def lnprob(self, model, data, sigma):
        model = np.clip(model, 0, 1)

        if model > self.threshold:
            return ((data - model) / sigma)**2
        else:
            return 0


class LikelihoodGreig(LikelihoodNeutralFraction, LikelihoodBaseFile):
    qso_redshift = 7.0842  # The redshift of the QSO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Read in data files.
        nf = np.load(path.expanduser(path.join("~", '.21CMMC', 'External_tables', "NeutralFractionsForPDF.npy")))
        pdf = np.load(path.expanduser(path.join("~", '.21CMMC', 'External_tables', "NeutralFractionPDF_SmallHII.npy")))

        # Normalising the PDF to have a peak probability of unity (consistent with how other priors are treated)
        # Ultimately, this step does not matter
        pdf /= np.amax(pdf)

        # Interpolate the QSO damping wing PDF
        self.spline_qso_damping_pdf = InterpolatedUnivariateSpline(nf, pdf)

    def computeLikelihood(self, model):
        """
        Constraints on the IGM neutral fraction at z = 7.1 from the IGM damping wing of ULASJ1120+0641
        Greig et al (2016) (http://arxiv.org/abs/1606.00441)
        """

        redshifts = model['redshifts']
        ave_nf = model['xHI']

        if self.qso_redshift in redshifts:
            nf_qso = redshifts.index(self.qso_redshift)

        elif len(redshifts) > 2:

            # Check the redshift range input by the user to determine whether to interpolate or extrapolate the IGM
            # neutral fraction to the QSO redshift
            if self.qso_redshift < np.min(redshifts):
                # The QSO redshift is outside the range set by the user. Need to extrapolate the reionisation history
                # to obtain the neutral fraction at the QSO redshift

                # The linear interpolation/extrapolation function, taking as input the redshift supplied by the user
                # and the corresponding neutral fractions recovered for the specific EoR parameter set
                global_nf_spl = InterpolatedUnivariateSpline(redshifts, ave_nf, k=1)

            else:
                # The QSO redshift is within the range set by the user. Can interpolate the reionisation history to
                # obtain the neutral fraction at the QSO redshift
                global_nf_spl = InterpolatedUnivariateSpline(redshifts, ave_nf, k=2 if len(redshifts)==3 else 3)

            nf_qso = global_nf_spl(self.qso_redshift)
        else:
            raise ValueError(
                """
                You cannot use the Greig prior likelihood with either less than 3 redshift or the redshift being 
                directly evaluated.""")

        # Ensure that the neutral fraction does not exceed unity, or go negative
        nf_qso = np.clip(nf_qso, 0, 1)

        qso_prob = self.spline_qso_damping_pdf(nf_qso)

        # Interpolating the PDF from the QSO damping wing might cause small negative values at the edges (i.e. x_HI ~ 0
        # or ~1) In case it is zero, or negative, set it to a very small non zero number (we take the log of this value,
        # it cannot be zero)
        # TODO: wouldn't it be better if we just returned -inf?
        if qso_prob <= 0.0:
            qso_prob = 0.000006

        # We work with the log-likelihood, therefore convert the IGM Damping wing PDF to log space
        return -2. * np.log(qso_prob)


class LikelihoodGlobalSignal(LikelihoodBaseFile):
    """
    A likelihood based on chi^2 comparison to Global Signal, where global signal is in mK as a function of MHz.
    """
    required_cores = [core.CoreLightConeModule]

    def simulate(self, ctx):
        return dict(
            frequencies=1420./(ctx.get("lightcone").node_redshifts+1),
            global_signal=ctx.get("lightcone").global_brightness_temp
        )

    def computeLikelihood(self, model):
        """
        Compute the likelihood, given the lightcone output from 21cmFAST.
        """
        model_spline = InterpolatedUnivariateSpline(model['frequencies'], model['global_signal'])

        lnl = -0.5 * np.sum((self.data['global_signal'] - model_spline(self.data['frequencies']))**2 / self.noise['sigma']**2)

        return lnl
import sys
import os
import subprocess
import keras
import json
import corner
import emcee
import warnings
import timeit
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import PyAstronomy.pyasl as pyasl
import paths_NN as ppp
import NN_wrapper_Hhe_split as fw
from scipy.signal import convolve
from scipy.signal import fftconvolve
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.optimize import minimize
from IPython.display import display, clear_output
from astropy.io import fits
from scipy import stats
from scipy.stats import binom

####################################################
## Importing and preparing all the imporant files ##
####################################################

# --- Load BLOeM spectrum ---
with fits.open(ppp.BLOeM_fits_path) as hdul:
    data = hdul[1].data

# Extract wavelength and flux from BLOeM
bloem_wl = data["WAVELENGTH"]   # Angstrom
bloem_flux = data["SCI_NORM"]   # Normalised flux


# --- Wavelength grid of Anja's emulator ---
df = pd.read_csv(ppp.master_wl_array_path)
master_wl_array = np.asarray(df['master wl Halpha_HeII6527 combined model 19002'])
order = np.argsort(master_wl_array)                                                 # indices to sort wavelength array
master_wl_ordered = master_wl_array[order]
master_wl_unique, unique_idx = np.unique(master_wl_ordered, return_index=True)      # deduplicated grid
wl_min, wl_max = master_wl_unique.min(), master_wl_unique.max()
UNIFORM_STEP = 0.2                                                                  # BLOeM resolution scale (~0.2 Å)
wl_uniform = np.arange(wl_min, wl_max, UNIFORM_STEP)

#import Anja's model for inference
model = keras.saving.load_model(ppp.keras_model_path)

# Load the normalization values
with open(ppp.norm_path) as f:
    normalization = json.load(f)


# --- Constants ---
cc = 2.99792458e10     #speed of light in a vacuum;
C_KMS = cc * 1e-5      #speed of light in km/s     
hh = 6.6260755e-27     #Planck constant
kk = 1.380658e-16      #Boltzmann constant
angstrom_to_cm = 1e-8  #Conversion factor from Angstrom to cm
_LMC_DISTANCE_RSUN = 50e3 * 3.08567758e18 / 6.96e10  # 50 kpc in solar radii


# --- Ks filter transmission ---
band = 'SPHERE_Ks'
filterfile = 'SPHERE_IRDIS_B_Ks.dat'
filterdir = ppp.filter_path
_KS_WAVE, _KS_TRANS = np.genfromtxt(filterdir + filterfile, comments='#').T
_KS_WAVE *= 10 # nm → Å
_KS_NORM = np.trapz(_KS_TRANS, _KS_WAVE) # Normalization factor for the filter transmission (integral over wavelength)
zpfile = filterdir + 'zero_points.dat'
zp_values = np.genfromtxt(zpfile, comments='#', dtype=str)
zp_system = 'vega' # Choose zero point system: 'vega', 'AB', or 'ST'
zpflux = ''
for afilter in zp_values:
    if afilter[0] == band:
        if zp_system == 'vega':
            zpflux = float(afilter[1])
        elif zp_system == 'AB':
            zpflux = float(afilter[2])
        elif zp_system == 'ST':
            zpflux = float(afilter[3])


# --- Parameter names (order must match theta vector throughout) ---
PARAM_NAMES = ['teff', 'logg', 'radius', 'logmdot', 'yhe', 'vsini']
PARAM_BOUNDS = np.array([
    [29000, 52000],  # teff
    [3.4,   4.3  ],  # logg
    [6,     21   ],  # radius
    [-7.5,  -5.2 ],  # logmdot
    [0.08,  0.15 ],  # yhe
    [50.0,  399.0],  # vsini
])

def clip_to_prior(theta):
    return np.clip(theta, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])

####################################################
########## Simulating bc nothing is real ###########
####################################################

# --- flux ---

def planck_wavelength(wave_angstrom, temp):
    ''' Calculate the Planck function as function of temperature and wavelength (in Angstrom. Output is then also in Angstrom).
    wave_angstrom: wavelength in Angstrom
    temp: temperature in Kelvin
    '''
    wave = wave_angstrom * angstrom_to_cm
    prefactor = 2.0 * hh * cc**2 / (wave**5)
    exponent = (hh * cc / kk) / (wave * temp)
    Blambda = prefactor * (1.0 / (np.exp(exponent)-1))
    Blambda = Blambda * angstrom_to_cm  #Blambda from per cm to per angstrom
    return Blambda

def flux_to_magnitude(obsflux):
    ''' Calculate magnitude from observed flux and zeropoint flux'''
    magnitude = -2.5 * np.log10(obsflux / zpflux)
    return magnitude

def compute_obs_flux(teff, radius, Tfrac=0.9, d=_LMC_DISTANCE_RSUN):
    ''' Calculate the observed flux in the K-band based on the given parameters.
    teff: effective temperature (K)
    radius: stellar radius (solar radii)
    Tfrac: fraction of teff to use for the blackbody calculation (default 0.9 to account for line formation in cooler layers)
    d: distance to the star (default 50 kpc in cm (distance to LMC), converted to solar radii)
    '''
    tBB = teff * Tfrac
    F_lambda = np.pi * planck_wavelength(_KS_WAVE, tBB)
    filtered_flux = np.trapz(_KS_TRANS * F_lambda, _KS_WAVE) / _KS_NORM
    return (radius / d)**2 * filtered_flux


# --- normalization ---

def normalize_theta(theta):
    mn = np.array([normalization[f'{n}_min'] for n in PARAM_NAMES])
    mx = np.array([normalization[f'{n}_max'] for n in PARAM_NAMES])
    return (theta - mn) / (mx - mn)


# --- rotational broadening baby ---

def _gray_kernel(v_grid, vsini, epsilon):
    """
    Evaluate the Gray rotational broadening profile on a uniform velocity grid
    and return a kernel normalised to **unit sum** (dimensionless).

    Parameters
    ----------
    v_grid : 1-D array  [km s⁻¹]
        Uniformly spaced velocity values, centred at 0.  The array length
        *must* be odd so that the kernel centre falls exactly on a grid point.
    vsini : float  [km s⁻¹]
        Projected equatorial rotation speed.
    epsilon : float
        Linear limb-darkening coefficient (0 <= e <= 1).

    Returns
    -------
    kernel : 1-D array
        Broadening profile normalised so that ``kernel.sum() == 1``.
        Convolve directly with a flux array; no extra ``* dv`` factor needed.
    """
    assert len(v_grid) % 2 == 1, "v_grid must have odd length (centred kernel)."

    x  = v_grid / vsini
    c1 = 2.0 * (1.0 - epsilon) / (np.pi * vsini * (1.0 - epsilon / 3.0))
    c2 = epsilon / (2.0 * vsini * (1.0 - epsilon / 3.0))

    kernel = np.zeros_like(v_grid, dtype=np.float64)
    inside = np.abs(x) < 1.0
    kernel[inside] = (c1 * np.sqrt(1.0 - x[inside] ** 2) + c2 * (1.0 - x[inside] ** 2))

    # Convert continuous density [per km/s] to discrete weights by
    # multiplying by the bin width, then renormalise for numerical safety.
    # After this step kernel.sum() == 1 exactly (up to floating-point noise).
    dv    = v_grid[1] - v_grid[0]
    kernel *= dv          # approximates integral of G(v) over each bin
    total  = kernel.sum()
    if total == 0.0:
        raise ValueError(
            f"Gray kernel is all zeros for vsini={vsini} km/s and "
            f"dv={dv:.4f} km/s.  Increase n_vel or oversample."
        )
    kernel /= total       # unit sum — convolve without extra factors
    return kernel

def vspace(wvl, flux, vsini, epsilon=0.6, n_vel=None, oversample=1):
    """
    Apply rotational broadening via a single convolution in velocity space.

    Parameters
    ----------
    wvl : array  [A]
        Wavelength array.  Does not need to be uniformly spaced.
    flux : array
        Flux array, same length as `wvl`.
    vsini : float  [km/s]
        Projected rotational velocity.
    epsilon : float
        Linear limb-darkening coefficient (0 to 1).  Default 0.6.
    n_vel : int, optional
        Number of points for the internal uniform log-lambda grid.  Defaults to
        ``len(wvl) * oversample``.  Must be large enough that the velocity
        pixel size ``dv = (ln lam_max - ln lam_min) / (n_vel-1) * c`` is
        smaller than ``vsini`` (otherwise the kernel degenerates to a delta).
    oversample : int
        Oversampling factor relative to ``len(wvl)`` when `n_vel` is not
        given.  Values > 1 improve accuracy when the input grid is coarse.

    Returns
    -------
    flux_broad : array
        Broadened flux, same length as `flux`.
    """
    wvl  = np.asarray(wvl,  dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if vsini <= 0.0:
        raise ValueError("vsini must be positive.")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be between 0 and 1.")
    if len(wvl) != len(flux):
        raise ValueError("wvl and flux must have the same length.")

    # Build a uniform grid in ln(lambda)  <->  velocity space
    ln_wvl = np.log(wvl)

    if n_vel is None:
        n_vel = len(wvl) * oversample

    ln_wvl_uniform = np.linspace(ln_wvl[0], ln_wvl[-1], n_vel)
    dv = (ln_wvl_uniform[1] - ln_wvl_uniform[0]) * C_KMS  # km/s per pixel

    # Interpolate flux onto the uniform ln-lambda grid
    interp_in    = interp1d(ln_wvl, flux, kind="linear", bounds_error=False, fill_value=(flux[0], flux[-1]))
    flux_uniform = interp_in(ln_wvl_uniform)

    # Build the kernel on a centred, odd-length velocity grid
    half_width = int(np.ceil(vsini / dv)) + 1                  # at least vsini/dv bins
    v_kernel   = np.arange(-half_width, half_width + 1) * dv   # odd length
    kernel     = _gray_kernel(v_kernel, vsini, epsilon)        # unit-sum

    # Reflection-pad to suppress edge ringing before convolving
    pad = len(kernel)
    flux_padded = np.pad(flux_uniform, pad, mode="reflect")
    conv_padded = fftconvolve(flux_padded, kernel, mode="same")
    flux_broad_uniform = conv_padded[pad:-pad]

    # Interpolate back onto the original wavelength grid
    interp_out = interp1d(ln_wvl_uniform, flux_broad_uniform, kind="linear", bounds_error=False, fill_value=(flux_broad_uniform[0], flux_broad_uniform[-1]))
    return interp_out(ln_wvl)

# JAX implementation of vspace (differentiable w.r.t. flux, vsini, epsilon)

def rotBroadJax(wvl, flux, vsini, vsini_max=399, epsilon=0.6, n_vel=None, oversample=1):
    """
    Differentiable rotational broadening via JAX.

    Gradients are available w.r.t. ``flux``, ``vsini``, and ``epsilon``
    using ``jax.grad`` / ``jax.jacobian`` / ``jax.value_and_grad``.

    Parameters
    ----------
    wvl : numpy array  [A]
        Wavelength array.  Treated as a **static constant** — pass plain
        NumPy, never a JAX tracer.  No gradient is computed for this arg.
    flux : jax array or array-like
        Flux array.  JAX will track gradients through this argument.
    vsini : float or scalar jax array  [km/s]
        Projected rotational velocity.  Differentiable.
    epsilon : float or scalar jax array
        Limb-darkening coefficient.  Differentiable.
    n_vel : int, optional
        Size of the internal uniform log-lambda grid.  Defaults to
        ``len(wvl) * oversample``.
    oversample : int
        Oversampling factor (used when `n_vel` is None).
    vsini_max : float
        A plain Python ``float`` (or anything that can be converted via
        ``float()`` **before** JAX begins tracing) used **only** to size
        the kernel grid.  It must be >= the largest ``vsini`` that will
        ever be evaluated.

        This parameter is **mandatory** and must always be a concrete
        number — never a JAX tracer.  The typical pattern is to pass the
        initial or maximum value of ``vsini`` as a bare Python float::

            rotBroadJax(wvl, flux, vsini=vsini_j,
                        epsilon=eps_j, n_vel=N,
                        vsini_max=float(vsini_init))

        For ``jax.jit``, pin it via ``functools.partial``::
            import functools, jax
            f = jax.jit(functools.partial(
                rotBroadJax, wvl, flux_template,
                n_vel=2000, vsini_max=150.0,
            ))
            result = f(vsini=jnp.array(30.0), epsilon=jnp.array(0.6))

    Returns
    -------
    flux_broad : jax array, same shape as `flux`.

    Notes
    -----
    The kernel is normalised to **unit sum** inside this function (using JAX
    ops, so normalisation is differentiable).  No extra scaling in the conv.

    Why ``vsini_max`` is needed
    ---------------------------
    JAX traces functions with *abstract* values that have a shape and dtype
    but no concrete number.  The kernel grid length (``2*half_width + 1``) is
    a Python ``int`` and must be known at trace time — it cannot depend on a
    traced ``vsini``.  The solution is to size the kernel with a plain float
    that is always >= the actual ``vsini``, so the grid is large enough for
    any value that will be evaluated during the trace.

    Examples
    --------
    >>> import jax, jax.numpy as jnp, numpy as np
    >>> from rotbroad_fast import rotBroadJax
    >>>
    >>> wvl  = np.linspace(5000, 5100, 1000)
    >>> flux = jnp.ones(1000).at[500].set(0.5)
    >>>
    >>> # jax.grad — vsini is concrete at call time, vsini_max not needed:
    >>> grad = jax.grad(
    ...     lambda vs: rotBroadJax(wvl, flux, vsini=vs, n_vel=1000).sum()
    ... )(jnp.array(30.0))
    >>>
    >>> # jax.jit — vsini is traced, so supply vsini_max:
    >>> f_jit = jax.jit(lambda vs: rotBroadJax(
    ...     wvl, flux, vsini=vs, n_vel=1000, vsini_max=150.0).sum()
    ... )
    >>> result = f_jit(jnp.array(30.0))
    """
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for rotBroadJax.  Install with:  pip install jax"
        ) from exc

    # ------------------------------------------------------------------
    # Static (non-traced) setup — plain Python / NumPy only, no JAX ops.
    #
    # IMPORTANT: nothing here may call float() / int() / np.ceil() on a
    # JAX tracer.  All shapes must be fully determined before tracing starts.
    # ------------------------------------------------------------------
    wvl_np    = np.asarray(wvl, dtype=np.float64)
    ln_wvl_np = np.log(wvl_np)

    if n_vel is None:
        n_vel = len(wvl_np) * oversample

    ln_wvl_uni_np = np.linspace(ln_wvl_np[0], ln_wvl_np[-1], n_vel)
    dv = float((ln_wvl_uni_np[1] - ln_wvl_uni_np[0]) * C_KMS)

    # vsini_max must be a plain Python float — never a JAX tracer.
    # It is used only to compute half_width (a Python int), which fixes
    # the kernel array shape at trace time.  The kernel *values* are
    # computed later with traced JAX ops and are fully differentiable.
    half_width  = int(np.ceil(vsini_max / dv)) + 2
    # Odd-length, static-shape kernel grid — the *values* will be traced,
    # but the *length* (2*half_width+1) is a plain Python int.
    v_kernel_np = np.arange(-half_width, half_width + 1, dtype=np.float64) * dv

    # Promote static arrays to JAX constants once
    ln_wvl_j      = jnp.array(ln_wvl_np)
    ln_wvl_uni_j  = jnp.array(ln_wvl_uni_np)
    v_kernel_j    = jnp.array(v_kernel_np)

    # Differentiable JAX pipeline starts here — all inputs may be traced, and all ops are JAX-compatible.
    flux_j = jnp.asarray(flux)

    # 1. Interpolate onto uniform log-lambda grid
    flux_uni = jnp.interp(ln_wvl_uni_j, ln_wvl_j, flux_j)

    # 2. Build the Gray kernel — fully differentiable in vsini and epsilon
    x  = v_kernel_j / vsini
    c1 = 2.0 * (1.0 - epsilon) / (jnp.pi * vsini * (1.0 - epsilon / 3.0))
    c2 = epsilon / (2.0 * vsini * (1.0 - epsilon / 3.0))

    kernel_density = jnp.where(
        jnp.abs(x) < 1.0,
        c1 * jnp.sqrt(jnp.clip(1.0 - x**2, 0.0)) + c2 * (1.0 - x**2),
        0.0,
    )

    # Convert density -> discrete weights (multiply by bin width), then
    # normalise to unit sum.  Both steps are differentiable.
    kernel = kernel_density * dv
    kernel = kernel / kernel.sum()   # unit sum — no extra factor in conv

    # 3. Convolve (jnp.convolve is differentiable via XLA)
    # Reflection-pad before convolving, then trim
    pad = len(kernel)
    flux_padded = jnp.pad(flux_uni, pad, mode="reflect")
    conv_padded = jnp.convolve(flux_padded, kernel, mode="same")
    flux_broad_uni = conv_padded[pad:-pad]

    # 4. Interpolate back onto the original grid
    return jnp.interp(ln_wvl_j, ln_wvl_uni_j, flux_broad_uni)


# --- Simulating the spectrum and adding noise ---

def simulate_model_spectrum(theta, broadening_method, model=model, limbdark=0.6, output_wl=None):
    ''' Simulate the model spectrum for given parameters, with optional rotational broadening and interpolation to an output wavelength grid.
    theta: array of parameters [teff, logg, radius, logmdot, yhe]
    broadening_method: methods to add rotational broadening (pyasl.rotBroad, pyasl.fastRotBroad, vspace or rotBroadJax)
    model: keras model to predict the spectrum
    vsini: projected rotational velocity (km/s) for rotational broadening (optional)
    limbdark: limb darkening coefficient for rotational broadening (optional)
    output_wl: wavelength grid to interpolate the final spectrum onto (optional; if None, returns on wl_uniform grid)
    '''
    theta_in_model=theta[:5]                                                                       # extract teff, logg, radius, logmdot, yhe
    norm_params = normalize_theta(theta_in_model)                                                  # normalize the input parameters to the [0,1] range expected by the model
    vsini=theta[5]                                                                                 # extract vsini
    flux_master = model(norm_params[None, :], training=False).numpy().ravel() 
    flux_master_ordered = flux_master[order]                                                       # reorder + deduplicate
    flux_unique = flux_master_ordered[unique_idx]                                                  # flux_master is now on the master_wl_unique grid
    flux_out = np.interp(wl_uniform,master_wl_unique,flux_unique)                                  # interpolate master → uniform grid
    if vsini == 0:
        flux_out = np.interp(output_wl,wl_uniform,flux_out) if output_wl is not None else flux_out # if no broadening, just interpolate to output grid if needed
    elif vsini > 0:
        if broadening_method == 'rotBroad':
            flux_out = pyasl.rotBroad(wl_uniform, flux_out, epsilon=limbdark, vsini=vsini)
        elif broadening_method == 'fastRotBroad':
            flux_out = pyasl.fastRotBroad(wl_uniform, flux_out, epsilon=limbdark, vsini=vsini)
        elif broadening_method == 'vspace':
            flux_out = vspace(wvl=wl_uniform, flux=flux_out, vsini=vsini, epsilon=limbdark)
        elif broadening_method == 'jax':
            flux_out = rotBroadJax(wvl=wl_uniform, flux=flux_out, vsini=vsini, vsini_max=399, epsilon=limbdark)
        if output_wl is not None:
            flux_out=np.interp(output_wl,wl_uniform,flux_out)                                      # interpolate master → output grid
    return flux_out

def add_noise_to_spectrum(prediction, spectral_snr):
    '''add Gaussian noise to the prediction'''
    spectral_sigma = 1/spectral_snr
    noisy_flux = prediction + np.random.normal(0.0, spectral_sigma, size=prediction.shape)
    return noisy_flux

def simulate_kband_magnitude(theta):
    ''' Simulate the K-band magnitude for given parameters using a simple model based on the Planck function and filter transmission.'''
    teff, logg, radius, logmdot, yhe, vsini = theta
    model_flux = compute_obs_flux(teff, radius)     # Calculate the observed flux in the K-band based on the given parameters
    kband_mag = flux_to_magnitude(model_flux)       # Convert the observed flux to a K-band magnitude
    return kband_mag



####################################################
####################### MCMC #######################
####################################################

def log_prior(theta):
    ''' Log-prior function that checks if the parameters are within the defined bounds. Returns 0 if within bounds (log(1)) and -inf if outside bounds (log(0)).'''
    if np.all((PARAM_BOUNDS[:, 0] < theta) & (theta < PARAM_BOUNDS[:, 1])):
        return 0.0
    return -np.inf

def make_log_posterior_with_kband(observed_wavelength,observed_flux,observed_kband,spectral_snr,broadening_method,kband_snr=20,limbdark=0.6,model=model):
    ''' Create a log-posterior function that combines the likelihood of the observed spectrum and the observed K-band magnitude, given the model predictions and uncertainties.
    model: keras model to predict the spectrum
    observed_wavelength: wavelength grid of the observed spectrum
    observed_flux: normalized observed spectrum
    observed_kband: observed K-band magnitude
    spectral_snr: signal-to-noise ratio of the observed spectrum (used to calculate the uncertainty for the spectral likelihood)
    broadening_method: which methid to use to broad the spectrum, options are: 'Sarah', 'rotBroad', 'fastRotBroad'
    kband_snr: signal-to-noise ratio of the K-band magnitude (used to calculate the uncertainty for the K-band likelihood)
    limbdark: limb darkening coefficient for rotational broadening
    '''
    spectral_sigma = 1.0 / spectral_snr                                                                                             # Use the spectral SNR to calculate the uncertainty for the spectral likelihood
    kband_sigma = 1.0 / kband_snr                                                                                                   # Use the K-band SNR to calculate the uncertainty

    def log_likelihood_kband(theta):
        sim_kband_mag = simulate_kband_magnitude(theta)                                                                             # Simulate the K-band magnitude for the given parameters using the model
        residual_kband = observed_kband - sim_kband_mag                                                                             # Calculate the residual between the observed and simulated K-band magnitudes
        ll_kband = -0.5 * (residual_kband / kband_sigma) ** 2                                                                       # Calculate the log-likelihood for the K-band magnitude assuming Gaussian errors
        return ll_kband if np.isfinite(ll_kband) else -np.inf                                                                       # Return -inf if the log-likelihood is not finite (e.g., due to numerical issues)

    def log_likelihood_lines(theta):
        sim_flux = simulate_model_spectrum(theta, broadening_method, model=model, limbdark=limbdark, output_wl=observed_wavelength) # Simulate the model spectrum with noise and optional rotational broadening
        valid = ~np.isnan(sim_flux)                                                                                                 # Only consider points where interpolation was valid (within the model's wavelength range)
        if not np.any(valid):                                                                                                       # If no valid points, return -inf to indicate zero likelihood
            return -np.inf

        residual_lines = observed_flux[valid] - sim_flux[valid]                                                                     # Calculate the residual between the observed and simulated spectra at the valid points
        ll_lines = -0.5 * np.sum((residual_lines / spectral_sigma) ** 2)                                                            # Calculate the log-likelihood for the spectral lines assuming Gaussian errors

        return ll_lines if np.isfinite(ll_lines) else -np.inf                                                                       # Return -inf if the log-likelihood is not finite (e.g., due to numerical issues)

    def log_posterior_with_kband(theta):
        lp = log_prior(theta)                                                                                                       # Calculate the log-prior for the given parameters
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood_lines(theta) + log_likelihood_kband(theta)                                                       # Combine the log-prior and log-likelihoods to get the log-posterior

    return log_posterior_with_kband

def run_mcmc_with_kband(observed_flux,observed_wavelength,observed_kband,spectral_snr,kband_snr,first_guess,broadening_method,
                        model=model,limbdark=0.6,ndim=6,nwalkers=32,nsteps=5000,theta_true = None):
    """
    Parameters
    ----------
    observed_flux       : array  — normalised observed spectrum
    observed_wavelength : array  — wavelength grid of observed spectrum (Å), also the grid on which the model spectrum will be evaluated and compared to the observed spectrum
    observed_kband      : float  — observed K-band magnitude
    spectral_snr        : float  — signal-to-noise ratio of the spectrum
    kband_snr           : float  — signal-to-noise ratio of the K-band magnitude
    first_guess         : array  — initial parameter guess [teff, logg, radius, logmdot, yhe, vsini]
    broadening_method   : string - which broadening method to use, either 'Sarah', 'rotBroad', or 'fastRotBroad'
    model               : keras model — neural network emulator
    limbdark            : limb darkening coefficient for rotational broadening (default 0.6)
    ndim                : int   — number of parameters (default 5)
    nwalkers            : int   — number of MCMC walkers (default 32)
    nsteps              : int   — number of MCMC steps (default 5000)
    theta_true          : array or None — ground truth parameters for validation plots

    Notes
    -----
    theta naming convention used throughout:
        first_guess  — user-supplied starting point
        theta_map    — MAP estimate from Nelder-Mead optimisation
        theta_50p    — median of posterior samples (returned as best estimate)
        theta_true   — known true values (only available for synthetic tests)
    """
    log_posterior = make_log_posterior_with_kband(model=model,observed_wavelength=observed_wavelength,observed_flux=observed_flux,observed_kband=observed_kband,spectral_snr=spectral_snr,
                broadening_method=broadening_method,kband_snr=kband_snr,limbdark=limbdark) # Create log-posterior function with the observed data (spectrum and kband magnitude) and model

    # MAP estimate
    call_count = [0]
    def objective(t):
        call_count[0] += 1
        if call_count[0] % 50 == 0:
            print(f"Nelder-Mead evaluation {call_count[0]}, theta={np.round(t, 3)}")
        return -log_posterior(t)

    res = minimize(objective, first_guess, method="Nelder-Mead",
                   options={"maxiter": 10_000, "xatol": 1.0, "fatol": 1.0})
    theta_map = res.x

    if not np.isfinite(log_posterior(theta_map)):
        print("Warning: MAP estimate outside prior bounds, falling back to first_guess.")
        theta_map = np.array(first_guess)
    theta_map = clip_to_prior(theta_map)

    print(f"Initial position (MAP estimate): {dict(zip(PARAM_NAMES, np.round(theta_map, 3)))}")

    # Initialise walkers
    scale = np.array([500, 0.05, 1, 0.1, 0.005, 1]) # Scale for initializing walkers around the MAP estimate (adjusted to be smaller than the prior range to ensure good starting positions)
    pos = []
    max_attempts = 100000
    attempts = 0
    while len(pos) < nwalkers:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not initialise {nwalkers} walkers after {max_attempts} attempts. "
                f"MAP estimate: {theta_map}. Check that log_posterior is finite near the starting point."
            )
        p = theta_map + scale * np.random.randn(ndim)
        p = clip_to_prior(p)
        if np.isfinite(log_posterior(p)):
            pos.append(p)
        attempts += 1
    print(f"Initialised {nwalkers} walkers in {attempts} attempts.")
    pos = np.array(pos)

    # Set up sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    print(f"Starting MCMC: {nwalkers} walkers × {nsteps} steps ({nwalkers * nsteps:,} total evaluations)")

    # Live chain plot
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        axes[i].set_ylabel(PARAM_NAMES[i])
        if theta_true is not None:
            axes[i].axhline(theta_true[i], color='red', linestyle='--', lw=1)
    axes[-1].set_xlabel("Step")
    display(fig)

    # Sampling loop
    checkpoint_interval = max(1, nsteps // 10)
    for i, _ in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i + 1) % checkpoint_interval == 0:
            acceptance = sampler.acceptance_fraction.mean()
            clear_output(wait=True)
            chain = sampler.get_chain()  # (steps_so_far, nwalkers, ndim)
            for i_param in range(ndim):
                axes[i_param].cla()
                axes[i_param].plot(chain[:, :, i_param], alpha=0.3, color='black', lw=0.5)
                axes[i_param].set_ylabel(PARAM_NAMES[i_param])
                if theta_true is not None:
                    axes[i_param].axhline(theta_true[i_param], color='red', linestyle='--', lw=1)
            axes[-1].set_xlabel("Step")
            fig.suptitle(f"Step {i+1}/{nsteps}  |  acceptance: {acceptance:.3f}")
            display(fig)
            plt.pause(0.01)
    print("Sampling complete.")

    # Post-processing
    chain = sampler.get_chain() # Extract the MCMC chain
    tau = sampler.get_autocorr_time(tol=0) # Estimate the autocorrelation time to determine how many steps to discard for burn-in
    max_tau = np.max(tau) # Use the maximum autocorrelation time across all parameters to be conservative in burn-in discarding
    if nsteps < 50 * max_tau:
        warnings.warn(
            f"Chain may not be converged: nsteps={nsteps} < 50 × max(tau)={50*max_tau:.0f}. "
            f"Consider running for at least {int(50*max_tau)} steps.",
            RuntimeWarning
        )
    discard = 3 * int(max_tau) # Discard 3 times the maximum autocorrelation time as burn-in
    thin = int(0.5*max_tau)
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True) # Flatten the chain after discarding burn-in and applying thinning
    print(f"\n--- Chain diagnostics ---")
    print(f"  Burn-in discarded : {discard} steps (3 × max τ)")
    print(f"  Thinning factor   : {thin}")
    print(f"  Samples remaining : {len(flat_samples)}")
    if len(flat_samples) < 1000:
        print("Warning: fewer than 1000 independent samples — consider running longer.")
    theta_50p = np.percentile(flat_samples, 50, axis=0) # Compute the median of the posterior samples for each parameter

    return {
        "observed_flux": observed_flux,
        "observed_wavelength": observed_wavelength,
        "observed_kband": observed_kband,
        "spectral_snr": spectral_snr,
        "first_guess": first_guess,
        "chain": chain,
        "flat_samples": flat_samples,
        "theta_map": theta_map,
        "theta_50p": theta_50p,
        "discard": discard,
        "theta_true": theta_true,
    }


# --- print, save, plot ---

def save_results(results, filename):
    np.savez(
        filename,
        observed_flux=results["observed_flux"],
        observed_wavelength=results["observed_wavelength"],
        observed_kband=results["observed_kband"],
        spectral_snr=results["spectral_snr"],
        first_guess=results["first_guess"],
        flat_samples=results["flat_samples"],
        theta_map=results["theta_map"],
        chain=results["chain"],
        theta_50p=results["theta_50p"],
        discard=results["discard"],
        theta_true=results["theta_true"] if "theta_true" in results else None,
    )

def load_results(filename):
    data = np.load(filename, allow_pickle=True)
    return {
        "observed_flux": data["observed_flux"],
        "observed_wavelength": data["observed_wavelength"],
        "observed_kband": data["observed_kband"],
        "spectral_snr": data["spectral_snr"],
        "first_guess": data["first_guess"],
        "flat_samples": data["flat_samples"],
        "theta_map": data["theta_map"],
        "chain": data["chain"],
        "discard": data["discard"],
        "theta_50p": data["theta_50p"],
        "theta_true": data["theta_true"] if "theta_true" in data else None,
    }

def print_posterior_summary(flat_samples, truths=None):
    print("Median and 1σ uncertainties:")
    for i, label in enumerate(PARAM_NAMES):
        p16, p50, p84 = np.percentile(flat_samples[:, i], [15.85, 50, 84.15]) # 16th, 50th, and 84th percentiles correspond to median and ±1σ for a Gaussian distribution
        print(f"{label} = {p50:.3f} -{p50-p16:.3f}/+{p84-p50:.3f}, true {label} = {truths[i]:.3f}" if truths is not None else f"{label} = {p50:.3f} -{p50-p16:.3f}/+{p84-p50:.3f}")

def plot_corner(flat_samples, truths=None):
    if truths is not None:
        corner.corner(flat_samples,labels=PARAM_NAMES, truths=truths)
    else:
        corner.corner(flat_samples,labels=PARAM_NAMES)
    plt.show()

def plot_chains(chain, ndim, discard, truths=None):
    fig, axes = plt.subplots(ndim, 1, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i].T, alpha=0.3, color='black')
        axes[i].axvline(discard, color='blue', linestyle='--', label='Discarded steps')
        if truths is not None:
            axes[i].axhline(truths[i], color='red', linestyle='-', label='True value')
        axes[i].set_ylabel(PARAM_NAMES[i])
        axes[i].legend()
    axes[-1].set_xlabel("Step")
    plt.show()

def plot_posterior_predictive(results,observed_flux,observed_wavelength,spectral_snr,broadening_method,model=model,limbdark=0.6, show_draws=True, n_draw=100):
    flat_samples = results["flat_samples"]
    theta_50p = results["theta_50p"]

    plt.figure(figsize=(15, 6))
    sim_50p_flux = simulate_model_spectrum(theta_50p, broadening_method=broadening_method, model=model, limbdark=limbdark, output_wl=observed_wavelength) # Simulate the 50th percentile theta spectrum with noise and optional rotational broadening
    sim_50p_flux = add_noise_to_spectrum(sim_50p_flux, spectral_snr) # Add noise to the simulated spectrum based on the specified SNR
    plt.plot(observed_wavelength, observed_flux, color='black', label='Observed spectrum', lw=0.8)
    if show_draws:
        idx = np.random.choice(len(flat_samples), n_draw, replace=False) # Randomly select n_draw samples from the posterior without replacement
        draw_samples = flat_samples[idx] # Extract the selected samples for plotting
        for i, theta in enumerate(draw_samples):
            sample_flux = simulate_model_spectrum(theta, broadening_method=broadening_method, model=model, limbdark=limbdark, output_wl=observed_wavelength) # Simulate the spectrum for this sample with noise and optional rotational broadening
            sample_flux = add_noise_to_spectrum(sample_flux, spectral_snr) # Add noise to the simulated spectrum based on the specified SNR
            plt.plot(observed_wavelength,sample_flux,color='tab:blue',alpha=0.25,linewidth=1,label=f"Posterior Predictive {n_draw} samples" if i == 0 else None)
    plt.plot(observed_wavelength, sim_50p_flux, color='orange', alpha=0.8, lw=1, label='Median posterior predictive')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Normalized Flux")
    plt.legend()
    plt.show()


####################################################
################ MCMC coverage test ################
####################################################

def get_credible_interval_mcmc(flat_samples, truth, param_index):
    """
    Compute the posterior percentile rank of the truth value for one parameter using unweighted MCMC samples.
    ----------
    flat_samples : array, shape (Nsamples, ndim)
        Posterior samples from emcee.
    truth : float
        True parameter value.
    param_index : int
        Which parameter column to use.
    -------
    float
        Fraction of posterior samples below the truth value.
    """
    samples_1d = flat_samples[:, param_index] # extract one parameter
    return np.mean(samples_1d < truth) # fraction of samples below truth

def run_coverage_test(wl_array_output, spectral_snr, kband_snr, broadening_method, N_sims=20, nsteps=5000, nwalkers=32, ndims=6, simulate=True, theta_true=None, model=model, limbdark=0.6):
    """
    Run a Bayesian coverage test (PP plot test) for all 5 parameters using repeated noisy simulations.
    ----------
    wl_array : array
        Model wavelength grid.
    spectral_snr : float
        Noise level used for simulations of spectra and inference.
    kband_snr : float
        Noise level used for simulations of kband magnitude and inference.
    broadening_method : string
        Which broadening method to use, either 'Sarah', 'rotBroad', or 'fastRotBroad'.
    N_sims : int
        Number of repeated simulations.
    nsteps, nwalkers, ndims: int
        MCMC control parameters.
    simulate : bool
        If True, simulate new true parameters for each run. If False, use provided theta_true for all runs. theta_true : array-like, shape (6,) If simulate=False, the true parameter values to use for all simulations.
    theta_true : array-like, shape (N_sims, 6) or (6,)
        If simulate=False, the true parameter values to use for all simulations. If shape is (6,), the same true parameters will be used for all simulations. If shape is (N_sims, 6), each row will be used as the true parameters for one simulation.
    model : keras model
        Your neural network spectral emulator.
    limbdark : float
        limbdarkening.
    -------
    credible_dict : dict
        Credible ranks for each parameter.
    """
    # Sample N true values for each parameter from the prior range
    if simulate:
        theta_true = np.column_stack([np.random.uniform(lo, hi, size=N_sims) for lo, hi in PARAM_BOUNDS])

    # Generate one random first guess per simulation
    guesses = np.column_stack([np.random.uniform(lo, hi, size=N_sims) for lo, hi in PARAM_BOUNDS])

    credible_dict = {name: [] for name in PARAM_NAMES} # Storage: one list per parameter
    flat_samples_all, chain_all, discard_all, results_all = [], [], [], [] # Storage for all flat samples, chains, discards, and results
    print(f"Running coverage test with N_sims = {N_sims}")

    for k in range(N_sims): # Loop over repeated simulations

        print(f"Simulation {k+1}/{N_sims}")

        observed_flux = simulate_model_spectrum(theta_true[k],broadening_method=broadening_method, model=model,limbdark=limbdark,output_wl=wl_array_output) # Simulate the observed spectrum for the true parameters on the output wavelength grid
        observed_flux = add_noise_to_spectrum(observed_flux, spectral_snr) # Add noise to the simulated spectrum based on the specified SNR
        observed_kband = simulate_kband_magnitude(theta_true[k]) + np.random.normal(0, 1/kband_snr) # Simulate the observed K-band magnitude for the true parameters and add noise based on the specified K-band SNR

        # Run inference
        results = run_mcmc_with_kband(
            observed_flux=observed_flux,
            observed_wavelength=wl_array_output,
            observed_kband=observed_kband,
            spectral_snr=spectral_snr,
            kband_snr=kband_snr,
            first_guess=guesses[k],
            broadening_method = broadening_method,
            model=model,
            limbdark=limbdark,
            ndim=ndims,
            nwalkers=nwalkers,
            nsteps=nsteps,
            theta_true = theta_true[k],
        )

        results_all.append(results)
        discard_all.append(results["discard"])
        chain_all.append(results["chain"])
        flat_samples_all.append(results["flat_samples"])

        for j, name in enumerate(PARAM_NAMES): # Compute credible ranks for each parameter

            u = get_credible_interval_mcmc(results["flat_samples"],truth=theta_true[k,j],param_index=j)
            credible_dict[name].append(u)

    return credible_dict, theta_true, flat_samples_all, guesses, discard_all, chain_all, results_all

#def plot_mcmc_sigma_regions(coverage, param_names, )

def plot_coverage_all_params(credible_dict):
    """
    Plot one PP coverage curve for each parameter, including expected binomial scatter bands.
    """

    fig, ax = plt.subplots(figsize=(7, 7))
    x_values = np.linspace(0.0, 1.0, 1001)
    ax.plot([0, 1], [0, 1], "--", color="black") # Ideal diagonal

    # Number of simulations
    first_param = list(credible_dict.keys())[0]
    N = len(credible_dict[first_param])

    # Expected statistical scatter bands
    bands = [0.68, 0.95, 0.997]
    band_alpha = [0.3, 0.15, 0.1]

    for ci, alpha in zip(bands, band_alpha):
            edge = (1.0 - ci) / 2.0 # Two-tailed
            lower = binom.ppf(edge, N, x_values) / N # Lower bound of binomial confidence interval
            upper = binom.ppf(1 - edge, N, x_values) / N # Upper bound of binomial confidence interval
            lower[0] = 0 # Ensure the bands start at (0,0) and end at (1,1)
            upper[0] = 0
            ax.fill_between(x_values, lower, upper, alpha=alpha, color="grey") # Plot binomial confidence bands

    for name in PARAM_NAMES: # Plot each parameter curve
        credible_intervals = np.asarray(credible_dict[name]) # Extract credible intervals
        pp = np.mean(credible_intervals[:, None] < x_values[None, :], axis=0) # Empirical coverage
        ax.plot(x_values, pp, lw=2, label=name) # Plot curve

    ax.set_xlabel("Credible interval")
    ax.set_ylabel("Fraction of events in credible interval")
    ax.set_title(f"Coverage test (PP plot) with {N} simulations")
    ax.legend()
    plt.show()
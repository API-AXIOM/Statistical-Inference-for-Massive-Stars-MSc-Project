import sys
import os
from emulator_inference import emulate_hg_spectrum, merge_line_spectra
try:
    from emulator_inference import resolve_device
except ImportError:
    def resolve_device(device=None):
        import torch
        if device is None or str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA was requested but torch.cuda.is_available() is False; using CPU.")
            return torch.device("cpu")
        return resolved
import json
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from IPython.display import display, clear_output
from scipy.stats import binom
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'KIWI-GA'))
try:
    import paths_NN as ppp
except ModuleNotFoundError:
    ppp = None

try:
    import ga_notebook_tools as gnt
except ModuleNotFoundError:
    gnt = None

try:
    import NN_wrapper_Hhe_split as fw
except ModuleNotFoundError:
    fw = None


def _require_paths_nn():
    if ppp is None:
        raise ModuleNotFoundError(
            "paths_NN.py is required for this helper. Put paths_NN.py on the "
            "notebook Python path, or avoid functions that save/load Anja/coverage paths."
        )
    return ppp

####################################################
## Importing and preparing all the imporant files ##
####################################################

bloem_wl = None
bloem_flux = None


def load_bloem_spectrum():
    """Load BLOeM spectrum lazily because it is not needed for every run."""
    global bloem_wl, bloem_flux

    if bloem_wl is not None and bloem_flux is not None:
        return bloem_wl, bloem_flux

    from astropy.io import fits

    paths = _require_paths_nn()
    with fits.open(paths.BLOeM_fits_path) as hdul:
        data = hdul[1].data

    bloem_wl = data["WAVELENGTH"]
    bloem_flux = data["SCI_NORM"]
    return bloem_wl, bloem_flux


model = None
normalization = None
order = None
unique_idx = None
master_wl_unique = None
wl_uniform = None


def load_anja_resources():
    """Load Anja-only resources lazily so Vasilis runs do not require Keras."""
    global model, normalization, order, unique_idx, master_wl_unique, wl_uniform

    if model is not None:
        return

    import keras

    paths = _require_paths_nn()
    df = pd.read_csv(paths.master_wl_array_path)
    master_wl_array = np.asarray(df['master wl Halpha_HeII6527 combined model 19002'])
    order = np.argsort(master_wl_array)
    master_wl_ordered = master_wl_array[order]
    master_wl_unique, unique_idx = np.unique(master_wl_ordered, return_index=True)
    wl_min, wl_max = master_wl_unique.min(), master_wl_unique.max()
    wl_uniform = np.arange(wl_min, wl_max, 0.2)

    model = keras.saving.load_model(paths.keras_model_path)
    with open(paths.norm_path) as f:
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
zp_system = 'vega' # Choose zero point system: 'vega', 'AB', or 'ST'
_KS_WAVE = None
_KS_TRANS = None
_KS_NORM = None
zpflux = None


def load_ks_filter():
    global _KS_WAVE, _KS_TRANS, _KS_NORM, zpflux

    if _KS_WAVE is not None:
        return _KS_WAVE, _KS_TRANS, _KS_NORM, zpflux

    paths = _require_paths_nn()
    filterdir = paths.filter_path
    _KS_WAVE, _KS_TRANS = np.genfromtxt(filterdir + filterfile, comments='#').T
    _KS_WAVE *= 10 # nm -> Angstrom
    _KS_NORM = np.trapz(_KS_TRANS, _KS_WAVE)
    zpfile = filterdir + 'zero_points.dat'
    zp_values = np.genfromtxt(zpfile, comments='#', dtype=str)
    for afilter in zp_values:
        if afilter[0] == band:
            if zp_system == 'vega':
                zpflux = float(afilter[1])
            elif zp_system == 'AB':
                zpflux = float(afilter[2])
            elif zp_system == 'ST':
                zpflux = float(afilter[3])
    return _KS_WAVE, _KS_TRANS, _KS_NORM, zpflux


# --- Parameter names (order must match theta vector throughout) ---
PARAM_NAMES_ANJA = ['teff', 'logg', 'radius', 'logmdot', 'yhe', 'vsini']
PARAM_BOUNDS_ANJA = np.array([
    [29000, 52000],  # teff
    [3.4,   4.3  ],  # logg
    [6,     21   ],  # radius
    [-7.5,  -5.2 ],  # logmdot
    [0.08,  0.15 ],  # yhe
    [0.0,  399.0],  # vsini
])

# Fixed parameters for Vasilis' emulator (set to None to free them)
VASILIS_FIXED = {
    "vinf":  2274.4,   # fix v_inf
    "vturb": 10.0,     # fix v_turb
}
VASILIS_EMULATOR_DIR = "emulators_per_line_hg"
VASILIS_DEVICE = str(resolve_device("auto"))
VASILIS_WAVELENGTHS_BY_LINE = None


def set_vasilis_fixed(vinf, vturb=10.0):
    """Set model-specific fixed parameters for the 6D Vasilis inference vector."""
    VASILIS_FIXED["vinf"] = float(vinf)
    VASILIS_FIXED["vturb"] = float(vturb)


def set_vasilis_device(device="auto"):
    """Set the device used by Vasilis' PyTorch per-line emulators."""
    global VASILIS_DEVICE
    VASILIS_DEVICE = str(resolve_device(device))
    return VASILIS_DEVICE


def set_vasilis_wavelengths_by_line(wavelengths_by_line=None):
    """Set fixed per-line wavelength grids for Vasilis' emulator calls."""
    global VASILIS_WAVELENGTHS_BY_LINE
    if wavelengths_by_line is None:
        VASILIS_WAVELENGTHS_BY_LINE = None
    else:
        VASILIS_WAVELENGTHS_BY_LINE = {
            str(name): np.asarray(wavelengths, dtype=np.float32)
            for name, wavelengths in wavelengths_by_line.items()
        }
    return VASILIS_WAVELENGTHS_BY_LINE

PARAM_NAMES_VASILIS  = ['teff', 'logg', 'radius', 'logmdot', 'yhe', 'vsini']  # 6 free params
PARAM_BOUNDS_VASILIS = np.array([
    [29000, 52000],  #teff
    [3.4,   4.3  ],  #logg
    [6,     21   ],  #radius
    [-7.5,  -5.2 ],  #logmdot
    [0.08,  0.15 ],  #yhe
    [0.0,   399.0],  #vsini
])

WALKER_SCALE = {
    "anja":    np.array([500, 0.05, 1, 0.1, 0.005, 1]),
    "vasilis": np.array([500, 0.05, 1, 0.1, 0.005, 1]),
}

GA_NAME_MAP = {
    "teff"   : "teff",
    "logg"   : "logg",
    "radius" : None,
    "logmdot": "mdot",
    "mdot"   : "mdot",
    "yhe"    : "yhe",
    "vsini"  : "vrot",
    "vinf"   : "vinf",
    "vturb"  : "vturb",
}

# Active set — change this one line to switch emulator throughout
ACTIVE_EMULATOR = "vasilis"   # "anja" or "vasilis"

PARAM_NAMES  = PARAM_NAMES_ANJA  if ACTIVE_EMULATOR == "anja" else PARAM_NAMES_VASILIS
PARAM_BOUNDS = PARAM_BOUNDS_ANJA if ACTIVE_EMULATOR == "anja" else PARAM_BOUNDS_VASILIS



def clip_to_prior(theta):
    return np.clip(theta, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])

####################################################
########## Simulating bc nothing is real ###########
####################################################

# --- normalization ---

def normalize_theta(theta):
    load_anja_resources()
    mn = np.array([normalization[f'{n}_min'] for n in PARAM_NAMES[:5]])
    mx = np.array([normalization[f'{n}_max'] for n in PARAM_NAMES[:5]])
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

def apply_rotational_broadening(wl, flux, broadening_method, vsini, limbdark=0.6):
    """Apply rotational broadening on one continuous wavelength segment."""
    if vsini <= 0:
        return flux

    if broadening_method == 'rotBroad':
        import PyAstronomy.pyasl as pyasl
        return pyasl.rotBroad(wl, flux, epsilon=limbdark, vsini=vsini)
    if broadening_method == 'fastRotBroad':
        import PyAstronomy.pyasl as pyasl
        return pyasl.fastRotBroad(wl, flux, epsilon=limbdark, vsini=vsini)
    if broadening_method == 'vspace':
        return vspace(wvl=wl, flux=flux, vsini=vsini, epsilon=limbdark)
    if broadening_method == 'jax':
        return np.asarray(rotBroadJax(wvl=wl, flux=flux, vsini=vsini,
                                      vsini_max=399, epsilon=limbdark))

    raise ValueError(
        "Unknown broadening_method. Choose 'rotBroad', 'fastRotBroad', 'vspace', or 'jax'."
    )


def simulate_model_spectrum(theta, broadening_method, emulator=ACTIVE_EMULATOR, limbdark=0.6, output_wl=None,
                            wavelengths_by_line=None, device=None):
    """
    Simulate a model spectrum for given parameters.

    Parameters
    ----------
    theta            : array
        For emulator="anja"    : [teff, logg, radius, logmdot, yhe, vsini]
        For emulator="vasilis" : [teff, logg, radius, logmdot, vinf, yhe, vturb, vsini]
    broadening_method : str — 'rotBroad', 'fastRotBroad', 'vspace', or 'jax'
    emulator          : str — 'anja' or 'vasilis'
    limbdark          : float
    output_wl         : array or None
    wavelengths_by_line : dict or None — fixed per-line grids for emulator="vasilis"
    device           : str or None — PyTorch device for emulator="vasilis"
    """

    if emulator == "anja":
        load_anja_resources()
        theta_in_model = theta[:5]
        norm_params    = normalize_theta(theta_in_model)
        vsini          = theta[5]

        flux_master         = model(norm_params[None, :], training=False).numpy().ravel()
        flux_master_ordered = flux_master[order]
        flux_unique         = flux_master_ordered[unique_idx]
        flux_out            = np.interp(wl_uniform, master_wl_unique, flux_unique)
        wl_out              = wl_uniform

    elif emulator == "vasilis":
        theta = np.asarray(theta, dtype=float)
        if theta.size == 6:
            teff, logg, radius, logmdot, yhe, vsini = theta
            vinf = VASILIS_FIXED["vinf"]
            vturb = VASILIS_FIXED["vturb"]
        elif theta.size == 8:
            teff, logg, radius, logmdot, vinf, yhe, vturb, vsini = theta
        else:
            raise ValueError(
                "For emulator='vasilis', theta must be either "
                "[teff, logg, radius, logmdot, yhe, vsini] with "
                "VASILIS_FIXED set from INDAT, or "
                "[teff, logg, radius, logmdot, vinf, yhe, vturb, vsini]."
            )

        params = {
            "Teff":   teff,
            "logg":   logg,
            "R":      radius,
            "Mdot":   10**logmdot,
            "v_inf":  vinf,
            "Y_He":   yhe,
            "v_turb": vturb,
        }
        if wavelengths_by_line is None:
            wavelengths_by_line = VASILIS_WAVELENGTHS_BY_LINE
        if device is None:
            device = VASILIS_DEVICE

        results = emulate_hg_spectrum(
            params,
            emulator_dir=VASILIS_EMULATOR_DIR,
            wavelengths_by_line=wavelengths_by_line,
            device=device,
        )

        if vsini > 0:
            results = {
                line_name: {
                    **line_data,
                    "flux": apply_rotational_broadening(
                        wl=line_data["wavelength"],
                        flux=line_data["flux"],
                        broadening_method=broadening_method,
                        vsini=vsini,
                        limbdark=limbdark,
                    ),
                }
                for line_name, line_data in results.items()
            }

        if output_wl is None:
            if wavelengths_by_line is not None:
                output_wl_for_merge = np.unique(
                    np.concatenate([np.asarray(w, dtype=np.float32) for w in wavelengths_by_line.values()])
                )
            else:
                all_wl = np.concatenate([d["wavelength"] for d in results.values()])
                output_wl_for_merge = np.linspace(all_wl.min(), all_wl.max(), 5000)
        else:
            output_wl_for_merge = output_wl

        merged = merge_line_spectra(results, output_wavelength=output_wl_for_merge, fill_value=1.0)
        wl_out = merged["wavelength"]
        flux_out = merged["flux"]

    else:
        raise ValueError(f"Unknown emulator '{emulator}'. Choose 'anja' or 'vasilis'.")

    # Anja's emulator is a single continuous spectrum. Vasilis' per-line
    # spectra are broadened before merging to avoid broadening across gaps.
    if vsini > 0 and emulator == "anja":
        flux_out = apply_rotational_broadening(
            wl=wl_out,
            flux=flux_out,
            broadening_method=broadening_method,
            vsini=vsini,
            limbdark=limbdark,
        )

    if output_wl is not None and emulator != "vasilis":
        flux_out = np.interp(output_wl, wl_out, flux_out)

    return flux_out

def add_noise_to_spectrum(prediction, spectral_snr):
    '''add Gaussian noise to the prediction'''
    spectral_sigma = 1/spectral_snr
    noisy_flux = prediction + np.random.normal(0.0, spectral_sigma, size=prediction.shape)
    return noisy_flux

def simulate_kband_magnitude(theta, kband_unc=1/10):
    ''' Simulate the K-band magnitude for given parameters using a simple model based on the Planck function and filter transmission.'''
    if gnt is None:
        raise ModuleNotFoundError(
            "ga_notebook_tools.py is required for simulate_kband_magnitude(). "
            "Put ga_notebook_tools.py on the notebook Python path, or skip the K-band likelihood."
        )
    if isinstance(theta, dict):
        teff   = theta["Teff"]
        radius = theta["R"]
    else:
        teff   = theta[0]
        radius = theta[2]
    model_flux = gnt.compute_obs_flux(teff, radius)      # Calculate the observed flux in the K-band based on the given parameters
    kband_mag = gnt.flux_to_magnitude(model_flux)        # Convert the observed flux to a K-band magnitude
    kband_noise = np.random.normal(0.0, kband_unc) # Calculate the noise level for the K-band magnitude based on the specified uncertainty (kband_unc)
    kband_mag_noisy = kband_mag + kband_noise  # Add Gaussian noise to the K-band magnitude based on the specified uncertainty
    return kband_mag, kband_mag_noisy, kband_noise  # Return both the noisy K-band magnitude and the noise on the K-band magnitude for reference



####################################################
################ Priors & Posteriors ###############
####################################################

def log_prior(theta):
    ''' Log-prior function that checks if the parameters are within the defined bounds. Returns 0 if within bounds (log(1)) and -inf if outside bounds (log(0)).'''
    if np.all((PARAM_BOUNDS[:, 0] <= theta) & (theta <= PARAM_BOUNDS[:, 1])):
        return 0.0
    return -np.inf

def make_log_likelihood_with_kband(observed_wavelength, observed_flux, observed_kband,
                                    spectral_snr, broadening_method,
                                    emulator=ACTIVE_EMULATOR, kband_unc=1/10, limbdark=0.6,
                                    wavelengths_by_line=None, device=None):
    spectral_sigma = 1.0 / spectral_snr

    # Capture these at closure-creation time, not at call time.
    # For Vasilis: fall back to module globals only if not explicitly passed.
    _wavelengths_by_line = wavelengths_by_line if wavelengths_by_line is not None \
                           else VASILIS_WAVELENGTHS_BY_LINE
    _device = device if device is not None else VASILIS_DEVICE

    def log_likelihood_kband(theta):
        sim_kband_mag, _, _ = simulate_kband_magnitude(theta, kband_unc=kband_unc)
        residual_kband = observed_kband - sim_kband_mag
        ll_kband = -0.5 * (residual_kband / kband_unc) ** 2
        return ll_kband if np.isfinite(ll_kband) else -np.inf

    def log_likelihood_lines(theta):
        sim_flux = simulate_model_spectrum(
            theta, broadening_method,
            emulator=emulator, limbdark=limbdark,
            output_wl=observed_wavelength,
            wavelengths_by_line=_wavelengths_by_line,   # explicit, not re-read from global
            device=_device,
        )
        valid = ~np.isnan(sim_flux)
        if not np.any(valid):
            return -np.inf
        residual_lines = observed_flux[valid] - sim_flux[valid]
        ll_lines = -0.5 * np.sum((residual_lines / spectral_sigma) ** 2)
        return ll_lines if np.isfinite(ll_lines) else -np.inf

    use_kband = observed_kband is not None and kband_unc is not None

    def log_likelihood(theta):
        ll = log_likelihood_lines(theta)
        if not np.isfinite(ll):
            return -np.inf
        if use_kband:
            ll += log_likelihood_kband(theta)
        return ll

    return log_likelihood


def make_log_posterior_with_kband(observed_wavelength, observed_flux, observed_kband,
                                   spectral_snr, broadening_method,
                                   emulator=ACTIVE_EMULATOR, kband_unc=1/10, limbdark=0.6,
                                   wavelengths_by_line=None, device=None):
    log_likelihood = make_log_likelihood_with_kband(
        observed_wavelength  = observed_wavelength,
        observed_flux        = observed_flux,
        observed_kband       = observed_kband,
        spectral_snr         = spectral_snr,
        broadening_method    = broadening_method,
        emulator             = emulator,
        kband_unc            = kband_unc,
        limbdark             = limbdark,
        wavelengths_by_line  = wavelengths_by_line,
        device               = device,
    )

    def log_posterior_with_kband(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    return log_posterior_with_kband

####################################################
####################### MCMC #######################
####################################################

def run_mcmc_with_kband(observed_flux,observed_wavelength,observed_kband,spectral_snr,kband_unc,first_guess,broadening_method,
                        emulator=ACTIVE_EMULATOR,limbdark=0.6,ndim=None,nwalkers=24,nsteps=3000,theta_true = None, wavelengths_by_line=None, device=None):
    """
    Parameters
    ----------
    observed_flux       : array  — normalised observed spectrum
    observed_wavelength : array  — wavelength grid of observed spectrum (Å), also the grid on which the model spectrum will be evaluated and compared to the observed spectrum
    observed_kband      : float  — observed K-band magnitude
    spectral_snr        : float  — signal-to-noise ratio of the spectrum
    kband_unc           : float  — uncertainty of the K-band magnitude
    first_guess         : array  — initial parameter guess [teff, logg, radius, (log)mdot, yhe, vsini]
    broadening_method   : string - which broadening method to use, either 'Sarah', 'rotBroad', or 'fastRotBroad'
    emulator            : string — active neural network emulator
    limbdark            : limb darkening coefficient for rotational broadening (default 0.6)
    ndim                : int   — number of parameters
    nwalkers            : int   — number of MCMC walkers (default 24)
    nsteps              : int   — number of MCMC steps (default 3000)
    theta_true          : array or None — ground truth parameters for validation plots

    Notes
    -----
    theta naming convention used throughout:
        first_guess  — user-supplied starting point
        theta_map    — MAP estimate from Nelder-Mead optimisation
        theta_50p    — median of posterior samples (returned as best estimate)
        theta_true   — known true values (only available for synthetic tests)
    """
    import emcee

    start = time.perf_counter()
    log_posterior = make_log_posterior_with_kband(observed_wavelength=observed_wavelength,observed_flux=observed_flux,observed_kband=observed_kband,spectral_snr=spectral_snr,
                broadening_method=broadening_method,emulator=emulator,kband_unc=kband_unc,limbdark=limbdark, wavelengths_by_line=wavelengths_by_line,device= device,) # Create log-posterior function with the observed data (spectrum and kband magnitude) and model

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
    if ndim is None:
        ndim = len(PARAM_BOUNDS_ANJA) if emulator == "anja" else len(PARAM_BOUNDS_VASILIS)
    scale = WALKER_SCALE[emulator] # Scale for initializing walkers around the MAP estimate (adjusted to be smaller than the prior range to ensure good starting positions)
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

    # Sampling loop
    checkpoint_interval = max(1, nsteps // 10)
    display_handle = None
    for i, _ in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i + 1) % checkpoint_interval == 0:
            acceptance = sampler.acceptance_fraction.mean()
            chain = sampler.get_chain()  # (steps_so_far, nwalkers, ndim)
            for i_param in range(ndim):
                axes[i_param].cla()
                axes[i_param].plot(chain[:, :, i_param], alpha=0.3, color='black', lw=0.5)
                axes[i_param].set_ylabel(PARAM_NAMES[i_param])
                if theta_true is not None:
                    axes[i_param].axhline(theta_true[i_param], color='red', linestyle='--', lw=1)
            axes[-1].set_xlabel("Step")
            fig.suptitle(f"Step {i+1}/{nsteps}  |  acceptance: {acceptance:.3f}")
            if display_handle is None:
                display_handle = display(fig, display_id=True)
            else:
                display_handle.update(fig)
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
    elapsed = time.perf_counter() - start
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
        "runtime_seconds": elapsed,
    }


# --- print, save, plot ---

def save_results(results, run, filename):
    np.savez(
        os.path.join(ppp.MCMC_results_path, run, filename),
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
        runtime_seconds=results["runtime_seconds"],
        theta_true=results["theta_true"] if "theta_true" in results else None,
        allow_pickle=True,
    )

def load_results(run, filename):
    data = np.load(os.path.join(ppp.MCMC_results_path, run, filename), allow_pickle=True)
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
        "runtime_seconds": data["runtime_seconds"],
        "theta_true": data["theta_true"] if "theta_true" in data else None,
    }

def print_posterior_summary(flat_samples, runtime_seconds,truths=None):
    print(f"Runtime {runtime_seconds:.1f} seconds")
    print("Median and 1σ uncertainties:")
    for i, label in enumerate(PARAM_NAMES):
        p16, p50, p84 = np.percentile(flat_samples[:, i], [15.85, 50, 84.15]) # 16th, 50th, and 84th percentiles correspond to median and ±1σ for a Gaussian distribution
        print(f"{label} = {p50:.3f} -{p50-p16:.3f}/+{p84-p50:.3f}, true {label} = {truths[i]:.3f}" if truths is not None else f"{label} = {p50:.3f} -{p50-p16:.3f}/+{p84-p50:.3f}")

def plot_corner(flat_samples, truths=None):
    import corner

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

def plot_posterior_predictive(results,observed_flux,observed_wavelength,spectral_snr,broadening_method,emulator=ACTIVE_EMULATOR, limbdark=0.6, show_draws=True, n_draw=100):
    flat_samples = results["flat_samples"]
    theta_50p = results["theta_50p"]

    plt.figure(figsize=(15, 6))
    sim_50p_flux = simulate_model_spectrum(theta_50p, broadening_method=broadening_method, emulator=emulator, limbdark=limbdark, output_wl=observed_wavelength) # Simulate the 50th percentile theta spectrum with noise and optional rotational broadening
    #sim_50p_flux = add_noise_to_spectrum(sim_50p_flux, spectral_snr) # Add noise to the simulated spectrum based on the specified SNR
    plt.plot(observed_wavelength, observed_flux, color='black', label='Observed spectrum', lw=0.8)
    if show_draws:
        idx = np.random.choice(len(flat_samples), n_draw, replace=False) # Randomly select n_draw samples from the posterior without replacement
        draw_samples = flat_samples[idx] # Extract the selected samples for plotting
        for i, theta in enumerate(draw_samples):
            sample_flux = simulate_model_spectrum(theta, broadening_method=broadening_method, emulator=emulator, limbdark=limbdark, output_wl=observed_wavelength) # Simulate the spectrum for this sample with noise and optional rotational broadening
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

def run_MCMC_coverage_test(wl_array_output, spectral_snr, kband_unc, broadening_method, emulator=ACTIVE_EMULATOR, N_sims=20, nsteps=3000, nwalkers=24, ndims=None, simulate=True, save=True, run=None, theta_true=None, limbdark=0.6):
    """
    Run a Bayesian coverage test (PP plot test) for all 6 parameters using repeated noisy simulations.
    ----------
    wl_array : array
        Model wavelength grid.
    spectral_snr : float
        Noise level used for simulations of spectra and inference.
    kband_unc : float
        Uncertainty of the K-band magnitude.
    broadening_method : string
        Which broadening method to use, either 'Sarah', 'rotBroad', or 'fastRotBroad'.
    emulator : str
        Active neural network emulator.
    N_sims : int
        Number of repeated simulations.
    nsteps, nwalkers, ndims: int
        MCMC control parameters.
    simulate : bool
        If True, simulate new true parameters for each run. If False, use provided theta_true for all runs. theta_true : array-like, shape (6,) If simulate=False, the true parameter values to use for all simulations.
    save : bool
        Whether to save individual simulation results and the summary. If True, requires a non-None run name to save under.
    run : str or None
        Name of the run for saving results. If None, results will not be saved even if save=True.
    theta_true : array-like, shape (N_sims, 6) or (6,)
        If simulate=False, the true parameter values to use for all simulations. If shape is (6,), the same true parameters will be used for all simulations. If shape is (N_sims, 6), each row will be used as the true parameters for one simulation.
    limbdark : float
        limbdarkening.
    -------
    credible_dict : dict
        Credible ranks for each parameter.
    theta_true : array
        True parameter values used in each simulation.
    flat_samples_all : list
        Flat MCMC samples for each simulation.
    guesses : array
        First guesses used in each simulation.
    discard_all : list
        Burn-in discards for each simulation.
    chain_all : list
        Full MCMC chains for each simulation.
    results_all : list
        Full result dicts for each simulation.
    runtime_seconds_all : list
        Wall-clock runtimes for each simulation.
    """
    # Sample N true values for each parameter from the prior range
    if simulate:
        theta_true = np.column_stack([np.random.uniform(lo, hi, size=N_sims) for lo, hi in PARAM_BOUNDS])

    # Generate one random first guess per simulation
    guesses = np.column_stack([np.random.uniform(lo, hi, size=N_sims) for lo, hi in PARAM_BOUNDS])

    credible_dict = {name: [] for name in PARAM_NAMES} # Storage: one list per parameter
    flat_samples_all, chain_all, discard_all, results_all, runtime_seconds_all = [], [], [], [], [] # Storage for all flat samples, chains, discards, and results
    print(f"Running coverage test with N_sims = {N_sims}")

    for k in range(N_sims): # Loop over repeated simulations

        print(f"Simulation {k+1}/{N_sims}")
        sim_filename = f"sim_{k:03d}.npz"

        # --- Simulate or load observed data ---
        if save and run is not None and os.path.exists(os.path.join(ppp.MCMC_results_path, run, sim_filename)):
            # Reuse existing simulated spectrum and kband
            print(f"  Loading existing simulation from {sim_filename}")
            existing = load_results(run, sim_filename)
            theta_true[k]    = existing["theta_true"]
            guesses[k]       = existing["first_guess"]
            results_all.append(existing)
            runtime_seconds_all.append(existing["runtime_seconds"])
            discard_all.append(existing["discard"])
            chain_all.append(existing["chain"])
            flat_samples_all.append(existing["flat_samples"])

            for j, name in enumerate(PARAM_NAMES): # Compute credible ranks for each parameter
                u = get_credible_interval_mcmc(existing["flat_samples"], truth=theta_true[k, j], param_index=j)
                credible_dict[name].append(u)
            continue

        # --- No saved result: simulate fresh data ---
        observed_flux  = simulate_model_spectrum(theta_true[k], broadening_method=broadening_method, emulator=emulator, limbdark=limbdark, output_wl=wl_array_output)
        observed_flux  = add_noise_to_spectrum(observed_flux, spectral_snr)
        _, observed_kband, kband_noise = simulate_kband_magnitude(theta_true[k], kband_unc=kband_unc)

        # Run inference
        start   = time.perf_counter()
        results = run_mcmc_with_kband(
            observed_flux      = observed_flux,
            observed_wavelength= wl_array_output,
            observed_kband     = observed_kband,
            spectral_snr       = spectral_snr,
            kband_unc          = kband_unc,
            first_guess        = guesses[k],
            broadening_method  = broadening_method,
            emulator           = emulator,
            limbdark           = limbdark,
            ndim               = ndims,
            nwalkers           = nwalkers,
            nsteps             = nsteps,
            theta_true         = theta_true[k],
        )
        runtime_seconds = time.perf_counter() - start

        results["observed_flux"]        = observed_flux
        results["observed_wavelength"]  = wl_array_output
        results["observed_kband"]       = observed_kband
        results["spectral_snr"]         = spectral_snr
        results["first_guess"]          = guesses[k]
        results["theta_true"]           = theta_true[k]
        results["runtime_seconds"]      = runtime_seconds
        
        results_all.append(results)
        runtime_seconds_all.append(runtime_seconds)
        discard_all.append(results["discard"])
        chain_all.append(results["chain"])
        flat_samples_all.append(results["flat_samples"])

        for j, name in enumerate(PARAM_NAMES): # Compute credible ranks for each parameter
            u = get_credible_interval_mcmc(results["flat_samples"],truth=theta_true[k,j],param_index=j)
            credible_dict[name].append(u)
        
        # --- Save per-simulation results ---
        if save and run is not None:
            save_results(results, run, sim_filename)
        
    # --- Save summary ---
    if save and run is not None:
        np.savez(
            os.path.join(ppp.MCMC_results_path, run, "coverage_summary.npz"),
            theta_true=theta_true,
            credible_dict=credible_dict,
            guesses=guesses,
        )
    return credible_dict, theta_true, flat_samples_all, guesses, discard_all, chain_all, results_all, runtime_seconds_all

def credible_to_sigma_regions(credible_dict, param_names=None):
    param_names = param_names or list(credible_dict.keys())

    coverage_dicts = {r: {p: [] for p in param_names} for r in [
        "below_2sig",
        "between_2sig_1sig_lo",
        "between_1sigs",
        "between_1sig_2sig_hi",
        "above_2sig",
    ]}

    for p in param_names:
        for u in credible_dict[p]:
            if np.isnan(u):
                for r in coverage_dicts:
                    coverage_dicts[r][p].append(None)
                continue

            coverage_dicts["below_2sig"][p].append(u < 0.025)
            coverage_dicts["between_2sig_1sig_lo"][p].append(0.025 <= u < 0.158)
            coverage_dicts["between_1sigs"][p].append(0.158 <= u <= 0.841)
            coverage_dicts["between_1sig_2sig_hi"][p].append(0.841 < u <= 0.975)
            coverage_dicts["above_2sig"][p].append(u > 0.975)

    return coverage_dicts

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


####################################################
################ Nested Sampling ###################
####################################################

def run_NS_with_kband(observed_flux, observed_wavelength, observed_kband, spectral_snr, kband_unc, broadening_method, emulator=ACTIVE_EMULATOR,
                      limbdark=0.6, theta_true=None, run_name=None, log_dir=None, n_live_points=400, plot_corner=False, viz_callback=None,
                      storage_backend="csv", resume="overwrite", wavelengths_by_line=None, device=None):
    """
    Run UltraNest nested sampling with a combined spectral + K-band likelihood.
    Returns a result dict compatible with save_ns_results / load_ns_results and
    the existing comparison functions (plot_posteriors_with_estimates,
    plot_bias_vs_true, print_comparison_summary).

    Parameters
    ----------
    observed_flux       : array  — normalised observed spectrum
    observed_wavelength : array  — wavelength grid
    observed_kband      : float  — observed K-band magnitude
    spectral_snr        : float  — SNR of the spectrum
    kband_unc           : float  — uncertainty of the K-band magnitude
    broadening_method   : str    — e.g. 'vspace', 'rotBroad', 'fastRotBroad'
    emulator            : str    — active neural network emulator
    limbdark            : float  — limb darkening coefficient (default 0.6)
    theta_true          : array or None — ground truth for validation
    run_name            : str or None — unique name for this run, used to set
                          log_dir automatically. If both run_name and log_dir
                          are given, log_dir takes precedence.
    log_dir             : str or None — directory for UltraNest output files.
                          If None, defaults to
                          ppp.nested_sampling_results_path / 'ultranest_runs' / run_name
                          (or 'default' if run_name is also None).
    n_live_points       : int   — number of live points (default 400;
                          increase for higher accuracy at greater cost)
    plot_corner         : bool — whether to plot corner plots
    viz_callback        : callable or None — passed to sampler.run().
    storage_backend     : str or backend — UltraNest point storage backend.
                          Use "csv" to avoid requiring h5py.
    resume              : UltraNest resume mode. Default "overwrite" avoids
                          reusing samples from a previous likelihood/grid.

    Notes
    -----
    theta naming convention (same as run_mcmc_with_kband):
        theta_map  — highest-likelihood posterior sample (posterior mode)
        theta_50p  — posterior median computed from flat_samples
        theta_true — known true values (synthetic tests only)

    The result dict contains MCMC-compatible keys (flat_samples, theta_map,
    theta_50p, chain, discard) so all existing comparison and plotting
    functions work without modification, plus NS-specific keys (ns_results,
    log_evidence, log_evidence_err, n_live_points).

    Returns
    -------
    result : dict
    """
    import ultranest
    import ultranest.plot

    start = time.perf_counter()

    # --- likelihood and prior ---
    log_likelihood = make_log_likelihood_with_kband(
        observed_wavelength = observed_wavelength,
        observed_flux       = observed_flux,
        observed_kband      = observed_kband,
        spectral_snr        = spectral_snr,
        kband_unc           = kband_unc,
        broadening_method   = broadening_method,
        emulator            = emulator,
        limbdark            = limbdark,
        wavelengths_by_line = wavelengths_by_line,
        device              = device,
    )

    def prior_transform(cube):
        params = cube.copy()
        for i, (lo, hi) in enumerate(PARAM_BOUNDS):
            params[i] = lo + (hi - lo) * cube[i]
        return params

    # --- set up log_dir — unique per run to avoid resume collisions ---
    if log_dir is None:
        if ppp is not None:
            ns_root = ppp.nested_sampling_results_path
        else:
            ns_root = os.path.join(os.getcwd(), "ns_runs")
        if run_name is not None:
            # expect run_name like "NS_coverage_v3_sim_019"
            # derive experiment folder from prefix, e.g. "NS_coverage_v3"
            parts = run_name.split("_")
            experiment = "_".join(parts[:-2]) if len(parts) > 2 else "vasilis"
            log_dir = os.path.join(ns_root, experiment, run_name, "ultranest")
        else:
            log_dir = os.path.join(ns_root, "default", "ultranest")
    os.makedirs(log_dir, exist_ok=True)

    # --- set up sampler ---

    sampler = ultranest.ReactiveNestedSampler(
        param_names = PARAM_NAMES,
        loglike     = log_likelihood,
        transform   = prior_transform,
        log_dir     = log_dir,
        resume      = resume,
        storage_backend = storage_backend,
    )

    print(f"Starting Nested Sampling with {n_live_points} live points.")
    ns_results = sampler.run(
        min_num_live_points = n_live_points,
        viz_callback        = viz_callback,
        show_status         = True,
    )
    sampler.print_results()

    elapsed = time.perf_counter() - start

    # --- extract posterior samples ---
    points  = np.array(ns_results['weighted_samples']['points'])  # shape (N, ndim)
    weights = np.array(ns_results['weighted_samples']['weights'])  # sums to 1

    # Resample to equally-weighted flat_samples for compatibility with
    # get_credible_interval_mcmc, plot_corner, plot_posteriors_with_estimates
    rng          = np.random.default_rng()
    n_draw       = min(10_000, len(points))
    idx          = rng.choice(len(points), size=n_draw, replace=True, p=weights / weights.sum())
    flat_samples = points[idx]

    # --- point estimates ---
    # theta_50p: posterior median, consistent with MCMC convention
    theta_50p = np.percentile(flat_samples, 50, axis=0)

    # theta_map: highest-likelihood sample (posterior mode)
    theta_map = points[np.argmax(ns_results['weighted_samples']['logl'])]

    # --- diagnostics ---
    log_evidence     = ns_results['logz']
    log_evidence_err = ns_results['logzerr']
    print(f"\n--- Nested Sampling diagnostics ---")
    print(f"  log Z            : {log_evidence:.2f} ± {log_evidence_err:.2f}")
    print(f"  Effective samples: {ns_results['ess']:.0f}")
    print(f"  Runtime          : {elapsed:.1f} s ({elapsed/60:.2f} min)")
    print(f"  Number of calls   : {ns_results['ncall']}")
    for i, name in enumerate(PARAM_NAMES):
        p16 = np.percentile(flat_samples[:, i], 15.87)
        p50 = np.percentile(flat_samples[:, i], 50)
        p84 = np.percentile(flat_samples[:, i], 84.13)
        truth_str = f", true = {theta_true[i]:.3f}" if theta_true is not None else ""
        print(f"  {name:<10s}: {p50:.4f}  -{p50-p16:.4f}/+{p84-p50:.4f}{truth_str}")

    # --- corner plot ---
    if plot_corner:
        ultranest.plot.cornerplot(ns_results, truths = theta_true, truth_color = 'red')
        plt.show()

    return {
        # MCMC-compatible keys
        "observed_flux"      : observed_flux,
        "observed_wavelength": observed_wavelength,
        "observed_kband"     : observed_kband,
        "spectral_snr"       : spectral_snr,
        "first_guess"        : theta_map,   # NS has no first_guess; MAP is closest equivalent
        "flat_samples"       : flat_samples,
        "theta_map"          : theta_map,
        "theta_50p"          : theta_50p,
        "discard"            : 0,           # no burn-in in NS; kept for compatibility
        "chain"              : np.expand_dims(points, axis=1),  # (N,1,ndim) for plot_chains
        "theta_true"         : theta_true,
        "runtime_seconds"    : elapsed,
        "n_calls"           : ns_results['ncall'],
        # NS-specific keys
        "ns_results"         : ns_results,
        "n_live_points"      : n_live_points,
        "log_evidence"       : log_evidence,
        "log_evidence_err"   : log_evidence_err,
    }

def save_ns_results(results, run, filename):
    """
    Save NS results to disk. MCMC-compatible keys go to an .npz file;
    ns_results (which contains nested dicts) goes to a separate .pkl file.
    """
    import pickle
    os.makedirs(os.path.join(ppp.nested_sampling_results_path, run), exist_ok=True)
    np.savez(
        os.path.join(ppp.nested_sampling_results_path, run, filename),
        observed_flux       = results["observed_flux"],
        observed_wavelength = results["observed_wavelength"],
        observed_kband      = results["observed_kband"],
        spectral_snr        = results["spectral_snr"],
        first_guess         = results["first_guess"],
        flat_samples        = results["flat_samples"],
        theta_map           = results["theta_map"],
        theta_50p           = results["theta_50p"],
        chain               = results["chain"],
        discard             = results["discard"],
        runtime_seconds     = results["runtime_seconds"],
        theta_true          = results["theta_true"] if results["theta_true"] is not None else np.array([]),
        log_evidence        = results["log_evidence"],
        log_evidence_err    = results["log_evidence_err"],
        n_live_points       = results["n_live_points"],
    )
    pkl_path = os.path.join(ppp.nested_sampling_results_path, run, filename.replace(".npz", "_ns_results.pkl"))
    with open(pkl_path, "wb") as f:
        pickle.dump(results["ns_results"], f)

def load_ns_results(run, filename):
    """
    Load NS results saved by save_ns_results. Returns a dict with the same
    keys as run_NS_with_kband, so all plotting functions work identically.
    """
    import pickle
    data = np.load(os.path.join(ppp.nested_sampling_results_path, run, filename), allow_pickle=True)
    pkl_path = os.path.join(ppp.nested_sampling_results_path, run, filename.replace(".npz", "_ns_results.pkl"))
    with open(pkl_path, "rb") as f:
        ns_results = pickle.load(f)
    return {
        "observed_flux"      : data["observed_flux"],
        "observed_wavelength": data["observed_wavelength"],
        "observed_kband"     : data["observed_kband"],
        "spectral_snr"       : float(data["spectral_snr"]),
        "first_guess"        : data["first_guess"],
        "flat_samples"       : data["flat_samples"],
        "theta_map"          : data["theta_map"],
        "theta_50p"          : data["theta_50p"],
        "chain"              : data["chain"],
        "discard"            : int(data["discard"]),
        "runtime_seconds"    : float(data["runtime_seconds"]),
        "theta_true"         : data["theta_true"] if data["theta_true"].shape != (0,) else None,
        "log_evidence"       : float(data["log_evidence"]),
        "log_evidence_err"   : float(data["log_evidence_err"]),
        "n_live_points"      : int(data["n_live_points"]),
        "ns_results"         : ns_results,
    }

####################################################
########### Nested Sampling Coverage Test###########
####################################################
def run_NS_coverage_test(spectral_snr, kband_unc, broadening_method, emulator=ACTIVE_EMULATOR, wl_array_output=None, mcmc_run=None, plot_corner=False, viz_callback=None,
                         N_sims=20, n_live_points=400, save=True, run=None, theta_true=None, limbdark=0.6):
    """
    Run a coverage test for Nested Sampling.

    Two modes:
    - mcmc_run is provided: reuses pre-simulated spectra from that MCMC coverage run. Recommended for a fair method comparison.
    - mcmc_run is None: generates fresh simulated spectra from scratch. In this case wl_array_output and theta_true must be provided or theta_true=None to draw randomly from PARAM_BOUNDS.

    Parameters
    ----------
    spectral_snr        : float  — SNR of the spectrum
    kband_unc           : float  — uncertainty of the K-band magnitude
    broadening_method   : str    — broadeing method e.g. 'vspace', 'rotBroad', 'fastRotBroad'
    emulator            : str    — active neural network emulator
    wl_array_output     : array or None
        Wavelength grid. Required when mcmc_run=None.
    mcmc_run            : str or None
        Name of the MCMC coverage run to load spectra from. If None, fresh spectra are simulated.
    N_sims              : int    — number of simulations
    n_live_points       : int    — UltraNest live points per simulation
    save                : bool   — whether to save results to disk
    run                 : str or None — name for this NS coverage run
    theta_true          : array, shape (N_sims, n_params), or None
        True parameter values to use when mcmc_run=None. If None, drawn randomly from PARAM_BOUNDS.
    limbdark            : float

    Returns
    -------
    credible_dict       : dict   — {param: [credible_rank_per_sim]}
    theta_true_all      : array  — shape (N_sims, n_params)
    results_all         : list   — one result dict per simulation
    runtime_seconds_all : list   — wall-clock runtime per simulation
    """

    # --- set up ground truth and observed data source ---
    if mcmc_run is not None:
        # Load all pre-simulated spectra from the MCMC coverage run
        print(f"Loading {N_sims} pre-simulated spectra from MCMC run '{mcmc_run}'")
        mcmc_sims = []
        for k in range(N_sims):
            sim_path = os.path.join(ppp.MCMC_results_path, mcmc_run, f"sim_{k:03d}.npz")
            if not os.path.exists(sim_path):
                raise FileNotFoundError(
                    f"MCMC simulation {k} not found at {sim_path}. "
                    f"Run the MCMC coverage test first with run='{mcmc_run}'."
                )
            mcmc_sims.append(load_results(mcmc_run, f"sim_{k:03d}.npz"))
        theta_true_all = np.array([s["theta_true"] for s in mcmc_sims])
    else:
        # Generate fresh spectra — wl_array_output is required
        if wl_array_output is None:
            raise ValueError("wl_array_output must be provided when mcmc_run=None.")
        mcmc_sims = None
        if theta_true is not None:
            theta_true_all = np.atleast_2d(theta_true)
            if theta_true_all.shape == (len(PARAM_NAMES),):
                # Single theta_true broadcast to all sims
                theta_true_all = np.tile(theta_true_all, (N_sims, 1))
        else:
            theta_true_all = np.column_stack([
                np.random.uniform(lo, hi, size=N_sims) for lo, hi in PARAM_BOUNDS
            ])
        print(f"Generating {N_sims} fresh simulated spectra")

    credible_dict       = {name: [] for name in PARAM_NAMES}
    results_all         = []
    runtime_seconds_all = []

    print(f"Running NS coverage test with N_sims={N_sims}, n_live_points={n_live_points}")

    for k in range(N_sims):
        print(f"\nSimulation {k+1}/{N_sims}")
        sim_filename = f"sim_{k:03d}.npz"

        # --- Check if this NS simulation is already complete ---
        if save and run is not None:
            ns_path = os.path.join(ppp.nested_sampling_results_path, run, sim_filename)
            if os.path.exists(ns_path):
                print(f"  Loading existing NS results from {sim_filename}")
                existing = load_ns_results(run, sim_filename)
                results_all.append(existing)
                runtime_seconds_all.append(existing["runtime_seconds"])
                for j, name in enumerate(PARAM_NAMES):
                    u = get_credible_interval_mcmc(
                        existing["flat_samples"], truth=theta_true_all[k, j], param_index=j
                    )
                    credible_dict[name].append(u)
                continue

        # --- Get or generate observed data ---
        if mcmc_sims is not None:
            # Reuse MCMC pre-simulated spectrum
            observed_flux       = mcmc_sims[k]["observed_flux"]
            observed_wavelength = mcmc_sims[k]["observed_wavelength"]
            observed_kband      = mcmc_sims[k]["observed_kband"]
        else:
            # Simulate fresh data
            observed_flux  = simulate_model_spectrum(theta_true_all[k], broadening_method=broadening_method, emulator=emulator, limbdark=limbdark, output_wl=wl_array_output)
            observed_flux  = add_noise_to_spectrum(observed_flux, spectral_snr)
            _, observed_kband, kband_noise = simulate_kband_magnitude(theta_true_all[k], kband_unc=kband_unc)
            observed_wavelength = wl_array_output

        # --- Run NS ---
        results = run_NS_with_kband(
            observed_flux       = observed_flux,
            observed_wavelength = observed_wavelength,
            observed_kband      = observed_kband,
            spectral_snr        = spectral_snr,
            kband_unc           = kband_unc,
            broadening_method   = broadening_method,
            emulator            = emulator,
            limbdark            = limbdark,
            theta_true          = theta_true_all[k],
            run_name            = f"{run}_sim_{k:03d}" if run is not None else f"ns_coverage_sim_{k:03d}",
            n_live_points       = n_live_points,
            plot_corner         = plot_corner,
            viz_callback        = viz_callback,
        )

        results_all.append(results)
        runtime_seconds_all.append(results["runtime_seconds"])

        for j, name in enumerate(PARAM_NAMES):
            u = get_credible_interval_mcmc(
                results["flat_samples"], truth=theta_true_all[k, j], param_index=j
            )
            credible_dict[name].append(u)

        if save and run is not None:
            os.makedirs(os.path.join(ppp.nested_sampling_results_path, run), exist_ok=True)
            save_ns_results(results, run, sim_filename)

    # --- Save summary ---
    if save and run is not None:
        np.savez(
            os.path.join(ppp.nested_sampling_results_path, run, "coverage_summary.npz"),
            theta_true    = theta_true_all,
            credible_dict = credible_dict,
        )

    return credible_dict, theta_true_all, results_all, runtime_seconds_all


def load_ns_coverage_summary(run, N_sims):
    """
    Reload a completed NS coverage test from disk without re-running.

    Parameters
    ----------
    run    : str — name of the NS coverage run
    N_sims : int — number of simulations in the run

    Returns
    -------
    credible_dict       : dict
    theta_true_all      : array
    results_all         : list of dicts
    runtime_seconds_all : list of floats
    """
    summary = np.load(os.path.join(ppp.nested_sampling_results_path, run, "coverage_summary.npz"),allow_pickle=True)
    credible_dict  = summary["credible_dict"].item()
    theta_true_all = summary["theta_true"]
    results_all, runtime_seconds_all = [], []
    for k in range(N_sims):
        r = load_ns_results(run, f"sim_{k:03d}.npz")
        results_all.append(r)
        runtime_seconds_all.append(r["runtime_seconds"])
    return credible_dict, theta_true_all, results_all, runtime_seconds_all


####################################################
############ Multi-method comparison plot ##########
####################################################


def plot_posteriors_with_estimates(
    sim_index,
    flat_samples_MCMC_all,
    theta_true_all,
    param_names=None,
    param_bounds=None,
    plot_spectrum=False,            # whether to plot the observed spectrum + best fits
    broadening_method='vspace',
    emulator=ACTIVE_EMULATOR,
    # --- MCMC ---
    mcmc_results_all=None,          # list of result dicts from run_mcmc_with_kband
    # --- GA ---
    ga_results_all=None,            # list of result dicts from gnt.get_ga_best_and_intervals
    ga_kband_all=None,              # needed for radius intervals
    kband_unc=0.0,           # uncertainty in K-band magnitudes
    # --- NS ---
    nested_sampling_results_all=None, # list of result dicts from run_NS_with_kband
    # --- add future methods here as keyword arguments ---
    # eg. nested_sampling_results_all=None,
):
    """
    For one simulation, plot the MCMC posterior for each parameter as a
    histogram, overlaid with point estimates and 1σ intervals from other
    methods (GA, etc.) and the true value.

    Parameters
    ----------
    sim_index : int
        Which simulation to plot (indexes into all the *_all lists).
    flat_samples_all : list of arrays, shape (N_sims, n_samples, n_params)
        MCMC posterior samples per simulation, from run_MCMC_coverage_test.
    theta_true_all : array, shape (N_sims, n_params)
        True parameter values per simulation.
    param_names : list of str or None
        Defaults to PARAM_NAMES.
    param_bounds : array, shape (n_params, 2) or None
        Defaults to PARAM_BOUNDS. Used to set x-axis limits.
    mcmc_results_all : list of result dicts or None
        If provided, the MCMC MAP estimate (theta_map) is shown as a vertical line.
    ga_results_all : list of result dicts or None
        If provided, GA best fit + 1σ interval is shown per parameter.
    ga_kband_all : array-like or None
        K-band magnitudes for each simulation, needed to derive GA radius intervals.
    nested_sampling_results_all : list of result dicts or None
        If provided, NS estimate and 1σ interval are shown per parameter.
    """
    param_names  = param_names  or PARAM_NAMES
    param_bounds = param_bounds if param_bounds is not None else PARAM_BOUNDS
    n_params     = len(param_names)

    flat_samples_MCMC = flat_samples_MCMC_all[sim_index]# (n_samples, n_params)
    theta_true   = theta_true_all[sim_index]            # (n_params,)

    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for j, name in enumerate(param_names):
        ax = axes[j]
        samples_1d = flat_samples_MCMC[:, j]

        # --- MCMC posterior histogram ---
        ax.hist(samples_1d, bins=40, density=True, color="steelblue",alpha=0.5, label="MCMC posterior")

        # --- MCMC 1σ interval (shaded) ---
        p16, p50, p84 = np.percentile(samples_1d, [15.85, 50, 84.15])
        ax.axvspan(p16, p84, alpha=0.15, color="steelblue", label="MCMC 1σ")
        ax.axvline(p50, color="steelblue", lw=1.5, linestyle="-", label="MCMC median")

        # --- MCMC MAP (optional) ---
        #if mcmc_results_all is not None:
        #    theta_map = mcmc_results_all[sim_index].get("theta_map")
        #    if theta_map is not None:
        #        ax.axvline(theta_map[j], color="steelblue", lw=1.5,linestyle="--", label="MCMC MAP")

        # --- GA best fit + 1σ (optional) ---
        if ga_results_all is not None and ga_results_all[sim_index] is not None:
            ga_result = ga_results_all[sim_index]
            if name == "radius":
                if ga_kband_all is not None:
                    ri = gnt.get_radius_intervals_from_kband(ga_result, ga_kband_all[sim_index], kband_unc=kband_unc)
                    ga_best, ga_lo1sig, ga_hi1sig = ri["best"], ri["lower_1sig"], ri["upper_1sig"]
                else:
                    ga_best, ga_lo1sig, ga_hi1sig = None, None, None    
            else:
                ga_name   = GA_NAME_MAP.get(name)
                ga_best   = ga_result["best"].get(ga_name) if ga_name else None
                ga_lo1sig = ga_result["lower_1sig"].get(ga_name) if ga_name else None
                ga_hi1sig = ga_result["upper_1sig"].get(ga_name) if ga_name else None
            if ga_best is not None:
                ax.axvline(ga_best, color="darkorange", lw=1.5,linestyle="-", label="GA best fit")
            if ga_lo1sig is not None and ga_hi1sig is not None:
                ax.axvspan(ga_lo1sig, ga_hi1sig, alpha=0.15,color="darkorange", label="GA 1σ")

        # NS best fit + 1σ
        if nested_sampling_results_all is not None and nested_sampling_results_all[sim_index] is not None:
            ns = nested_sampling_results_all[sim_index]
            ns_samples1d = ns["flat_samples"][:, j]
            ns_p16, ns_p50, ns_p84 = np.percentile(ns_samples1d, [15.87, 50, 84.13])
            ax.hist(ns_samples1d, bins=40, density=True, color="green", alpha=0.4, label="NS posterior")
            ax.axvspan(ns_p16, ns_p84, alpha=0.15, color="green", label="NS 1σ")
            ax.axvline(ns_p50, color="green", lw=1.5, linestyle="-", label="NS median")

        # --- add future methods here, e.g.:
        # if nested_sampling_results_all is not None:
        #     ns = nested_sampling_results_all[sim_index]
        #     ax.axvline(ns["median"][j], color="green", lw=1.5, label="NS median")
        #     ax.axvspan(ns["lo1sig"][j], ns["hi1sig"][j], alpha=0.15, color="green")

        # --- true value ---
        ax.axvline(theta_true[j], color="red", lw=2,linestyle="--", label="True value")

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_xlim(param_bounds[j])
        ax.set_title(name)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
    
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    if mcmc_results_all is not None:
        print(f"Runtime MCMC: {mcmc_results_all[sim_index]['runtime_seconds']:.1f} s")
    if ga_results_all is not None:
        print(f"Runtime GA:   {ga_results_all[sim_index]['runtime_seconds']:.1f} s")
    if nested_sampling_results_all is not None:
        print(f"Runtime NS:   {nested_sampling_results_all[sim_index]['runtime_seconds']:.1f} s")

    fig.suptitle(f"Simulation {sim_index}  —  posterior comparison", fontsize=13)
    plt.tight_layout()
    plt.show()

    if plot_spectrum:
        obs_wl   = mcmc_results_all[sim_index]['observed_wavelength']
        obs_flux = mcmc_results_all[sim_index]['observed_flux']

        fig_spec, ax_spec = plt.subplots(figsize=(14, 6))
        ax_spec.plot(obs_wl, obs_flux, lw=0.8, label='Observed', color='black', alpha=0.7)

        if mcmc_results_all is not None and mcmc_results_all[sim_index] is not None:
            mcmc_flux = simulate_model_spectrum(mcmc_results_all[sim_index]['theta_50p'], broadening_method=broadening_method, emulator=emulator, limbdark=0.6, output_wl=obs_wl)
            ax_spec.plot(obs_wl, mcmc_flux, lw=1, label='Best fit MCMC (median)', color='steelblue')

        if ga_results_all is not None and ga_results_all[sim_index] is not None:
            ga_b = ga_results_all[sim_index]['best']
            if 'radius' in ga_b:
                best_radius_GA = ga_b['radius']
            else:
                ri = gnt.get_radius_intervals_from_kband(ga_results_all[sim_index],ga_kband_all[sim_index], kband_unc=kband_unc)
                best_radius_GA = ri['best']
            best_theta_GA = []
            for name in PARAM_NAMES:
                if name == 'radius':
                    best_theta_GA.append(best_radius_GA)
                else:
                    ga_col = GA_NAME_MAP.get(name)
                    best_theta_GA.append(ga_b[ga_col] if ga_col else 0.0)
            best_theta_GA = np.array(best_theta_GA)
            ga_flux = simulate_model_spectrum(best_theta_GA, broadening_method=broadening_method, emulator=emulator, limbdark=0.6, output_wl=obs_wl)
            ax_spec.plot(obs_wl, ga_flux, lw=1, label='Best fit GA', color='darkorange')

        if nested_sampling_results_all is not None and nested_sampling_results_all[sim_index] is not None:
            ns_flux = simulate_model_spectrum(nested_sampling_results_all[sim_index]['theta_50p'], broadening_method=broadening_method, emulator=emulator, limbdark=0.6, output_wl=obs_wl)
            ax_spec.plot(obs_wl, ns_flux, lw=1, label='Best fit NS (median)', color='green')

        ax_spec.set_xlabel('Wavelength (nm)')
        ax_spec.set_ylabel('Flux')
        ax_spec.set_title(f'Observed spectrum — simulation {sim_index}')
        ax_spec.legend()
        plt.tight_layout()
        plt.show()
        return fig_spec, ax_spec
    else:
        return None, None

def plot_bias_vs_true(theta_true_all, mcmc_results_all, ga_results_all, kband_all, kband_unc, ns_results_all):
    n_params = len(PARAM_NAMES)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols / 3, 8 * n_rows / 2))
    axes = axes.flatten()

    for j, name in enumerate(PARAM_NAMES):
        ax = axes[j]
        ax.axhline(0, color='grey', lw=1, linestyle='--')

        # MCMC: use theta_50p as the point estimate
        mcmc_bias = [r["theta_50p"][j] - theta_true_all[i, j] for i, r in enumerate(mcmc_results_all) if r is not None]
        mcmc_true = [theta_true_all[i, j] for i, r in enumerate(mcmc_results_all) if r is not None]
        ax.scatter(mcmc_true, mcmc_bias, label='MCMC median', alpha=0.7, marker='o')

        # GA: use best-fit value
        ga_bias, ga_true = [], []
        for i, r in enumerate(ga_results_all):
            if r is None: continue
            if name == "radius":
                ri = gnt.get_radius_intervals_from_kband(r, kband_all[i], kband_unc=kband_unc)
                best = ri["best"]
            else:
                ga_name = GA_NAME_MAP.get(name)
                best = r["best"].get(ga_name) if ga_name else None
            if best is not None:
                ga_bias.append(best - theta_true_all[i, j])
                ga_true.append(theta_true_all[i, j])
        ax.scatter(ga_true, ga_bias, label='GA best fit', alpha=0.7, marker='D')

        # NS bias
        if ns_results_all is not None:
            ns_bias, ns_true = [], []
            for i, r in enumerate(ns_results_all):
                if r is None: continue
                ns_bias.append(r["theta_50p"][j] - theta_true_all[i, j])
                ns_true.append(theta_true_all[i, j])
            ax.scatter(ns_true, ns_bias, label='NS median', alpha=0.7, marker='^', color='green')

        ax.set_xlabel(f'true {name}')
        ax.set_ylabel('bias (recovered − true)')
        ax.set_title(name)
        if j == 0:
            ax.legend()
    fig.suptitle(f"Bias vs. true value for MCMC, GA and NS — {len(mcmc_results_all)} simulations", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

def print_comparison_summary(
    theta_true_all,
    mcmc_results_all,
    ga_results_all,
    kband_all,
    kband_unc,
    mcmc_credible_dict,
    ga_coverage_dicts,
    mcmc_runtime_all=None,
    ga_runtime_all=None,
    ns_results_all=None,
    ns_credible_dict=None,
    ns_runtime_all=None,
    param_names=None,
    ga_name_map=None,
):
    """
    Print a summary table comparing MCMC, GA, and NS performance across all simulations.

    For each parameter, reports:
        - Median bias      : median(recovered - true)
        - RMS scatter      : RMS of (recovered - true)
        - 1sigma coverage  : fraction of simulations where true value falls within 1sigma interval
        - 2sigma coverage  : fraction of simulations where true value falls within 2sigma interval
    Plus a final row with median runtime per simulation.

    Parameters
    ----------
    theta_true_all : np.ndarray, shape (N_sims, n_params)
        True parameter values for each simulation.
    mcmc_results_all : list of dicts
        Output of run_MCMC_coverage_test — one result dict per simulation.
    ga_results_all : list of dicts
        Output of run_ga_coverage_test — one result dict per simulation.
    kband_all : list of floats
        Observed K-band magnitude for each simulation (needed for GA radius).
    mcmc_credible_dict : dict
        {param_name: [credible_rank_per_sim]} from run_MCMC_coverage_test.
    ga_coverage_dicts : list of dicts
        Per-simulation coverage dicts from run_ga_coverage_test.
    mcmc_runtime_all : list of floats or None
        Wall-clock runtime in seconds for each MCMC simulation.
    ga_runtime_all : list of floats or None
        Wall-clock runtime in seconds for each GA simulation.
    ns_results_all : list of dicts or None
        Output of run_NS_coverage_test — one result dict per simulation.
    ns_credible_dict : dict or None
        {param_name: [credible_rank_per_sim]} from run_NS_coverage_test.
    ns_runtime_all : list of floats or None
        Wall-clock runtime in seconds for each NS simulation.
    param_names : list of str or None
        Parameter names to include. Defaults to PARAM_NAMES.
    ga_name_map : dict or None
        Mapping from PARAM_NAMES to GA column names, e.g. {"teff": "Teff"}.
        Defaults to identity mapping.
    """
    param_names = param_names or PARAM_NAMES
    ga_name_map = ga_name_map or {n: n for n in param_names}
    use_ns      = ns_results_all is not None and ns_credible_dict is not None

    col_w  = 14   # width per method column
    n_meth = 3 if use_ns else 2

    # --- header rows ---
    def make_sep():
        return "+" + "-"*14 + ("+" + "-"*col_w*n_meth)*4 + "+"

    def make_col_hdr():
        return (
            f"{'Parameter':<14}|"
            f"{'Median bias':>{col_w*n_meth}}|"
            f"{'RMS scatter':>{col_w*n_meth}}|"
            f"{'1σ coverage':>{col_w*n_meth}}|"
            f"{'2σ coverage':>{col_w*n_meth}}|"
        )

    def make_method_hdr():
        methods = f"{'MCMC':>{col_w}}{'GA':>{col_w}}" + (f"{'NS':>{col_w}}" if use_ns else "")
        return f"{'':14}|{methods}|{methods}|{methods}|{methods}|"

    sep = make_sep()
    print(sep)
    print(make_col_hdr())
    print(make_method_hdr())
    print(sep)

    for j, name in enumerate(param_names):

        # --- MCMC ---
        mcmc_best  = np.array([r["theta_50p"][j] for r in mcmc_results_all if r is not None])
        true_vals  = theta_true_all[:len(mcmc_best), j]
        mcmc_bias  = mcmc_best - true_vals
        mcmc_med_b = np.median(mcmc_bias)
        mcmc_rms   = np.sqrt(np.mean(mcmc_bias**2))
        mcmc_ranks = np.array(mcmc_credible_dict[name])
        mcmc_1sig  = np.mean((mcmc_ranks >= 0.159) & (mcmc_ranks <= 0.841))
        mcmc_2sig  = np.mean((mcmc_ranks >= 0.023) & (mcmc_ranks <= 0.977))

        # --- GA ---
        ga_best_vals, ga_true_vals = [], []
        ga_1sig_count, ga_2sig_count, ga_n = 0, 0, 0
        for k, r in enumerate(ga_results_all):
            if r is None:
                continue
            if name == "radius":
                ri   = gnt.get_radius_intervals_from_kband(r, kband_all[k], kband_unc=kband_unc)
                best = ri["best"]
                lo1, hi1 = ri["lower_1sig"], ri["upper_1sig"]
                lo2, hi2 = ri["lower_2sig"], ri["upper_2sig"]
            else:
                ga_col = ga_name_map.get(name, name)
                if ga_col not in r["best"]:
                    continue
                best = r["best"][ga_col]
                lo1, hi1 = r["lower_1sig"][ga_col], r["upper_1sig"][ga_col]
                lo2, hi2 = r["lower_2sig"][ga_col], r["upper_2sig"][ga_col]
            truth = theta_true_all[k, j]
            ga_best_vals.append(best)
            ga_true_vals.append(truth)
            ga_1sig_count += int(lo1 <= truth <= hi1)
            ga_2sig_count += int(lo2 <= truth <= hi2)
            ga_n += 1

        if ga_n > 0:
            ga_bias  = np.array(ga_best_vals) - np.array(ga_true_vals)
            ga_med_b = np.median(ga_bias)
            ga_rms   = np.sqrt(np.mean(ga_bias**2))
            ga_1sig  = ga_1sig_count / ga_n
            ga_2sig  = ga_2sig_count / ga_n
        else:
            ga_med_b = ga_rms = ga_1sig = ga_2sig = float("nan")

        # --- NS ---
        if use_ns:
            ns_best  = np.array([r["theta_50p"][j] for r in ns_results_all if r is not None])
            ns_true  = theta_true_all[:len(ns_best), j]
            ns_bias  = ns_best - ns_true
            ns_med_b = np.median(ns_bias)
            ns_rms   = np.sqrt(np.mean(ns_bias**2))
            ns_ranks = np.array(ns_credible_dict[name])
            ns_1sig  = np.mean((ns_ranks >= 0.159) & (ns_ranks <= 0.841))
            ns_2sig  = np.mean((ns_ranks >= 0.023) & (ns_ranks <= 0.977))
            ns_cols  = f"{ns_med_b:>{col_w}.4f}{ns_rms:>{col_w}.4f}{ns_1sig:>{col_w}.3f}{ns_2sig:>{col_w}.3f}"
        else:
            ns_cols = ""

        # --- print row ---
        # build each section then join with |
        bias_sec = f"{mcmc_med_b:>{col_w}.4f}{ga_med_b:>{col_w}.4f}" + (f"{ns_med_b:>{col_w}.4f}" if use_ns else "")
        rms_sec  = f"{mcmc_rms:>{col_w}.4f}{ga_rms:>{col_w}.4f}"     + (f"{ns_rms:>{col_w}.4f}"   if use_ns else "")
        s1_sec   = f"{mcmc_1sig:>{col_w}.3f}{ga_1sig:>{col_w}.3f}"   + (f"{ns_1sig:>{col_w}.3f}"  if use_ns else "")
        s2_sec   = f"{mcmc_2sig:>{col_w}.3f}{ga_2sig:>{col_w}.3f}"   + (f"{ns_2sig:>{col_w}.3f}"  if use_ns else "")
        print(f"{name:<14}|{bias_sec}|{rms_sec}|{s1_sec}|{s2_sec}|")

    print(sep)

    # --- Runtime row ---
    mcmc_rt = f"{np.median(mcmc_runtime_all)/60:.1f} min" if mcmc_runtime_all else "N/A"
    ga_rt   = f"{np.median(ga_runtime_all)/60:.1f} min"   if ga_runtime_all   else "N/A"
    ns_rt   = f"{np.median(ns_runtime_all)/60:.1f} min"   if ns_runtime_all   else "N/A"

    rt_sec  = f"{mcmc_rt:>{col_w}}{ga_rt:>{col_w}}" + (f"{ns_rt:>{col_w}}" if use_ns else "")
    blank   = f"{'':>{col_w}}{'':>{col_w}}"          + (f"{'':>{col_w}}"    if use_ns else "")
    print(f"{'Runtime (med)':<14}|{rt_sec}|{blank}|{blank}|{blank}|")
    print(sep)

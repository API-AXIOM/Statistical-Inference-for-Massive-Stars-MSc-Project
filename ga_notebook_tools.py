"""
ga_notebook_tools.py
====================
Utilities for running the Kiwi-GA from a Jupyter notebook and performing
a coverage (calibration) test on the GA uncertainties.

Sections
--------
1. Input-file writers  – write spectrum.norm, parameter_space.txt, radius_info.txt, line_list.txt (optional)
2. GA launcher         – run the GA via mpirun from the notebook
3. GA result reader    – read chi2.txt and extract best-fit + 1/2σ bounds (replicating the logic in func_GA_analysis.get_uncertainties)
4. Coverage test       – run N simulations and check whether the truth falls inside the GA 1σ interval, then plot the result
"""
import sys
import os
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom
from scipy.interpolate import interp1d
import paths_NN as ppp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'KIWI-GA'))
import NN_wrapper_Hhe_split as fw

# ---------------------------------------------------------------------------
# You will need to set these paths to match your system.
#
# Kiwi-GA directory layout (from the manual):
#   <GA_BASE_DIR>/
#     kiwiga_Hhe_split.py          ← GA_SCRIPT
#     input/<run_name>/            ← INPUT_DIR + run_name  (input files go here)
#     output/<run_name>/chi2.txt   ← OUTPUT_DIR + run_name (results come out here)
# ---------------------------------------------------------------------------
GA_BASE_DIR  = ppp.KIWI_GA_path
INPUT_DIR    = os.path.join(GA_BASE_DIR, "input") # input/<run_name>/ lives here
OUTPUT_DIR   = os.path.join(GA_BASE_DIR, "output") # output/<run_name>/chi2.txt lives here
GA_SCRIPT    = os.path.join(GA_BASE_DIR, "kiwiga_Hhe_split.py")

# Keep RUNS_DIR as an alias for INPUT_DIR for backwards compatibility
RUNS_DIR      = INPUT_DIR

# example input dir
EXAMPLE_INPUT_DIR = os.path.join(GA_BASE_DIR, "example_input")

# Read parameter_space
theparamfile = os.path.join(EXAMPLE_INPUT_DIR, 'parameter_space.txt')
# PARAM_NAMES here reflects the GA's parameter_space.txt, not the emulator's theta vector.
# When using Vasilis' emulator, ensure parameter_space.txt includes vinf and vturb
# (or fix them) so the GA column names align with GA_NAME_MAP in comparing_everything.py.
pspacedata = fw.read_paramspace(theparamfile)
PARAM_NAMES, PARAM_SPACE, FIX_NAMES, FIX_VALS = pspacedata
nfree = len(PARAM_NAMES)
RMSEA_THRESHOLD = 1.5 # mirrors func_GA_analysis.py

cc = 2.99792458e10 #speed of light in a vacuum in cm/s;
C_KMS = cc * 1e-5 # speed of light in km/s

# --- Ks filter transmission (used by compute_obs_flux / flux_to_magnitude) ---
_band       = 'SPHERE_Ks'
_filterfile = 'SPHERE_IRDIS_B_Ks.dat'
_KS_WAVE, _KS_TRANS = np.genfromtxt(os.path.join(ppp.filter_path, _filterfile), comments='#').T
_KS_WAVE *= 10  # nm → Å
_KS_NORM = np.trapz(_KS_TRANS, _KS_WAVE)
_zp_values = np.genfromtxt(os.path.join(ppp.filter_path, 'zero_points.dat'), comments='#', dtype=str)
zpflux = next(float(row[1]) for row in _zp_values if row[0] == _band) # vega zero-point flux for SPHERE_Ks


def check_paths(run_name=None):
    """
    Print a diagnostic summary of paths and optionally check a specific run.
    Call this first if you get FileNotFoundErrors.

    Example
    -------
    ga.check_paths("automated")   # checks all paths + the 'automated' run
    """
    print("=== ga_notebook_tools path check ===")
    for label, p in [("GA_BASE_DIR", GA_BASE_DIR),
                     ("INPUT_DIR  ", INPUT_DIR),
                     ("OUTPUT_DIR ", OUTPUT_DIR),
                     ("GA_SCRIPT  ", GA_SCRIPT)]:
        exists = os.path.exists(p)
        print(f"  {label}: {p}  {'OK' if exists else 'NOT FOUND'}")
    if run_name:
        print(f"\n  Run: '{run_name}'")
        for label, p in [
            ("input dir ", os.path.join(INPUT_DIR, run_name)),
            ("output dir", os.path.join(OUTPUT_DIR, run_name)),
            ("chi2.txt  ", os.path.join(OUTPUT_DIR, run_name, "chi2.txt")),
        ]:
            exists = os.path.exists(p)
            print(f"    {label}: {p}  {'OK' if exists else 'NOT FOUND'}")



# ===========================================================================
# 1.  INPUT-FILE WRITERS
# ===========================================================================

def write_spectrum_norm(run_name, wavelengths, fluxes, spectral_snr, errors=None):
    """
    Write the spectrum.norm input file for the GA.

    Parameters
    ----------
    run_name : str
        Name of the run (determines the subfolder inside RUNS_DIR).
    wavelengths : array-like
        Wavelength array [Å].
    fluxes : array-like
        Normalised flux array.
    errors : array-like or None
        Per-pixel flux uncertainties.  If None, uniform 1/spectral_snr is used.
    spectral_snr : float
        Used only when `errors` is None.
    """
    run_dir = _ensure_run_dir(run_name)
    outfile = os.path.join(run_dir, "spectrum.norm")

    wavelengths = np.asarray(wavelengths)
    fluxes      = np.asarray(fluxes)
    errors      = (np.ones_like(fluxes) / spectral_snr if errors is None else np.asarray(errors))

    data = np.column_stack([wavelengths, fluxes, errors])
    np.savetxt(outfile, data, fmt="%.6f", header="wave flux err")
    print(f"[write_spectrum_norm] Written: {outfile}")
    return outfile


def write_parameter_space(run_name, free_params, fixed_params=None):
    """
    Write the parameter_space.txt file.

    Parameters
    ----------
    run_name : str
    free_params : dict
        Keys are parameter names; values are (min, max, step) tuples.
        Example: {"teff": (29000, 52000, 1000), "logg": (3.4, 4.3, 0.05)}
    fixed_params : dict or None
        Keys are parameter names; values are the fixed scalar value.
        These are written with a flag of -1 (GA convention for fixed).
        Example: {"beta": 1.0, "fclump": 1.0}
    """
    run_dir = _ensure_run_dir(run_name)
    outfile = os.path.join(run_dir, "parameter_space.txt")
    fixed_params = fixed_params or {}

    lines = []
    for name, (lo, hi, step) in free_params.items():
        lines.append(f"{name}   {lo}   {hi}   {step}")
    for name, val in fixed_params.items():
        lines.append(f"{name}   {val}   -1   0")

    with open(outfile, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[write_parameter_space] Written: {outfile}")
    return outfile

def copy_template_inputs(run_name, template_run_name, files=None):
    """
    Copy input files from an existing (template) run to a new run directory.
    Useful for files you don't want to regenerate (control.txt, defaults_fastwind.txt …).

    Looks for source files first in input/<template_run_name>/, then in
    output/<template_run_name>/input_copy/ (where Kiwi-GA stores a copy after a run).

    Parameters
    ----------
    run_name : str
        Destination run name.
    template_run_name : str
        Source run name.
    files : list of str, str, or None
        Which files to copy. Accepts a single filename string or a list of strings.
        Defaults to all standard input files.
    """
    if files is None:
        files = ["control.txt", "defaults_fastwind.txt", "line_list.txt",
                 "parameter_space.txt", "radius_info.txt"]
    elif isinstance(files, str):
        files = [files]

    src_candidates = [
        os.path.join(INPUT_DIR,  template_run_name),
        os.path.join(OUTPUT_DIR, template_run_name, "input_copy"),
    ]
    dst_dir = _ensure_run_dir(run_name)

    for fname in files:
        copied = False
        for src_dir in src_candidates:
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(dst_dir, fname)
                os.system(f"cp '{src}' '{dst}'")
                print(f"[copy_template_inputs] Copied {fname} from {src_dir}")
                copied = True
                break
        if not copied:
            print(f"[copy_template_inputs] WARNING: {fname} not found in any "
                  f"source directory, skipped.")

def write_radius_info(run_name, band, magnitude, zp_system="vega"):
    """
    Write the radius_info.txt file.

    Parameters
    ----------
    run_name : str
    band : str
        Photometric band name, e.g. '2MASS_Ks', 'VISTA_Ks'.
    magnitude : float
        Observed absolute magnitude in that band.
    system : str
        Zero-point system: 'vega', 'AB', or 'ST'.
    """
    run_dir = _ensure_run_dir(run_name)
    outfile = os.path.join(run_dir, "radius_info.txt")
    with open(outfile, "w") as f:
        f.write(f"{band}\n{magnitude}\n{zp_system}\n")
    print(f"[write_radius_info] Written: {outfile}")
    return outfile


def write_line_list(run_name, line_list):
    """
    Write (or overwrite) the line_list.txt file.

    Parameters
    ----------
    run_name : str
    line_list : list of tuples
        Each tuple: (name, something, left_bound, right_bound).
        Lines starting with '#' are treated as comments by the GA.
        Pass them as plain strings in the list if needed.
        Example:
            [("HALPHA", 0, 6530.0, 6590.0),
             ("HBETA",  0, 4840.0, 4880.0)]
    """
    run_dir = _ensure_run_dir(run_name)
    outfile = os.path.join(run_dir, "line_list.txt")
    with open(outfile, "w") as f:
        for entry in line_list:
            if isinstance(entry, str):          # raw comment line
                f.write(entry + "\n")
            else:
                name, flag, left, right = entry
                f.write(f"{name}   {flag}   {left:.2f}   {right:.2f}\n")
    print(f"[write_line_list] Written: {outfile}")
    return outfile


# ===========================================================================
# 2.  GA LAUNCHER
# ===========================================================================

def run_ga(run_name, n_cores=4, block=True, logfile=None):
    """
    Launch the GA via mpirun from the notebook.

    Parameters
    ----------
    run_name : str
    n_cores : int
        Number of MPI cores (the -n argument).
    block : bool
        If True (default), wait for the GA to finish before returning.
        If False, launch in the background and return the Popen object.
    logfile : str or None
        If given, stdout+stderr are redirected to this file path.
        If None, output goes to the notebook cell.

    Returns
    -------
    subprocess.CompletedProcess  (if block=True)
    subprocess.Popen             (if block=False)
    """
    cmd = ["mpirun", "-n", str(n_cores), "python", GA_SCRIPT, run_name]
    print(f"[run_ga] Launching: {' '.join(cmd)}")

    # Add the project directory to PYTHONPATH so that paths_NN and other
    # local modules can be found by the MPI subprocess workers
    env = os.environ.copy()
    project_dir = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_dir + (":" + existing_pythonpath if existing_pythonpath else "")

    if logfile:
        log_fh = open(logfile, "w")
        kwargs = dict(stdout=log_fh, stderr=subprocess.STDOUT, cwd=GA_BASE_DIR, env=env)
    else:
        kwargs = dict(cwd=GA_BASE_DIR, env=env)

    if block:
        start = time.perf_counter()
        result = subprocess.run(cmd, **kwargs)
        elapsed = time.perf_counter() - start
        # write a small timing file next to chi2.txt
        timing_file = os.path.join(OUTPUT_DIR, run_name, "runtime.txt")
        with open(timing_file, "w") as f:
            f.write(str(elapsed))
        print(f"[run_ga] Finished with return code {result.returncode}")
        print(f"[run_ga] Elapsed time: {elapsed:.1f} seconds")
        return result, elapsed
    else:
        start = time.perf_counter()
        proc = subprocess.Popen(cmd, **kwargs)
        print(f"[run_ga] Running in background (PID {proc.pid})")
        return proc, start

# ===========================================================================
# 3.  GA RESULT READER  (replicates func_GA_analysis.get_uncertainties)
# ===========================================================================

def read_ga_results(run_name):
    """
    Read the chi2.txt output file for a completed GA run.

    Returns
    -------
    df : pd.DataFrame
        All models with their parameters and fit statistics.
    """
    chi2_file = os.path.join(OUTPUT_DIR, run_name, "chi2.txt")
    if not os.path.isfile(chi2_file):
        raise FileNotFoundError(f"chi2.txt not found at {chi2_file}")

    with open(chi2_file) as f:
        header = f.readline().strip("# \n").split()

    df = pd.read_csv(chi2_file, sep=r"\s+", comment="#", header=None)
    df.columns = header
    df["gen"] = df["gen"].astype(int)
    return df

def read_ga_runtime(run_name):
    timing_file = os.path.join(OUTPUT_DIR, run_name, "runtime.txt")
    if os.path.exists(timing_file):
        with open(timing_file) as f:
            return float(f.read())
    return None  # not available for old runs

def get_ga_best_and_intervals(run_name, param_names=None, param_space=None, dof_tot=None, npspec=None):
    """
    Read GA output and compute the best-fit values and 1σ/2σ intervals,
    using the same logic as func_GA_analysis.get_uncertainties.

    Parameters
    ----------
    run_name : str
    param_names : list of str
        Parameter names to extract (must match column names in chi2.txt).
        Defaults to PARAM_NAMES.
    param_space : list of (min, max, step) tuples  or  None
        If provided, the step is subtracted/added to the interval edges
        (as the original code does).  If None, step = 0 for all params.
    dof_tot : int or None
        Degrees of freedom (n_data_points - n_free_params).
        Required for the P-value method.  If None, npspec is also ignored
        and the RMSEA method is always used.
    npspec : int or None
        Number of spectral data points used.  Required for RMSEA calculation.

    Returns
    -------
    result : dict with keys:
        'best'      : dict  {param: best_fit_value}
        'lower_1sig': dict  {param: lower bound}
        'upper_1sig': dict  {param: upper bound}
        'lower_2sig': dict  {param: lower bound}
        'upper_2sig': dict  {param: upper bound}
        'method'    : str   'Pval_chi2' or 'RMSEA'
        'best_run_id': str
        'df'        : pd.DataFrame (the full GA population)
    """
    param_names = param_names or PARAM_NAMES

    # Build step-size lookup (default 0 if param_space not given)
    if param_space is not None:
        steps = {name: sp[2] for name, sp in zip(param_names, param_space)}
    else:
        steps = {name: 0 for name in param_names}

    df = read_ga_results(run_name)

    # ---- choose statistic ----
    best_rchi2 = df["rchi2"].min()
    if dof_tot is not None and npspec is not None and best_rchi2 <= RMSEA_THRESHOLD:
        method = "Pval_chi2"
    else:
        method = "RMSEA"

    xbest      = df["rchi2"].idxmin()
    best_run_id = df.loc[xbest, "run_id"]

    # ---- build selection masks ----
    if method == "Pval_chi2":
        # P-value: normalise chi2 by its minimum, compute chi2 survival function
        chi2_vals  = df["chi2"].values
        scaling    = chi2_vals.min()
        chi2_norm  = chi2_vals * dof_tot / scaling
        pvals      = np.array([stats.chi2.sf(c, dof_tot) for c in chi2_norm])
        df["P-value"] = pvals
        ind_1sig = pvals >= 0.317
        ind_2sig = pvals >= 0.0455
    else:
        # RMSEA fallback (used when best rchi2 > 1.5 or dof unknown)
        if npspec is not None and dof_tot is not None:
            rmsea = np.sqrt(np.maximum(df["chi2"].values - dof_tot, 0)/ (dof_tot * (npspec - 1)))
        else:
            # Approximate RMSEA from rchi2 alone
            rmsea = np.sqrt(np.maximum(df["rchi2"].values - 1.0, 0))
        df["RMSEA"]  = rmsea
        min_rmsea    = rmsea.min()
        ind_1sig = rmsea <= min_rmsea * 1.04
        ind_2sig = rmsea <= min_rmsea * 1.09

    # ---- extract intervals ----
    best       = {}
    lower_1sig = {}
    upper_1sig = {}
    lower_2sig = {}
    upper_2sig = {}

    for name in param_names:
        if name not in df.columns:
            print(f"WARNING: '{name}' not in chi2.txt columns, skipping.")
            continue
        step = steps[name]
        vals_1 = df.loc[ind_1sig, name]
        vals_2 = df.loc[ind_2sig, name]
        best[name]       = df.loc[xbest, name]
        lower_1sig[name] = vals_1.min() - step
        upper_1sig[name] = vals_1.max() + step
        lower_2sig[name] = vals_2.min() - step
        upper_2sig[name] = vals_2.max() + step

    return {
        "best":       best,
        "lower_1sig": lower_1sig,
        "upper_1sig": upper_1sig,
        "lower_2sig": lower_2sig,
        "upper_2sig": upper_2sig,
        "method":     method,
        "best_run_id": best_run_id,
        "best_rchi2": best_rchi2,
        "best_chi2": df.loc[xbest, "chi2"],
        "df":         df,
        "runtime_seconds": read_ga_runtime(run_name),
        "ind_1sig": ind_1sig,
        "ind_2sig": ind_2sig,
        "npspec": npspec,
        "dof_tot": dof_tot,
    }


def planck_wavelength(wave_angstrom, temp):
    ''' Calculate the Planck function as function of temperature and wavelength (in Angstrom. Output is then also in Angstrom).'''
    angstrom_to_cm = 1e-8
    wave = wave_angstrom * angstrom_to_cm
    # All units in cgs
    hh = 6.6260755e-27 #Planck constant;
    kk = 1.380658e-16 #Boltzmann constant;

    prefactor = 2.0 * hh * cc**2 / (wave**5)
    exponent = (hh * cc / kk) / (wave * temp)
    Blambda = prefactor * (1.0 / (np.exp(exponent)-1))
    Blambda = Blambda * angstrom_to_cm #Blambda from per cm to per angstrom
    return Blambda

# Distance constant: 10 pc expressed in solar radii (the reference distance
# for absolute magnitudes).  All flux <-> magnitude conversions use this.
_RSUN_CM     = 6.96e10
_PARSEC_CM   = 3.08567758e18
_D_10PC_RSUN = 10.0 * _PARSEC_CM / _RSUN_CM   # 10 pc in solar radii


def flux_to_magnitude(obsflux, zp=None):
    '''Convert a flux at 10 pc (in the same units as the zeropoint) to an
    absolute magnitude.
 
    Parameters
    ----------
    obsflux : float or array
        Band-integrated flux at 10 pc distance (erg/s/cm²/Å, surface-flux
        diluted by (R / d_10pc)²).
    zp : float or None
        Zeropoint flux in the same units.  Defaults to the SPHERE_Ks Vega
        zeropoint loaded at module level.
    '''
    if zp is None:
        zp = zpflux
    return -2.5 * np.log10(obsflux / zp)


def compute_obs_flux(teff, radius, Tfrac=0.9, d=50e3*3.08567758e18/6.96e10):
    ''' Calculate the observed flux in the K-band based on the given parameters.
    teff: effective temperature (K)
    radius: stellar radius (solar radii)
    Tfrac: fraction of teff to use for the blackbody calculation (default 0.9 to account for line formation in cooler layers)
    d: distance to the star (default 50 kpc in cm (distance to LMC), converted to solar radii)
    '''
    tBB = teff * Tfrac
    F_lambda = np.pi * planck_wavelength(_KS_WAVE, tBB)
    filtered_flux = np.trapz(_KS_TRANS * F_lambda, _KS_WAVE) / _KS_NORM
    return (radius / _D_10PC_RSUN)**2 * filtered_flux

def simulate_magnitude(teff, radius, band=None, zp_system='vega',
                       Tfrac=0.9, filterdir=None):
    '''Simulate the absolute magnitude of a star in a given photometric band.
 
    Uses the same forward model as compute_obs_flux / flux_to_magnitude for
    the default SPHERE_Ks band, or loads an arbitrary filter when `band` is
    given.  Absolute magnitude is defined at d = 10 pc.
 
    Parameters
    ----------
    teff      : float – effective temperature in K
    radius    : float – stellar radius in solar radii
    band      : str or None
        Photometric band name (e.g. '2MASS_Ks', 'VISTA_Ks', 'Johnson_V').
        If None, the module-level SPHERE_Ks filter is used (fast path).
    zp_system : str – 'vega', 'AB', or 'ST'
    Tfrac     : float – blackbody temperature scaling (default 0.9)
    filterdir : str or None – path to filter directory; defaults to ppp.filter_path
 
    Returns
    -------
    float : absolute magnitude in the requested band
    '''
    if band is None:
        # Fast path: use the pre-loaded SPHERE_Ks arrays
        obs_flux = compute_obs_flux(teff, radius, Tfrac=Tfrac)
        return flux_to_magnitude(obs_flux, zp=zpflux)
 
    # General path: load the requested filter
    wave, trans, zp = _load_filter(band, zp_system,
                                   filterdir or ppp.filter_path)
    tBB = teff * Tfrac
    F_lambda = np.pi * planck_wavelength(wave, tBB)
    filtered_flux = np.trapz(trans * F_lambda, wave) / np.trapz(trans, wave)
    obs_flux = (radius / _D_10PC_RSUN)**2 * filtered_flux
    return flux_to_magnitude(obs_flux, zp=zp)
 
 
def magnitude_to_radius(teff, band, obsmag, zp_system='vega',
                        Tfrac=0.9, filterdir=None):
    '''Estimate the stellar radius given an absolute magnitude and temperature.
 
    Inverse of simulate_magnitude: solves for R such that
        simulate_magnitude(teff, R, band, ...) == obsmag
 
    Parameters
    ----------
    teff      : float – effective temperature in K
    band      : str   – photometric band name
    obsmag    : float – observed absolute magnitude (dereddened)
    zp_system : str   – 'vega', 'AB', or 'ST'
    Tfrac     : float – blackbody temperature scaling (default 0.9)
    filterdir : str or None
 
    Returns
    -------
    float : stellar radius in solar radii
    '''
    wave, trans, zp = _load_filter(band, zp_system, filterdir or ppp.filter_path)
    tBB = teff * Tfrac
    F_lambda = np.pi * planck_wavelength(wave, tBB)
    filtered_flux = np.trapz(trans * F_lambda, wave) / np.trapz(trans, wave)
 
    # Invert:  obsmag = -2.5*log10( (R/d_10pc)^2 * filtered_flux / zp )
    # => (R/d_10pc)^2 = zp * 10^(-obsmag/2.5) / filtered_flux
    obs_flux = zp * 10**(-obsmag / 2.5)
    radius_rsun = _D_10PC_RSUN * np.sqrt(obs_flux / filtered_flux)
    return radius_rsun
 
 
def _load_filter(band, zp_system, filterdir):
    '''Load filter transmission curve and zeropoint flux.
 
    Returns
    -------
    wave  : np.ndarray – wavelengths in Angstrom
    trans : np.ndarray – transmission (0–1)
    zp    : float      – zeropoint flux in erg/s/cm²/Å
    '''
    band_map = {
        'SPHERE_Ks': ('SPHERE_IRDIS_B_Ks.dat', 'nm'),
        'HST_555w':  ('HST_ACS_HRC.F555W.dat',  'angstrom'),
        '2MASS_Ks':  ('2MASS_Ks.dat',            'angstrom'),
        'VISTA_Ks':  ('Paranal_VISTA.Ks.dat',    'angstrom'),
        'Johnson_V': ('GCPD_Johnson.V.dat',       'angstrom'),
        'Johnson_J': ('Generic_Johnson.J.dat',    'angstrom'),
    }
    if band not in band_map:
        raise ValueError(f'Unknown band: {band!r}. '
                         f'Available: {list(band_map.keys())}')
 
    filterfile, waveunit = band_map[band]
    wave, trans = np.genfromtxt(os.path.join(filterdir, filterfile),
                                comments='#').T
    if waveunit == 'nm':
        wave *= 10  # nm → Å
 
    col = {'vega': 1, 'AB': 2, 'ST': 3}.get(zp_system)
    if col is None:
        raise ValueError(f'Unknown zp_system: {zp_system!r}. '
                         f"Choose from 'vega', 'AB', 'ST'.")
 
    zp_values = np.genfromtxt(os.path.join(filterdir, 'zero_points.dat'),
                               comments='#', dtype=str)
    for row in zp_values:
        if row[0] == band:
            return wave, trans, float(row[col])
 
    raise ValueError(f'Zero point for band {band!r} not found in '
                     f'zero_points.dat')
 
 
def get_radius_intervals_from_kband(result, kband_mag, distance_kpc=None,
                                    band='SPHERE_Ks', extinction_ak=0.0, kband_unc=0.0):
    """
    Compute best-fit radius and 1σ/2σ intervals from a K-band magnitude.

    Parameters
    ----------
    result : dict
        Output of get_ga_best_and_intervals().
    kband_mag : float
        Observed (apparent) K-band magnitude, or absolute if distance_kpc=None.
    distance_kpc : float or None
        Distance in kpc. If None, kband_mag is treated as already absolute.
    band : str
        Photometric band name.
    extinction_ak : float
        K-band extinction in magnitudes.
    kband_unc : float
        1σ uncertainty on the (absolute) magnitude.  When > 0, the resulting
        radius uncertainty is added in quadrature to the spectral interval,
        mirroring func_GA_analysis.add_anchor_magnitude_uncertainty.

    Returns
    -------
    dict with keys: best, lower_1sig, upper_1sig, lower_2sig, upper_2sig
    """
    df       = result["df"]
    ind_1sig = result["ind_1sig"]
    ind_2sig = result["ind_2sig"]

    if distance_kpc is not None:
        distance_modulus = 5.0 * np.log10(distance_kpc * 1e3) - 5.0
        abs_kband_mag = kband_mag - distance_modulus - extinction_ak
    else:
        abs_kband_mag = kband_mag

    def invert_radius(teff):
        try:
            return magnitude_to_radius(teff, band, abs_kband_mag)
        except Exception:
            return np.nan

    xbest       = df["rchi2"].idxmin()
    best_radius = invert_radius(df.loc[xbest, "teff"])

    radii_1sig = np.array([invert_radius(t) for t in df.loc[ind_1sig, "teff"]])
    radii_2sig = np.array([invert_radius(t) for t in df.loc[ind_2sig, "teff"]])

    radii_1sig = radii_1sig[~np.isnan(radii_1sig)]
    radii_2sig = radii_2sig[~np.isnan(radii_2sig)]

    if radii_1sig.size == 0:
        raise ValueError("No valid radii in 1σ interval.")
    if radii_2sig.size == 0:
        raise ValueError("No valid radii in 2σ interval.")

    # --- spectral uncertainty half-widths (from teff spread) ---
    lo1_spec = best_radius - radii_1sig.min()
    hi1_spec = radii_1sig.max() - best_radius
    lo2_spec = best_radius - radii_2sig.min()
    hi2_spec = radii_2sig.max() - best_radius

    # --- magnitude uncertainty contribution ---
    # R ∝ 10^(-m/5)  =>  δR/R = (ln10/5) * δm
    # Evaluate at ±1σ and ±2σ in magnitude to get δR, matching the
    # max_rad / min_rad pattern in func_GA_analysis.
    if kband_unc  > 0.0:
        r_mag_plus1  = magnitude_to_radius(df.loc[xbest, "teff"], band,
                                               abs_kband_mag + kband_unc)
        r_mag_minus1 = magnitude_to_radius(df.loc[xbest, "teff"], band,
                                               abs_kband_mag - kband_unc)
        r_mag_plus2  = magnitude_to_radius(df.loc[xbest, "teff"], band,
                                               abs_kband_mag + 2 * kband_unc)
        r_mag_minus2 = magnitude_to_radius(df.loc[xbest, "teff"], band,
                                               abs_kband_mag - 2 * kband_unc)

        # half-range at 1σ and 2σ in magnitude
        delta_r_mag1 = 0.5 * abs(r_mag_minus1 - r_mag_plus1)
        delta_r_mag2 = 0.5 * abs(r_mag_minus2 - r_mag_plus2)

        # add in quadrature (symmetric assumption)
        lo1 = np.sqrt(lo1_spec**2 + delta_r_mag1**2)
        hi1 = np.sqrt(hi1_spec**2 + delta_r_mag1**2)
        lo2 = np.sqrt(lo2_spec**2 + delta_r_mag2**2)
        hi2 = np.sqrt(hi2_spec**2 + delta_r_mag2**2)
    else:
        lo1, hi1 = lo1_spec, hi1_spec
        lo2, hi2 = lo2_spec, hi2_spec

    return {
        "best":       best_radius,
        "lower_1sig": best_radius - lo1,
        "upper_1sig": best_radius + hi1,
        "lower_2sig": best_radius - lo2,
        "upper_2sig": best_radius + hi2,
    }

import matplotlib.pyplot as plt

def plot_seds_for_radii(teff, radii, band='SPHERE_Ks', Tfrac=0.9,
                         wave_min=10000, wave_max=30000, n_wave=500,
                         filterdir=None, ax=None):
    '''Plot SEDs (surface flux scaled to 10 pc) for a fixed Teff and a list
    of radii, overplotted, with a vertical line marking the band's
    effective/pivot wavelength used for the K-band magnitude.

    Parameters
    ----------
    teff      : float - effective temperature in K
    radii     : list/array of float - stellar radii in solar radii
    band      : str - photometric band name (used to load wavelength grid
                for the vertical line and, if not None, for flux computation
                with that filter's wavelength range)
    Tfrac     : float - blackbody temperature scaling (default 0.9)
    wave_min, wave_max : float - wavelength range (Å) for the SED curve
    n_wave    : int - number of wavelength points
    filterdir : str or None - path to filter directory
    ax        : matplotlib axis or None - if provided, plot on this axis

    Returns
    -------
    ax : matplotlib axis with the plot
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    tBB = teff * Tfrac
    wave = np.linspace(wave_min, wave_max, n_wave)

    # Determine the reference wavelength for the vertical line.
    # Use the transmission-weighted mean wavelength of the filter.
    if band == 'SPHERE_Ks' or band is None:
        line_wave = np.trapz(_KS_TRANS * _KS_WAVE, _KS_WAVE) / np.trapz(_KS_TRANS, _KS_WAVE)
    else:
        filt_wave, filt_trans, _ = _load_filter(band, 'vega', filterdir or ppp.filter_path)
        line_wave = np.trapz(filt_trans * filt_wave, filt_wave) / np.trapz(filt_trans, filt_wave)

    for radius in radii:
        F_lambda = np.pi * planck_wavelength(wave, tBB)
        flux_10pc = (radius / _D_10PC_RSUN)**2 * F_lambda
        ax.plot(wave, flux_10pc, label=f'R = {radius:.2f} R$_\\odot$')

    ax.axvline(line_wave, color='k', linestyle='--', alpha=0.6,
               label=f'{band} ({line_wave:.0f} Å)')

    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel(r'Flux at 10 pc (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
    ax.set_yscale('log')
    ax.set_title(f'SED for $T_\\mathrm{{eff}}$ = {teff} K (BB scaled at {Tfrac}$\\times T_\\mathrm{{eff}}$)')
    ax.legend()

    return ax

# ===========================================================================
# 4.  COVERAGE TEST
# ===========================================================================

# ── Region labels (order must match check_ga_coverage_single's return order) ──
REGION_KEYS = [
    "below_2sig",
    "between_2sig_1sig_lo",
    "between_1sigs",
    "between_1sig_2sig_hi",
    "above_2sig",
]
REGION_LABELS = {
    "below_2sig":             "< 2σ low",
    "between_2sig_1sig_lo":   "2σ–1σ lo",
    "between_1sigs":          "inside 1σ",
    "between_1sig_2sig_hi":   "1σ–2σ hi",
    "above_2sig":             "> 2σ hi",
}
# Expected fractions for a well-calibrated Gaussian
REGION_EXPECTED = {
    "below_2sig":             0.025,
    "between_2sig_1sig_lo":   0.135,
    "between_1sigs":          0.683,
    "between_1sig_2sig_hi":   0.135,
    "above_2sig":             0.025,
}

def check_ga_coverage_single(result, theta_true, param_names=None):
    """
    For one completed GA run, classify each parameter's true value into
    one of five regions relative to the 1σ and 2σ intervals.

    Parameters
    ----------
    result : dict
        Output of get_ga_best_and_intervals(). Must contain keys
        'lower_1sig', 'upper_1sig', 'lower_2sig', 'upper_2sig',
        each mapping param name -> float.
    theta_true : array-like, shape (n_params,)
        True parameter values in the same order as param_names.
    param_names : list of str or None

    Returns
    -------
    hits_below_2sig : dict {param: bool}
        True if truth < lower_2sig  (outside 2σ on the low side).
    hits_between_2sig_1sig_lo : dict {param: bool}
        True if lower_2sig <= truth <= lower_1sig.
    hits_between_1sigs : dict {param: bool}
        True if lower_1sig <= truth <= upper_1sig  (inside 1σ).
    hits_between_1sig_2sig_hi : dict {param: bool}
        True if upper_1sig <= truth <= upper_2sig.
    hits_above_2sig : dict {param: bool}
        True if truth > upper_2sig  (outside 2σ on the high side).
    """

    param_names = param_names or PARAM_NAMES
    hits_below_2sig             = {}
    hits_between_2sig_1sig_lo   = {}
    hits_between_1sigs          = {}
    hits_between_1sig_2sig_hi   = {}
    hits_above_2sig             = {}

    for j, name in enumerate(param_names):
        if (name not in result.get("lower_1sig", {}) or
                name not in result.get("upper_1sig", {}) or
                name not in result.get("lower_2sig", {}) or
                name not in result.get("upper_2sig", {})):
            continue
        lo_2sig = result["lower_2sig"][name]
        lo_1sig = result["lower_1sig"][name]
        hi_1sig = result["upper_1sig"][name]
        hi_2sig = result["upper_2sig"][name]
        hits_below_2sig[name] = (theta_true[j] < lo_2sig)
        hits_between_2sig_1sig_lo[name] = (lo_2sig <=theta_true[j] <= lo_1sig)
        hits_between_1sigs[name] = (lo_1sig <= theta_true[j] <= hi_1sig)
        hits_between_1sig_2sig_hi[name] = (hi_1sig <= theta_true[j] <= hi_2sig)
        hits_above_2sig[name] = (theta_true[j] > hi_2sig)

    return (hits_below_2sig, hits_between_2sig_1sig_lo, hits_between_1sigs, hits_between_1sig_2sig_hi, hits_above_2sig)

def _print_coverage_summary(coverage_dicts: dict[str, dict[str, list]], param_names: list) -> None:
    """
    Print a coverage summary table to stdout.

    Parameters
    ----------
    coverage_dicts : dict {region_key: {param: list of bool or None}}
        Keyed by the five REGION_KEYS; built inside run_ga_coverage_test.
    param_names : list of str
    """
    col_w = 14  # width of each region column

    header_parts = [f"{'Param':10s}"] + [f"{REGION_LABELS[r]:>{col_w}s}" for r in REGION_KEYS] + [f"{'N':>5s}"]
    divider_parts = ["-" * 10] + ["-" * col_w for _ in REGION_KEYS] + ["-" * 5]

    print(f"\n{'='*60}")
    print("Coverage summary (fraction of simulations per region):")
    print("  Expected (well-calibrated Gaussian): " + "  ".join(f"{REGION_LABELS[r]}={REGION_EXPECTED[r]*100:.1f}%" for r in REGION_KEYS))
    print("  " + "  ".join(header_parts))
    print("  " + "  ".join(divider_parts))

    for name in param_names:
        # Use the 'between_1sigs' region to determine N (all regions have same length)
        raw = coverage_dicts["between_1sigs"].get(name, [])
        n = sum(1 for h in raw if h is not None)

        if n == 0: #If every simulation failed for this parameter, we can't compute any fractions, so we print a row of N/As and skip to the next parameter.
            na_parts = [f"{'N/A':>{col_w}s}" for _ in REGION_KEYS]
            print("  " + "  ".join([f"{name:10s}"] + na_parts + [f"{0:>5d}"]))
            continue

        row_parts = [f"{name:10s}"]
        for r in REGION_KEYS:
            hits = [h for h in coverage_dicts[r].get(name, []) if h is not None]
            f  = np.mean(hits) # the fraction of simulations where the truth fell in that region
            se = np.sqrt(f * (1 - f) / n) # the binomial standard error, which tells the uncertainty on that fraction given n simulations
            row_parts.append(f"{f*100:.1f}±{se*100:.1f}%".rjust(col_w))
        row_parts.append(f"{n:>5d}")
        print("  " + "  ".join(row_parts))


def run_ga_coverage_test(
    run_name_prefix,
    write_inputs_fn,
    simulate_fn=None,
    N_sims=20,
    param_names=None,
    param_space=None,
    prior=None,
    theta_true_all=None,
    dof_tot=None,
    npspec=None,
    n_cores=4,
    template_run=None,
    logdir=None,
):
    """
    Run a coverage test for the GA: simulate N datasets, run the GA on each,
    and record whether the truth falls inside the 1σ and 2σ interval.

    Because each GA run takes a long time, this function runs them
    sequentially and prints progress.  Results are saved after every
    simulation so you can inspect partial output.

    Parameters
    ----------
    run_name_prefix : str
        Each simulation gets a run name like "<prefix>_sim00", "_sim01", …
    write_inputs_fn : callable
        A function with signature:
            write_inputs_fn(run_name, theta_true, observed_flux, observed_wavelength, observed_kband)
        that writes all required input files for one simulation.
        You need to implement this yourself because it depends on your
        specific file formats (control.txt, line_list.txt etc.).
        See the docstring example below.
    N_sims : int
        Number of repeated simulations.
    param_names : list of str or None
    param_space : list of (min, max, step) or None
    dof_tot, npspec : int or None
        Passed through to get_ga_best_and_intervals.
    n_cores : int
        MPI cores per GA run.
    template_run : str or None
        If given, control.txt, defaults_fastwind.txt etc. are copied from
        this template run into each new run directory before launching.
    logdir : str or None
        If given, GA stdout is saved to <logdir>/<run_name>.log.

    Returns
    -------
    coverage_dict : dict {param: list of bool}
        For each parameter, a list of N_sims True/False values.
    theta_true_all : np.ndarray, shape (N_sims, n_params)
    results_all : list of result dicts from get_ga_best_and_intervals

    Example write_inputs_fn
    -----------------------
    def my_write_inputs(run_name, theta_true, observed_flux, observed_wavelength, observed_kband):
        # 1. Write the simulated spectrum
        write_spectrum_norm(run_name, observed_wavelength, observed_flux)
        # 2. Write radius_info using the simulated K-band magnitude
        write_radius_info(run_name, "2MASS_Ks", observed_kband)
        # 3. Copy static files from a template run
        copy_template_inputs(run_name, "my_template_run", files=["control.txt", "defaults_fastwind.txt", "line_list.txt", "parameter_space.txt"])
    """
    param_names = param_names or ["teff", "logg", "radius", "mdot", "yhe", "vrot"]
    n_params    = len(param_names)
    if logdir:
        os.makedirs(logdir, exist_ok=True)

    # Prior ranges — used to draw true parameters
    if prior is None:
        prior = {
            "teff":   (29000, 52000),
            "logg":   (3.4,   4.3),
            "radius": (6,     21),
            "mdot":   (-7.5,  -5.2),
            "yhe":    (0.08,  0.15),
            "vrot":   (50.0,  399.0),
        }

    # Draw true parameters from the prior
    if theta_true_all is None:
        theta_true_all = np.zeros((N_sims, n_params))
        for j, name in enumerate(param_names):
            lo, hi = prior.get(name, (0, 1))
            theta_true_all[:, j] = np.random.uniform(lo, hi, size=N_sims)

    coverage_dicts = {r: {name: [] for name in param_names} for r in REGION_KEYS} 
    results_all   = []
    kband_all     = []
    runtime_seconds_all = []

    print(f"[run_ga_coverage_test] Starting {N_sims} simulations with "
          f"prefix '{run_name_prefix}'")

    for k in range(N_sims):
        run_name = f"{run_name_prefix}_sim{k:02d}"
        theta_k  = theta_true_all[k]

        print(f"\n{'='*60}")
        print(f"Simulation {k+1}/{N_sims}  |  run: {run_name}")
        print(f"  True params: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(param_names, theta_k)))

        # --- simulate data ---
        if simulate_fn is not None:
            observed_flux, observed_wavelength, observed_kband = simulate_fn(theta_k)
        else:
            observed_flux, observed_wavelength, observed_kband = _simulate_data_for_coverage(theta_k)
        kband_all.append(observed_kband)

        # --- write input files ---
        write_inputs_fn(run_name, theta_k, observed_flux, observed_wavelength, observed_kband)

        # --- optionally copy static inputs from template ---
        if template_run is not None:
            copy_template_inputs(run_name, template_run, files=["control.txt", "defaults_fastwind.txt"])

        # --- run the GA ---
        logfile = (f"{logdir}/{run_name}.log" if logdir else None)
        run_ga(run_name, n_cores=n_cores, block=True, logfile=logfile)

        # --- read results and check coverage ---
        try:
            result = get_ga_best_and_intervals(
                run_name,
                param_names=param_names,
                param_space=param_space,
                dof_tot=dof_tot,
                npspec=npspec,
            )
            (hits_below_2sig, hits_between_2sig_1sig_lo, hits_between_1sigs, hits_between_1sig_2sig_hi, 
             hits_above_2sig) = check_ga_coverage_single(result, theta_k, param_names=param_names)

            region_hits = {
                "below_2sig":             hits_below_2sig,
                "between_2sig_1sig_lo":   hits_between_2sig_1sig_lo,
                "between_1sigs":          hits_between_1sigs,
                "between_1sig_2sig_hi":   hits_between_1sig_2sig_hi,
                "above_2sig":             hits_above_2sig,
            }

            results_all.append(result)
            runtime_seconds_all.append(result.get("runtime_seconds", None))

            # Derive 1σ / 2σ hits from the regions for the per-sim print
            hits_1sig = hits_between_1sigs
            hits_2sig = {
                name: (hits_between_2sig_1sig_lo.get(name, False)
                       or hits_between_1sigs.get(name, False)
                       or hits_between_1sig_2sig_hi.get(name, False))
                for name in param_names
            }

            print("  Best fit:  " + ", ".join(f"{n}={result['best'].get(n, float('nan')):.4g}" for n in param_names))
            print("  Coverage 1σ:  " + ", ".join(f"{n}={'✓' if v else '✗'}" for n, v in hits_1sig.items()))
            print("  Coverage 2σ:  " +  ", ".join(f"{n}={'✓' if v else '✗'}" for n, v in hits_2sig.items()))
            print(f" Runtime: {result.get('runtime_seconds', 'N/A')} seconds")


            for name in param_names:
                for r in REGION_KEYS:
                    coverage_dicts[r][name].append(region_hits[r].get(name))

        except Exception as e:
            print(f"  WARNING: Could not read results for {run_name}: {e}")
            results_all.append(None)
            for name in param_names:
                for r in REGION_KEYS:
                    coverage_dicts[r][name].append(None)
    
    # Print summary for both 1σ and 2σ
    _print_coverage_summary(coverage_dicts, param_names)
    print(f"Average runtime per simulation: {np.mean([r.get('runtime_seconds', 0) for r in results_all if r is not None]):.1f} seconds")

    np.savez(
        os.path.join(OUTPUT_DIR, run_name_prefix + "_coverage_summary.npz"),
        theta_true_all=theta_true_all,
        kband_all=np.array(kband_all),
        coverage_dicts=coverage_dicts,  # needs allow_pickle=True on load
        results_all=results_all,        # needs allow_pickle=True on load
        runtime_seconds_all=runtime_seconds_all,
    )
    return coverage_dicts, theta_true_all, results_all, np.array(kband_all)

def load_ga_coverage_summary(run_name_prefix):
    path = os.path.join(OUTPUT_DIR, run_name_prefix + "_coverage_summary.npz")
    d = np.load(path, allow_pickle=True)
    return d["coverage_dicts"].item(), d["theta_true_all"], d["kband_all"], d["results_all"], d["runtime_seconds_all"]

def plot_sigma_regions(coverage_dicts, kband_unc=0.0, param_names=None, results=None, kband_all=None, theta_true_all=None, radius_param_index=2):
    """
    One subplot per parameter. Each subplot shows a histogram-style bar chart
    across the five interval regions, with a reference Gaussian curve overlaid.

    Parameters
    ----------
    coverage_dicts : dict {region_key: {param: list of bool or None}}
        Output of run_ga_coverage_test.
    param_names : list of str or None
    """
    param_names  = param_names or PARAM_NAMES
    n_params     = len(param_names)
    n_cols       = 3
    n_rows       = int(np.ceil(n_params / n_cols))

    region_colors = {
        "below_2sig":             "#d62728",
        "between_2sig_1sig_lo":   "#ff7f0e",
        "between_1sigs":          "#2ca02c",
        "between_1sig_2sig_hi":   "#ff7f0e",
        "above_2sig":             "#d62728",
    }
    region_x = {r: i for i, r in enumerate(REGION_KEYS)}   # x position per bar
    x_ticks   = list(range(len(REGION_KEYS)))
    x_labels  = ["< 2σ lo", "2σ–1σ lo", "inside 1σ", "1σ–2σ hi", "> 2σ hi"]

    # Expected Gaussian fractions (smooth curve reference)
    expected = np.array([REGION_EXPECTED[r] * 100 for r in REGION_KEYS])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=False)
    axes = np.array(axes).flatten()   # always 1-D for easy indexing

    for idx, name in enumerate(param_names):
        ax = axes[idx]

        # Special handling for radius
        if (name == "radius" and results is not None and kband_all is not None and theta_true_all is not None):

            region_counts = {r: 0 for r in REGION_KEYS}
            n = 0
            for k, result in enumerate(results):
                if result is None:
                    continue
                ri = get_radius_intervals_from_kband(result, kband_all[k], kband_unc=kband_unc)
                truth = theta_true_all[k, radius_param_index]
                lo2, lo1 = ri["lower_2sig"], ri["lower_1sig"]
                hi1, hi2 = ri["upper_1sig"], ri["upper_2sig"]

                if truth < lo2:
                    region_counts["below_2sig"] += 1
                elif lo2 <= truth <= lo1:
                    region_counts["between_2sig_1sig_lo"] += 1
                elif lo1 <= truth <= hi1:
                    region_counts["between_1sigs"] += 1
                elif hi1 <= truth <= hi2:
                    region_counts["between_1sig_2sig_hi"] += 1
                else:
                    region_counts["above_2sig"] += 1
                n += 1

            fracs = [region_counts[r] / n * 100 if n > 0 else 0.0 for r in REGION_KEYS]
            errs  = [np.sqrt((f/100) * (1 - f/100) / n) * 100 if n > 0 else 0.0 for f in fracs]

        # ── All other parameters: use coverage_dicts as before ──────────────
        else:
            raw = coverage_dicts["between_1sigs"].get(name, [])
            n   = sum(1 for h in raw if h is not None)
            fracs, errs = [], []
            for r in REGION_KEYS:
                hits = [h for h in coverage_dicts[r].get(name, []) if h is not None]
                if n == 0:
                    fracs.append(0.0); errs.append(0.0)
                else:
                    f  = np.mean(hits)
                    se = np.sqrt(f * (1 - f) / n)
                    fracs.append(f * 100); errs.append(se * 100)

        # Plotting (shared)
        bars = ax.bar(x_ticks, fracs, color=[region_colors[r] for r in REGION_KEYS], yerr=errs, capsize=4, alpha=0.8, ecolor="black", width=0.7)
        ax.plot(x_ticks, expected, color="black", linestyle="--", linewidth=1.5, label="Expected (Gaussian)")
        ax.set_title(name, fontsize=13)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Fraction of simulations (%)", fontsize=9)
        ax.set_ylim(0, max(max(fracs) + max(errs) + 12, 45))
        ax.legend(fontsize=7, loc="upper right")
        if n > 0:
            ax.text(0.02, 0.97, f"N = {n}", transform=ax.transAxes, fontsize=8, va="top", color="gray")

    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("coverage test — sigma region histogram", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_ga_coverage_pp(coverage_dicts, kband_unc=0.0, param_names=None, results=None, kband_all=None, theta_true_all=None, radius_param_index=2):
    param_names = param_names or PARAM_NAMES
    fig, ax = plt.subplots(figsize=(7, 7))

    x_theoretical = np.array([0.025, 0.158, 0.841, 0.975])
    x_fine = np.linspace(0, 1, 100)
    ax.plot([0, 1], [0, 1], "--", color="black", label="Ideal")

    # Get N from any non-radius param for the binomial bands
    first_p = next(p for p in param_names if p != "radius")
    N = sum(1 for h in coverage_dicts["between_1sigs"][first_p] if h is not None)

    for ci, alpha in zip([0.68, 0.95], [0.2, 0.1]):
        edge = (1.0 - ci) / 2.0
        lower = binom.ppf(edge, N, x_fine) / N
        upper = binom.ppf(1 - edge, N, x_fine) / N
        ax.fill_between(x_fine, lower, upper, color="grey", alpha=alpha, zorder=0)

    for name in param_names:

        # Special handling for radius
        if (name == "radius"
                and results is not None
                and kband_all is not None
                and theta_true_all is not None):

            below_2sig = 0
            between_2sig_1sig_lo = 0
            between_1sigs = 0
            between_1sig_2sig_hi = 0
            n = 0
            for k, result in enumerate(results):
                if result is None:
                    continue
                ri = get_radius_intervals_from_kband(result, kband_all[k], kband_unc=kband_unc)
                truth = theta_true_all[k, radius_param_index]
                lo2, lo1 = ri["lower_2sig"], ri["lower_1sig"]
                hi1, hi2 = ri["upper_1sig"], ri["upper_2sig"]

                if truth < lo2:
                    below_2sig += 1
                elif lo2 <= truth <= lo1:
                    between_2sig_1sig_lo += 1
                elif lo1 <= truth <= hi1:
                    between_1sigs += 1
                elif hi1 <= truth <= hi2:
                    between_1sig_2sig_hi += 1
                n += 1

            if n == 0:
                continue
            f1 = below_2sig / n
            f2 = f1 + between_2sig_1sig_lo / n
            f3 = f2 + between_1sigs / n
            f4 = f3 + between_1sig_2sig_hi / n

        # ── All other parameters: use coverage_dicts as before ───────────────
        else:
            f1 = np.mean(coverage_dicts["below_2sig"][name])
            f2 = f1 + np.mean(coverage_dicts["between_2sig_1sig_lo"][name])
            f3 = f2 + np.mean(coverage_dicts["between_1sigs"][name])
            f4 = f3 + np.mean(coverage_dicts["between_1sig_2sig_hi"][name])

        y_empirical = np.array([f1, f2, f3, f4])
        x_plot = np.concatenate(([0], x_theoretical, [1]))
        y_plot = np.concatenate(([0], y_empirical, [1]))
        ax.plot(x_plot, y_plot, marker='o', markersize=4, label=name, lw=2)

    ax.set_xlabel("Expected Credible Interval (Gaussian)")
    ax.set_ylabel("Measured Fraction of Simulations")
    ax.set_title(f"GA Coverage PP-Plot ({N} sims)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_ga_best_vs_true(results, theta_true_all, kband_unc=0.0, kband_all=None, param_names=None):
    param_names = param_names or PARAM_NAMES
    n_params = len(param_names)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 4*n_rows), sharex=False)
    axes = np.array(axes).flatten()

    for p_idx, param in enumerate(param_names):
        ax = axes[p_idx]
        best = []
        lower_1 = []
        upper_1 = []
        lower_2 = []
        upper_2 = []
        true_vals = []
        for i, r in enumerate(results):
            if r is None:
                continue

            # for radius
            if param == "radius":
                ri = get_radius_intervals_from_kband(r,kband_all[i], kband_unc=kband_unc)
                best.append(ri["best"])
                lower_1.append(ri["lower_1sig"])
                upper_1.append(ri["upper_1sig"])
                lower_2.append(ri["lower_2sig"])
                upper_2.append(ri["upper_2sig"])

            # other parameters
            else:
                best.append(r["best"][param])
                lower_1.append(r["lower_1sig"][param])
                upper_1.append(r["upper_1sig"][param])
                lower_2.append(r["lower_2sig"][param])
                upper_2.append(r["upper_2sig"][param])

            true_vals.append(theta_true_all[i, p_idx])

        best = np.array(best)
        lower_1 = np.array(lower_1)
        upper_1 = np.array(upper_1)
        lower_2 = np.array(lower_2)
        upper_2 = np.array(upper_2)
        true_vals = np.array(true_vals)
        x = np.arange(len(best))  # simulations on x-axis

        # --- plot ---
        for i in range(len(x)):
            ax.vlines(x[i], lower_2[i], upper_2[i], linewidth=1, alpha=0.4) # 2σ (thin vertical)
            ax.vlines(x[i], lower_1[i], upper_1[i], linewidth=2) # 1σ (thick vertical)
            ax.plot(x[i], best[i], 'o', color='purple', markersize=4) # best
            ax.plot(x[i], true_vals[i], 'x', color='red', markersize=4) # true

        ax.set_title(param)
        ax.set_xlabel("Simulation")
        ax.set_xticks(x)
        ax.set_ylabel("Value")

    # remove unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    # legend (only once)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], lw=4, label="1σ"),
        Line2D([0], [0], lw=1, alpha=0.4, label="2σ"),
        Line2D([0], [0], marker='o', linestyle='None', label="Best", color='purple'),
        Line2D([0], [0], marker='x', linestyle='None', label="True", color='red'),
    ]
    fig.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()

# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _ensure_run_dir(run_name):
    """Create the run input directory (input/<run_name>/) if needed and return its path."""
    run_dir = os.path.join(INPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _simulate_data_for_coverage(theta_true):
    """
    Placeholder: replace with your actual simulate_model_spectrum,
    add_noise_to_spectrum, and simulate_kband_magnitude calls.
    theta_true shape depends on the active emulator:
      anja    : (6,)  [teff, logg, radius, logmdot, yhe, vsini]
      vasilis : (8,)  [teff, logg, radius, mdot, vinf, yhe, vturb, vsini]

    Returns
    -------
    observed_flux : np.ndarray
    observed_wavelength : np.ndarray
    observed_kband : float
    """
    raise NotImplementedError(
        "Replace _simulate_data_for_coverage with your own simulation calls.\n"
        "It should return (observed_flux, observed_wavelength, observed_kband).\n"
        "Example:\n"
        "  flux = simulate_model_spectrum(theta_true, output_wl=wl_array_output)\n"
        "  noisy_flux = add_noise_to_spectrum(flux, spectral_snr)\n"
        "  kband = simulate_kband_magnitude(theta_true) + np.random.normal(0, kband_err)\n"
        "  return noisy_flux, wl_array_output, kband"
    )

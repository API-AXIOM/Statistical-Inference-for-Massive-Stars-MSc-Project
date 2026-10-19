#!/usr/bin/env python3
"""Fit the 15-parameter model to a spectrum on one GPU, with emcee (MCMC) or
nested sampling.

README.md and GETTING_STARTED.md explain how to run it, what it reads and
what it writes; the options are listed below.
"""

import argparse
import json
import os
import sys
import time

# The repository folder: the one holding this file.
REPOSITORY = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPOSITORY, "src"))

import numpy as np

import data_io
from emulators import load_emulators
from gpu_config import BOUNDS_ARRAY, LMC_DISTANCE_PC, N_DIM, PARAMETER_NAMES
from map_estimate import initialise_walkers, multi_start_map
from samplers import resolve_move
import version


DEFAULT_TRUE_PARAMETERS = np.array([
    40000.0,            # teff
    3.8,                # logg
    15.0,               # radius
    np.log10(3e-7),     # log10 mdot
    0.12,               # yhe
    7.5,                # C
    7.8,                # N
    8.6,                # O
    1.2,                # beta
    2200.0,             # vinf
    0.3,                # fic
    0.5,                # fvel
    15.0,               # fclump
    100.0,              # vsini
    50.0,               # vmacro
])

class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Adds each option's default to its help text, when it has a real one."""

    def _get_help_string(self, action):
        text = action.help or ""
        default = action.default
        if (default in (None, False, argparse.SUPPRESS)
                or "default" in text or not action.option_strings):
            return text
        if isinstance(default, (list, tuple)):
            default = " ".join(str(item) for item in default)
        return f"{text} (default: {str(default).replace('%', '%%')})".strip()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=_HelpFormatter)
    # The 15 parameter names, as placeholders for options taking one value each.
    parameter_metavar = tuple(name.upper() for name in PARAMETER_NAMES)

    data = parser.add_argument_group("observation")
    data.add_argument("--data",
                      help="observed spectrum, .fits, .npz, or plain text "
                           "(.txt, .dat, .ascii, .norm). A .npz needs "
                           "'wavelength' and 'flux' arrays; a .fits holds a "
                           "table with those columns; a text file has columns "
                           "wavelength, flux and optionally error. Run "
                           "inspect_spectrum.py on it first to check how it "
                           "will be read.")
    data.add_argument("--fits-extension", type=int, default=None,
                      help="FITS extension holding the spectrum "
                           "(default: find it automatically)")
    data.add_argument("--wavelength-column", default=None,
                      help="name of the wavelength column in a FITS table")
    data.add_argument("--flux-column", default=None,
                      help="name of the flux column in a FITS table")
    data.add_argument("--mock", action="store_true",
                      help="generate a synthetic observation from --true-params")
    data.add_argument("--instrument", nargs="+", default=["dense"],
                      help="one or more entries in config/instruments.json "
                           "defining the wavelength grid and resolution; "
                           "name more than one to combine instruments that "
                           "measured different parts of the same spectrum "
                           "(e.g. --instrument LMC_stars_R45800 "
                           "LMC_stars_R6700)")
    data.add_argument("--instruments-file", default=None,
                      help="instrument definitions to use instead of "
                           "config/instruments.json")
    data.add_argument("--resolution", type=float, default=None,
                      help="override the instrument's spectral resolution")
    data.add_argument("--lines", nargs="*", default=None,
                      help="restrict the fit to these line names, as "
                           "inspect_spectrum.py lists them (default: every "
                           "line the data covers)")
    data.add_argument("--snr", type=float, default=100.0,
                      help="signal-to-noise per pixel: the fit assumes a flux "
                           "uncertainty of 1/SNR at every point (unless "
                           "--use-errors). Do NOT use the calculated SNR given "
                           "by inspect_spectrum.py if the spectrum does not "
                           "contain at least 500 points in the continuum! "
                           "--mock adds noise at SNR=100 if no SNR is given.")
    data.add_argument("--true-params", type=float, nargs=N_DIM, default=None,
                      metavar=parameter_metavar,
                      help="the true parameters for --mock, in this order "
                           "(mdot as log10); default: a built-in O star")
    data.add_argument("--no-noise", action="store_true",
                      help="mock without observational noise (debugging)")


    prior_group = parser.add_argument_group("prior")
    prior_group.add_argument("--prior", default="flat",
                             choices=["flat", "informative"],
                             help="'flat' is uniform over the parameter "
                                  "bounds; 'informative' uses LMC abundances, "
                                  "helium and wind relations")
    prior_group.add_argument("--no-relations", action="store_true",
                             help="drop the physical relations on mass, "
                                  "v_inf/v_esc and mass loss")
    prior_group.add_argument("--priors-file", default=None,
                             help="prior settings to use instead of "
                                  "config/priors.json (with --prior informative)")

    likelihood = parser.add_argument_group("likelihood")
    likelihood.add_argument("--use-errors", action="store_true",
                            help="weight each wavelength by its own "
                                 "uncertainty from the data file instead of "
                                 "assuming 1/SNR everywhere; needs an "
                                 "uncertainty column in the file")
    likelihood.add_argument("--error-column", default=None,
                            help="name of the uncertainty column in a FITS "
                                 "table (default: find it automatically)")

    photometry = parser.add_argument_group("photometry")
    photometry.add_argument("--no-magnitude", action="store_true",
                            help="fit the spectrum only; skip the SED/magnitude term")
    photometry.add_argument("--filter-name", default="2MASS_Ks",
                            help="photometric band of --magnitude: 2MASS_Ks, "
                                 "VISTA_Ks, SPHERE_Ks, Johnson_V or HST_555w")
    photometry.add_argument("--zp-system", default="vega",
                            choices=["vega", "AB", "ST"],
                            help="magnitude system of --magnitude")
    photometry.add_argument("--distance", type=float, default=LMC_DISTANCE_PC,
                            help="distance in parsec (default: the LMC, "
                                 "49.59 kpc, Sander et al. 2024)")
    photometry.add_argument("--magnitude", type=float, default=None,
                            help="observed magnitude in --filter-name; needed "
                                 "with --data unless --no-magnitude is given")
    photometry.add_argument("--magnitude-uncertainty", type=float, default=0.1,
                            help="1-sigma uncertainty of --magnitude, in "
                                 "magnitudes")
    photometry.add_argument("--extinction", type=float, default=0.0,
                            help="extinction A_lambda in the chosen filter, in "
                                 "magnitudes; the model SED is dimmed by this "
                                 "before comparison (default 0)")
    photometry.add_argument("--filter-dir", default=None,
                            help="folder of filter curves and zero_points.dat "
                                 "(default: filter_transmissions/ in the emulator "
                                 "directory)")

    sampler = parser.add_argument_group("sampler")
    sampler.add_argument("--sampler", default="emcee", choices=["emcee", "nested"],
                         help="emcee (default): ensemble MCMC from a MAP "
                              "estimate; nested: nested sampling from the "
                              "whole prior, which also gives the evidence")
    sampler.add_argument("--seed", type=int, default=42,
                         help="random seed for the mock noise, the MAP search "
                              "and the samplers")
    sampler.add_argument("--resume", action="store_true",
                         help="continue from an existing chain .h5 (emcee) or "
                              "ultranest/ checkpoint (nested) in --outdir")

    mcmc = parser.add_argument_group("MCMC (--sampler emcee)")
    mcmc.add_argument("--nwalkers", type=int, default=512,
                      help="number of emcee walkers; even, and at least "
                           f"{2 * N_DIM}")
    mcmc.add_argument("--nsteps", type=int, default=20000,
                      help="chain length per walker; with --resume, the "
                           "total length to reach")
    mcmc.add_argument("--move", default="stretch",
                      choices=["stretch", "de", "de-snooker"],
                      help="emcee proposal: stretch (default), de, or "
                           "de-snooker (an 80/20 DE + DE-snooker mixture)")
    mcmc.add_argument("--first-guess", type=float, nargs=N_DIM, default=None,
                      metavar=parameter_metavar,
                      help="starting point for the walkers, same order as "
                           "--true-params; only used together with "
                           "--skip-map (the MAP search draws its own random "
                           "starts)")
    mcmc.add_argument("--skip-map", action="store_true",
                      help="skip the MAP search and start the walkers "
                           "around --first-guess")
    mcmc.add_argument("--map-prescan", type=int, default=20000,
                      help="random prior draws evaluated before L-BFGS-B")
    mcmc.add_argument("--map-prescan-width", type=float, default=None,
                      help="if set, pre-scan a Gaussian ball of this "
                           "fraction of the prior range around the guess")
    mcmc.add_argument("--map-fd-step", type=float, default=1e-5,
                      help="finite-difference step, as a fraction of the "
                           "prior range of each parameter")
    mcmc.add_argument("--walker-scale", type=float, default=0.01,
                      help="initial walker ball, fraction of the prior range")
    mcmc.add_argument("--progress-every", type=int, default=200,
                      help="steps between progress lines in the log")

    nested = parser.add_argument_group(
        "nested sampling (--sampler nested)",
        "Defaults: fewer slice steps than 8 gave posteriors that missed the "
        "truth on test problems; extra walkers mostly add discarded work "
        "without shortening the run.")
    nested.add_argument("--n-live", type=int, default=600,
                        help="number of live points")
    nested.add_argument("--ns-method", default="slice",
                        choices=["slice", "randomwalk"],
                        help="how walkers move: slice (default) or randomwalk "
                             "(Metropolis jumps). randomwalk is experimental: "
                             "not validated, and can stop with 'walkers never "
                             "moved'. The 8-step default was "
                             "tuned for slice; randomwalk needs its own "
                             "--ns-steps")
    nested.add_argument("--ns-steps", type=int, default=8,
                        help="steps per new live point")
    nested.add_argument("--ns-direction", default=None,
                        help="step direction generator (see "
                             "samplers.STEP_DIRECTIONS); default mixture for "
                             "slice, region-random for randomwalk")
    nested.add_argument("--ns-scale", type=float, default=None,
                        help="initial step length, in units of the direction "
                             "vector (default 0.05 for slice, 1 for randomwalk)")
    nested.add_argument("--ns-walk-adapt", type=float, default=0.9,
                        help="randomwalk only (experimental): factor by which the step length "
                             "is adjusted after each set of steps; smaller "
                             "adapts faster (default 0.9)")
    nested.add_argument("--ns-walkersize", type=int, default=50,
                        help="walkers the population sampler advances "
                             "together; sets the GPU batch, not the statistics")
    nested.add_argument("--ns-max-calls", type=int, default=None,
                        help="stop after this many likelihood calls (for "
                             "timing tests; the posterior is then not trustworthy)")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=None, help="cuda, cuda:0, cpu ...")
    runtime.add_argument("--dtype", default="float32",
                         choices=["float32", "float64"],
                         help="precision of the emulator evaluation on the GPU")
    runtime.add_argument("--no-cuda-graph", action="store_true",
                         help="disable CUDA graph capture of the posterior "
                              "(it is verified against eager evaluation "
                              "before use, and falls back automatically)")
    runtime.add_argument("--tf32", action="store_true",
                         help="allow TF32 matmuls (faster, ~1e-3 relative "
                              "error in the emulator; off by default)")
    runtime.add_argument("--outdir", default=None,
                         help="name of the run, e.g. my_star: its outputs go "
                              "in results/mcmc/<name>/ or "
                              "results/nested_sampling/<name>/ inside the "
                              "repository folder (default: run). A full path "
                              "is used as given instead")
    runtime.add_argument("--no-plots", action="store_true",
                         help="skip the figures; results.npz and summary.txt "
                              "are still written")
    runtime.add_argument("--benchmark-only", action="store_true",
                         help="time the posterior and exit without sampling")

    return parser.parse_args(argv)


def resolve_device(requested):
    import torch
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; "
                               "is this running on a GPU node?")
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        print(f"[device] {device} -> {name} ({total:.1f} GiB)")
    else:
        print(f"[device] {device} (no GPU found; this will be slow)")
    return device


def timed_posterior_benchmark(posterior, n_walkers, repeats=5):
    """Report the batched throughput of the posterior."""
    import torch
    rng = np.random.default_rng(0)
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    batch = rng.uniform(lower, upper, size=(n_walkers, N_DIM))

    posterior(batch)  # warm-up: cuFFT plans, cuBLAS handles, autotuning
    if posterior.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        posterior(batch)
    if posterior.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats

    print(f"[bench] batch of {n_walkers}: {elapsed * 1e3:.2f} ms "
          f"({elapsed / n_walkers * 1e6:.1f} us per walker, "
          f"{n_walkers / elapsed:,.0f} evaluations/s)")
    return elapsed


def emcee_backend_has_samples(path):
    """True if an HDF5 backend at `path` already holds at least one step."""
    if not os.path.exists(path):
        return False
    try:
        import emcee
        return emcee.backends.HDFBackend(path).iteration > 0
    except Exception:
        return False


def write_outputs(arguments, posterior, observation, true_parameters,
                  flat_samples, sampler_results, sampling_line,
                  provenance=None, plot_extras=None):
    """Summarise the posterior samples and write the run's outputs.

    Prints the percentile table and the prior-boundary check, then writes
    results.npz, summary.txt and the figures. Everything both samplers store
    is filled in here; ``sampler_results`` adds what only one of them has.
    ``sampling_line`` is the cost line of summary.txt. ``provenance`` is
    (tau, discard, thin) for an MCMC run and None for nested sampling.
    ``plot_extras`` holds what the figures need beyond results.npz.
    """
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    boundary_records = data_io.prior_boundary_report(flat_samples)
    print(data_io.format_parameter_table(percentiles, true_parameters,
                                         boundary_records))
    boundary_warning = data_io.format_boundary_warning(boundary_records)
    if boundary_warning:
        print(boundary_warning)

    results = {
        **sampler_results,
        "flat_samples": flat_samples,
        "percentiles": percentiles,
        "parameters_50p": percentiles[1],
        "true_parameters": true_parameters,
        "observed_wavelength": observation["wavelength"],
        "observed_flux": observation["flux"],
        "observed_magnitude": observation["magnitude"],
        "resolution": observation["resolution"],
        "spectrum_snr": arguments.snr,
        "flux_uncertainty": posterior.flux_uncertainty,
        "parameter_names": PARAMETER_NAMES,
        "n_covered_points": posterior.n_covered_points,
        "pipeline_version": version.describe(),
        "parameters_at_prior_bound": [record["name"] for record in
                                      boundary_records if record["status"]],
        "settings": vars(arguments),
    }
    results_path = os.path.join(arguments.outdir, "results.npz")
    data_io.save_results(results_path, results)
    print(f"[post] results written to {results_path}")

    if provenance is None:
        provenance_block = data_io.format_provenance(
            arguments, posterior, results["pipeline_version"], None, None,
            None, len(flat_samples), nested=results)
    else:
        provenance_block = data_io.format_provenance(
            arguments, posterior, results["pipeline_version"], *provenance,
            len(flat_samples))
    with open(os.path.join(arguments.outdir, "summary.txt"), "w") as handle:
        handle.write(data_io.format_parameter_table(
            percentiles, true_parameters, boundary_records))
        if boundary_warning:
            handle.write("\n" + boundary_warning + "\n")
        handle.write(f"\n\nsampling: {sampling_line}\n")
        handle.write(version.summary_line() + "\n")
        handle.write(provenance_block + "\n")

    if not arguments.no_plots:
        import plots
        plots.make_all({**results, **(plot_extras or {})}, posterior,
                       arguments.outdir)
        print(f"[post] figures written to {arguments.outdir}")
    return 0


def fit_nested(arguments, posterior, rng, true_parameters, observation):
    """Nested sampling of the posterior, written out in the same form as an MCMC fit.

    Nested sampling explores from the whole prior, so it needs no MAP estimate
    or starting point. UltraNest checkpoints to <outdir>/ultranest/ as it goes,
    so a run cut off by a job time limit continues with --resume.
    """
    import samplers

    log_dir = os.path.join(arguments.outdir, "ultranest")
    if arguments.ns_direction is None:
        arguments.ns_direction = ("region-random"
                                  if arguments.ns_method == "randomwalk"
                                  else "mixture")
    method = ("random-walk" if arguments.ns_method == "randomwalk"
              else "slice")
    print(f"[nested] {arguments.n_live} live points; population {method} "
          f"sampler, {arguments.ns_steps} steps, {arguments.ns_walkersize} "
          f"walkers, {arguments.ns_direction} directions; "
          f"checkpoints in {log_dir}")
    result = samplers.run_nested(
        posterior, rng, n_live=arguments.n_live, log_dir=log_dir,
        max_calls=arguments.ns_max_calls,
        use_cuda_graph=(not arguments.no_cuda_graph
                        and posterior.device.type == "cuda"),
        step_sampler_steps=arguments.ns_steps,
        step_method=arguments.ns_method,
        step_scale=arguments.ns_scale,
        walk_adapt_factor=arguments.ns_walk_adapt,
        step_direction=arguments.ns_direction,
        walkersize=arguments.ns_walkersize, posterior_only=True,
        progress=False, verbose=True,
        resume="resume-similar" if arguments.resume else "overwrite")

    flat_samples = result["flat_samples"]
    capped = (arguments.ns_max_calls is not None
              and result["n_evaluations"] >= arguments.ns_max_calls)
    print(f"[post] {len(flat_samples)} equally weighted samples drawn from the "
          f"nested-sampling posterior")

    sampler_results = {
        "sampler": "nested",
        "weighted_points": result["weighted_points"],
        "weighted_weights": result["weighted_weights"],
        "parameters_map": result["parameters_map"],
        "log_evidence": result["log_evidence"],
        "log_evidence_error": result["log_evidence_error"],
        "effective_sample_size": result["effective_sample_size"],
        "information": result["information"],
        "insertion_test_passed": result["insertion_test_passed"],
        "n_live": result["n_live"],
        "n_evaluations": result["n_evaluations"],
        "stopped_at_call_cap": capped,
        "sampling_seconds": result["seconds"],
    }
    sampling_line = (f"{result['seconds'] / 60:.2f} min, nested sampling with "
                     f"{result['n_live']} live points, "
                     f"{result['n_evaluations']:,} likelihood calls")
    return write_outputs(arguments, posterior, observation, true_parameters,
                         flat_samples, sampler_results, sampling_line)


def fit_emcee(arguments, posterior, rng, true_parameters, observation):
    """Ensemble MCMC with emcee, from a multi-start MAP estimate.

    The chain is written to <outdir>/chain.h5 after every step, so a run cut
    off by a job time limit continues with --resume.
    """
    import emcee

    # ---- timing ----------------------------------------------------------
    half_ensemble = max(1, arguments.nwalkers // 2)
    batch_time = timed_posterior_benchmark(posterior, half_ensemble)
    estimated = 2 * batch_time * arguments.nsteps
    print(f"[bench] posterior-only estimate for {arguments.nwalkers} walkers "
          f"x {arguments.nsteps} steps: {estimated / 60:.1f} min "
          "(excludes emcee and HDF5 overhead)")
    if arguments.benchmark_only:
        timed_posterior_benchmark(posterior, 1)
        return 0

    # ---- starting point --------------------------------------------------
    first_guess = np.asarray(
        arguments.first_guess if arguments.first_guess is not None
        else rng.uniform(low=BOUNDS_ARRAY[:, 0], high=BOUNDS_ARRAY[:, 1],
                         size=N_DIM))

    resuming = arguments.resume and emcee_backend_has_samples(
        os.path.join(arguments.outdir, "chain.h5"))
    map_estimate_path = os.path.join(arguments.outdir, "map_estimate.npz")

    # The MAP estimate is cached to disk as soon as it exists, so a resumed
    # run reports the same starting point as the run it continues.
    map_info = {}
    if arguments.skip_map:
        map_parameters = np.clip(first_guess, BOUNDS_ARRAY[:, 0] + 1e-9,
                                 BOUNDS_ARRAY[:, 1] - 1e-9)
    elif resuming:
        if os.path.exists(map_estimate_path):
            map_parameters = np.load(
                map_estimate_path, allow_pickle=False)["parameters"]
            print(f"[MAP] resuming: reusing the stored MAP estimate from "
                  f"{map_estimate_path}")
        else:
            map_parameters = np.clip(first_guess, BOUNDS_ARRAY[:, 0] + 1e-9,
                                     BOUNDS_ARRAY[:, 1] - 1e-9)
            print(f"[MAP] resuming without a stored MAP estimate "
                  f"({map_estimate_path} not found); reporting the first "
                  f"guess as the starting point.")
    else:
        map_parameters, map_info = multi_start_map(
            posterior, rng, n_starts=10,
            step_fraction=arguments.map_fd_step,
            prescan=arguments.map_prescan,
            prescan_width=arguments.map_prescan_width)

    if not resuming:
        np.savez_compressed(map_estimate_path, parameters=map_parameters)

    print("[MAP] starting point: "
          + json.dumps(dict(zip(PARAMETER_NAMES,
                                np.round(map_parameters, 4).tolist()))))

    # ---- sampling --------------------------------------------------------

    backend_path = os.path.join(arguments.outdir, "chain.h5")
    backend = emcee.backends.HDFBackend(backend_path)
    if resuming:
        print(f"[mcmc] resuming from {backend_path} "
              f"at step {backend.iteration}")
        # emcee only knows how to continue from an explicit State, so hand it
        # the last one the backend stored.
        start_positions = backend.get_last_sample()
        remaining = max(0, arguments.nsteps - backend.iteration)
        if remaining == 0:
            print(f"[mcmc] the chain already has {backend.iteration} steps; "
                  f"raise --nsteps to extend it. Post-processing only.")
    else:
        backend.reset(arguments.nwalkers, N_DIM)
        start_positions = initialise_walkers(
            posterior, map_parameters, arguments.nwalkers, rng,
            scale_fraction=arguments.walker_scale)
        remaining = arguments.nsteps

    # emcee's red-blue move always proposes one half of the ensemble at a
    # time, so the batch size is fixed and the posterior can be graph captured.
    if not arguments.no_cuda_graph and posterior.device.type == "cuda":
        if posterior.enable_cuda_graph(arguments.nwalkers // 2):
            batch_time = timed_posterior_benchmark(posterior, half_ensemble)
            print(f"[bench] with the graph, posterior-only estimate for "
                  f"{arguments.nwalkers} walkers x {arguments.nsteps} steps: "
                  f"{2 * batch_time * arguments.nsteps / 60:.1f} min "
                  "(excludes emcee and HDF5 overhead)")

    chosen_moves = resolve_move(arguments.move)
    sampler = emcee.EnsembleSampler(
        arguments.nwalkers, N_DIM, posterior,
        vectorize=True, backend=backend,
        **({"moves": chosen_moves} if chosen_moves else {}))

    sampling_seconds = 0.0
    if remaining > 0:
        print(f"[mcmc] {arguments.nwalkers} walkers x {remaining} steps "
              f"({arguments.nwalkers * remaining:,} model evaluations)")
        sample_start = time.perf_counter()
        # tqdm's bar is redraw spam in a SLURM log; the periodic
        # [mcmc] lines below carry the same information.
        show_bar = sys.stderr.isatty()
        for step, _ in enumerate(sampler.sample(start_positions,
                                                iterations=remaining,
                                                progress=show_bar), start=1):
            if arguments.progress_every and step % arguments.progress_every == 0:
                acceptance = float(np.mean(sampler.acceptance_fraction))
                log_probability = sampler.get_last_sample().log_prob
                elapsed = time.perf_counter() - sample_start
                print(f"[mcmc] step {step}/{remaining} | "
                      f"acceptance {acceptance:.3f} | "
                      f"median log_posterior {np.median(log_probability):.2f} | "
                      f"{elapsed / step * 1e3:.1f} ms/step")
        sampling_seconds = time.perf_counter() - sample_start
        print(f"[mcmc] sampling finished in {sampling_seconds / 60:.2f} min "
              f"({sampling_seconds / remaining * 1e3:.1f} ms/step)")

    # ---- post-processing -------------------------------------------------
    chain = sampler.get_chain()
    try:
        tau = sampler.get_autocorr_time(tol=0)
        max_tau = float(np.nanmax(tau))
    except Exception as error:                      # very short chains
        print(f"[post] autocorrelation estimate failed: {error}")
        tau, max_tau = np.full(N_DIM, np.nan), np.nan

    if np.isfinite(max_tau) and max_tau > 0:
        discard = min(int(3 * max_tau), chain.shape[0] - 1)
        thin = max(1, int(0.5 * max_tau))
        if chain.shape[0] < 50 * max_tau:
            print(f"[post] WARNING chain may not be converged: "
                  f"{chain.shape[0]} steps < 50 x max(tau) = {50 * max_tau:.0f}")
    else:
        discard, thin = chain.shape[0] // 2, 1

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    print(f"[post] burn-in {discard} steps, thinning {thin}, "
          f"{len(flat_samples)} samples retained")
    if len(flat_samples) < 1000:
        print("[post] WARNING fewer than 1000 independent samples")

    # The full chain and its per-step log-posterior live in chain.h5 (the
    # emcee backend) and are not duplicated in results.npz, which holds what
    # is derived from them. Read them back with emcee.backends.HDFBackend.
    sampler_results = {
        "sampler": "emcee",
        "n_steps": int(chain.shape[0]),
        "acceptance_fraction": sampler.acceptance_fraction,
        "autocorrelation_time": tau,
        "parameters_map": map_parameters,
        "first_guess": first_guess,
        "discard": discard,
        "thin": thin,
        "sampling_seconds": sampling_seconds,
        "seconds_per_step": sampling_seconds / max(remaining, 1),
        "map_info": map_info,
    }
    sampling_line = (f"{sampling_seconds / 60:.2f} min for "
                     f"{arguments.nwalkers} walkers x {remaining} steps")
    return write_outputs(arguments, posterior, observation, true_parameters,
                         flat_samples, sampler_results, sampling_line,
                         provenance=(tau, discard, thin),
                         plot_extras={"chain": chain,
                                      "log_probability": sampler.get_log_prob()})


# Where each sampler's runs are kept, inside <repository>/results/.
SAMPLER_FOLDERS = {"emcee": "mcmc", "nested": "nested_sampling"}


def resolve_outdir(outdir, sampler):
    """Full path of a run's output folder.

    A run name such as ``my_star`` (or ``LMC/my_star``) goes in
    ``results/mcmc/`` or ``results/nested_sampling/`` inside the repository
    folder, according to the sampler, whichever folder the program is started
    from. A full path (``/scratch/...``, ``~/...``) is used as given.
    """
    outdir = os.path.expanduser(outdir or "run")
    if os.path.isabs(outdir):
        return outdir
    first = os.path.normpath(outdir).split(os.sep)[0]
    if first in ("results", ".."):
        raise SystemExit(
            f"--outdir {outdir}: give just the run's name, e.g. --outdir "
            "my_star; it is placed in results/mcmc/ or "
            "results/nested_sampling/ automatically. To write anywhere else, "
            "give a full path.")
    return os.path.join(REPOSITORY, "results", SAMPLER_FOLDERS[sampler], outdir)


def main(argv=None):
    arguments = parse_arguments(argv)
    arguments.outdir = resolve_outdir(arguments.outdir, arguments.sampler)
    if arguments.sampler == "emcee":
        if arguments.nwalkers % 2:
            raise SystemExit("--nwalkers must be even: emcee's red-blue move "
                             "splits the ensemble into two equal halves")
        if arguments.nwalkers < 2 * N_DIM:
            raise SystemExit(f"--nwalkers must be at least 2 x ndim = {2 * N_DIM}")
    os.makedirs(arguments.outdir, exist_ok=True)
    print(f"[setup] {version.summary_line()}")
    print(f"[setup] output folder: {arguments.outdir}")

    import torch
    torch.backends.cuda.matmul.allow_tf32 = arguments.tf32
    torch.backends.cudnn.allow_tf32 = arguments.tf32

    device = resolve_device(arguments.device)
    dtype = getattr(torch, arguments.dtype)
    rng = np.random.default_rng(arguments.seed)
    # emcee and UltraNest draw from NumPy's global generator rather than from
    # rng, so it is seeded too for the run to be reproducible.
    np.random.seed(arguments.seed)

    entries, resolution, description = data_io.load_instruments(
        arguments.instrument, arguments.instruments_file)
    if arguments.resolution is not None:
        if isinstance(resolution, list):
            raise SystemExit(
                "--resolution can't override more than one instrument's "
                "resolving power at once; drop --resolution, or narrow "
                "--instrument down to a single entry.")
        resolution = arguments.resolution
    print(f"[data] instrument '{', '.join(arguments.instrument)}': {description}")
    print(f"[data] resolution {data_io.format_resolution(resolution)}")

    print("[setup] loading emulator checkpoints ...")
    load_start = time.perf_counter()
    line_emulators, sed_emulator = load_emulators(
        load_sed=not arguments.no_magnitude,
        only_lines=arguments.lines)
    print(f"[setup] {len(line_emulators)} line emulators loaded in "
          f"{time.perf_counter() - load_start:.1f} s")

    # ---- observation -----------------------------------------------------
    from gpu_posterior import GPUPosterior

    true_parameters = (np.asarray(arguments.true_params, dtype=np.float64)
                       if arguments.true_params is not None else None)
    flux_uncertainty = None
    if arguments.data:
        observation = data_io.load_observation(
            arguments.data,
            extension=arguments.fits_extension,
            wavelength_column=arguments.wavelength_column,
            flux_column=arguments.flux_column,
            error_column=arguments.error_column,
            verbose=True)
        wavelength = observation["wavelength"]
        flux = observation["flux"]
        magnitude = arguments.magnitude
        print(f"[data] {wavelength.size} points, "
              f"{wavelength.min():.1f}-{wavelength.max():.1f} A, "
              f"median flux {np.median(flux):.4g}")
        if not 0.8 < float(np.median(flux)) < 1.2:
            print("[data] WARNING the median flux is far from 1.0, so this "
                  "spectrum does not look continuum normalised. The likelihood "
                  "compares it with a normalised model, so normalise it first.")
        if arguments.use_errors:
            if observation["error"] is None:
                raise SystemExit(
                    "--use-errors was given but the file has no uncertainty "
                    "column. Name it with --error-column, supply one in a .npz "
                    "as 'error', or drop --use-errors and set --snr instead.")
            flux_uncertainty = observation["error"]
            usable = np.isfinite(flux_uncertainty) & (flux_uncertainty > 0)
            print(f"[data] per-pixel uncertainties: median "
                  f"{np.median(flux_uncertainty[usable]):.5f}, "
                  f"range {flux_uncertainty[usable].min():.5f}-"
                  f"{flux_uncertainty[usable].max():.5f} "
                  f"({int((~usable).sum())} unusable)")
        if magnitude is None and not arguments.no_magnitude:
            raise SystemExit(
                "the magnitude term is on but no magnitude was given: pass "
                "--magnitude, or use --no-magnitude to fit the spectrum alone.")
    else:
        if not arguments.mock:
            raise SystemExit("give either --data <file> or --mock")
        if arguments.use_errors:
            raise SystemExit("--use-errors needs a real observation; a mock "
                             "spectrum is generated at the SNR given by --snr")
        if true_parameters is None:
            true_parameters = np.asarray(DEFAULT_TRUE_PARAMETERS, dtype=np.float64)
        wavelength = data_io.build_multi_instrument_grid(entries)
        flux = np.ones_like(wavelength)
        # Filled in from the mock below when the magnitude term is on.
        magnitude = None

    import priors as priors_module
    if arguments.prior == "flat":
        prior = None
        print("[setup] prior: uniform over the parameter bounds")
    else:
        prior = priors_module.informative_prior(
            config=(priors_module.load_prior_config(arguments.priors_file)
                    if arguments.priors_file else None),
            use_relations=not arguments.no_relations,
        )
        print("[setup] " + prior.describe().replace("\n", "\n[setup] "))

    print("[setup] building the batched posterior "
          "(caching trunk bases and FFT plans) ...")
    build_start = time.perf_counter()
    posterior = GPUPosterior(
        observed_wavelength=wavelength,
        observed_flux=flux,
        observed_magnitude=magnitude,
        spectrum_snr=arguments.snr,
        resolution=resolution,
        line_emulators=line_emulators,
        sed_emulator=sed_emulator,
        filter_name=arguments.filter_name,
        zp_system=arguments.zp_system,
        distance_pc=arguments.distance,
        magnitude_uncertainty=arguments.magnitude_uncertainty,
        device=device,
        dtype=dtype,
        use_magnitude=not arguments.no_magnitude,
        filter_directory=arguments.filter_dir,
        flux_uncertainty=flux_uncertainty,
        prior=prior,
        extinction=arguments.extinction,
    )
    print(f"[setup] posterior ready in {time.perf_counter() - build_start:.1f} s")

    if arguments.mock:
        print("[data] generating the mock observation from the true parameters")
        mock = data_io.make_mock_observation(
            posterior, true_parameters, arguments.snr, rng,
            add_noise=not arguments.no_noise)
        flux = mock["flux"]
        magnitude = mock["magnitude"]
        # In place: the cached bases and FFT plans depend only on the grid.
        posterior.set_observed_flux(flux, observed_magnitude=magnitude)
        mock_arrays = dict(
            wavelength=mock["wavelength"], flux=flux,
            clean_flux=mock["clean_flux"],
            true_parameters=true_parameters)
        # No magnitude is stored when the magnitude term is off.
        if magnitude is not None:
            mock_arrays["magnitude"] = magnitude
        np.savez_compressed(
            os.path.join(arguments.outdir, "mock_observation.npz"),
            **mock_arrays)
        if arguments.no_magnitude:
            print("[data] magnitude term disabled")
        else:
            print(f"[data] mock magnitude ({arguments.filter_name}, "
                  f"{arguments.zp_system}) = {magnitude:.4f}")

        truth_log_posterior = float(posterior(true_parameters[None, :])[0])
        ideal = -0.5 * posterior.n_covered_points
        print(f"[data] log_posterior at the true parameters: "
              f"{truth_log_posterior:,.1f}")
        print(f"[data] expected for a perfect fit at this SNR: {ideal:,.1f}")
        if truth_log_posterior < 2 * ideal:
            print("[data] WARNING the truth sits well below the ideal value. "
                  "Check the mock-generation and likelihood configuration.")

    print(f"[data] {wavelength.size} observed points, "
          f"{posterior.n_covered_points} residuals in the likelihood")

    # ---- sampling --------------------------------------------------------
    observation = {"wavelength": wavelength, "flux": flux,
                   "magnitude": magnitude, "resolution": resolution}
    if arguments.sampler == "nested":
        timed_posterior_benchmark(posterior, 64)
        if arguments.benchmark_only:
            return 0
        return fit_nested(arguments, posterior, rng, true_parameters,
                          observation)
    return fit_emcee(arguments, posterior, rng, true_parameters, observation)


if __name__ == "__main__":
    raise SystemExit(main())

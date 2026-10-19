#!/usr/bin/env python3
"""Regenerate the figures and summary.txt for a finished run, without
re-sampling.

    python3 scripts/replot.py results/mcmc/my_run

Reads results.npz (and mock_observation.npz when present), rebuilds the
posterior so the posterior-predictive panels can be drawn, and overwrites the
PNGs and summary.txt in place. Works for emcee and nested-sampling runs of
run_sampler_gpu.py alike; the sampler is read from the stored results.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

import data_io
import plots
import priors
from emulators import load_emulators
from gpu_posterior import GPUPosterior
import version


def stored_flux_uncertainty(stored, settings):
    """The per-pixel flux uncertainties the run was fitted with, or None.

    Taken from results.npz when stored there. A run made with --use-errors
    before they were stored has them re-read from its data file, provided the
    file still has the same wavelengths; otherwise the run cannot be rebuilt
    faithfully and the replot stops.
    """
    if "flux_uncertainty" in stored.files:
        return stored["flux_uncertainty"]
    if not settings.get("use_errors"):
        return None
    observation = data_io.load_observation(
        settings["data"],
        extension=settings.get("fits_extension"),
        wavelength_column=settings.get("wavelength_column"),
        flux_column=settings.get("flux_column"),
        error_column=settings.get("error_column"))
    if (observation["error"] is None or not np.array_equal(
            observation["wavelength"], stored["observed_wavelength"])):
        raise SystemExit(
            f"this run was fitted with the per-pixel uncertainties of "
            f"{settings['data']}, which results.npz does not store and the "
            f"file no longer provides on the same wavelengths")
    print(f"[replot] per-pixel uncertainties re-read from {settings['data']}")
    return observation["error"]


def stored_prior(settings):
    """The prior the run used, rebuilt from its settings (None means flat)."""
    if settings.get("prior", "flat") == "flat":
        return None
    priors_file = settings.get("priors_file")
    return priors.informative_prior(
        config=priors.load_prior_config(priors_file) if priors_file else None,
        use_relations=not settings.get("no_relations", False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir")
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args(argv)

    stored = np.load(os.path.join(arguments.outdir, "results.npz"),
                     allow_pickle=False)
    settings = json.loads(str(stored["settings"]))

    torch.backends.cuda.matmul.allow_tf32 = settings.get("tf32", False)
    device = torch.device(arguments.device
                          or ("cuda" if torch.cuda.is_available() else "cpu"))

    line_emulators, sed_emulator = load_emulators(
        load_sed=not settings["no_magnitude"],
        only_lines=settings.get("lines"))

    posterior = GPUPosterior(
        observed_wavelength=stored["observed_wavelength"],
        observed_flux=stored["observed_flux"],
        observed_magnitude=(float(stored["observed_magnitude"])
                            if "observed_magnitude" in stored.files else None),
        spectrum_snr=float(stored["spectrum_snr"]),
        flux_uncertainty=stored_flux_uncertainty(stored, settings),
        resolution=data_io.parse_stored_resolution(stored["resolution"]),
        line_emulators=line_emulators, sed_emulator=sed_emulator,
        filter_name=settings["filter_name"], zp_system=settings["zp_system"],
        distance_pc=settings["distance"],
        magnitude_uncertainty=settings["magnitude_uncertainty"],
        device=device, dtype=getattr(torch, settings["dtype"]),
        use_magnitude=not settings["no_magnitude"],
        filter_directory=settings.get("filter_dir"),
        prior=stored_prior(settings),
        extinction=settings.get("extinction", 0.0),
        verbose=True)

    # Stored results without a "sampler" entry are emcee runs.
    sampler = str(stored["sampler"]) if "sampler" in stored.files else "emcee"

    results = {key: stored[key] for key in stored.files
               if key not in ("settings", "map_info", "parameter_names")}
    if sampler == "emcee":
        results["discard"] = int(stored["discard"])
        # The raw chain and its log-posterior live in chain.h5 rather than in
        # results.npz; the chain and log-posterior plots need them.
        chain_path = os.path.join(arguments.outdir, "chain.h5")
        if "chain" not in results and os.path.exists(chain_path):
            import emcee
            backend = emcee.backends.HDFBackend(chain_path, read_only=True)
            results["chain"] = backend.get_chain()
            results["log_probability"] = backend.get_log_prob()
    if "true_parameters" in stored.files:
        results["true_parameters"] = stored["true_parameters"]

    plots.make_all(results, posterior, arguments.outdir)
    print(f"figures rewritten in {arguments.outdir}")

    percentiles = results["percentiles"]
    true_parameters = results.get("true_parameters")
    flat_samples = results["flat_samples"]
    boundary_records = data_io.prior_boundary_report(flat_samples)
    boundary_warning = data_io.format_boundary_warning(boundary_records)

    sampling_seconds = float(results["sampling_seconds"])
    if sampler == "nested":
        nested = {
            "log_evidence": float(results["log_evidence"]),
            "log_evidence_error": float(results["log_evidence_error"]),
            "effective_sample_size": float(results["effective_sample_size"]),
            "n_evaluations": int(results["n_evaluations"]),
            "stopped_at_call_cap": bool(results["stopped_at_call_cap"]),
        }
        tau = discard = thin = None
        sampling_line = (f"sampling: {sampling_seconds / 60:.2f} min, nested "
                         f"sampling with {int(results['n_live'])} live points, "
                         f"{nested['n_evaluations']:,} likelihood calls")
    else:
        nested = None
        tau = results["autocorrelation_time"]
        discard = int(results["discard"])
        thin = int(results["thin"])
        seconds_per_step = float(results["seconds_per_step"])
        # Recovers the step count actually sampled in the original run
        # (matters for --resume, where it can differ from settings["nsteps"])
        # without storing it separately -- it is the other factor of the
        # stored product.
        remaining = (round(sampling_seconds / seconds_per_step) if seconds_per_step
                     else int(results["n_steps"]) if "n_steps" in results
                     else int(results["chain"].shape[0]))
        sampling_line = (f"sampling: {sampling_seconds / 60:.2f} min for "
                         f"{settings['nwalkers']} walkers x {remaining} steps")

    # The provenance block names the code version and timestamp of the run
    # that actually produced these samples, not of this replot -- reusing the
    # pipeline_version recorded at run time (rather than calling
    # version.summary_line() again here) keeps that true after the fact.
    pipeline_version = json.loads(str(stored["pipeline_version"]))
    version_line = version.format_line(pipeline_version)
    run_args = argparse.Namespace(**settings)

    with open(os.path.join(arguments.outdir, "summary.txt"), "w") as handle:
        handle.write(data_io.format_parameter_table(
            percentiles, true_parameters, boundary_records))
        if boundary_warning:
            handle.write("\n" + boundary_warning + "\n")
        handle.write(f"\n\n{sampling_line}\n")
        handle.write(version_line + "\n")
        handle.write(data_io.format_provenance(
            run_args, posterior, pipeline_version, tau, discard,
            thin, len(flat_samples), nested=nested) + "\n")
    print(f"summary rewritten in {arguments.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

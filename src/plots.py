"""Diagnostic figures. Written to PNG - SLURM jobs have no display."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpu_config import PARAMETER_NAMES


def plot_chains(results, path):
    chain = results["chain"]
    n_dim = chain.shape[2]
    truth = results.get("true_parameters")

    figure, axes = plt.subplots(n_dim, 1, figsize=(12, 1.6 * n_dim), sharex=True)
    for index in range(n_dim):
        axis = axes[index]
        axis.plot(chain[:, :, index], color="black", alpha=0.25, lw=0.4)
        axis.set_ylabel(PARAMETER_NAMES[index], fontsize=9)
        if truth is not None:
            axis.axhline(truth[index], color="red", ls="--", lw=1)
        if results.get("discard"):
            axis.axvline(results["discard"], color="tab:blue", ls=":", lw=1)
    axes[-1].set_xlabel("step")
    figure.suptitle("walker traces (blue: burn-in cut, red: truth)")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_corner(results, path):
    try:
        import corner
    except ImportError:
        return
    figure = corner.corner(
        results["flat_samples"], labels=PARAMETER_NAMES,
        truths=results.get("true_parameters"),
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 8})
    figure.savefig(path, dpi=100)
    plt.close(figure)


def plot_log_probability(results, path):
    log_probability = results["log_probability"]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(np.median(log_probability, axis=1), color="black",
              label="median over walkers")
    axis.fill_between(np.arange(log_probability.shape[0]),
                      np.percentile(log_probability, 16, axis=1),
                      np.percentile(log_probability, 84, axis=1),
                      color="black", alpha=0.2, label="16-84%")
    axis.set_xlabel("step")
    axis.set_ylabel("log posterior")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def plot_posterior_predictive(results, posterior, path, n_draws=100):
    """Observed vs. model spectrum, one panel per line window.

    Every line window gets its own panel, ordered left-to-right/top-to-bottom
    by ascending wavelength.
    """
    import torch

    flat_samples = results["flat_samples"]
    rng = np.random.default_rng(0)
    draws = flat_samples[rng.choice(len(flat_samples),
                                    min(n_draws, len(flat_samples)),
                                    replace=False)]

    true_parameters = results.get("true_parameters")

    with torch.no_grad():
        batch = torch.as_tensor(draws, dtype=torch.float64,
                                device=posterior.device)
        best = torch.as_tensor(results["parameters_50p"][None, :],
                               dtype=torch.float64, device=posterior.device)
        # Panels show the merged model, i.e. exactly what the likelihood
        # compares against the data at each wavelength.
        merged_draws = posterior.merged_model(batch).cpu().numpy()
        # Posterior predictive: each draw gets observation noise at the level
        # the fit assumed (1/SNR, or each pixel's own uncertainty with
        # --use-errors), so the blue band shows where the data should fall if
        # the model is right. Points the likelihood ignores (zero weight) have
        # no predicted value and are left blank.
        if posterior.inverse_variance is None:
            sigma = np.full(merged_draws.shape[1], posterior.spectrum_sigma)
        else:
            weight = posterior.inverse_variance.cpu().numpy()
            with np.errstate(divide="ignore"):
                sigma = np.where(weight > 0, 1.0 / np.sqrt(weight), np.nan)
        merged_draws += rng.normal(size=merged_draws.shape) * sigma
        merged_median = posterior.merged_model(best)[0].cpu().numpy()
        merged_truth = None
        if true_parameters is not None:
            truth = torch.as_tensor(np.asarray(true_parameters)[None, :],
                                    dtype=torch.float64, device=posterior.device)
            merged_truth = posterior.merged_model(truth)[0].cpu().numpy()
        positions = [np.searchsorted(posterior.covered_indices, line.indices)
                     for line in posterior.lines]
        spectra = [merged_draws[:, p] for p in positions]
        median_spectra = [merged_median[p] for p in positions]
        truth_spectra = ([merged_truth[p] for p in positions]
                         if merged_truth is not None else None)

    lines = posterior.lines
    order = sorted(range(len(lines)), key=lambda i: lines[i].wavelength[0])

    n_panels = len(order)
    n_columns = 3
    n_rows = int(np.ceil(n_panels / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns,
                                figsize=(5 * n_columns, 2.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for panel, line_index in enumerate(order):
        line = lines[line_index]
        axis = axes[panel]
        axis.plot(line.wavelength, posterior.observed_flux[line.indices],
                  color="black", lw=0.7, label="observed")
        for draw in spectra[line_index]:
            axis.plot(line.wavelength, draw, color="tab:blue", alpha=0.06, lw=0.8)
        axis.plot(line.wavelength, median_spectra[line_index],
                  color="tab:red", lw=1.0, label="posterior median")
        if truth_spectra is not None:
            axis.plot(line.wavelength, truth_spectra[line_index],
                      color="tab:green", lw=0.8, linestyle="dotted", label="true (noiseless)")
        axis.set_title(line.name, fontsize=9)
        axis.tick_params(labelsize=7)
    for panel in range(n_panels, len(axes)):
        axes[panel].axis("off")
    axes[0].legend(fontsize=7)
    figure.suptitle("posterior predictive (blue: draws from the posterior, "
                    "with the observation noise the fit assumed)")
    figure.tight_layout()
    figure.savefig(path, dpi=300)
    plt.close(figure)


def plot_full_spectrum(results, posterior, path, dpi=300):
    """Observed flux across the entire wavelength range, one continuous
    panel, with the posterior-median merged model overlaid wherever a line
    window covers it (gaps outside any window show observed data only --
    there is no model prediction to compare against there).
    """
    import torch
    from matplotlib.ticker import MultipleLocator

    wavelength = posterior.observed_wavelength
    flux = posterior.observed_flux

    true_parameters = results.get("true_parameters")

    with torch.no_grad():
        best = torch.as_tensor(results["parameters_50p"][None, :],
                               dtype=torch.float64, device=posterior.device)
        merged_median = posterior.merged_model(best)[0].cpu().numpy()
        merged_truth = None
        if true_parameters is not None:
            truth = torch.as_tensor(np.asarray(true_parameters)[None, :],
                                    dtype=torch.float64, device=posterior.device)
            merged_truth = posterior.merged_model(truth)[0].cpu().numpy()

    covered = posterior.covered_indices

    def on_full_grid(model):
        # NaN wherever no line window covers the data, so matplotlib leaves
        # a gap there instead of drawing a straight line from one window to
        # the next.
        full = np.full(wavelength.size, np.nan)
        full[covered] = model
        return full

    span = float(wavelength[-1] - wavelength[0])
    width = max(15.0, min(80.0, span / 50.0))
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.plot(wavelength, flux, color="black", lw=0.4, label="observed")
    axis.plot(wavelength, on_full_grid(merged_median), color="tab:red", lw=0.6,
              label="posterior median (covered wavelengths only)")
    if merged_truth is not None:
        # Only for --mock runs: the noiseless model at the true parameters
        # used to generate the mock, i.e. what the observed flux would be
        # without the SNR noise draw added on top.
        axis.plot(wavelength, on_full_grid(merged_truth), color="tab:green",
                  lw=0.6, ls="dotted",
                  label="true, noiseless (covered wavelengths only)")
    axis.set_xlabel("wavelength (A)")
    axis.set_ylabel("normalised flux")
    axis.set_xlim(wavelength[0], wavelength[-1])
    axis.xaxis.set_major_locator(MultipleLocator(50))
    axis.legend(fontsize=8, loc="upper right")
    figure.suptitle("full observed spectrum vs. posterior-median model")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def make_all(results, posterior, outdir):
    os.makedirs(outdir, exist_ok=True)
    # Chain traces and the log-posterior trace only exist for MCMC.
    if "log_probability" in results:
        plot_chains(results, os.path.join(outdir, "chains.png"))
        plot_log_probability(results, os.path.join(outdir, "log_posterior.png"))
    plot_corner(results, os.path.join(outdir, "corner.png"))
    plot_posterior_predictive(
        results, posterior, os.path.join(outdir, "posterior_predictive.png"))
    plot_full_spectrum(
        results, posterior, os.path.join(outdir, "full_spectrum.png"))

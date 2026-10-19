#!/usr/bin/env python3
"""Report the structure and content of a spectrum file.

Run this on a new observation before fitting it. It prints how the file is
laid out, what the pipeline would read from it, and whether the flux looks
continuum normalised.

    python3 inspect_spectrum.py my_star.npz --plot check.png
"""

import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np


# Settings of the signal-to-noise estimate in `summarise`. It fits a straight
# continuum through the spectrum, rejects pixels that stand out from it
# (so spectral lines) and measures the scatter of the rest.
#
# Fewest continuum pixels for an estimate to be reported.
SNR_MIN_POINTS = 400
# Pixels more than this many standard deviations from the continuum are
# rejected. Pure Gaussian noise stays within 3 sigma 99.7% of the time, so
# almost no genuine continuum pixels are lost.
SNR_CLIP_SIGMA = 3.0
# Upper limit on fit-and-reject rounds; the loop normally stops sooner, once
# the set of kept pixels no longer changes.
SNR_MAX_ITERATIONS = 10
# For Gaussian noise the median absolute deviation (MAD) is 0.6745 sigma, so
# sigma = MAD / 0.6745 = 1.4826 x MAD. The MAD is used because a few line
# pixels barely move a median but inflate a standard deviation.
MAD_TO_SIGMA = 1.4826
# Check that the scatter really is noise. Spectral lines are smooth over
# several pixels while noise changes from one pixel to the next, so the
# differences between neighbouring pixels keep the noise but cancel the line
# structure (the idea behind the DER_SNR estimator, Stoehr et al. 2008). For
# pure noise the two estimates agree; when the continuum scatter is more than
# this many times the pixel-to-pixel scatter, the window is dominated by lines
# rather than continuum and the S/N estimate is too low. Noise that is
# correlated between neighbouring pixels (e.g. after resampling) also raises
# the ratio, so the warning is a prompt to check, not proof.
SNR_STRUCTURE_RATIO = 1.5


def describe_fits(path):
    """Print every extension, its type, and its columns or axes."""
    from astropy.io import fits

    print(f"=== FITS structure: {os.path.basename(path)} ===")
    with fits.open(path) as hdus:
        hdus.info()
        print()
        for index, hdu in enumerate(hdus):
            kind = type(hdu).__name__
            print(f"--- extension {index} ({kind}) ---")
            if getattr(hdu, "columns", None) is not None:
                for column in hdu.columns:
                    print(f"    column {column.name!r:20s} format={column.format} "
                          f"unit={column.unit}")
            if hdu.data is not None and getattr(hdu, "columns", None) is None:
                print(f"    data shape {np.shape(hdu.data)} "
                      f"dtype {getattr(hdu.data, 'dtype', '?')}")
            general = (
                "OBJECT", "INSTRUME", "TELESCOP", "EXPTIME", "DATE-OBS",
                "MJD-OBS", "SNR", "SN", "SNR_PPL", "SNR_EMP",
                "SNR_WMIN", "SNR_WMAX", "SPEC_RES", "RESOLUTN",
                "HIERARCH ESO INS EXPMODE", "BUNIT",
            )
            wcs = ("NAXIS1", "CRVAL1", "CDELT1", "CD1_1", "CRPIX1", "CTYPE1",
                   "CUNIT1")
            keys = general if getattr(hdu, "columns", None) is not None \
                else general + wcs
            for key in keys:
                if key in hdu.header:
                    print(f"    {key:10s} = {hdu.header[key]}")
            print(f"    ({len(hdu.header)} header cards in total; "
                  f"--header {index} prints them all)")
            print()


def summarise(wavelength, flux, label, snr_window=None, lines=None):
    """Print the ranges, sampling, normalisation state, and estimated S/N."""
    print(f"=== {label} ===")
    print(f"  points          : {wavelength.size}")
    print(
        f"  wavelength      : {wavelength.min():.2f} to {wavelength.max():.2f} "
        f"(units look like "
        f"{'Angstrom' if wavelength.max() > 1000 else 'nm or micron'})"
    )

    from gpu_broadening import detect_sampling
    from gpu_config import line_window

    for name, emulator in lines.items():
        try: 
            low, high = line_window(name, emulator)
            mask = (wavelength >= low) & (wavelength < high)
            wl_line = wavelength[mask]
            if mask.sum() < 3:
                print(f"{name} contains less than 3 points, skipping")
            else:
                spacing = np.diff(wl_line)
                velocity = spacing / wl_line[:-1] * 299792.458
                print(f"  d(lambda)       : {spacing.min():.5f} to {spacing.max():.5f} A")
                print(f"  velocity step   : {velocity.min():.4f} to {velocity.max():.4f} km/s")
                
                mode, _, irregularity = detect_sampling(wl_line)
                described = (
                    f"evenly spaced in VELOCITY (constant d(lambda)/lambda) for {name}" if mode == "log"
                    else f"evenly spaced in WAVELENGTH (constant d(lambda)) for {name}")
                print(f"  sampling        : {described}")
                print(f"                    regular to {irregularity:.3f} of a pixel")

                if mode == "log":
                    step = np.median(velocity)
                    print(f"use sampling='velocity' with a step of "
                        f"{step:.3f} km/s in config/instruments.json")
            
        except ValueError as error:
            print(f"  sampling        : IRREGULAR for {name}")
            print(textwrap.fill(str(error), width=78,
                                 initial_indent=" " * 20,
                                 subsequent_indent=" " * 20))
            print(textwrap.fill(
                "The fit leaves this line out: it can only broaden a line "
                "whose whole window is evenly sampled.",
                width=78, initial_indent=" " * 20,
                subsequent_indent=" " * 20))

    spacing = np.diff(wavelength)
    gaps = np.flatnonzero(spacing > 3 * np.median(spacing))
    if gaps.size:
        print(f"  gaps            : {gaps.size}; a line window containing "
              f"one is left out of the fit")
        for index in gaps[:5]:
            print(
                f"                    {wavelength[index]:.2f} to "
                f"{wavelength[index + 1]:.2f} A"
            )

    valid = np.isfinite(wavelength) & np.isfinite(flux)
    print(f"  flux finite     : {valid.sum()}/{flux.size}")

    if valid.any():
        good = flux[valid]

        print(f"  flux range      : {good.min():.4g} to {good.max():.4g}")
        print(f"  flux median     : {np.median(good):.4g}")

        normalised = 0.8 < np.median(good) < 1.2
        print(
            f"  looks normalised: {normalised}"
            + (
                ""
                if normalised
                else "   <-- the fit needs continuum-normalised flux "
                     "(continuum = 1)"
            )
        )

        # Choose either the entire valid spectrum or a user-specified interval.
        if snr_window is None:
            wmin = wavelength[valid].min()
            wmax = wavelength[valid].max()
            window_label = "whole spectrum"
        else:
            wmin, wmax = snr_window
            window_label = f"{wmin:.1f}–{wmax:.1f} Å"
        snr_mask = valid & (wavelength >= wmin) & (wavelength <= wmax)
        x = wavelength[snr_mask]
        y = flux[snr_mask]

        if x.size < SNR_MIN_POINTS:
            print(
                f"  implied SNR     : unavailable "
                f"({x.size} valid points in {window_label}, need at least "
                f"{SNR_MIN_POINTS}; widen --snr-window)"
            )
        else:
            # Estimate the noise by repeatedly fitting the continuum and
            # rejecting the pixels that stand out from it. `keep` marks the
            # pixels currently treated as continuum; at first, all of them.
            keep = np.ones(x.size, dtype=bool)
            noise = np.nan

            for _ in range(SNR_MAX_ITERATIONS):
                if keep.sum() < SNR_MIN_POINTS:
                    break
                # 1. Fit a straight line through the kept pixels: the
                #    continuum estimate, evaluated at every pixel.
                coefficients = np.polyfit(x[keep], y[keep], deg=1)
                continuum = np.polyval(coefficients, x)
                # 2. What is left after removing the continuum: noise, plus
                #    any lines and bad pixels.
                residual = y - continuum
                # 3. The noise level from the kept pixels' residuals: their
                #    median absolute deviation, converted to a standard
                #    deviation.
                residual_centre = np.median(residual[keep])
                noise = MAD_TO_SIGMA * np.median(
                    np.abs(residual[keep] - residual_centre))
                if not np.isfinite(noise) or noise <= 0:
                    break
                # 4. Keep only the pixels within SNR_CLIP_SIGMA of the
                #    continuum; lines and bad pixels fall outside and are
                #    dropped for the next round.
                new_keep = (np.abs(residual - residual_centre)
                            < SNR_CLIP_SIGMA * noise)
                # 5. Stop once the same pixels are kept twice in a row.
                if np.array_equal(new_keep, keep):
                    break
                keep = new_keep

            if keep.sum() >= SNR_MIN_POINTS and np.isfinite(noise) and noise > 0:
                # Refit on the final set of continuum pixels and measure
                # their scatter once more: that is the noise per pixel.
                coefficients = np.polyfit(x[keep], y[keep], deg=1)
                continuum = np.polyval(coefficients, x[keep])
                residual = y[keep] - continuum

                residual_centre = np.median(residual)
                noise = MAD_TO_SIGMA * np.median(
                    np.abs(residual - residual_centre)
                )

                # S/N per pixel: the continuum level over the noise. For a
                # normalised spectrum the continuum is close to 1, so this is
                # close to 1 / noise.
                continuum_level = np.median(continuum)
                snr = continuum_level / noise

                # The same noise level from differences between neighbouring
                # continuum pixels; dividing by sqrt(2) because the
                # difference of two noisy pixels has sqrt(2) times the noise.
                differences = np.diff(y[keep])
                pixel_noise = MAD_TO_SIGMA * np.median(
                    np.abs(differences - np.median(differences))) / np.sqrt(2)

                print(f"  S/N window      : {window_label}")
                print(f"  continuum points: {keep.sum()}/{x.size}")
                print(f"  noise per pixel : {noise:.4f}")
                print(
                    f"  implied SNR     : ~{snr:.0f} "
                    f"(robust residual scatter; per pixel)"
                )
                if pixel_noise > 0 and noise > SNR_STRUCTURE_RATIO * pixel_noise:
                    print(textwrap.fill(
                        f"WARNING: the scatter around the continuum is "
                        f"{noise / pixel_noise:.1f}x the pixel-to-pixel "
                        f"scatter, so it is dominated by spectral lines, not "
                        f"noise, and the S/N above is too low. The "
                        f"pixel-to-pixel scatter suggests S/N ~"
                        f"{continuum_level / pixel_noise:.0f}. Choose a "
                        f"--snr-window with more line-free continuum, or "
                        f"use --use-errors if the file has uncertainties.",
                        width=78, initial_indent=" " * 20,
                        subsequent_indent=" " * 20))
            else:
                print(
                    "  implied SNR     : unavailable "
                    f"(fewer than {SNR_MIN_POINTS} continuum pixels left "
                    "after rejecting lines; widen --snr-window)"
                )

    print()

def coverage_report(wavelength, lines=None):
    """Which line windows the fit will use, where they lie, and why any
    others are left out (the same check the fit applies)."""
    from gpu_config import line_window
    from gpu_posterior import check_window

    print("=== line windows covered ===")
    used, left_out, total = [], [], np.zeros(wavelength.size, dtype=int)
    for name, emulator in lines.items():
        low, high = line_window(name, emulator)
        mask = (wavelength >= low) & (wavelength < high)
        problem, _ = check_window(wavelength[mask], low, high)
        if problem:
            left_out.append((name, low, high, problem))
        else:
            used.append((name, low, high, int(mask.sum())))
            total[mask] += 1

    print(f"  {len(used)} of {len(lines)} line windows will be fitted")
    for name, low, high, count in sorted(used, key=lambda item: item[1]):
        print(f"    {name:22s} {low:9.2f} - {high:9.2f} A {count:7d} points")
    reached = [item for item in left_out if not item[3].startswith("fewer")]
    if reached:
        print("  left out although the data reach them:")
        for name, low, high, problem in sorted(reached, key=lambda item: item[1]):
            print(f"    {name:22s} {low:9.2f} - {high:9.2f} A  {problem}")
    missed = [item[0] for item in left_out if item[3].startswith("fewer")]
    if missed:
        print(f"  not covered: {', '.join(sorted(missed))}")
    if used:
        print(f"  {int((total > 0).sum())} observed points enter the likelihood")
        if (total > 1).any():
            print(f"  WARNING {int((total > 1).sum())} points lie in more than "
                  f"one window; the fit will refuse overlapping windows "
                  f"(edit LINE_WINDOW_OVERRIDES in src/gpu_config.py)")
    else:
        print("  WARNING no line window is covered - check the wavelength units")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a .fits, .npz or .txt spectrum")
    parser.add_argument("--fits-extension", type=int, default=None,
                        help="FITS extension to read (default: auto-detect)")
    parser.add_argument("--wavelength-column", default=None)
    parser.add_argument("--flux-column", default=None)
    parser.add_argument("--snr-window", nargs=2, type=float,
                        metavar=("WMIN", "WMAX"), default=None,
                        help="line-free wavelength interval for a robust per-pixel S/N estimate",)
    parser.add_argument("--plot", default=None, help="write a PNG of the spectrum")
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--header", type=int, default=None,
                        metavar="EXT",
                        help="print every header card of this FITS extension")
    arguments = parser.parse_args(argv)

    if arguments.path.lower().endswith((".fits", ".fit", ".fits.gz")):
        describe_fits(arguments.path)
        if arguments.header is not None:
            from astropy.io import fits
            with fits.open(arguments.path) as hdus:
                print(f"=== full header of extension {arguments.header} ===")
                print(repr(hdus[arguments.header].header))
                print()

    import data_io
    try:
        observation = data_io.load_observation(
            arguments.path, extension=arguments.fits_extension,
            wavelength_column=arguments.wavelength_column,
            flux_column=arguments.flux_column, verbose=True)
    except Exception as error:
        print(f"Could not read a spectrum automatically: {error}")
        print("Pass --fits-extension / --wavelength-column / --flux-column using the "
              "structure printed above.")
        return 1
    from emulators import load_emulators
    try:
        lines, _ = load_emulators(load_sed=False)
    except Exception as error:
        lines = {}
        print(f"Could not load lines {error}")
        print("Coverage report cannot be made.")
    summarise(observation["wavelength"], observation["flux"],
              "what the pipeline would read", snr_window=arguments.snr_window, lines=lines)

    if not arguments.no_coverage and len(lines) > 0:
        coverage_report(observation["wavelength"], lines=lines)

    if arguments.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(14, 4))
        axis.plot(observation["wavelength"], observation["flux"], lw=0.5,
                  color="black")
        axis.axhline(1.0, color="tab:red", lw=0.8, ls="--",
                     label="continuum = 1")
        axis.set_xlabel("wavelength")
        axis.set_ylabel("flux")
        axis.legend()
        figure.tight_layout()
        figure.savefig(arguments.plot, dpi=300)
        print(f"plot written to {arguments.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

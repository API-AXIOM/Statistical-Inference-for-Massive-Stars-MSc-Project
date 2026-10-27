"""Reading spectra and instrument setups, and writing results.

* Instruments and wavelength grids: ``load_instruments`` and
  ``build_multi_instrument_grid``, from ``config/instruments.json``.
* Observations: ``load_observation`` reads a spectrum from .fits, .npz or a
  text file; ``make_mock_observation`` simulates one from the model.
* Line windows: ``read_line_windows`` reads the fitting window of each line
  from a Kiwi-GA ``line_list.txt``.
* Results: ``save_results`` writes ``results.npz``; ``prior_boundary_report``,
  ``format_parameter_table`` and ``format_provenance`` build the text of
  ``summary.txt``.
"""

import json
import os
import numpy as np
from gpu_config import BOUNDS_ARRAY, C_KMS, N_THETA, PARAMETER_NAMES
import version

DEFAULT_INSTRUMENTS = os.path.join(
    os.environ.get("MCMC_GPU_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "instruments.json")


def load_instruments(names, path=None):
    """Read one or more instrument setups from config/instruments.json.

    ``names`` is an instrument name, or a list of names to combine into a
    single fit -- e.g. a spectrum stitched from more than one arm or
    instrument. Returns ``(entries, resolution, description)``:

    * ``entries`` is one ``(segments, sampling)`` pair per named instrument,
      for building a mock wavelength grid with ``build_multi_instrument_grid``.
    * ``resolution`` is a single resolving power if every named instrument
      shares one (including the common case of a single name), or otherwise a
      list of ``(low, high, value)`` wavelength bands -- one per instrument --
      for ``gpu_config.resolve_resolution`` to assign per line.
    * ``description`` joins each instrument's own description, prefixed by
      name when more than one was given.
    """
    if isinstance(names, str):
        names = [names]
    path = path or DEFAULT_INSTRUMENTS
    with open(path) as handle:
        catalogue = json.load(handle)

    entries = []
    bands = []
    descriptions = []
    for name in names:
        if name not in catalogue:
            available = [k for k in catalogue if not k.startswith("_")]
            raise KeyError(f"unknown instrument {name!r}; available: {available}")
        setup = catalogue[name]
        segments = setup["segments"]
        instrument_resolution = float(setup["resolution"])
        sampling = setup.get("sampling", "wavelength")
        entries.append((segments, sampling))
        descriptions.append(setup.get("description", ""))
        for low, high, _step in segments:
            bands.append((float(low), float(high), instrument_resolution))

    bands.sort(key=lambda band: band[0])
    for previous, current in zip(bands, bands[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"instrument segments overlap between {previous[1]:.1f} and "
                f"{current[0]:.1f} A ({names}); a wavelength cannot have two "
                f"resolving powers. Narrow the segments so they meet, not "
                f"overlap.")

    distinct = {band[2] for band in bands}
    resolution = distinct.pop() if len(distinct) == 1 else bands

    if len(names) == 1:
        description = descriptions[0]
    else:
        description = "; ".join(f"{name}: {text}"
                                for name, text in zip(names, descriptions))
    return entries, resolution, description


def build_multi_instrument_grid(entries):
    """Wavelength grid combining one or more instruments' own grids.

    ``entries`` is the list of ``(segments, sampling)`` pairs returned by
    ``load_instruments``. Each instrument's grid is built with its own
    sampling mode, and the pieces are combined and deduplicated -- two
    instruments meeting exactly at a shared boundary wavelength would
    otherwise contribute that point twice.
    """
    pieces = [build_wavelength_grid(segments, sampling)
             for segments, sampling in entries]
    return np.unique(np.concatenate(pieces))


def format_resolution(resolution):
    """Human-readable resolving power, for a scalar or a list of bands."""
    if isinstance(resolution, list):
        return ", ".join(
            f"{low:.0f}-{high:.0f} A: R={value:g}"
            for low, high, value in resolution)
    return f"R = {resolution:g}"


def parse_stored_resolution(value):
    """Undo how ``save_results`` encodes the 'resolution' field.

    A single resolving power round-trips through ``results.npz`` as a plain
    number; more than one round-trips as a JSON string of ``[low, high,
    value]`` bands (see ``save_results``, which JSON-encodes any list or
    tuple field).
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return [tuple(band) for band in json.loads(str(value))]


def build_wavelength_grid(segments, sampling="wavelength"):
    """Evenly sampled wavelength grid from ``[min, max, step]`` segments.

    With ``sampling="wavelength"`` the step is a spacing in Angstrom, giving a
    grid evenly spaced in wavelength. With ``sampling="velocity"`` it is a step
    in km/s, giving a grid evenly spaced in velocity — equivalently in
    ``ln(lambda)`` — which is what many pipelines produce.

    Segments are generated independently, so each keeps its own sampling.
    """
    pieces = []
    for low, high, step in segments:
        if sampling == "velocity":
            log_step = step / C_KMS
            count = int(np.floor((np.log(high) - np.log(low)) / log_step)) + 1
            pieces.append(np.exp(np.log(low) + log_step * np.arange(count)))
        else:
            count = int(np.floor((high - low) / step)) + 1
            pieces.append(low + step * np.arange(count))
    grid = np.concatenate(pieces)
    grid = np.unique(np.round(grid, 9))
    return grid.astype(np.float64)


def make_mock_observation(posterior, true_parameters, snr, rng,
                          add_noise=True):
    """Synthetic observation drawn from the model itself."""
    import torch

    parameters = np.asarray(true_parameters, dtype=np.float64)[None, :]
    with torch.no_grad():
        batch = torch.as_tensor(parameters, dtype=torch.float64,
                                device=posterior.device)
        # Ask the posterior for the very spectrum it will compare against the
        # data, so the simulated observation cannot disagree with the model
        # that will be fitted to it.
        model = posterior.merged_model(batch)[0].cpu().numpy().astype(np.float64)
        magnitude = None
        if posterior.photometry is not None:
            magnitude = float(
                posterior.photometry.magnitude(batch[:, :N_THETA])[0].cpu())

    # Points no line window covers stay at the continuum.
    clean = np.ones(posterior.observed_wavelength.size, dtype=np.float64)
    clean[posterior.covered_indices] = model

    noisy = clean.copy()
    if add_noise:
        noisy = noisy + rng.normal(0.0, 1.0 / snr, size=clean.shape)

    return {
        "wavelength": posterior.observed_wavelength,
        "flux": noisy,
        "clean_flux": clean,
        "magnitude": magnitude,
    }


# Column names seen in the wild for the two spectral axes, lower-cased.
_WAVELENGTH_NAMES = ("wave", "wavelength", "lambda", "wav", "awav", "lam",
                     "wave_vac", "wave_air", "spectral_axis")
_FLUX_NAMES = ("sci_norm", "flux_norm", "norm_flux", "normalised_flux",
               "normalized_flux", "fnorm", "flux", "sci", "sci_flux",
               "spectrum", "intensity", "counts", "data")
_ERROR_NAMES = ("err", "error", "sigma", "flux_err", "err_norm", "noise",
                "uncertainty", "stat_err")


def _pick_column(columns, candidates, override=None):
    """Find a column by name, case-insensitively."""
    names = {name.lower(): name for name in columns}
    if override is not None:
        if override in columns:
            return override
        if override.lower() in names:
            return names[override.lower()]
        raise KeyError(f"column {override!r} not found; available: {list(columns)}")
    for candidate in candidates:
        if candidate in names:
            return names[candidate]
    # Fall back to a substring match, so SCI_NORM_1 or FLUX_REDUCED still work.
    for candidate in candidates:
        for lowered, original in names.items():
            if candidate in lowered:
                return original
    return None


def _numeric_columns(hdu):
    """Names of columns holding a single number per row."""
    numeric = []
    for column in hdu.columns:
        code = "".join(character for character in str(column.format)
                       if character.isalpha())
        repeat = "".join(character for character in str(column.format)
                         if character.isdigit())
        if code in ("E", "D", "J", "I", "K", "F") and repeat in ("", "1"):
            numeric.append(column.name)
    return numeric


def _guess_spectral_columns(hdu, wave_name, flux_name):
    """Fill in whichever of the two columns was not matched by name.

    A spectrum table usually holds just the two arrays, so if one is
    identified the other is unambiguous. If neither matched, the wavelength
    axis is the one that increases monotonically.
    """
    numeric = _numeric_columns(hdu)
    if wave_name is not None and flux_name is None:
        remaining = [name for name in numeric if name != wave_name]
        return wave_name, (remaining[0] if len(remaining) == 1 else None)
    if wave_name is None and flux_name is not None:
        remaining = [name for name in numeric if name != flux_name]
        return (remaining[0] if len(remaining) == 1 else None), flux_name
    if wave_name is None and flux_name is None and len(numeric) == 2:
        first, second = (np.asarray(hdu.data[name], np.float64).ravel()
                         for name in numeric)
        first_sorted = np.all(np.diff(first) > 0)
        second_sorted = np.all(np.diff(second) > 0)
        if first_sorted and not second_sorted:
            return numeric[0], numeric[1]
        if second_sorted and not first_sorted:
            return numeric[1], numeric[0]
    return wave_name, flux_name


def _convert_to_angstrom(wavelength, unit, verbose=False):
    """Scale a wavelength axis to Angstrom, guessing the unit if need be."""
    factors = {"angstrom": 1.0, "a": 1.0, "aa": 1.0, "ang": 1.0,
               "nm": 10.0, "nanometer": 10.0,
               "um": 1e4, "micron": 1e4, "micrometer": 1e4,
               "m": 1e10}
    if unit:
        factor = factors.get(str(unit).strip().lower())
        if factor is not None:
            if verbose and factor != 1.0:
                print(f"[data] converting wavelengths from {unit} to Angstrom")
            return wavelength * factor, str(unit)

    # No usable unit keyword: guess from the range. Optical/UV spectra of OB
    # stars sit at 1e3-1e4 Angstrom, 1e2-1e3 nm, or 0.1-1 micron.
    median = float(np.median(wavelength))
    if median < 10.0:
        guess, factor = "micron", 1e4
    elif median < 1000.0:
        guess, factor = "nm", 10.0
    else:
        guess, factor = "Angstrom", 1.0
    if verbose:
        print(f"[data] no wavelength unit in the file; values look like "
              f"{guess}, using them as such")
    return wavelength * factor, guess + " (guessed)"


def _load_fits(path, extension=None, wavelength_column=None,
               flux_column=None, error_column=None, verbose=False):
    """Read a 1-D spectrum from a binary table in a FITS file.

    The first extension with recognisable wavelength and flux columns is used,
    unless ``extension`` names one.
    """
    from astropy.io import fits

    with fits.open(path) as hdus:
        candidates = range(len(hdus)) if extension is None else [extension]

        for index in candidates:
            hdu = hdus[index]
            if getattr(hdu, "columns", None) is None or hdu.data is None:
                continue
            names = list(hdu.columns.names)
            wave_name = _pick_column(names, _WAVELENGTH_NAMES, wavelength_column)
            flux_name = _pick_column(names, _FLUX_NAMES, flux_column)
            if wave_name is None or flux_name is None:
                wave_name, flux_name = _guess_spectral_columns(
                    hdu, wave_name, flux_name)
            if wave_name is None or flux_name is None:
                continue
            wavelength = np.asarray(hdu.data[wave_name], np.float64).ravel()
            flux = np.asarray(hdu.data[flux_name], np.float64).ravel()
            error_name = _pick_column(names, _ERROR_NAMES, error_column)
            error = (np.asarray(hdu.data[error_name], np.float64).ravel()
                     if error_name else None)
            unit = None
            for column in hdu.columns:
                if column.name == wave_name:
                    unit = column.unit
            if verbose:
                print(f"[data] FITS extension {index}: table columns "
                      f"{wave_name!r} and {flux_name!r}"
                      + (f", errors {error_name!r}" if error_name else ""))
            wavelength, _ = _convert_to_angstrom(wavelength, unit, verbose)
            return wavelength, flux, error

    raise ValueError(
        f"could not find a 1-D spectrum in {path}. Run "
        f"`python3 inspect_spectrum.py {path}` to see how the file is laid "
        f"out, then pass --fits-extension, --wavelength-column and --flux-column.")


def _load_ascii_spectrum(path, verbose=False):
    """Read a whitespace-delimited ASCII spectrum.

    Expected columns are:
        wavelength[Angstrom]  normalised_flux  [flux_error]

    The error column is optional. Lines beginning with '#' are treated as
    comments, so a commented header is allowed. ``ndmin=2`` makes even a
    one-row file a 2-D table.
    """
    try:
        table = np.loadtxt(path, comments="#", ndmin=2)
    except OSError as error:
        raise OSError(
            f"could not open ASCII spectrum {path!r}: {error}") from error
    except ValueError as error:
        raise ValueError(
            f"could not parse {path!r} as a whitespace-delimited numeric "
            "spectrum. Expected columns: wavelength flux [error].") from error

    if table.shape[1] < 2:
        raise ValueError(f"{path!r} has {table.shape[1]} column(s); "
                         "expected at least wavelength and flux.")

    wavelength = np.asarray(table[:, 0], dtype=np.float64)
    flux = np.asarray(table[:, 1], dtype=np.float64)
    error = (np.asarray(table[:, 2], dtype=np.float64)
             if table.shape[1] >= 3 else None)

    if verbose:
        print("[data] ASCII spectrum: column 0 = wavelength, column 1 = flux"
              + (", column 2 = error" if error is not None else ""))
        print("[data] ASCII wavelengths assumed to be in Angstrom")

    return wavelength, flux, error


def load_observation(path, extension=None, wavelength_column=None,
                     flux_column=None, error_column=None, verbose=False):
    """Load an observed spectrum from a .npz, .fits, or ASCII text file.

    A .npz must contain ``wavelength`` and ``flux`` arrays, and may also
    contain ``error``.

    A .fits must hold the spectrum as a binary table with wavelength and flux
    columns.

    A plain-text spectrum (.txt, .dat, .ascii, or .norm) must contain
    whitespace-delimited numeric columns::

        wavelength[Angstrom]  continuum_normalised_flux  [flux_error]

    Lines beginning with ``#`` are ignored, so a commented header is allowed.
    For text spectra, wavelengths are assumed to already be in Angstrom.

    Wavelengths are returned in Angstrom, sorted ascending, with invalid
    wavelength/flux points removed. If errors are supplied, non-finite or
    non-positive-error pixels are also removed. Flux must already be
    continuum normalised.
    """
    error = None
    lower_path = str(path).lower()

    if lower_path.endswith((".fits", ".fit", ".fits.gz")):
        wavelength, flux, error = _load_fits(
            path,
            extension=extension,
            wavelength_column=wavelength_column,
            flux_column=flux_column,
            error_column=error_column,
            verbose=verbose,
        )

    elif lower_path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            for key in ("wavelength", "flux"):
                if key not in data:
                    raise KeyError(f"{path} must contain a '{key}' array")

            wavelength = np.asarray(data["wavelength"], dtype=np.float64)
            flux = np.asarray(data["flux"], dtype=np.float64)

            error = (
                np.asarray(data["error"], dtype=np.float64)
                if "error" in data
                else None
            )

        if verbose:
            print(
                "[data] NPZ spectrum: arrays 'wavelength' and 'flux'"
                + (", 'error'" if error is not None else "")
            )

    elif lower_path.endswith((".txt", ".dat", ".ascii", ".norm")):
        wavelength, flux, error = _load_ascii_spectrum(path, verbose)

    else:
        raise ValueError(
            f"unsupported spectrum format for {path!r}. Supported formats: "
            ".fits, .fit, .fits.gz, .npz, .txt, .dat, .ascii, and .norm.")

    # Ensure one-dimensional float arrays.
    wavelength = np.asarray(wavelength, dtype=np.float64).ravel()
    flux = np.asarray(flux, dtype=np.float64).ravel()

    if wavelength.size != flux.size:
        raise ValueError(
            f"wavelength and flux have different lengths "
            f"({wavelength.size} vs {flux.size})")

    if error is not None:
        error = np.asarray(error, dtype=np.float64).ravel()
        if error.size != wavelength.size:
            raise ValueError(
                f"wavelength and error have different lengths "
                f"({wavelength.size} vs {error.size})")

    # A likelihood using supplied errors cannot handle zero, negative, or
    # non-finite uncertainties. Remove such pixels here, alongside invalid
    # wavelength or flux values.
    good = np.isfinite(wavelength) & np.isfinite(flux)

    if error is not None:
        good &= np.isfinite(error) & (error > 0)

    if not good.all() and verbose:
        print(f"[data] dropping {int((~good).sum())} invalid point(s)")

    wavelength = wavelength[good]
    flux = flux[good]
    if error is not None:
        error = error[good]

    if wavelength.size < 2:
        raise ValueError(
            f"{path!r} contains fewer than two valid spectral pixels after "
            "filtering.")

    # The rest of the pipeline expects an increasing wavelength axis.
    order = np.argsort(wavelength)
    if not np.all(order == np.arange(order.size)):
        if verbose:
            print("[data] sorting into ascending wavelength order")
        wavelength = wavelength[order]
        flux = flux[order]
        if error is not None:
            error = error[order]

    return {
        "wavelength": wavelength,
        "flux": flux,
        "error": error,
    }


def read_line_windows(path):
    """Read the fitting window of each line from a Kiwi-GA ``line_list.txt``.

    Each non-empty line not starting with ``#`` holds the line name, its
    resolving power, and the low and high edge of its window in Angstrom,
    followed by columns that are not used here. The names must be the
    emulators' line names (as ``inspect_spectrum.py`` lists them).

    Returns a dictionary from line name to ``(low, high)``, in file order.
    """
    windows = {}
    with open(path) as handle:
        for number, text in enumerate(handle, start=1):
            columns = text.split()
            if not columns or columns[0].startswith("#"):
                continue
            if len(columns) < 4:
                raise ValueError(f"{path}, line {number}: expected a name, a "
                                 f"resolving power and two window edges")
            name = columns[0]
            low, high = float(columns[2]), float(columns[3])
            if not low < high:
                raise ValueError(f"{path}, line {number}: the window of "
                                 f"{name} has low edge {low} >= high edge {high}")
            if name in windows:
                raise ValueError(f"{path}, line {number}: {name} is listed twice")
            windows[name] = (low, high)
    if not windows:
        raise ValueError(f"{path} lists no line windows")
    return windows


def save_results(path, results):
    """Write the run's results to one compressed .npz.

    Dictionaries, lists and tuples (other than ``percentiles``) are stored as
    JSON strings; ``None`` values are skipped.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {}
    for key, value in results.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)) and key != "percentiles":
            payload[key] = np.asarray(json.dumps(value, default=str))
        else:
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)


def prior_boundary_report(flat_samples, tolerance=0.02):
    """Find parameters whose posterior is limited by the prior, not the data.

    A parameter piled against the edge of its prior box has not been measured:
    the data wanted to go further and the prior stopped it. The reported value
    is then an artefact of where the box was drawn.

    ``tolerance`` is the distance from a bound, as a fraction of the prior
    range, that counts as "at" it. Returns one record per parameter, with a
    ``status`` of ``""`` (fine), ``"reaches"`` (one tail touches the bound)
    or ``"pinned"`` (the median itself sits on the bound).
    """
    samples = np.asarray(flat_samples)
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    span = upper - lower
    low, mid, high = np.percentile(samples, [16, 50, 84], axis=0)

    records = []
    for index, name in enumerate(PARAMETER_NAMES):
        margin = tolerance * span[index]
        at_lower = (mid[index] - lower[index]) <= margin
        at_upper = (upper[index] - mid[index]) <= margin
        tail_lower = (low[index] - lower[index]) <= margin
        tail_upper = (upper[index] - high[index]) <= margin

        if at_lower or at_upper:
            status, edge = "pinned", "lower" if at_lower else "upper"
        elif tail_lower or tail_upper:
            status, edge = "reaches", "lower" if tail_lower else "upper"
        else:
            status, edge = "", ""

        piled = float(np.mean(
            (samples[:, index] - lower[index] <= margin)
            | (upper[index] - samples[:, index] <= margin)))
        records.append({"name": name, "status": status, "edge": edge,
                        "fraction_at_bound": piled, "tolerance": tolerance,
                        "position": float((mid[index] - lower[index]) / span[index])})
    return records


def format_boundary_warning(records):
    """Render the prior-boundary report, or an empty string if all is well."""
    flagged = [record for record in records if record["status"]]
    if not flagged:
        return ""

    phrase = {"pinned": "pinned at the {edge}", "reaches": "reaches the {edge}"}
    lines = ["", "PRIOR BOUNDARY WARNING",
             "These parameters are limited by the prior, not by the data.",
             "Their values are set by where the box was drawn and should not",
             "be quoted as measurements:", ""]
    for record in sorted(flagged, key=lambda item: -item["fraction_at_bound"]):
        lines.append(
            f"    {record['name']:>8s}  "
            f"{phrase[record['status']].format(edge=record['edge']) + ' bound':>26s}   "
            f"({record['fraction_at_bound']:.0%} of samples within "
            f"{record.get('tolerance', 0.02):.0%} of the range from it)")
    lines.append("")
    lines.append("A parameter reaches a bound when the data cannot constrain it,")
    lines.append("when the prior is too narrow, or when the model cannot")
    lines.append("reproduce the observation. Check which before interpreting.")
    return "\n".join(lines)


def format_parameter_table(percentiles, true_parameters=None,
                           boundary_records=None):
    """Render the 16/50/84 posterior summary as aligned text."""
    status = {}
    if boundary_records is not None:
        status = {record["name"]: record["status"] for record in boundary_records}

    header = f"{'parameter':>10s} {'16%':>14s} {'50%':>14s} {'84%':>14s}"
    if true_parameters is not None:
        header += f" {'truth':>14s}"
    if any(status.values()):
        header += "  note"
    lines = [header]

    for index, name in enumerate(PARAMETER_NAMES):
        low, mid, high = percentiles[:, index]
        row = f"{name:>10s} {low:14.5g} {mid:14.5g} {high:14.5g}"
        if true_parameters is not None:
            row += f" {true_parameters[index]:14.5g}"
        if any(status.values()):
            note = status.get(name, "")
            row += ("  <-- " + ("pinned at" if note == "pinned" else "reaches")
                    + " the prior bound") if note else ""
        lines.append(row.rstrip())
    return "\n".join(lines)


def format_provenance(arguments, posterior, pipeline_version, tau, discard,
                      thin, n_samples, nested=None):
    """Render the "HOW THIS FIT WAS MADE" block appended to every summary.txt.

    Pulls directly from the run's own settings and the posterior actually
    built from them, rather than restating anything by hand, so it stays
    correct as flags and defaults change. For a nested-sampling fit, pass its
    results dictionary as ``nested``; ``tau``, ``discard`` and ``thin`` are then
    unused.
    """
    lines = ["", "HOW THIS FIT WAS MADE", "-" * 22]

    instrument_names = ", ".join(arguments.instrument)
    if arguments.mock:
        lines.append(f"data       : mock spectrum (--mock), "
                     f"instrument {instrument_names}, seed {arguments.seed}")
    else:
        lines.append(f"data       : {arguments.data} "
                     f"(instrument {instrument_names})")

    res_note = ("override (--resolution)" if arguments.resolution is not None
               else f"instrument {instrument_names} default")
    lines.append(f"resolution : {format_resolution(posterior.resolution)}, "
                f"from {res_note}")

    if arguments.use_errors:
        lines.append(f"noise      : the data file's own per-pixel "
                     f"uncertainties (--use-errors), median "
                     f"{posterior.median_uncertainty:.4g}; --snr not used")
    else:
        lines.append(f"SNR        : {arguments.snr:g}, an assumed uniform "
                     f"noise level (--snr)")

    if arguments.no_magnitude:
        lines.append("photometry : off (--no-magnitude)")
    else:
        magnitude = ("synthetic (mock)" if arguments.mock
                     else f"{arguments.magnitude:g}")
        lines.append(f"photometry : {arguments.filter_name} = {magnitude} "
                     f"+/- {arguments.magnitude_uncertainty:g} mag, "
                     f"distance {arguments.distance:g} pc")

    prior_line = f"prior      : {arguments.prior}"
    if arguments.prior == "informative":
        prior_line += ", LMC"
        if arguments.no_relations:
            prior_line += " (--no-relations: standard-relation terms off)"
    lines.append(prior_line)

    covered_lines = len(posterior.lines)
    total_lines = covered_lines + len(posterior.skipped_lines)
    lines.append(
        f"lines      : {covered_lines}/{total_lines} line windows usable, "
        f"{posterior.n_covered_points} points enter the likelihood")
    lines.append(f"code       : {version.format_line(pipeline_version)}")

    if nested is not None:
        # Stored settings without an ns_method entry used slice sampling.
        method = ("random-walk"
                  if getattr(arguments, "ns_method", "slice") == "randomwalk"
                  else "slice")
        sampling_line = (f"sampling   : nested sampling (UltraNest), "
                         f"{arguments.n_live} live points, population "
                         f"{method} sampler with {arguments.ns_steps} steps and "
                         f"{arguments.ns_walkersize} walkers, seed {arguments.seed}")
        if arguments.resume:
            sampling_line += " (--resume)"
        lines.append(sampling_line)
        convergence_line = (
            f"convergence: log Z = {nested['log_evidence']:.2f} +- "
            f"{nested['log_evidence_error']:.2f}, effective sample size "
            f"{nested['effective_sample_size']:.0f}, "
            f"{nested['n_evaluations']:,} likelihood calls, "
            f"{n_samples} equally weighted samples")
        if nested["stopped_at_call_cap"]:
            convergence_line += (f" -- WARNING stopped at the "
                                 f"{arguments.ns_max_calls:,}-call cap before "
                                 f"converging; the posterior is not trustworthy")
        lines.append(convergence_line)
        return "\n".join(lines)

    # Stored settings without a move entry used emcee's default, stretch.
    move = getattr(arguments, "move", "stretch")
    sampling_line = (f"sampling   : {arguments.nwalkers} walkers x "
                     f"{arguments.nsteps} steps, seed {arguments.seed}, "
                     f"move={move}")
    if arguments.resume:
        sampling_line += " (--resume)"
    lines.append(sampling_line)

    max_tau = float(np.max(tau)) if np.all(np.isfinite(tau)) else float("nan")
    convergence_line = (
        f"convergence: max(tau)={max_tau:.0f} steps, burn-in {discard}, "
        f"thin {thin}, {n_samples} samples retained")
    if np.isfinite(max_tau) and arguments.nsteps < 50 * max_tau:
        convergence_line += (f" -- WARNING {arguments.nsteps} steps < "
                             f"50 x max(tau) = {50 * max_tau:.0f}, "
                             f"read the table above with that in mind")
    lines.append(convergence_line)

    return "\n".join(lines)

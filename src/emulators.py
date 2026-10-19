"""Loading of the FASTWIND emulator checkpoints.

The emulators are read from the folder named by the ``EMULATOR_ROOT``
environment variable, or from ``emulators/`` at the top of the repository if
it is not set. Everything else that reads from that folder (the filter curves
in ``gpu_photometry``, the emulator fingerprint in ``version``) uses the same
``DEFAULT_EMULATOR_ROOT``, so it all comes from one place.
"""

import os
import sys

DEFAULT_EMULATOR_ROOT = os.environ.get(
    "EMULATOR_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "emulators"))


def _missing(path):
    """True if ``path`` does not exist, or is a folder without .pth files."""
    if not os.path.exists(path):
        return True
    return os.path.isdir(path) and not any(
        name.endswith(".pth") for name in os.listdir(path))


def load_emulators(load_sed=True, only_lines=None):
    """Return ``(line_emulators, sed_emulator)`` with weights still on the CPU.

    load_sed   : also load the continuum (SED) emulator, needed for the
                 magnitude term; otherwise ``sed_emulator`` is None.
    only_lines : names of the lines to keep, e.g. ``["HEII4541", "HBETA"]``;
                 None keeps all of them. Every line emulator is read either
                 way, which takes a few seconds.

    ``line_emulators`` is a dictionary from line name to emulator. The
    batched wrappers in ``gpu_emulator`` later copy them to the GPU.
    """
    line_models = os.path.join(DEFAULT_EMULATOR_ROOT, "line_emulators_13par",
                               "models")
    sed_model = os.path.join(DEFAULT_EMULATOR_ROOT, "fluxcont", "model",
                             "emulator_FLUXCONT.pth")
    needed = [line_models] + ([sed_model] if load_sed else [])
    absent = [path for path in needed if _missing(path)]
    if absent:
        raise FileNotFoundError(
            "trained emulator files not found: " + ", ".join(absent) + ". They "
            "are available from the author of FASTWIND_emulator "
            "(https://github.com/API-AXIOM/FASTWIND_emulator); see "
            "emulators/README.md for where to put them, or set EMULATOR_ROOT.")
    # The two emulator packages are imported from the emulator folder.
    if DEFAULT_EMULATOR_ROOT not in sys.path:
        sys.path.insert(0, DEFAULT_EMULATOR_ROOT)

    from line_emulators_13par.line_emulator import load_all_lines
    from fluxcont.fluxcont_emulator import FluxcontEmulator

    line_emulators = load_all_lines(line_models)
    if only_lines:
        wanted = set(only_lines)
        unknown = wanted - set(line_emulators)
        if unknown:
            raise KeyError(
                f"unknown line(s): {sorted(unknown)}; available: "
                f"{', '.join(sorted(line_emulators))}")
        line_emulators = {name: emulator
                          for name, emulator in line_emulators.items()
                          if name in wanted}

    sed_emulator = FluxcontEmulator(sed_model) if load_sed else None
    return line_emulators, sed_emulator

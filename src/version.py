"""Provenance: what produced a set of results.

Fits made with different code, settings or emulators are not comparable, so
every run records enough to identify what made it. ``describe()`` returns a
dictionary that is stored in ``results.npz`` under ``pipeline_version``.

Two fingerprints identify a fit. ``source_fingerprint`` is a hash of the code
and configuration files that determine a fit's result, and
``emulator_fingerprint`` a hash of the trained emulator checkpoints. Two
results with the same pair of fingerprints came from identical code,
configuration and emulators. ``git_revision`` names the commit the repository
was at, with "-dirty" appended when tracked files had uncommitted changes.
"""

import glob
import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Code and configuration that determine a fit's result, relative to the
# repository folder.
_SOURCE_FILES = (
    "src/gpu_posterior.py", "src/gpu_broadening.py", "src/gpu_emulator.py",
    "src/gpu_photometry.py", "src/gpu_config.py", "src/map_estimate.py",
    "src/data_io.py", "src/priors.py", "src/physics.py", "src/samplers.py",
    "src/emulators.py", "run_sampler_gpu.py",
    "config/priors.json", "config/instruments.json",
)


def _hash_files(paths, names):
    """Short SHA-256 over the files' names and contents, in the given order.

    A missing file contributes its name and the word "missing", so it changes
    the hash instead of being skipped silently.
    """
    digest = hashlib.sha256()
    for path, name in zip(paths, names):
        digest.update(name.encode())
        if os.path.exists(path):
            with open(path, "rb") as handle:
                digest.update(handle.read())
        else:
            digest.update(b"missing")
    return digest.hexdigest()[:12]


def source_fingerprint():
    """A short hash of the code and configuration files that define a fit.

    Two runs with the same fingerprint used byte-identical files.
    """
    names = sorted(_SOURCE_FILES)
    return _hash_files([os.path.join(_REPOSITORY, name) for name in names],
                       names)


def emulator_fingerprint():
    """A short hash of the trained emulator checkpoints (*.pth).

    Two runs with the same fingerprint used byte-identical emulators.
    """
    from emulators import DEFAULT_EMULATOR_ROOT

    paths = sorted(glob.glob(os.path.join(DEFAULT_EMULATOR_ROOT, "**", "*.pth"),
                             recursive=True))
    if not paths:
        return "none found"
    names = [os.path.relpath(path, DEFAULT_EMULATOR_ROOT) for path in paths]
    return _hash_files(paths, names)


def _git_revision():
    """The current commit, with "-dirty" if tracked files have changes."""
    try:
        result = subprocess.run(
            ["git", "-C", _REPOSITORY, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return ""
        revision = result.stdout.strip()
        status = subprocess.run(
            ["git", "-C", _REPOSITORY, "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=5)
        if status.returncode == 0 and status.stdout.strip():
            revision += "-dirty"
        return revision
    except Exception:
        return ""


def _package_version(name):
    """Installed version of a package, or "" if it is not installed."""
    try:
        return __import__(name).__version__
    except Exception:
        return ""


def describe():
    """Everything needed to identify the code and machine behind a run."""
    try:
        import torch
        device_name = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu")
    except Exception:
        device_name = ""

    return {
        "source_fingerprint": source_fingerprint(),
        "emulator_fingerprint": emulator_fingerprint(),
        "git_revision": _git_revision(),
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "device": device_name,
        "python": sys.version.split()[0],
        **{package: _package_version(package)
           for package in ("numpy", "scipy", "torch", "emcee", "ultranest")},
    }


def format_line(info):
    """One line identifying the code behind a run, from a ``describe()`` dict."""
    commit = info.get("git_revision")
    emulators = info.get("emulator_fingerprint")
    return (f"source {info['source_fingerprint']}"
            + (f", emulators {emulators}" if emulators else "")
            + (f", commit {commit}" if commit else "")
            + f", run {info['run_at']} on {info['hostname']}")


def summary_line():
    """One line identifying the code behind this run, for logs and summaries."""
    return format_line(describe())

"""Batched broadband photometry from the FLUXCONT continuum emulator.

A predicted magnitude is the model continuum integrated through a filter
transmission curve and referred to a zero point. The transmission curve, the
zero point and the emulator's wavelength basis are all fixed for a given
filter, so ``PhotometryPlan`` reads and builds them once and then converts a
whole batch of parameter vectors into magnitudes with one matrix multiply and
one weighted sum.

The integral runs in float64. ``F_lambda`` reaches ~1e37 erg/s/A, which would
overflow float32, and it is only a few hundred wavelength points per parameter
vector, so the cost is negligible.

Filter files are read from ``FILTER_DIRECTORY`` unless another directory is
given, and follow the layout described in ``FILTER_FILES``: one
whitespace-separated ``wavelength transmission`` file per filter, plus a
``zero_points.dat`` table with one row per filter and columns for the Vega,
AB and ST systems. The transmission curves come from the SVO Filter Profile
Service (http://svo2.cab.inta-csic.es/theory/fps/); curves and zero points
are taken from Kiwi-GA (Brands et al. 2022, A&A 663, A36), and the band
average is computed the same way as there.
"""

import os

import numpy as np
import torch

from emulators import DEFAULT_EMULATOR_ROOT
from gpu_emulator import BatchedSedEmulator

PARSEC_CM = 3.0857e18

# filter name -> (transmission file, wavelength unit of that file)
FILTER_FILES = {
    "SPHERE_Ks": ("SPHERE_IRDIS_B_Ks.dat", "nm"),
    "HST_555w": ("HST_ACS_HRC.F555W.dat", "angstrom"),
    "2MASS_Ks": ("2MASS_Ks.dat", "angstrom"),
    "VISTA_Ks": ("Paranal_VISTA.Ks.dat", "angstrom"),
    "Johnson_V": ("GCPD_Johnson.V.dat", "angstrom"),
}

ZERO_POINT_COLUMN = {"vega": 1, "AB": 2, "ST": 3}

# Filter curves and zero_points.dat ship inside the emulator directory.
FILTER_DIRECTORY = os.path.join(DEFAULT_EMULATOR_ROOT, "filter_transmissions")


def load_filter(filter_name, zp_system, filter_directory=None):
    """Return (wavelength_angstrom, transmission, zero_point_flux)."""
    if filter_name not in FILTER_FILES:
        raise ValueError(
            f"unknown filter {filter_name!r}; known: {sorted(FILTER_FILES)}")
    if zp_system not in ZERO_POINT_COLUMN:
        raise ValueError(f"unknown zero-point system {zp_system!r}")

    directory = filter_directory or FILTER_DIRECTORY
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"filter directory {directory!r} does not exist. The magnitude "
            f"term needs zero_points.dat plus the filter transmission curves. "
            f"Pass the directory holding them with --filter-dir, or run with "
            f"--no-magnitude to fit the spectrum alone.")

    filename, wavelength_unit = FILTER_FILES[filter_name]
    transmission_path = os.path.join(directory, filename)
    zero_point_path = os.path.join(directory, "zero_points.dat")
    for path in (transmission_path, zero_point_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing photometry input: {path}")

    zero_points = np.genfromtxt(zero_point_path, comments="#", dtype=str)
    zero_point_flux = None
    for row in np.atleast_2d(zero_points):
        if row[0] == filter_name:
            zero_point_flux = float(row[ZERO_POINT_COLUMN[zp_system]])
            break
    if zero_point_flux is None:
        raise ValueError(f"zero point for {filter_name} not found in {zero_point_path}")

    wavelength, transmission = np.genfromtxt(transmission_path, comments="#").T
    if wavelength_unit == "nm":
        wavelength = wavelength * 10.0
    return wavelength, transmission, zero_point_flux


def _trapezoid_weights(x):
    """Weights w with sum(w * y) == np.trapezoid(y, x).

    The trapezoid rule written as a weighted sum. The weights depend only on
    the wavelength grid, which is fixed for a filter, so they are computed
    once; integrating a whole batch of model fluxes is then a single
    multiply-and-sum on the GPU, without copying to NumPy or recomputing the
    interval widths on every call.
    """
    weights = np.zeros_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


class PhotometryPlan:
    """Everything needed to turn a batch of parameter vectors into magnitudes."""

    def __init__(self, sed_emulator, filter_name, zp_system, distance_pc,
                 device, dtype=torch.float32, filter_directory=None,
                 extinction=0.0):
        wavelength, transmission, zero_point_flux = load_filter(
            filter_name, zp_system, filter_directory)

        # Dust between the star and us dims it. In magnitudes the effect is a
        # constant offset in the band, added to the intrinsic magnitude, so it
        # is applied at the end rather than to the spectrum. A value of zero
        # leaves the magnitude untouched.
        self.extinction = float(extinction)

        self.filter_name = filter_name
        self.zp_system = zp_system
        self.distance_pc = float(distance_pc)
        self.zero_point_flux = float(zero_point_flux)
        self.device = device

        self.emulator = BatchedSedEmulator(sed_emulator, wavelength, device, dtype)

        weights = _trapezoid_weights(np.asarray(wavelength, np.float64))
        weighted_transmission = weights * np.asarray(transmission, np.float64)
        self.weighted_transmission = torch.as_tensor(
            weighted_transmission, dtype=torch.float64, device=device)
        self.transmission_integral = float(weighted_transmission.sum())

        distance_cm = self.distance_pc * PARSEC_CM
        self.geometric_dilution = 4.0 * np.pi * distance_cm ** 2

    def magnitude(self, theta_log_mdot):
        """(N, 13) -> (N,) observed magnitude in the configured filter."""
        log_flux = self.emulator.predict_log_flux(theta_log_mdot)
        flux = torch.pow(10.0, log_flux) / self.geometric_dilution
        filtered = (flux * self.weighted_transmission).sum(dim=-1) \
            / self.transmission_integral
        intrinsic = -2.5 * torch.log10(filtered / self.zero_point_flux)
        return intrinsic + self.extinction

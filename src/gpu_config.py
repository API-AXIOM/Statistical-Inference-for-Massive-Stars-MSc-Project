"""Shared constants: the parameter vector, its bounds, and model defaults.

Every module agrees on the same 15-parameter vector and the same prior box
through this file. Change a bound here and the whole pipeline follows.
"""

import numpy as np

C_KMS = 2.99792458e5  # speed of light in km/s
# Distance to the LMC in parsec (Sander et al. 2024, A&A 689, A30).
LMC_DISTANCE_PC = 49.59e3

# Sampled parameters, in order. Index 3 is log10(mdot); the emulators want
# mdot linear, so the conversion happens inside the forward model.
PARAMETER_BOUNDS = {
    "teff": (25000.044921875, 54999.77734375),
    "logg": (2.5011658668518066, 4.499993801116943),
    "radius": (5.0000715255737305, 29.999823570251465),
    "mdot": (np.log10(3.1628540727421585e-10), np.log10(9.999747472818132e-06)),
    "yhe": (0.08000079542398453, 0.19999997317790985),
    "C": (6.40001106262207, 8.999998092651367),
    "N": (6.500001430511475, 8.999996185302734),
    "O": (6.500010967254639, 9.49998950958252),
    "beta": (0.5000482201576233, 2.999984085559845),
    "vinf": (500.0116271972656, 3999.9855041503906),
    "fic": (3.4524546208558604e-06, 0.9999991013155523),
    "fvel": (5.602571491181152e-06, 0.9999821779460945),
    "fclump": (1.0120052099227905, 39.99984014034271),
    "vsini": (0.0, 600.0),
    "vmacro": (0.0, 200.0),
}

PARAMETER_NAMES = list(PARAMETER_BOUNDS)
BOUNDS_ARRAY = np.asarray(list(PARAMETER_BOUNDS.values()), dtype=np.float64)
N_DIM = len(PARAMETER_NAMES)
N_THETA = 13  # the emulators' parameter vector length
MDOT_INDEX = 3
VSINI_INDEX = 13
VMACRO_INDEX = 14

# The fitting window of each line, (low, high) in Angstrom, covering
# low <= wavelength < high. Windows must not overlap, though neighbours may
# share an edge; building a posterior from overlapping windows raises an
# error. Each window is broadened on its own, so its edges should lie in the
# continuum: a line wing that crosses an edge is cut off there. Check the
# windows against your spectrum before fitting (inspect_spectrum.py lists
# them). Lines marked "no GA_example_input linewindow" use the edges of their
# neighbouring windows; a line not listed here uses its emulator's own range.
LINE_WINDOW_OVERRIDES = {
    "PV1118": (1070.0, 1150.0), #no GA_example_input linewindow
    "CIV1169b": (1150.0, 1210.0), #no GA_example_input linewindow
    "NV1240": (1210.0, 1270.0), #no GA_example_input linewindow
    "OIV1340": (1310.0, 1360.0), #no GA_example_input linewindow
    "OIV1371": (1360.0, 1380.0), #no GA_example_input linewindow
    "SiIV1400": (1380.0, 1435.0), #no GA_example_input linewindow
    "CIV1550": (1500.0, 1580.0), #no GA_example_input linewindow
    "HEII1640": (1615.0, 1665.0), #no GA_example_input linewindow
    "NIV1718": (1690.0, 1735.0), #no GA_example_input linewindow
    "NIII1750": (1735.0, 1780.0), #no GA_example_input linewindow

    "NIV3480": (3475.753, 3488.738),

    "HEPS": (3959.789, 3981.164),
    "HEI4026": (4017.922, 4032.904),
    "NIV4058": (4051.283, 4062.47),
    "CIII4070": (4062.47, 4085.043),  #no GA_example_input linewindow
    "HDELTA": (4085.043, 4119.004),
    "HEII4200": (4192.119, 4207.102),
    "HGAMMA": (4324.565, 4358.525),
    "HEI4387": (4376.504, 4392.086),
    "HEI4471": (4466.4, 4476.388),
    "HEII4541": (4531.524, 4549.103),
    "SiIII4552": (4549.103, 4599.045),  #no GA_example_input linewindow
    "NV4603": (4599.045, 4630.808),
    "CIIINIIIcombined": (4630.808, 4653.781),
    "HEII4686": (4674.956, 4719.904),
    "HBETA": (4845.558, 4877.521),
    "HEI4922": (4916.076, 4928.062),

    "HEII5411": (5402.11, 5422.086),
    "OIII5592": (5586.096, 5599.081),
    "CIII5696": (5689.175, 5703.159),
    "CIV5801": (5790.458, 5820.223),
    "HEI5875": (5869.965, 5880.953),

    "NIV6380": (6375.377, 6387.163),
    "HEII6406": (6392.157, 6421.123),
    "HALPHAHEII6527": (6517.611, 6605.11),
    "HEII6683": (6667.037, 6695.005),
    "HEI7065": (7058.382, 7072.765),
}

# Broadening defaults.
LIMB_DARKENING = 0.6
INSTRUMENT_NSIG = 4.0
# The macroturbulence profile has no edge; cut it at this multiple of vmacro.
# At 3 the discarded wings hold about 2e-6 of its area.
MACRO_EXTENT = 3.0


def line_window(name, emulator):
    """Wavelength window used for fitting one line."""
    if name in LINE_WINDOW_OVERRIDES:
        return LINE_WINDOW_OVERRIDES[name]
    return tuple(float(x) for x in emulator.lambda_range)


def resolve_resolution(resolution, wavelength_points, line_name):
    """Resolving power to broaden one line at.

    ``resolution`` is either a single value for the whole spectrum, or a list
    of ``(low, high, value)`` wavelength bands assigning resolving power by
    region (built by ``data_io.load_instruments`` when a fit names more than
    one instrument). For a scalar, this is a no-op. For bands, every point in
    ``wavelength_points`` must fall in the same band -- a line whose window
    straddles two bands has no single well-defined resolving power, and
    raises rather than guessing which one to use.
    """
    if isinstance(resolution, (int, float, np.floating)):
        return float(resolution)

    values = set()
    uncovered = None
    for point in wavelength_points:
        match = next(((low, high, value) for low, high, value in resolution
                     if low <= point <= high), None)
        if match is None:
            uncovered = point
            break
        values.add(match[2])

    if uncovered is not None:
        raise ValueError(
            f"line {line_name!r} has an observed point at {uncovered:.2f} A "
            f"not covered by any of the given instruments' wavelength "
            f"ranges; add an instrument covering it, or drop the data there.")
    if len(values) > 1:
        raise ValueError(
            f"line {line_name!r} spans {wavelength_points.min():.1f}-"
            f"{wavelength_points.max():.1f} A, which straddles more than one "
            f"instrument's resolving power ({sorted(values)}). Move the "
            f"boundary between the instruments so no line window crosses it.")
    return float(values.pop())

"""Derived stellar quantities and mass-loss recipes.

The sampled parameter vector holds Teff, log g and radius, so mass,
luminosity, escape velocity and the predicted mass-loss rate are all
derivable. Physical priors act on these derived quantities rather than on the
sampled parameters directly, which is what lets them forbid unphysical
*combinations* rather than merely unphysical values.

Everything here works on batches and on plain NumPy arrays, so it can be
evaluated for a whole ensemble at once.
"""

import numpy as np

# Solar and physical constants in the units used throughout.
TEFF_SUN = 5772.0                  # K
GRAVITY_CONSTANT = 6.67430e-8      # cgs
SOLAR_MASS = 1.98892e33            # g
SOLAR_RADIUS = 6.957e10            # cm
SOLAR_LUMINOSITY = 3.828e33        # erg/s
SPEED_OF_LIGHT = 2.99792458e10     # cm/s
THOMSON_OPACITY = 0.34             # cm^2/g, electron scattering, X ~ 0.7

# Metallicity relative to solar. The emulators were trained at LMC
# metallicity, so every star is treated as an LMC star.
LMC_METALLICITY = 0.5


def luminosity(teff, radius):
    """L / L_sun from effective temperature and radius in solar units."""
    return radius ** 2 * (teff / TEFF_SUN) ** 4


def mass(logg, radius):
    """M / M_sun from log g (cgs) and radius in solar units."""
    surface_gravity = 10.0 ** logg
    return (surface_gravity * (radius * SOLAR_RADIUS) ** 2
            / GRAVITY_CONSTANT / SOLAR_MASS)


def eddington_factor(luminosity_solar, mass_solar, opacity=THOMSON_OPACITY):
    """Gamma_e, the ratio of radiative to gravitational acceleration."""
    return (opacity * luminosity_solar * SOLAR_LUMINOSITY
            / (4.0 * np.pi * SPEED_OF_LIGHT * GRAVITY_CONSTANT
               * mass_solar * SOLAR_MASS))


def escape_velocity(mass_solar, radius_solar, gamma=0.0):
    """Effective escape velocity in km/s, reduced by radiation pressure.

    A parameter combination above the Eddington limit has no escape velocity;
    those are clipped to a small positive value so the caller gets a finite
    number to compare against, and the prior can reject them on its own terms.
    """
    reduced = np.clip(np.nan_to_num(1.0 - gamma, nan=1e-6), 1e-6, None)
    inside = np.clip(2.0 * GRAVITY_CONSTANT * mass_solar * SOLAR_MASS
                     * reduced / (radius_solar * SOLAR_RADIUS), 0.0, None)
    return np.sqrt(inside) / 1.0e5


def bjorklund2023_mass_loss(teff, luminosity_solar, mass_solar, metallicity):
    """log10 of the predicted mass-loss rate, in M_sun/yr.

    Bjorklund, Sundqvist, Singh, Puls & Najarro (2023, A&A 676, A109), Eq. 7 -
    self-consistent, steady-state CMF wind models. Successor to Vink et al.
    (2001); finds no bi-stability jump and rates roughly a factor ~3 lower
    than Vink et al. on the hot side.

    The effective mass is M (1 - Gamma_e), with Gamma_e computed from a
    constant electron-scattering opacity of 0.34 cm^2/g. This is how the
    recipe was fitted: their Eq. 6 gives that value for doubly ionised
    helium at N(He)/N(H) = 0.1, the helium abundance of every model in their
    grid.

    Valid ranges: 4.5 <= log(L/Lsun) <= 6.0, 15 <= M/Msun <= 80,
                  15000 K <= Teff <= 50000 K, 0.2 <= Z/Zsun <= 1.0.
    """
    gamma_e = eddington_factor(luminosity_solar, mass_solar)
    mass_eff_solar = mass_solar * (1.0 - gamma_e)

    log_teff_scaled = np.log10(teff / 45000.0)

    return (-5.52
            + 2.39 * np.log10(luminosity_solar / 1.0e6)
            - 1.48 * np.log10(np.clip(mass_eff_solar, 1e-6, None) / 45.0)
            + 2.12 * log_teff_scaled
            + (0.75 - 1.87 * log_teff_scaled) * np.log10(metallicity))


def derived_quantities(theta, metallicity=LMC_METALLICITY):
    """Every derived quantity, for a batch of parameter vectors.

    theta : (N, 15) with log10(mdot) at index 3 and v_inf at index 9, as sampled.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    teff, logg, radius = theta[:, 0], theta[:, 1], theta[:, 2]
    log_mdot, vinf = theta[:, 3], theta[:, 9]

    stellar_luminosity = luminosity(teff, radius)
    stellar_mass = mass(logg, radius)
    gamma = eddington_factor(stellar_luminosity, stellar_mass)
    v_escape = escape_velocity(stellar_mass, radius, gamma)
    predicted = bjorklund2023_mass_loss(teff, stellar_luminosity, stellar_mass,
                                        metallicity)

    # Keep every derived quantity finite: a prior must be able to score even
    # an absurd parameter combination without raising.
    predicted = np.nan_to_num(predicted, nan=-30.0, posinf=30.0, neginf=-30.0)

    return {
        "teff": teff,
        "luminosity": stellar_luminosity,
        "log_luminosity": np.log10(stellar_luminosity),
        "mass": stellar_mass,
        "eddington": gamma,
        "escape_velocity": v_escape,
        "vinf": vinf,
        "log_mdot": log_mdot,
        "log_mdot_predicted": predicted,
        "log_mdot_excess": log_mdot - predicted,
    }

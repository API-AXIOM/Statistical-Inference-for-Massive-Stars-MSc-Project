"""Priors: flat by default, or informed by what is known of LMC stars.

A prior here is an object with three abilities, so that one definition serves
every sampler:

    log_prob(theta)      -> (N,) log prior density, for MCMC
    transform(cube)      -> (N, 15) unit hypercube to parameters, for nested
                            sampling
    sample(n, rng)       -> (n, 15) draws, for simulating spectra with known
                            parameters

``realistic_truths`` gives a separate distribution of realistic LMC O stars to
draw such parameters from.

The emulators were trained at LMC metallicity, so every star is treated as an
LMC star (``physics.LMC_METALLICITY``). Two kinds of information can be
supplied.

**Marginal priors** put a distribution on one parameter at a time. They are
invertible, so they work with nested sampling through ``transform``.

**Relations** put a distribution on a *derived* quantity -- the mass implied 
by log g and radius, the terminal velocity expected from Teff, the mass-loss
rate a recipe predicts. These forbid unphysical *combinations* that no 
per-parameter range can exclude. They cannot be inverted analytically, so 
for nested sampling they should be moved into the likelihood instead (see 
``Prior.relation_log_prob``).

**Caution.** Several of the LMC values and relations used by the informed
prior (set in ``config/priors.json``) have not yet been extensively checked:
the abundance baselines and widths, the helium prior, the mass-loss offset,
the mass bounds, and the coefficients of the wind relations. Verify them against the
literature before using an informed prior for scientific results. The same
values shape the stars drawn by ``realistic_truths``. The flat prior does not
depend on them.
"""

import json
import os

import numpy as np
from scipy import stats

import physics
from gpu_config import BOUNDS_ARRAY, N_DIM, PARAMETER_NAMES

DEFAULT_PRIOR_CONFIG = os.path.join(
    os.environ.get("MCMC_GPU_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "priors.json")


# ---------------------------------------------------------------- marginals
class Uniform:
    """Flat between two bounds."""

    def __init__(self, low, high):
        self.low, self.high = float(low), float(high)

    def log_prob(self, x):
        inside = (x > self.low) & (x < self.high)
        return np.where(inside, -np.log(self.high - self.low), -np.inf)

    def ppf(self, u):
        return self.low + u * (self.high - self.low)

    def describe(self):
        return f"uniform [{self.low:.4g}, {self.high:.4g}]"


class TruncatedNormal:
    """Gaussian confined to a range."""

    def __init__(self, centre, width, low, high):
        self.centre, self.width = float(centre), float(width)
        self.low, self.high = float(low), float(high)
        a = (self.low - self.centre) / self.width
        b = (self.high - self.centre) / self.width
        self._frozen = stats.truncnorm(a, b, loc=self.centre, scale=self.width)

    def log_prob(self, x):
        with np.errstate(divide="ignore"):
            return self._frozen.logpdf(x)

    def ppf(self, u):
        return self._frozen.ppf(np.clip(u, 1e-12, 1 - 1e-12))

    def describe(self):
        return (f"normal {self.centre:.4g} +- {self.width:.4g} "
                f"truncated to [{self.low:.4g}, {self.high:.4g}]")


class TruncatedGamma:
    """Gamma distribution with a given mean and standard deviation, confined
    to a range.

    Used for vsini: a gamma distribution is zero at zero, rises to a single
    peak and has a tail towards high values, which is the shape of observed
    vsini distributions of O stars.
    """

    def __init__(self, mean, sd, low, high):
        self.mean, self.sd = float(mean), float(sd)
        self.low, self.high = float(low), float(high)
        self._gamma = stats.gamma((self.mean / self.sd) ** 2,
                                  scale=self.sd ** 2 / self.mean)
        self._cdf_low = self._gamma.cdf(self.low)
        self._mass = self._gamma.cdf(self.high) - self._cdf_low

    def log_prob(self, x):
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore"):
            inside = (x >= self.low) & (x <= self.high)
            return np.where(inside, self._gamma.logpdf(x) - np.log(self._mass),
                            -np.inf)

    def ppf(self, u):
        return self._gamma.ppf(self._cdf_low + np.asarray(u) * self._mass)

    def describe(self):
        return (f"gamma, mean {self.mean:.4g} sd {self.sd:.4g}, "
                f"truncated to [{self.low:.4g}, {self.high:.4g}]")


# ---------------------------------------------------------------- relations
class Relation:
    """A Gaussian expectation on a quantity derived from several parameters.

    ``quantity`` maps the derived-quantity dictionary to a value per sample.
    The log density is ``-0.5 * ((value - centre) / width) ** 2``, peaking at
    zero, which keeps rejection sampling in ``Prior.sample`` exact.
    """

    def __init__(self, name, quantity, centre, width, description=""):
        self.name = name
        self.quantity = quantity
        self.centre, self.width = float(centre), float(width)
        self.description = description

    def log_prob(self, derived):
        value = self.quantity(derived)
        return -0.5 * ((value - self.centre) / self.width) ** 2

    def describe(self):
        return (f"{self.name}: {self.centre:.4g} +- {self.width:.4g}"
                + (f"  ({self.description})" if self.description else ""))


class LinearRelation:
    """A Gaussian expectation on y around a straight line in x.

    The expected value is ``slope * (x / x_scale) - intercept`` and the width
    is ``scatter``, the root-mean-square spread of stars about the line. The
    width is the same at every x, so the log density
    ``-0.5 * ((y - expected) / scatter) ** 2`` differs from a normalised
    Gaussian only by a constant. It peaks at zero, which keeps rejection
    sampling in ``Prior.sample`` exact.
    """

    def __init__(self, name, x_quantity, y_quantity, slope, intercept,
                 scatter, x_scale=1.0, description=""):
        self.name = name
        self.x_quantity = x_quantity
        self.y_quantity = y_quantity
        self.slope, self.intercept = float(slope), float(intercept)
        self.scatter = float(scatter)
        self.x_scale = float(x_scale)
        self.description = description

    def log_prob(self, derived):
        x = self.x_quantity(derived) / self.x_scale
        y = self.y_quantity(derived)
        expected = self.slope * x - self.intercept
        return -0.5 * ((y - expected) / self.scatter) ** 2

    def describe(self):
        return (f"{self.name}: {self.slope:.4g} x - {self.intercept:.4g}, "
                f"scatter {self.scatter:.4g}"
                + (f"  ({self.description})" if self.description else ""))


class HardBoundsRelation:
    """A top-hat constraint on a quantity derived from model parameters.

    Returns zero log-prior inside [low, high] and -inf outside. It therefore
    rejects invalid parameter combinations without preferring any value within
    the permitted interval.
    """

    def __init__(self, name, quantity, low, high, description=""):
        self.name = name
        self.quantity = quantity
        self.low = float(low)
        self.high = float(high)
        self.description = description

        if not self.high > self.low:
            raise ValueError(
                f"{self.name}: high={self.high} must exceed low={self.low}"
            )

    def log_prob(self, derived):
        value = self.quantity(derived)
        inside = (value >= self.low) & (value <= self.high)
        return np.where(inside, 0.0, -np.inf)

    def describe(self):
        result = f"{self.name}: hard bounds [{self.low:.4g}, {self.high:.4g}]"
        if self.description:
            result += f"  ({self.description})"
        return result


def _wind_relation_coefficients(config, key):
    """Slope ``a``, intercept ``b`` and scatter ``rms`` of a v_inf relation."""
    block = config.get("lmc", {}).get(key, {})
    missing = [name for name in ("a", "b", "rms") if name not in block]
    if missing:
        raise ValueError(
            f"the '{key}' entry of the prior config's 'lmc' block needs "
            f"{', '.join(missing)}; set relations.{key}.use to false to "
            f"disable the relation.")
    return float(block["a"]), float(block["b"]), float(block["rms"])


def standard_relations(metallicity, config, use_mass_loss=None,
                       use_vinf_vesc=None):
    """The physical relations worth imposing on a hot massive star.

    Each relation has an entry in the ``relations`` block of the prior config,
    whose ``use`` flag switches it on or off:

    - ``mass`` (on unless ``use`` is false): hard bounds ``low`` and ``high``
      on the spectroscopic mass M = g R^2 / G.
    - ``vinf_teff`` (on unless ``use`` is false): a Gaussian on v_inf around
      v_inf = a * (Teff / 100 K) - b, with width ``rms``.
    - ``vinf_vesc`` (off unless ``use`` is true): a Gaussian on v_inf around
      v_inf = a * v_esc - b, with width ``rms``.
    - ``mass_loss`` (off unless ``use`` is true): a Gaussian with ``centre``
      and ``width`` in dex on the offset of log mdot from the Bjorklund et
      al. (2023) recipe.

    The coefficients of the two v_inf relations sit in the ``lmc`` block.
    Both relations were fitted to the same stars (Hawcroft et al. 2024), so
    they are not independent; switch on one of them at a time.

    Parameters
    ----------
    metallicity : float
        Z / Z_sun, passed to the mass-loss recipe.
    config : dict
        The prior config, as returned by ``load_prior_config``.
    use_mass_loss, use_vinf_vesc : bool or None
        Switch the mass-loss or the v_inf(v_esc) relation on or off,
        overriding the config. ``None`` uses the config's ``use`` flag.

    Returns
    -------
    list of relation objects, each with ``log_prob(derived)`` and ``describe()``.
    """
    settings = config.get("relations", {})
    relations = []

    block = settings.get("mass", {})
    if block.get("use", True):
        low, high = block.get("low"), block.get("high")
        if None in (low, high):
            raise ValueError(
                "the mass relation needs 'low' and 'high' (the 'mass' entry "
                "of the prior config's 'relations' block); set "
                "relations.mass.use to false to disable it.")
        relations.append(HardBoundsRelation(
            "current spectroscopic mass", lambda d: d["mass"], low, high,
            "M_spec = g R^2 / G; derived physicality constraint"))

    if settings.get("vinf_teff", {}).get("use", True):
        slope, intercept, scatter = _wind_relation_coefficients(
            config, "vinf_teff")
        relations.append(LinearRelation(
            "vinf(Teff)", lambda d: d["teff"], lambda d: d["vinf"],
            slope, intercept, scatter, x_scale=100.0,
            description="x = Teff / 100 K; Hawcroft et al. (2024)"))

    vinf_vesc_on = (use_vinf_vesc if use_vinf_vesc is not None
                    else settings.get("vinf_vesc", {}).get("use", False))
    if vinf_vesc_on:
        slope, intercept, scatter = _wind_relation_coefficients(
            config, "vinf_vesc")
        relations.append(LinearRelation(
            "vinf(vesc)", lambda d: d["escape_velocity"], lambda d: d["vinf"],
            slope, intercept, scatter,
            description="x = v_esc in km/s; Hawcroft et al. (2024)"))

    block = settings.get("mass_loss", {})
    mass_loss_on = (use_mass_loss if use_mass_loss is not None
                    else block.get("use", False))
    if mass_loss_on:
        relations.append(Relation(
            "log mdot excess", lambda d: d["log_mdot_excess"],
            block.get("centre", 0.0), block.get("width", 0.5),
            f"mass loss against the Bjorklund et al. (2023) recipe at "
            f"Z = {metallicity:.2f}"))

    return relations


# -------------------------------------------------------------------- prior
class Prior:
    """Marginal distributions per parameter, plus optional relations.

    Parameters
    ----------
    marginals : list
        One distribution per sampled parameter, in ``PARAMETER_NAMES`` order,
        each with ``log_prob``, ``ppf`` and ``describe``.
    relations : sequence
        Relations on derived quantities (see ``standard_relations``).
    metallicity : float
        Z / Z_sun used when computing derived quantities.
    name : str
        Label used by ``describe``.
    """

    def __init__(self, marginals, relations=(),
                 metallicity=physics.LMC_METALLICITY, name="flat"):
        if len(marginals) != N_DIM:
            raise ValueError(f"expected {N_DIM} marginals, got {len(marginals)}")
        self.marginals = list(marginals)
        self.relations = list(relations)
        self.metallicity = float(metallicity)
        self.name = name

    # -- densities ---------------------------------------------------------
    def marginal_log_prob(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        total = np.zeros(theta.shape[0])
        for index, marginal in enumerate(self.marginals):
            total = total + marginal.log_prob(theta[:, index])
        return total

    def relation_log_prob(self, theta):
        """The relation terms alone.

        A nested sampler that cannot invert these should add this to its
        log-likelihood instead, and use ``transform`` for the marginals only.
        """
        if not self.relations:
            theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
            return np.zeros(theta.shape[0])
        derived = physics.derived_quantities(theta, self.metallicity)
        total = np.zeros(np.atleast_2d(theta).shape[0])
        for relation in self.relations:
            total = total + relation.log_prob(derived)
        return total

    def log_prob(self, theta):
        single = np.asarray(theta).ndim == 1
        value = self.marginal_log_prob(theta)
        finite = np.isfinite(value)
        if finite.any() and self.relations:
            value = value + np.where(finite, self.relation_log_prob(theta), 0.0)
        return float(value[0]) if single else value

    # -- for nested sampling ----------------------------------------------
    def transform(self, unit_cube):
        """Unit hypercube to parameters, using the marginals only."""
        cube = np.atleast_2d(np.asarray(unit_cube, dtype=np.float64))
        out = np.empty_like(cube)
        for index, marginal in enumerate(self.marginals):
            out[:, index] = marginal.ppf(cube[:, index])
        return out[0] if np.asarray(unit_cube).ndim == 1 else out

    # -- for simulating spectra with known parameters ----------------------
    def sample(self, n_samples, rng, max_attempts=20000):
        """Draw from the full prior, relations included.

        Marginal draws are accepted with probability exp(relation log-prob),
        which is exact because the relation terms peak at zero.
        """
        collected = []
        total = 0
        for _ in range(max_attempts):
            if total >= n_samples:
                break
            block = max(n_samples - total, 256)
            cube = rng.random((block, N_DIM))
            candidate = self.transform(cube)
            if self.relations:
                accept = np.log(rng.random(block)) < self.relation_log_prob(candidate)
                candidate = candidate[accept]
            if candidate.size:
                collected.append(candidate)
                total += candidate.shape[0]
        if total < n_samples:
            raise RuntimeError(
                f"only drew {total}/{n_samples} prior samples; the relations "
                f"may be too tight for the marginal ranges")
        return np.vstack(collected)[:n_samples]

    # -- reporting ---------------------------------------------------------
    def describe(self):
        lines = [f"prior '{self.name}' (Z = {self.metallicity:.2f} solar)", ""]
        for name, marginal in zip(PARAMETER_NAMES, self.marginals):
            lines.append(f"  {name:>8s}  {marginal.describe()}")
        if self.relations:
            lines.append("")
            lines.append("  relations on derived quantities:")
            for relation in self.relations:
                lines.append(f"    {relation.describe()}")
        return "\n".join(lines)


# ------------------------------------------------------------------ builders
def load_prior_config(path=None):
    """Read the prior config (``config/priors.json`` unless ``path`` is given)."""
    with open(path or DEFAULT_PRIOR_CONFIG) as handle:
        return json.load(handle)


def informative_prior(config=None, use_relations=True, use_mass_loss=None,
                      use_vinf_vesc=None):
    """Build a prior from what is known of LMC stars.

    The ``lmc`` block of the prior config sets the abundance baselines,
    helium and the coefficients of the terminal-velocity relations; the
    ``relations`` block switches each relation on or off (see
    ``standard_relations``). This information is independent of the spectrum
    being fitted, so it adds genuine constraint.

    ``use_mass_loss`` turns the Bjorklund et al. (2023) mass-loss relation on
    or off, overriding ``config``. It is **off by default** (``None`` falls
    back to ``relations.mass_loss.use`` in the config, itself ``False``): the
    sampled parameter box extends well outside the recipe's calibrated range,
    so whether this relation helps or hurts recovery is an open question, not
    a settled constraint to impose by default.

    ``use_vinf_vesc`` turns the v_inf(v_esc) relation on or off in the same
    way, overriding ``relations.vinf_vesc.use`` in the config (``False``).
    It is off by default because it was fitted to the same stars as the
    v_inf(Teff) relation, which is on; use one of the two at a time.
    ``use_relations=False`` drops every relation, leaving only the marginals.
    """
    config = config or load_prior_config()
    block = config["lmc"]
    marginals = [Uniform(low, high) for low, high in BOUNDS_ARRAY]
    bounds = {name: (low, high)
              for name, (low, high) in zip(PARAMETER_NAMES, BOUNDS_ARRAY)}
    metallicity = float(block.get("metallicity", physics.LMC_METALLICITY))

    # Abundances: flat on one side of the LMC baseline, zero on the other.
    # The CNO cycle turns carbon and oxygen into nitrogen, and mixing brings
    # the processed material to the surface, so a star's nitrogen can only
    # rise above its birth value and its carbon and oxygen only fall below it.
    # The cut is moved past the baseline by a margin, because measurements
    # scatter and stars are not all born with exactly the baseline values.
    margins = block.get("baseline_margin", {})
    for element, direction in (("C", "down"), ("N", "up"), ("O", "down")):
        baseline = float(block["abundances"][element])
        margin = float(margins.get(element, 0.0))
        low, high = bounds[element]
        if direction == "up":
            cut = float(np.clip(baseline - margin, low, high))
            marginal = Uniform(cut, high)
        else:
            cut = float(np.clip(baseline + margin, low, high))
            marginal = Uniform(low, cut)
        marginals[PARAMETER_NAMES.index(element)] = marginal

    helium = block.get("yhe", {})
    if helium:
        low, high = bounds["yhe"]
        marginals[PARAMETER_NAMES.index("yhe")] = TruncatedNormal(
            float(helium["centre"]), float(helium["width"]),
            max(float(helium.get("low", low)), low), high)

    relations = (standard_relations(metallicity, config, use_mass_loss,
                                    use_vinf_vesc)
                 if use_relations else [])
    return Prior(marginals, relations, metallicity, name="LMC")


def realistic_truths(config=None):
    """A distribution of realistic LMC O stars, for drawing the parameters of
    simulated spectra while the fits themselves keep a flat prior.

    It is the informative prior with the Bjorklund et al. (2023) mass-loss
    relation switched on, vsini drawn from a truncated gamma distribution
    with the mean and standard deviation observed for LMC O stars, and the
    electron-scattering Eddington factor Gamma_e capped (both set in the
    ``realistic_truths`` block of the prior config). Teff, log g, radius,
    beta, the clumping parameters and vmacro stay uniform over the box; the
    mass bounds and the terminal-velocity relations act on their combinations.

    Only ``sample`` is meant to be used: the distribution describes which
    stars to simulate, not what a fit should assume.
    """
    config = config or load_prior_config()
    block = config["realistic_truths"]
    prior = informative_prior(config=config, use_mass_loss=True)

    rotation = block["rotation"]
    vsini = PARAMETER_NAMES.index("vsini")
    low, high = BOUNDS_ARRAY[vsini]
    prior.marginals[vsini] = TruncatedGamma(
        rotation["mean"], rotation["sd"], low, high)
    prior.relations.append(HardBoundsRelation(
        "Gamma_e", lambda d: d["eddington"], 0.0, block["eddington_cap"],
        "electron-scattering Eddington factor"))
    prior.name = "realistic-LMC"
    return prior

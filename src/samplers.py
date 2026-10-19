"""Sampler-specific code for the two inference methods.

``run_nested`` runs nested sampling of a ``GPUPosterior`` with UltraNest and
returns a dictionary with

    flat_samples : (n_samples, 15) equally weighted posterior samples
    parameters_50p, parameters_map : point estimates
    seconds, n_evaluations : cost

plus the evidence and UltraNest's own diagnostics.

``resolve_move`` builds the emcee proposal moves for the MCMC fit, which
itself runs in ``fit_emcee`` in ``run_sampler_gpu.py``.
"""

import time

import numpy as np

from gpu_config import PARAMETER_NAMES


# How emcee walkers propose their next position:
#   "stretch" (emcee's default) proposes a point on the line through two
#       walkers. It works well when the posterior is roughly round, but is
#       slow when parameters are strongly correlated, as stellar parameters
#       are (temperature with radius, the wind parameters with each other).
#   "de" (differential evolution) steps along the difference between two
#       other walkers, so its steps follow the shape of the correlations.
#   "de-snooker" mixes "de" with emcee's snooker variant of it.
# The thesis used only "stretch"; "de" and "de-snooker" are untested.
def resolve_move(name):
    import emcee.moves as m
    if name in (None, "stretch"):
        return None
    if name == "de":
        return [(m.DEMove(), 1.0)]
    if name == "de-snooker":
        return [(m.DEMove(), 0.8), (m.DESnookerMove(), 0.2)]
    raise ValueError(f"unknown move {name!r}; use stretch, de or de-snooker")


# Batch sizes at which the likelihood is recorded as a CUDA graph (see
# `run_nested`). Each recorded graph keeps its own GPU memory, so the sizes
# stay modest.
NESTED_GRAPH_SIZES = (64, 128, 256, 512, 1024)

# Starting step length of the slice sampler, as a fraction of the unit cube.
# Proposals that land outside the cube are thrown away, and in 15 dimensions
# a long step lands outside nearly every time. Starting short is safe,
# because the slice sampler lengthens its steps when needed, and the scale
# adapts during the run anyway.
DEFAULT_STEP_SCALE = 0.05

# Number of walkers the population sampler moves together; their proposals
# go to the GPU in one batched call. This only affects speed, not the
# statistics of the run (those are set by the number of live points).
# More walkers is not faster: each call still returns at most one new live
# point, and walkers waiting longer for their turn have to redo more steps.
DEFAULT_WALKERSIZE = 50


# Ways the slice sampler can choose a direction to step in. "mixture" (the
# default) combines random directions with directions that follow the shape
# of the live points, which helps when parameters are correlated.
STEP_DIRECTIONS = {
    "mixture": "generate_mixture_random_direction",
    "region": "generate_region_oriented_direction",
    "region-random": "generate_region_random_direction",
    "cube": "generate_cube_oriented_direction",
    "random": "generate_random_direction",
    "differential": "generate_differential_direction",
}

# How a step sampler moves a walker:
#   "slice" (default) searches along a line until it finds a point above the
#       likelihood threshold, so every step moves the walker.
#   "randomwalk" makes random jumps and keeps only those that land above the
#       threshold. Experimental: not validated, and on test problems it can
#       stop with UltraNest's "some walkers never moved" error.
STEP_METHODS = ("slice", "randomwalk")


def run_nested(posterior, rng, n_live=600, log_dir=None, max_calls=None,
               use_cuda_graph=False, batch_size=16384,
               graph_sizes=NESTED_GRAPH_SIZES, step_sampler_steps=None,
               step_direction="mixture", adaptive_nsteps=False,
               population=True, walkersize=None, step_scale=None,
               step_method="slice", walk_adapt_factor=0.9,
               benchmark=False, posterior_only=False,
               progress=False, verbose=True, resume="overwrite",
               warmstart_max_tau=0.5, **_):
    """Nested sampling with UltraNest.

    Nested sampling uses the likelihood and the prior separately. The prior
    enters through ``transform``, which maps points in the unit cube to
    parameter values; ``GPUPosterior`` provides both.

    Prior relations between parameters, such as the mass implied by log g and
    radius, cannot be written as such a transform, so they are added to the
    likelihood instead. The posterior is the same.

    UltraNest runs vectorised: it sends the GPU a batch of points at a time.
    ``batch_size`` is how many candidate points it proposes per round. Late in
    a run only a few per cent of them pass UltraNest's cheap pre-check and
    reach the likelihood, so a large ``batch_size`` keeps the GPU batches
    big, while rejected candidates cost almost nothing. This only matters
    when no step sampler is used (``step_sampler_steps=None``).

    To continue a run, ``run_sampler_gpu.py`` passes
    ``resume="resume-similar"``. A resumed run may be on different hardware,
    where the likelihood can come out very slightly different. Plain
    ``"resume"`` stops on any difference; ``"resume-similar"`` accepts small
    ones (set by ``warmstart_max_tau``) and carries on.

    ``step_method`` chooses how walkers move: ``"slice"`` (default) or
    ``"randomwalk"``; see ``STEP_METHODS``. Both need ``step_sampler_steps``,
    and they need different numbers of steps for a correct posterior.
    """
    import ultranest

    started = time.perf_counter()
    relations = (posterior.prior is not None
                 and bool(getattr(posterior.prior, "relations", ())))

    def transform(cube):
        return posterior.prior_transform(cube)

    # Every GPU call has a fixed start-up cost, and the population slice
    # sampler gains at most one live point per call, so this cost limits the
    # speed of the run. A CUDA graph records the GPU work once and replays it,
    # which removes most of that cost. A graph only fits one batch size, so
    # graphs are recorded at several sizes and each call is padded up to the
    # nearest one; calls larger than the biggest graph run without one.
    if use_cuda_graph:
        posterior.enable_cuda_graph(*graph_sizes, kind="likelihood",
                                    verbose=verbose)

    # A progress line is printed every 30 s. The likelihood wrapper below
    # counts calls and batch sizes; UltraNest's visualisation hook
    # (`record_state`) adds how far the run has got, but UltraNest only calls
    # it when `log_dir` is set. Throughput is measured over the last 30 s, not
    # since the start, because the batch size can drop a lot during a run.
    counter = {"calls": 0, "batches": 0, "reported": time.perf_counter(),
               "calls_at_report": 0, "batches_at_report": 0}
    state = {}

    def record_state(points, info, region, transformLayer, region_fresh=False):
        logl = points["logl"]
        state["it"] = info["it"]
        state["logl_range"] = (float(np.min(logl)), float(np.max(logl)))
        # Estimated share of the posterior still beyond the live points;
        # falling below `frac_remain` is one of the conditions for stopping.
        # It stays near 100% for most of a run and drops quickly at the end.
        gap = np.clip(info["logz"] - info["logz_remain"], -700.0, 700.0)
        state["remaining"] = 1.0 / (1.0 + np.exp(gap))
        # Slice sampler: the share of walkers that have not yet finished their
        # step, and the current step scale. Many unfinished walkers together
        # with small batches means the GPU is waiting on a few slow walkers.
        logstat = getattr(sampler.stepsampler, "logstat", None)
        if population and logstat and step_method == "slice":
            completed, scale = logstat[-1][0], logstat[-1][1]
            state["slice"] = (float(scale), 1.0 - float(completed))
        # Random walk: the share of all jumps accepted in the last set of
        # steps, and the jump length, which UltraNest tunes towards 23.4%.
        if population and logstat and step_method == "randomwalk":
            accepted, scale = logstat[-1][1], logstat[-1][2]
            state["walk"] = (float(scale), float(accepted))

    def report():
        now = time.perf_counter()
        interval = now - counter["reported"]
        calls = counter["calls"] - counter["calls_at_report"]
        batches = counter["batches"] - counter["batches_at_report"]
        hours, seconds = divmod(int(now - started), 3600)
        parts = [f"{hours}h{seconds // 60:02d}m"]
        if "it" in state:
            parts.append(f"it {state['it']:,}")
        parts.append(f"{counter['calls']:,} calls")
        parts.append(f"last {interval:.0f} s: {calls / interval:,.0f} calls/s, "
                     f"batch {calls / max(batches, 1):.0f}")
        if "logl_range" in state:
            low, high = state["logl_range"]
            parts.append(f"logL live {low:.1f}..{high:.1f}")
            parts.append(f"remaining {100 * state['remaining']:.0f}% "
                         f"(target <{100 * state['stop_remain']:.0f}%)")
        if "slice" in state:
            scale, unfinished = state["slice"]
            parts.append(f"slice scale {scale:.3g}, "
                         f"{100 * unfinished:.0f}% unfinished")
        if "walk" in state:
            scale, accepted = state["walk"]
            parts.append(f"walk scale {scale:.3g}, "
                         f"{100 * accepted:.0f}% accepted")
        print("[nested] " + " | ".join(parts), flush=True)
        counter["reported"] = now
        counter["calls_at_report"] = counter["calls"]
        counter["batches_at_report"] = counter["batches"]

    def log_likelihood(theta):
        theta = np.atleast_2d(theta)
        counter["calls"] += theta.shape[0]
        counter["batches"] += 1
        if verbose and time.perf_counter() - counter["reported"] > 30.0:
            report()
        value = posterior.log_likelihood(theta)
        if relations:
            value = value + posterior.prior.relation_log_prob(theta)
        return np.where(np.isfinite(value), value, -1e100)

    sampler = ultranest.ReactiveNestedSampler(
        param_names=list(PARAMETER_NAMES),
        loglike=log_likelihood,
        transform=transform,
        vectorized=True,
        ndraw_min=batch_size // 4,
        ndraw_max=batch_size,
        log_dir=log_dir,
        resume=resume if log_dir else "overwrite",
        warmstart_max_tau=warmstart_max_tau,
    )

    # In 15 dimensions, drawing new live points at random from a region
    # around the current ones barely works: almost every draw falls below the
    # likelihood threshold. A step sampler instead starts from an existing
    # live point and walks it to a new one in `nsteps` steps.
    #
    # The catch: correctness now depends on `nsteps`. With too few steps a new
    # point stays close to where it started, the live points slowly bunch
    # together, and the posterior comes out too narrow, without any warning.
    # Tune `nsteps` and check the result on spectra with known parameters.
    if step_method not in STEP_METHODS:
        raise ValueError(f"unknown step method {step_method!r}; "
                         f"choose from {STEP_METHODS}")
    if step_method == "randomwalk" and not (step_sampler_steps and population):
        raise ValueError("the random-walk step sampler needs a step count and "
                         "the population sampler")
    if step_sampler_steps:
        name = STEP_DIRECTIONS.get(step_direction, step_direction)

        # Each slice step depends on the one before, so a single walker sends
        # the GPU one point per call, and one point costs about as much as a
        # few hundred. The population sampler moves `walkersize` walkers
        # together, so every step is one batched call.
        if population:
            from ultranest import popstepsampler as stepsampler
            if not hasattr(stepsampler, name):
                raise ValueError(
                    f"direction {step_direction!r} is not available for the "
                    f"population sampler")
            size = int(walkersize or DEFAULT_WALKERSIZE)
        if population and step_method == "randomwalk":
            # The random walk moves all walkers `nsteps` times. Each step is a
            # random jump, kept only if it lands above the likelihood
            # threshold, and is one batched GPU call. Afterwards the whole
            # population is handed out as new live points, one per iteration,
            # without further calls (UltraNest skips any the rising threshold
            # has overtaken). A rejected jump leaves the walker where it was,
            # so a random walk needs more steps than slice sampling.
            #
            # UltraNest tunes the jump length towards 23.4% accepted jumps,
            # starting from `step_scale` (default 1): after each set of steps
            # it multiplies or divides the length by `walk_adapt_factor`. The
            # run stops with an error if a walker has every jump rejected; a
            # smaller factor adapts faster and makes this less likely.
            scale = float(step_scale) if step_scale else 1.0
            sampler.stepsampler = stepsampler.PopulationRandomWalkSampler(
                popsize=size,  # UltraNest's own constructor argument name
                nsteps=int(step_sampler_steps),
                generate_direction=getattr(stepsampler, name),
                scale=scale,
                scale_adapt_factor=float(walk_adapt_factor),
            )
            if verbose:
                print(f"[nested] population random-walk sampler: "
                      f"{step_sampler_steps} steps, {size} walkers evolved "
                      f"together, {step_direction} directions, "
                      f"initial step scale {scale:.3f}, adapting by "
                      f"{walk_adapt_factor:g} per set of steps", flush=True)
        elif population:
            # Proposals outside the unit cube are thrown away before the
            # likelihood is called, so a step that is too long leaves only
            # small batches for the GPU. Start from a short step unless told
            # otherwise.
            scale = float(step_scale) if step_scale else DEFAULT_STEP_SCALE
            sampler.stepsampler = stepsampler.PopulationSliceSampler(
                popsize=size,  # UltraNest's own constructor argument name
                nsteps=int(step_sampler_steps),
                generate_direction=getattr(stepsampler, name),
                scale=scale,
            )
            if verbose:
                print(f"[nested] population slice sampler: "
                      f"{step_sampler_steps} steps, {size} walkers evolved "
                      f"together, {step_direction} directions, "
                      f"step scale {scale:.3f}", flush=True)
        else:
            from ultranest import stepsampler
            if not hasattr(stepsampler, name):
                raise ValueError(
                    f"unknown step direction {step_direction!r}; "
                    f"choose from {sorted(STEP_DIRECTIONS)}")
            sampler.stepsampler = stepsampler.SliceSampler(
                nsteps=int(step_sampler_steps),
                generate_direction=getattr(stepsampler, name),
                adaptive_nsteps="move-distance" if adaptive_nsteps else False,
            )
            if verbose:
                print(f"[nested] serial slice step sampler: "
                      f"{step_sampler_steps} steps, {step_direction} "
                      f"directions"
                      + (" (adaptive)" if adaptive_nsteps else ""), flush=True)
    # `progress` shows UltraNest's live progress bar. It keeps redrawing one
    # line, which in a log file turns into one enormous line, so use it only in
    # a terminal; the report every 30 s covers batch jobs.
    #
    # The run stops once all of these are met:
    #   dlogz       - the evidence is precise enough;
    #   dKL         - the posterior's information content is precise enough;
    #   frac_remain - at most this share of the posterior is estimated to lie
    #                 beyond the live points, so the run has found the
    #                 high-likelihood region;
    #   min_ess     - at least 400 effective posterior samples (UltraNest's
    #                 default, not set here).
    # When only the posterior is wanted (`posterior_only`), dlogz and dKL are
    # loosened. frac_remain stays at 1%: on the 3-line test problem, stopping
    # at 50% sometimes ended a run before it had found the highest likelihoods,
    # and repeated fits then disagreed (in log Z by ~3).
    evidence_tolerance = 10.0 if posterior_only else 0.5
    frac_remain = 0.01
    kl_tolerance = 2.0 if posterior_only else 0.5
    state["stop_remain"] = frac_remain
    results = sampler.run(min_num_live_points=n_live,
                          max_ncalls=max_calls,
                          dlogz=evidence_tolerance,
                          frac_remain=frac_remain,
                          dKL=kl_tolerance,
                          show_status=progress,
                          viz_callback=record_state if verbose else False)
    if max_calls is not None and results["ncall"] >= max_calls:
        if benchmark:
            print(f"[nested] stopped at the {max_calls:,} call cap as asked "
                  f"(timing run, so the posterior is not meant to be used).",
                  flush=True)
        else:
            print(f"[nested] WARNING stopped at the {max_calls:,} call cap "
                  f"before converging; the posterior is not trustworthy. "
                  f"Effective sample size {results['ess']:.0f}.", flush=True)
    elapsed = time.perf_counter() - started

    points = np.asarray(results["weighted_samples"]["points"], dtype=np.float64)
    weights = np.asarray(results["weighted_samples"]["weights"], dtype=np.float64)
    log_likelihoods = np.asarray(results["weighted_samples"]["logl"])

    # Resample to equal weights so downstream code needs no special case.
    # The weighted samples are kept as well, for anything that can use them.
    n_draw = min(20000, max(len(points), 1))
    index = rng.choice(len(points), size=n_draw, replace=True,
                       p=weights / weights.sum())
    flat = points[index]

    if verbose:
        print(f"[nested] log Z = {results['logz']:.2f} +- "
              f"{results['logzerr']:.2f}, {results['ncall']:,} likelihood "
              f"calls, effective sample size {results['ess']:.0f}", flush=True)

    return {
        "sampler": "nested",
        "flat_samples": flat,
        "weights": None,
        "weighted_points": points,
        "weighted_weights": weights,
        "parameters_50p": np.percentile(flat, 50, axis=0),
        # The sample with the highest likelihood (plus any relation terms).
        # The marginal priors are not included, since nested sampling keeps
        # them out of the likelihood, so with an informative prior this is not
        # exactly the MAP; with the flat prior it is the MAP within the box.
        "parameters_map": points[int(np.argmax(log_likelihoods))],
        "log_evidence": float(results["logz"]),
        "log_evidence_error": float(results["logzerr"]),
        "effective_sample_size": float(results["ess"]),
        "n_live": int(n_live),
        "step_sampler_steps": (int(step_sampler_steps)
                               if step_sampler_steps else 0),
        # The information H (the KL divergence from prior to posterior) sets
        # both the run time and the evidence uncertainty, so it shows how hard
        # the problem was.
        "information": float(results.get("H", float("nan"))),
        "insertion_test_passed": bool(
            results.get("insertion_order_MWW_test", {}).get("converged", True)),
        "step_direction": step_direction if step_sampler_steps else "",
        "step_method": step_method if step_sampler_steps else "",
        "posterior_only": bool(posterior_only),
        "n_evaluations": int(results["ncall"]),
        "mean_batch": float(counter["calls"] / max(counter["batches"], 1)),
        "seconds": elapsed,
        # kept so the same plotting code works for both methods
        "chain": points[:, None, :],
        "discard": 0, "thin": 1,
        "acceptance_fraction": np.nan,
        "autocorrelation_time": np.nan,
    }


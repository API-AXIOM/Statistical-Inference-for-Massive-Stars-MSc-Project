"""Starting point for the fits: the maximum a posteriori (MAP) estimate.

``multi_start_map`` runs ``find_map`` from several independent starts, keeps
the best result, and refines that once more with a strict convergence
tolerance. ``find_map`` first evaluates a batch of random points (the
pre-scan) and starts from the best of them, then runs SciPy's L-BFGS-B
optimiser. The optimiser needs the gradient of the posterior; it is estimated
by finite differences, with all 16 posterior evaluations in one batched GPU
call. ``initialise_walkers`` then places emcee's walkers in a small ball
around the result.
"""

import time

import numpy as np
from scipy.optimize import minimize

from gpu_config import BOUNDS_ARRAY, N_DIM


def prior_prescan(log_posterior, n_samples, rng, batch_size=4096,
                  centre=None, width=None):
    """Evaluate random points and return the best one found.

    With ``centre``/``width`` the draw is a Gaussian ball around a first guess
    (clipped to the prior); otherwise it is uniform over the whole prior box.
    """
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    best_value = -np.inf
    best_point = None
    remaining = int(n_samples)

    while remaining > 0:
        count = min(batch_size, remaining)
        remaining -= count
        if centre is None:
            points = rng.uniform(lower, upper, size=(count, N_DIM))
        else:
            scale = width * (upper - lower)
            points = centre + scale * rng.standard_normal((count, N_DIM))
        # Strict inequalities in the prior: keep samples just inside the box.
        points = np.clip(points, lower + 1e-9 * (upper - lower),
                         upper - 1e-9 * (upper - lower))
        values = np.asarray(log_posterior(points))
        index = int(np.nanargmax(values))
        if values[index] > best_value:
            best_value = float(values[index])
            best_point = points[index].copy()

    return best_point, best_value


def _batched_negative_and_gradient(log_posterior, x, step_fraction):
    """(-logpost, gradient) using one batched forward-difference call."""
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    span = upper - lower
    steps = step_fraction * span

    points = np.repeat(x[None, :], N_DIM + 1, axis=0)
    signs = np.ones(N_DIM)
    for i in range(N_DIM):
        # Step inwards when the forward step would leave the prior box.
        if x[i] + steps[i] >= upper[i]:
            signs[i] = -1.0
        points[i + 1, i] = x[i] + signs[i] * steps[i]
    points = np.clip(points, lower + 1e-12 * span, upper - 1e-12 * span)

    values = np.asarray(log_posterior(points), dtype=np.float64)
    base = values[0]
    if not np.isfinite(base):
        return np.inf, np.zeros(N_DIM)

    gradient = np.zeros(N_DIM)
    finite = np.isfinite(values[1:])
    # Row i of points[1:] differs from x only in column i, so the step actually
    # taken (clipping may have shortened it) is the diagonal of that block.
    perturbed = points[1:]
    actual_steps = perturbed[np.arange(N_DIM), np.arange(N_DIM)] - x
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient[finite] = -(values[1:][finite] - base) / actual_steps[finite]
    return -base, gradient


def find_map(log_posterior, first_guess, step_fraction=1e-5, maxiter=2000,
             ftol=1e-3, prescan=0, prescan_width=None, rng=None, verbose=True):
    """Maximise the log-posterior from ``first_guess``.

    prescan        : number of random points evaluated first; the best one
                     replaces ``first_guess`` if it is better. 0 skips it.
    prescan_width  : if given, the pre-scan draws a Gaussian ball of this
                     fraction of the prior range around ``first_guess``;
                     otherwise it draws uniformly over the whole prior box.
    step_fraction  : finite-difference step, as a fraction of each
                     parameter's prior range.
    ftol           : L-BFGS-B stops once an iteration improves -log-posterior
                     by less than this fraction of its value.

    Returns ``(map_parameters, info)``.
    """
    rng = rng or np.random.default_rng(0)
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    start = np.clip(np.asarray(first_guess, dtype=np.float64),
                    lower + 1e-9 * (upper - lower),
                    upper - 1e-9 * (upper - lower))

    info = {"prescan_samples": int(prescan)}
    if prescan > 0:
        centre = start if prescan_width else None
        point, value = prior_prescan(
            log_posterior, prescan, rng, centre=centre, width=prescan_width)
        info["prescan_best_log_posterior"] = value
        start_value = float(np.asarray(log_posterior(start[None, :]))[0])
        info["first_guess_log_posterior"] = start_value
        if verbose:
            print(f"[MAP] pre-scan best log-posterior {value:.3f} "
                  f"(first guess {start_value:.3f})")
        if point is not None and value > start_value:
            start = point

    evaluations = {"count": 0}

    def objective(x):
        evaluations["count"] += 1
        value, gradient = _batched_negative_and_gradient(
            log_posterior, x, step_fraction)
        if verbose and evaluations["count"] % 25 == 0:
            print(f"[MAP] L-BFGS-B call {evaluations['count']}: "
                  f"-log_posterior = {value:.4f}")
        return value, gradient

    result = minimize(
        objective, start, jac=True, method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": maxiter, "ftol": ftol},
    )

    parameters = np.clip(result.x, lower + 1e-9 * (upper - lower),
                         upper - 1e-9 * (upper - lower))
    final = float(np.asarray(log_posterior(parameters[None, :]))[0])
    if not np.isfinite(final):
        if verbose:
            print("[MAP] optimiser left the prior support; keeping the start point.")
        parameters = start
        final = float(np.asarray(log_posterior(parameters[None, :]))[0])

    info.update({
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective_calls": evaluations["count"],
        "log_posterior": final,
    })
    if verbose:
        print(f"[MAP] {result.message} after {result.nit} iterations "
              f"({evaluations['count']} batched gradient calls)")
        print(f"[MAP] log_posterior = {final:.4f}")
    return parameters, info


def initialise_walkers(log_posterior, centre, n_walkers, rng,
                       scale_fraction=0.01, max_attempts=200, verbose=True):
    """Gaussian ball of walkers around ``centre``, all inside the prior.

    Candidate positions are drawn and tested in batches, so the rejection
    loop costs a handful of posterior evaluations rather than one per walker.
    A candidate outside the prior box is drawn again rather than moved onto
    the edge, so walkers never share a coordinate, which would stop emcee's
    moves from mixing in that parameter.
    """
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    span = upper - lower
    scale = scale_fraction * span

    accepted = []
    for _ in range(max_attempts):
        needed = n_walkers - sum(len(a) for a in accepted)
        if needed <= 0:
            break
        draw = centre + scale * rng.standard_normal((max(needed * 2, 64), N_DIM))
        draw = draw[np.all((draw > lower) & (draw < upper), axis=1)]
        if not draw.size:
            continue
        values = np.asarray(log_posterior(draw))
        good = draw[np.isfinite(values)]
        if good.size:
            accepted.append(good[:needed])
    positions = np.concatenate(accepted, axis=0) if accepted else np.empty((0, N_DIM))

    if positions.shape[0] < n_walkers:
        raise RuntimeError(
            f"only {positions.shape[0]}/{n_walkers} walkers landed inside the "
            f"prior support around the starting point; widen --walker-scale or "
            f"check the MAP estimate.")
    if verbose:
        print(f"[init] {n_walkers} walkers initialised "
              f"(ball of {scale_fraction:.1%} of the prior range)")
    return positions[:n_walkers]


def multi_start_map(log_posterior, rng, n_starts=10, step_fraction=1e-5,
                    prescan=20000, prescan_width=None, polish=True,
                    polish_prescan=2000, polish_width=0.02, polish_ftol=1e-8):
    """Best of several independent MAP searches, refined at the end.

    A single L-BFGS-B search can settle on a local optimum, so ``n_starts``
    searches are run. Each begins from a random point inside the prior box,
    but its ``prescan`` random draws over the whole box (see ``find_map``)
    almost always contain a better point, which it starts from instead. The
    run with the highest log-posterior is kept.

    These runs use a loose convergence tolerance, which is enough to tell a
    good optimum from a poor one but stops short of the maximum. With
    ``polish`` the best run is therefore optimised once more, with a small
    pre-scan around it (``polish_prescan`` points, ``polish_width`` of the
    prior range) and the strict tolerance ``polish_ftol``; the polished point
    is kept if it is at least as good.

    Returns ``(map_parameters, info)``; ``info`` records every run.
    """
    started = time.perf_counter()
    lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
    margin = 1e-9 * (upper - lower)

    # All starts are random and strictly inside the box bounds.
    starts = rng.uniform(low=lower + margin, high=upper - margin,
                         size=(n_starts, N_DIM))

    best_parameters = None
    best_log_posterior = -np.inf
    best_run = None
    all_runs = []

    for run_index, random_start in enumerate(starts, start=1):
        print(f"\n[MAP] multi-start run {run_index}/{n_starts}")

        candidate_parameters, candidate_info = find_map(
            log_posterior, random_start,
            step_fraction=step_fraction,
            prescan=prescan,
            prescan_width=prescan_width,
            rng=rng)

        candidate_log_posterior = candidate_info["log_posterior"]

        all_runs.append({"run": run_index, "start": random_start.copy(),
                         "parameters": candidate_parameters.copy(),
                         **candidate_info})

        print(f"[MAP] run {run_index}/{n_starts}: "
              f"log_posterior = {candidate_log_posterior:.4f}")

        # Also accept the first result, even if it is -inf, so that
        # best_parameters is always defined before the validity check.
        if (best_parameters is None or
                candidate_log_posterior > best_log_posterior):
            best_parameters = candidate_parameters
            best_log_posterior = candidate_log_posterior
            best_run = run_index

    if not np.isfinite(best_log_posterior):
        raise RuntimeError(
            "All MAP optimisation runs ended outside finite posterior "
            "support. Increase --map-prescan, revise the bounds/prior, "
            "or provide a more appropriate initialisation strategy."
        )

    info = {"n_starts": n_starts, "best_run": best_run,
            "log_posterior": best_log_posterior, "runs": all_runs}

    if polish:
        print(f"\n[MAP] refining run {best_run} with ftol = {polish_ftol:g}")
        polished, polish_info = find_map(
            log_posterior, best_parameters,
            step_fraction=step_fraction,
            prescan=polish_prescan,
            prescan_width=polish_width,
            ftol=polish_ftol,
            rng=rng)
        info["polish"] = polish_info
        if polish_info["log_posterior"] >= best_log_posterior:
            print(f"[MAP] refined log_posterior = "
                  f"{polish_info['log_posterior']:.4f} (was "
                  f"{best_log_posterior:.4f})")
            best_parameters = polished
            best_log_posterior = polish_info["log_posterior"]
            info["log_posterior"] = best_log_posterior
        else:
            print("[MAP] refinement did not improve; keeping the best run")

    print(f"\n[MAP] selected run {best_run}/{n_starts}, "
          f"log_posterior = {best_log_posterior:.4f}")
    print(f"[MAP] total multi-start optimisation took "
          f"{time.perf_counter() - started:.1f} s")
    return best_parameters, info

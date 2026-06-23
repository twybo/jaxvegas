"""A from-scratch JAX implementation of Lepage's VEGAS+ integrator.

VEGAS+ combines two adaptive mechanisms (Lepage, arXiv:2009.05112):

1. An adaptive *importance-sampling map*: a separable per-axis grid that
   concentrates integration points where ``|f|`` is large.
2. Adaptive *stratified sampling*: a beta-damped reallocation of integrand
   evaluations across a grid of hypercubes.

The integrand must be a pure, batched JAX callable ``f(x) -> y`` mapping an
array of shape ``(n, D)`` to an array of shape ``(n,)``.  It may itself be the
output of ``jax.grad`` / ``jax.jit`` / ``jax.vmap``.  The per-iteration kernel
runs under ``jax.jit``.

The public entry point is :func:`integrate`.
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Mirror the package's tiny-number guards.
TINY = 1e-30
EPSILON = 1e-8

__all__ = ["VegasResult", "integrate"]


class VegasResult(NamedTuple):
    """Result of an :func:`integrate` call.

    Attributes:
        mean: inverse-variance weighted estimate of the integral.
        sdev: standard deviation of ``mean``.
        chi2: chi-squared of the weighted average across iterations.
        dof: degrees of freedom (``nitn - 1``).
        itn_results: list of ``(mean, sdev)`` tuples, one per accumulated
            iteration.
    """

    mean: float
    sdev: float
    chi2: float
    dof: int
    itn_results: list


# --------------------------------------------------------------------------- #
# Adaptive importance-sampling map
# --------------------------------------------------------------------------- #
def init_grid(domain, ninc: int) -> jnp.ndarray:
    """Build a uniform grid ``(D, ninc + 1)`` spanning each ``[a, b]``.

    Args:
        domain: array / list of ``(a, b)`` pairs, shape ``(D, 2)``.
        ninc: number of increments per axis.

    Returns:
        ``grid`` of shape ``(D, ninc + 1)``; ``grid[d]`` is the increasing
        sequence of node positions along axis ``d``.
    """
    domain = jnp.asarray(domain, dtype=jnp.float64)
    a = domain[:, 0]
    b = domain[:, 1]
    t = jnp.linspace(0.0, 1.0, ninc + 1)  # (ninc+1,)
    grid = a[:, None] + (b - a)[:, None] * t[None, :]  # (D, ninc+1)
    return grid


def map_y_to_x(grid: jnp.ndarray, y: jnp.ndarray):
    """Map ``y in [0, 1)^D`` to ``x`` via the separable grid, with Jacobian.

    Args:
        grid: shape ``(D, ninc + 1)``.
        y: shape ``(n, D)``.

    Returns:
        ``(x, jac)`` with ``x`` of shape ``(n, D)`` and ``jac`` of shape
        ``(n,)`` equal to ``prod_d dx_d / dy_d``.
    """
    ninc = grid.shape[1] - 1
    inc = jnp.diff(grid, axis=1)  # (D, ninc)

    pos = y * ninc  # (n, D)
    i = jnp.floor(pos).astype(jnp.int32)
    i = jnp.clip(i, 0, ninc - 1)  # guard y == 1.0
    frac = pos - i

    # Gather per-point, per-dim grid node and increment.
    #   grid is (D, ninc+1), index i along axis 1 -> need transpose to (n, D).
    gridT = grid.T  # (ninc+1, D)
    incT = inc.T  # (ninc, D)
    g_lo = jnp.take_along_axis(gridT, i, axis=0)  # (n, D)
    inc_i = jnp.take_along_axis(incT, i, axis=0)  # (n, D)

    x = g_lo + frac * inc_i
    jac = jnp.prod(inc_i * ninc, axis=1)  # (n,)
    return x, jac


@functools.partial(jax.jit, static_argnums=(2,))
def smooth_and_adapt(grid: jnp.ndarray, train: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Adapt the grid to accumulated training data (vectorized over axes).

    Implements the paper's recipe: 3-point smoothing, normalization,
    alpha-compression, and redistribution of boundaries so each new interval
    holds equal cumulative training weight.

    Args:
        grid: current grid, shape ``(D, ninc + 1)``.
        train: per-interval accumulated ``(J f)^2``, shape ``(D, ninc)``.
        alpha: adaptation rate (``0`` leaves the grid unchanged).

    Returns:
        the new grid, shape ``(D, ninc + 1)``.
    """
    D, ninc = train.shape
    if alpha <= 0 or ninc <= 1:
        return grid

    d = train  # (D, ninc), == avg_f in the package (n_f is uniform here)

    # 3-point smoothing with edge weighting (matches AdaptiveMap.adapt).
    left = jnp.concatenate([d[:, :1], d[:, :-1]], axis=1)
    right = jnp.concatenate([d[:, 1:], d[:, -1:]], axis=1)
    sm = (6.0 * d + left + right) / 8.0
    # Edges use a 7:1 weighting rather than 6:1:1.
    sm = sm.at[:, 0].set((7.0 * d[:, 0] + d[:, 1]) / 8.0)
    sm = sm.at[:, -1].set((7.0 * d[:, -1] + d[:, -2]) / 8.0)
    sm = jnp.abs(sm)

    # Normalize per axis.
    ssum = jnp.sum(sm, axis=1, keepdims=True)
    sm = jnp.where(ssum > 0, sm / ssum + TINY, TINY)

    # alpha-compression: d <- ((1 - d) / ln(1/d))^alpha.
    safe = (sm > 0) & (sm <= 0.99999999)
    comp = jnp.power(
        -(1.0 - sm) / jnp.log(jnp.where(safe, sm, 0.5)), alpha
    )
    sm = jnp.where(safe, comp, sm)

    # Redistribute boundaries: place new nodes so cumulative `sm` is equal per
    # new interval.  cumsum gives the cumulative weight at each old boundary;
    # interpolate old node positions against equally-spaced cumulative targets.
    cum = jnp.cumsum(sm, axis=1)  # (D, ninc), cumulative at old right-edges
    cum0 = jnp.concatenate([jnp.zeros((D, 1)), cum], axis=1)  # (D, ninc+1)
    total = cum0[:, -1:]  # (D, 1)
    targets = jnp.linspace(0.0, 1.0, ninc + 1)[None, :]  # (1, ninc+1)
    targets = targets * total  # (D, ninc+1)

    # jnp.interp is 1-D; vmap over axes.
    def interp_axis(cum0_d, grid_d, targets_d):
        return jnp.interp(targets_d, cum0_d, grid_d)

    new_grid = jax.vmap(interp_axis)(cum0, grid, targets)
    # Pin endpoints exactly.
    new_grid = new_grid.at[:, 0].set(grid[:, 0])
    new_grid = new_grid.at[:, -1].set(grid[:, -1])
    return new_grid


# --------------------------------------------------------------------------- #
# Stratification bookkeeping
# --------------------------------------------------------------------------- #
def choose_nstrat(neval: int, D: int, neval_frac: float):
    """Choose ``nstrat`` strata per axis and return ``(nstrat, Nh)``.

    ``nstrat = max(1, floor((neval * neval_frac / 2)^(1/D)))`` clamped so that
    ``2 * nstrat^D <= neval`` (so the ``n_min = 2`` floor per hypercube fits).
    """
    target = (neval * neval_frac / 2.0) ** (1.0 / D)
    nstrat = max(1, int(np.floor(target)))
    # Clamp so 2 * nstrat^D <= neval.
    while nstrat > 1 and 2 * nstrat**D > neval:
        nstrat -= 1
    Nh = nstrat**D
    return nstrat, Nh


def hcube_corners(nstrat: int, D: int) -> np.ndarray:
    """Multi-index corner (in stratum units) for each of the ``nstrat^D`` hcubes.

    Returns an array of shape ``(Nh, D)`` of integer corner coordinates so that
    hypercube ``h`` spans ``[corner/nstrat, (corner + 1)/nstrat]`` per axis.
    """
    Nh = nstrat**D
    idx = np.arange(Nh)
    corners = np.stack(np.unravel_index(idx, (nstrat,) * D), axis=1)
    return corners.astype(np.int32)


def allocate(sigf: np.ndarray, neval: int, neval_frac: float, Nh: int):
    """Host-side allocation of evaluations across hypercubes.

    Produces an integer per-hcube count summing *exactly* to ``neval`` (so the
    jitted kernel sees a fixed total and compiles once), with a floor of
    ``n_min = 2`` per hcube.

    Args:
        sigf: per-hcube ``sigma_h`` weights, shape ``(Nh,)``.
        neval: total target evaluations.
        neval_frac: fraction of evaluations distributed by ``sigf``; the rest
            go to the uniform ``n_min`` floor.
        Nh: number of hypercubes.

    Returns:
        ``(neval_hcube, point_hcube_idx)`` where ``neval_hcube`` has shape
        ``(Nh,)`` and ``point_hcube_idx`` has shape ``(sum(neval_hcube),)``
        mapping each point to its hypercube.
    """
    n_min = 2
    sigf = np.asarray(sigf, dtype=np.float64)
    ssum = sigf.sum()
    if not np.isfinite(ssum) or ssum <= 0:
        sigf = np.ones(Nh)
        ssum = float(Nh)

    # Points to distribute beyond the n_min floor.  choose_nstrat guarantees
    # 2 * Nh <= neval, so this is non-negative for the automatic nstrat; clamp
    # defensively for a user-supplied nstrat.
    budget = max(neval - n_min * Nh, 0)

    # Real-valued target per hcube: floor + an adaptive (neval_frac, proportional
    # to sigf) part + a uniform (1 - neval_frac) part.  Summing the targets gives
    # exactly neval, so floor(target) leaves a leftover strictly less than Nh
    # which the fractional-part pass below distributes exactly -> sum == neval
    # every iteration, hence a fixed kernel shape (compile once).
    target = (
        n_min
        + neval_frac * budget * (sigf / ssum)
        + (1.0 - neval_frac) * budget / Nh
    )
    neval_hcube = np.floor(target).astype(np.int64)

    leftover = neval - int(neval_hcube.sum())  # in [0, Nh) by construction
    if leftover > 0:
        order = np.argsort(-(target - np.floor(target)))
        neval_hcube[order[:leftover]] += 1

    point_hcube_idx = np.repeat(np.arange(Nh, dtype=np.int32), neval_hcube)
    return neval_hcube.astype(np.int64), point_hcube_idx.astype(np.int32)


# --------------------------------------------------------------------------- #
# Jitted per-iteration kernel
# --------------------------------------------------------------------------- #
@functools.partial(jax.jit, static_argnums=(0, 6, 7))
def run_iteration(f, key, grid, point_hcube_idx, hcube_corner, neval_hcube, Nh, nstrat):
    """One VEGAS+ iteration (jitted; ``f`` and ``Nh`` are static).

    Args:
        f: pure batched JAX integrand ``(n, D) -> (n,)``.
        key: PRNG key.
        grid: current map grid, shape ``(D, ninc + 1)``.
        point_hcube_idx: per-point hypercube index, shape ``(neval,)``.
        hcube_corner: per-hcube integer corner, shape ``(Nh, D)``.
        neval_hcube: per-hcube point count, shape ``(Nh,)``.
        Nh: number of hypercubes (static).
        nstrat: strata per axis (traced scalar is fine).

    Returns:
        ``(I, var, sigf, train)``:
          - ``I``: scalar integral estimate for this iteration.
          - ``var``: scalar variance estimate.
          - ``sigf``: per-hcube ``sigma_h`` (before the beta exponent is
            applied; see :func:`integrate`), shape ``(Nh,)``.  Actually returns
            ``Omega * sqrt(s2_h)`` so the caller raises to ``beta``.
          - ``train``: per-axis training data ``(D, ninc)``.
    """
    ninc = grid.shape[1] - 1
    D = grid.shape[0]
    neval = point_hcube_idx.shape[0]
    Omega = 1.0 / Nh

    # 1) Stratified uniform draws: y = (corner + u) / nstrat.
    u = jax.random.uniform(key, shape=(neval, D), dtype=jnp.float64)
    corner = hcube_corner[point_hcube_idx]  # (neval, D)
    y = (corner + u) / nstrat

    # 2) Map to x, evaluate integrand.
    x, jac = map_y_to_x(grid, y)
    fx = f(x)
    jf = jac * fx  # (neval,)

    # 3) Per-point weight w = Omega * J f / n_h.
    nh_pt = neval_hcube[point_hcube_idx].astype(jnp.float64)  # (neval,)
    w = Omega * jf / nh_pt

    # 4) Per-hcube reductions via segment_sum.
    seg = point_hcube_idx
    sum_jf = jax.ops.segment_sum(jf, seg, num_segments=Nh)
    sum_jf2 = jax.ops.segment_sum(jf * jf, seg, num_segments=Nh)
    nh = neval_hcube.astype(jnp.float64)  # (Nh,)

    mean_jf = sum_jf / nh
    # Bessel-corrected per-hcube sample variance of (J f).
    s2 = (sum_jf2 - nh * mean_jf**2) / jnp.maximum(nh - 1.0, 1.0)
    s2 = jnp.abs(s2)

    # Iteration integral and variance.
    I = jnp.sum(w)
    var = jnp.sum(Omega**2 * s2 / nh)

    # sigma_h = Omega * sqrt(s2); caller applies the beta exponent.
    sigf = Omega * jnp.sqrt(s2)

    # 5) Training data for the map: scatter |jf|^2 into per-axis intervals.
    jf2 = jf * jf
    iy = jnp.clip(jnp.floor(y * ninc).astype(jnp.int32), 0, ninc - 1)  # (neval, D)

    def per_axis(i_d):
        return jnp.zeros(ninc, dtype=jnp.float64).at[i_d].add(jf2)

    train = jax.vmap(per_axis, in_axes=1)(iy)  # (D, ninc)

    return I, var, sigf, train


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def integrate(
    f: Callable,
    domain,
    *,
    neval: int = 10000,
    nitn: int = 10,
    alpha: float = 0.5,
    beta: float = 0.75,
    neval_frac: float = 0.75,
    ninc: int = 1000,
    nstrat: int | None = None,
    seed: int = 0,
    nitn_warmup: int = 10,
) -> VegasResult:
    """Integrate a pure batched JAX function with VEGAS+.

    Args:
        f: integrand ``f(x) -> y`` mapping ``(n, D)`` to ``(n,)``.  Must be a
            pure JAX callable (it is traced under ``jit``).
        domain: ``(D, 2)`` array / list of ``(a, b)`` integration bounds.
        neval: target integrand evaluations per iteration.
        nitn: number of accumulated iterations.
        alpha: grid adaptation rate.
        beta: stratification damping exponent.
        neval_frac: fraction of evaluations distributed adaptively.
        ninc: grid increments per axis.
        nstrat: strata per axis; if ``None``, chosen automatically.
        seed: PRNG seed.
        nitn_warmup: number of discarded warmup iterations used to adapt the
            grid and stratification before accumulating results.

    Returns:
        a :class:`VegasResult`.
    """
    domain = jnp.asarray(domain, dtype=jnp.float64)
    if domain.ndim == 1:
        domain = domain[None, :]
    D = domain.shape[0]

    # ninc must divide cleanly into the interp; keep it as given.
    grid = init_grid(domain, ninc)

    if nstrat is None:
        nstrat, Nh = choose_nstrat(neval, D, neval_frac)
    else:
        Nh = nstrat**D
    hcube_corner = jnp.asarray(hcube_corners(nstrat, D))

    key = jax.random.PRNGKey(seed)

    # Uniform sigf to start.
    sigf_host = np.ones(Nh)

    # Mutable state carried across iterations (closure-friendly containers).
    grid_state = [grid]

    def step(key):
        neval_hcube_np, point_hcube_idx_np = allocate(sigf_state[0], neval, neval_frac, Nh)
        neval_hcube = jnp.asarray(neval_hcube_np)
        point_hcube_idx = jnp.asarray(point_hcube_idx_np)
        I, var, sigf, train = run_iteration(
            f, key, grid_state[0], point_hcube_idx, hcube_corner,
            neval_hcube, Nh, nstrat,
        )
        # Update map and stratification weights for next iteration.
        grid_state[0] = smooth_and_adapt(grid_state[0], train, alpha)
        if beta > 0:
            sigf_b = np.asarray(jnp.power(sigf, beta))
            if not np.all(np.isfinite(sigf_b)) or sigf_b.sum() <= 0:
                sigf_b = np.ones(Nh)
            sigf_state[0] = sigf_b
        return float(I), float(var)

    sigf_state = [sigf_host]

    # Warmup: adapt only, discard results.
    for _ in range(nitn_warmup):
        key, sub = jax.random.split(key)
        step(sub)

    # Accumulate.
    means, vars, itn_results = [], [], []
    for _ in range(nitn):
        key, sub = jax.random.split(key)
        I, var = step(sub)
        means.append(I)
        vars.append(max(var, TINY))
        itn_results.append((I, float(np.sqrt(max(var, TINY)))))

    means = np.asarray(means)
    vars = np.asarray(vars)

    # Inverse-variance weighted average.
    w = 1.0 / np.maximum(vars, TINY)
    wsum = w.sum()
    mean = float((w * means).sum() / wsum)
    var_mean = float(1.0 / wsum)
    sdev = float(np.sqrt(var_mean))
    chi2 = float((w * (means - mean) ** 2).sum())
    dof = max(nitn - 1, 0)

    return VegasResult(mean=mean, sdev=sdev, chi2=chi2, dof=dof, itn_results=itn_results)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests (no pytest required, but pytest-compatible)
uv run python test_jaxvegas.py

# Run a single test
uv run python -c "from test_jaxvegas import test_constant; test_constant(); print('OK')"

# Run the benchmark/comparison against vegas and torchquad
uv run python compare.py
```

## Architecture

This is a from-scratch JAX implementation of Lepage's **VEGAS+** adaptive Monte Carlo integrator (arXiv:2009.05112). The algorithm combines two adaptive mechanisms:

1. **Adaptive importance-sampling map** — a separable per-axis grid that concentrates samples where `|f|` is large.
2. **Adaptive stratified sampling** — beta-damped reallocation of evaluations across a grid of hypercubes.

### Package layout

All logic lives in a single file: `jaxvegas/integrator.py`.

The public API is just two exports: `integrate(f, domain, ...)` and `VegasResult`.

#### Call flow inside `integrate`

```
integrate()
  └── init_grid()                  # uniform grid over the domain
  └── choose_nstrat() / hcube_corners()   # host-side stratification setup
  └── per iteration:
        allocate()                 # host-side: fixed neval per hcube (sum = neval)
        run_iteration()            # jit-compiled kernel
          └── map_y_to_x()        # stratified y -> physical x + Jacobian
          └── f(x)                # user integrand
          └── segment_sum()       # per-hcube variance / training data
        smooth_and_adapt()        # update grid (jit-compiled)
```

#### Key design constraints

- `jax.config.update("jax_enable_x64", True)` is applied at import time — all arrays are `float64`.
- `run_iteration` is `jax.jit`-compiled with `f` and `Nh` as **static** args. The total point count `neval` must be fixed between iterations so JAX compiles once. This is guaranteed by `allocate()` running on the host and always summing exactly to `neval`.
- `allocate()` is intentionally *not* jitted: it uses `np.floor` / integer arithmetic to produce a fixed-shape `point_hcube_idx` array, which keeps the jitted kernel shape-stable across iterations.
- The `vegas/` directory is a symlink to `../vegas` (the reference `vegas` Python package), used only by `compare.py` for benchmarking.

#### Integrand contract

The integrand `f` must be a pure, batched JAX callable: `f(x: (n, D)) -> (n,)`. It may be the output of `jax.grad` / `jax.jit` / `jax.vmap` — the kernel traces through it.

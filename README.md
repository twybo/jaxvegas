# jaxvegas

A JAX implementation of Lepage's **VEGAS+** adaptive Monte Carlo integrator
([arXiv:2009.05112](https://arxiv.org/abs/2009.05112)).

VEGAS+ combines adaptive importance sampling (a per-axis grid that concentrates
points where `|f|` is large) with adaptive stratified sampling (beta-damped
reallocation of evaluations across hypercubes). The per-iteration kernel runs
under `jax.jit`, so the integrand may itself be built from `jax.grad`,
`jax.jit`, or `jax.vmap`.

This was made by prompting Claude Opus 4.8 with the (vegas)[https://github.com/gplepage/vegas] repo and the arXiv TEX source.

## Installation

From source:

```bash
git clone https://github.com/twybo/jaxvegas
pip install ./jaxvegas
```

To also install the dependencies needed by `compare.py` (torch, torchquad,
matplotlib, vegas):

```bash
pip install "jaxvegas[compare]"
```

## Quick start

```python
import jax.numpy as jnp
import jaxvegas

# Integrand: pure, batched JAX callable  f(x: (n, D)) -> (n,)
def f(x):
    return jnp.prod(jnp.exp(x), axis=1)   # exact: (e-1)^D

result = jaxvegas.integrate(f, domain=[[0, 1], [0, 1]])

print(result.mean, "±", result.sdev)   # → 2.952 ± 0.001
print("chi2/dof =", result.chi2 / result.dof)
```

## Integrating your own function

Your integrand must satisfy one contract: **pure, batched JAX callable**
`f(x) -> y` where `x` has shape `(n, D)` and `y` has shape `(n,)`.

```python
# my_integrand.py
import jax.numpy as jnp

def sharp_peak(x):
    """Gaussian peak exp(-200 * sum (x - 0.5)^2) over [0,1]^D."""
    return jnp.exp(-200.0 * jnp.sum((x - 0.5) ** 2, axis=1))
```

```python
# run.py
import jaxvegas
from my_integrand import sharp_peak

result = jaxvegas.integrate(
    sharp_peak,
    domain=[[0, 1], [0, 1], [0, 1]],  # D=3, any (a,b) bounds work
    neval=20_000,   # evaluations per iteration
    nitn=20,        # accumulated iterations
    nitn_warmup=10, # discarded warmup iterations (grid adapts during these)
    seed=42,
)
print(result)
```

Integrands built from `jax.grad` or `jax.jit` work directly:

```python
import jax
import jaxvegas

g = jax.vmap(jax.grad(lambda t: t ** 3))   # 3t^2, integral over [0,1] = 1
result = jaxvegas.integrate(lambda x: g(x[:, 0]), domain=[[0, 1]])
```

## `integrate` parameters

| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| `f`           | —       | Batched JAX integrand `(n, D) → (n,)`                        |
| `domain`      | —       | `(D, 2)` list/array of `[a, b]` bounds per axis              |
| `neval`       | `10000` | Target evaluations per iteration                             |
| `nitn`        | `10`    | Number of accumulated iterations                             |
| `nitn_warmup` | `10`    | Discarded warmup iterations (grid adapts, results discarded) |
| `alpha`       | `0.5`   | Grid adaptation rate (0 = no adaptation)                     |
| `beta`        | `0.75`  | Stratification damping exponent                              |
| `ninc`        | `1000`  | Grid increments per axis                                     |
| `nstrat`      | `None`  | Strata per axis; chosen automatically if `None`              |
| `seed`        | `0`     | PRNG seed                                                    |

## `VegasResult` fields

| Field         | Description                                           |
| ------------- | ----------------------------------------------------- |
| `mean`        | Inverse-variance weighted integral estimate           |
| `sdev`        | Standard deviation of `mean`                          |
| `chi2`        | Chi-squared of the weighted average across iterations |
| `dof`         | Degrees of freedom (`nitn - 1`)                       |
| `itn_results` | List of `(mean, sdev)` per accumulated iteration      |

A `chi2/dof` near 1 indicates consistent estimates across iterations.
Values significantly above 1 suggest the integrand is still varying between
iterations and more warmup or a larger `neval` may help.

## Benchmarks

`compare.py` benchmarks jaxvegas against the reference `vegas` package and
torchquad across several integrands. Run with:

```bash
pip install "jaxvegas[compare]"
python compare.py   # writes comparison.pdf
```

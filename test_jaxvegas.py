"""Sanity tests for jaxvegas.

Run with ``uv run python test_jaxvegas.py`` (no pytest required, but the
functions are pytest-compatible).
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxvegas


def test_constant():
    """Integral of f = 1 over [0,1]^D must be 1 (Jacobian/volume bookkeeping)."""
    for D in (1, 2, 3):
        r = jaxvegas.integrate(lambda x: jnp.ones(x.shape[0]), [[0, 1]] * D,
                               neval=4000, nitn=8, seed=1)
        assert abs(r.mean - 1.0) < 1e-2, (D, r.mean)


def test_prod_exp_calibrated():
    """prod_d exp(x_d) over [0,1]^D == (e-1)^D; check pull and chi2/dof."""
    for D in (1, 2, 4):
        exact = (np.e - 1) ** D
        # Average pull over several seeds should be small and well-calibrated.
        pulls = []
        for seed in range(5):
            r = jaxvegas.integrate(lambda x: jnp.prod(jnp.exp(x), axis=1),
                                   [[0, 1]] * D, neval=8000, nitn=10, seed=seed)
            pulls.append((r.mean - exact) / r.sdev)
            assert abs(r.mean - exact) < 3 * r.sdev, (D, seed, r.mean, r.sdev)
            assert r.chi2 / r.dof < 3.0, (D, seed, r.chi2 / r.dof)
        assert abs(np.mean(pulls)) < 2.0, (D, np.mean(pulls))


def test_pure_jax_grad_integrand():
    """Integrand produced by jax.grad runs under jit without tracer leaks."""
    g = jax.vmap(jax.grad(lambda t: jnp.exp(t)))  # d/dt exp = exp
    f = lambda x: g(x[:, 0])
    r = jaxvegas.integrate(f, [[0, 1]], neval=4000, nitn=8, seed=3)
    assert abs(r.mean - (np.e - 1)) < 3 * r.sdev + 1e-3


def test_pure_jax_jit_integrand():
    """Integrand produced by jax.jit runs cleanly."""
    f = jax.jit(lambda x: jnp.prod(jnp.exp(x), axis=1))
    r = jaxvegas.integrate(f, [[0, 1]] * 2, neval=4000, nitn=8, seed=4)
    assert abs(r.mean - (np.e - 1) ** 2) < 3 * r.sdev + 1e-2


if __name__ == "__main__":
    test_constant()
    print("test_constant: OK")
    test_prod_exp_calibrated()
    print("test_prod_exp_calibrated: OK")
    test_pure_jax_grad_integrand()
    print("test_pure_jax_grad_integrand: OK")
    test_pure_jax_jit_integrand()
    print("test_pure_jax_jit_integrand: OK")
    print("\nAll tests passed.")

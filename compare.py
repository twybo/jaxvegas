"""Validation / benchmark: jaxvegas vs. the vegas package vs. torchquad VEGAS.

Four integrands, each defined once as a pure batched JAX function and wrapped
for the NumPy (vegas) and torch (torchquad) backends:

    1. Sharp Gaussian peak       exp(-a * sum (x-0.5)^2)
    2. Oscillatory diagonal ridge cos(k * sum x)^2
    3. Closed-form analytic       prod exp(x_d)            == (e-1)^D
    4. Gaussian peak in D = 4, 6, 8

For each we report mean, sdev, |error vs exact|, chi2/dof (where available) and
wall-time, print a table, and save a bar chart to ``comparison.png``.

A note on evaluation budgets: jaxvegas and the vegas package take ``neval`` per
iteration over ``nitn`` iterations (total ~ neval * nitn).  torchquad takes a
single total budget ``N`` that it grows internally across its own iterations.
We pass ``N = neval * nitn`` so the total number of integrand evaluations is
comparable, and we report torchquad's actual eval count where it exposes it.

Run with:  uv run python compare.py
"""

import math
import time

import jax.numpy as jnp
import numpy as np

import jaxvegas

# --------------------------------------------------------------------------- #
# Backends: vegas package + torchquad (quiet logging)
# --------------------------------------------------------------------------- #
import vegas

import torch
from torchquad import VEGAS, MonteCarlo, set_up_backend
from loguru import logger as _tq_logger

_tq_logger.remove()  # silence torchquad's per-iteration DEBUG/INFO spam
set_up_backend("torch", data_type="float64")


# --------------------------------------------------------------------------- #
# Integrand definitions  (jax / numpy / torch flavours + exact value)
# --------------------------------------------------------------------------- #
def gaussian_peak(a, D):
    """Sharp Gaussian peak exp(-a * sum (x-0.5)^2) over [0,1]^D."""
    # 1-D integral: ∫_0^1 exp(-a (x-0.5)^2) dx = sqrt(pi/a) * erf(sqrt(a)/2)
    one_d = math.sqrt(math.pi / a) * math.erf(math.sqrt(a) / 2.0)
    exact = one_d ** D

    def fjax(x):
        return jnp.exp(-a * jnp.sum((x - 0.5) ** 2, axis=1))

    def fnp(x):
        return np.exp(-a * np.sum((x - 0.5) ** 2, axis=1))

    def ftorch(x):
        return torch.exp(-a * torch.sum((x - 0.5) ** 2, dim=1))

    return fjax, fnp, ftorch, exact


def diagonal_ridge(k, D):
    """Oscillatory diagonal ridge cos(k * sum x)^2 over [0,1]^D."""
    # cos^2 = (1 + cos(2k s))/2 ; ∫ cos(2k sum x) = Re[((e^{2ik}-1)/(2ik))^D]
    z = (np.exp(2j * k) - 1.0) / (2j * k)
    exact = 0.5 + 0.5 * np.real(z ** D)

    def fjax(x):
        return jnp.cos(k * jnp.sum(x, axis=1)) ** 2

    def fnp(x):
        return np.cos(k * np.sum(x, axis=1)) ** 2

    def ftorch(x):
        return torch.cos(k * torch.sum(x, dim=1)) ** 2

    return fjax, fnp, ftorch, float(exact)


def prod_exp(D):
    """Closed-form analytic prod_d exp(x_d) over [0,1]^D == (e-1)^D."""
    exact = (math.e - 1.0) ** D

    def fjax(x):
        return jnp.prod(jnp.exp(x), axis=1)

    def fnp(x):
        return np.prod(np.exp(x), axis=1)

    def ftorch(x):
        return torch.prod(torch.exp(x), dim=1)

    return fjax, fnp, ftorch, exact


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_jaxvegas(fjax, D, neval, nitn, seed=0):
    domain = [[0.0, 1.0]] * D
    # Warm up jit so timing reflects steady state.
    _ = jaxvegas.integrate(fjax, domain, neval=neval, nitn=1, nitn_warmup=0, seed=seed)
    t0 = time.perf_counter()
    r = jaxvegas.integrate(fjax, domain, neval=neval, nitn=nitn, seed=seed)
    dt = time.perf_counter() - t0
    return dict(mean=r.mean, sdev=r.sdev, chi2dof=r.chi2 / max(r.dof, 1), time=dt)


def run_vegas(fnp, D, neval, nitn):
    domain = [[0.0, 1.0]] * D
    bf = vegas.batchintegrand(fnp)
    integ = vegas.Integrator(domain)
    t0 = time.perf_counter()
    integ(bf, nitn=nitn, neval=neval)        # adapt (discarded)
    r = integ(bf, nitn=nitn, neval=neval)    # accumulate
    dt = time.perf_counter() - t0
    return dict(mean=r.mean, sdev=r.sdev, chi2dof=r.chi2 / max(r.dof, 1), time=dt)


def run_torchquad(ftorch, D, N):
    domain = [[0.0, 1.0]] * D
    vq = VEGAS()
    t0 = time.perf_counter()
    val = vq.integrate(ftorch, dim=D, N=N, integration_domain=domain, seed=0)
    dt = time.perf_counter() - t0
    return dict(mean=float(val), sdev=float("nan"), chi2dof=float("nan"), time=dt)


def run_plain_mc(ftorch, D, N):
    """Plain (non-adaptive) Monte Carlo, for the 'beats MC' check."""
    domain = [[0.0, 1.0]] * D
    mc = MonteCarlo()
    val = mc.integrate(ftorch, dim=D, N=N, integration_domain=domain, seed=0)
    return float(val)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    neval, nitn = 10000, 10
    N = neval * nitn  # comparable total budget for torchquad

    cases = []
    cases.append(("Gaussian peak a=200 D=3", *gaussian_peak(200.0, 3), 3))
    cases.append(("Diagonal ridge k=8 D=3", *diagonal_ridge(8.0, 3), 3))
    cases.append(("prod exp(x) D=3 (analytic)", *prod_exp(3), 3))
    for D in (4, 6, 8):
        cases.append((f"Gaussian peak a=200 D={D}", *gaussian_peak(200.0, D), D))

    rows = []
    print("\nRunning comparisons (neval=%d, nitn=%d, torchquad N=%d)\n" % (neval, nitn, N))
    for name, fjax, fnp, ftorch, exact, D in cases:
        jv = run_jaxvegas(fjax, D, neval, nitn)
        vg = run_vegas(fnp, D, neval, nitn)
        tq = run_torchquad(ftorch, D, N)
        rows.append((name, exact, jv, vg, tq))

        def relerr(m):
            return abs(m - exact) / abs(exact) if exact != 0 else abs(m - exact)

        print(f"=== {name}   (exact = {exact:.8g}) ===")
        print(f"  {'method':<11} {'mean':>14} {'sdev':>11} {'relerr':>10} "
              f"{'chi2/dof':>9} {'time[s]':>8}")
        for label, r in (("jaxvegas", jv), ("vegas", vg), ("torchquad", tq)):
            print(f"  {label:<11} {r['mean']:>14.8g} {r['sdev']:>11.2e} "
                  f"{relerr(r['mean']):>10.2e} {r['chi2dof']:>9.2f} {r['time']:>8.3f}")
        print()

    # 'beats plain MC on the sharp peak' check.
    fjax, fnp, ftorch, exact = gaussian_peak(200.0, 3)
    mc_val = run_plain_mc(ftorch, 3, N)
    jv = run_jaxvegas(fjax, 3, neval, nitn)
    jv_err = abs(jv["mean"] - exact)
    mc_err = abs(mc_val - exact)
    print(f"Sharp-peak vs plain MC (same budget N={N}):")
    print(f"  jaxvegas |err| = {jv_err:.3e}   plain MC |err| = {mc_err:.3e}   "
          f"-> jaxvegas {'WINS' if jv_err < mc_err else 'loses'}\n")

    # ----------------------------------------------------------------------- #
    # Plot: relative error and wall-time per method.
    # ----------------------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0] for r in rows]
    methods = ["jaxvegas", "vegas", "torchquad"]
    colors = {"jaxvegas": "#1f77b4", "vegas": "#ff7f0e", "torchquad": "#2ca02c"}

    def relerr_of(exact, m):
        return abs(m - exact) / abs(exact) if exact != 0 else abs(m - exact)

    rel = {meth: [] for meth in methods}
    tim = {meth: [] for meth in methods}
    for name, exact, jv, vg, tq in rows:
        for meth, r in zip(methods, (jv, vg, tq)):
            rel[meth].append(max(relerr_of(exact, r["mean"]), 1e-12))
            tim[meth].append(r["time"])

    x = np.arange(len(names))
    w = 0.25
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    for i, meth in enumerate(methods):
        ax1.bar(x + (i - 1) * w, rel[meth], w, label=meth, color=colors[meth])
        ax2.bar(x + (i - 1) * w, tim[meth], w, label=meth, color=colors[meth])
    ax1.set_yscale("log")
    ax1.set_ylabel("relative error |mean - exact| / |exact|")
    ax1.set_title("VEGAS+ comparison: accuracy")
    ax1.legend()
    ax2.set_ylabel("wall time [s]")
    ax2.set_title("VEGAS+ comparison: timing")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig("comparison.pdf")
    print("Wrote comparison.pdf")


if __name__ == "__main__":
    main()

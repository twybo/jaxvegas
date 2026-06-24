"""jaxvegas: a JAX implementation of the VEGAS+ adaptive Monte Carlo integrator."""

from .integrator import VegasResult, integrate, report

__all__ = ["VegasResult", "integrate", "report"]
__version__ = "0.1.0"

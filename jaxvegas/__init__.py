"""jaxvegas: a JAX implementation of the VEGAS+ adaptive Monte Carlo integrator."""

from .integrator import VegasResult, integrate

__all__ = ["VegasResult", "integrate"]
__version__ = "0.1.0"

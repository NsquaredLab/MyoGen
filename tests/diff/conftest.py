"""Shared setup for the differentiable-PoC tests.

Skips the whole directory unless the ``diff`` extra (jax + jaxley) is installed, and enables
float64 before any JAX array is created (the cable-equation solve wants the precision).
"""

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jaxley")

jax.config.update("jax_enable_x64", True)

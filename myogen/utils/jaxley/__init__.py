"""
Utility functions for Jaxley simulations.

This module provides high-level convenience functions that mirror
the NEURON utilities for seamless workflow compatibility.
"""

from myogen.utils.jaxley.inject_currents import (
    inject_currents_and_simulate_spike_trains,
    inject_currents_into_populations,
    simulation_result_to_neo_block,
)

__all__ = [
    "inject_currents_and_simulate_spike_trains",
    "inject_currents_into_populations",
    "simulation_result_to_neo_block",
]

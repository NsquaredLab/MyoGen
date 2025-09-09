"""
Descending drive neuron populations for cortical input.

This module contains the population class for descending drive neurons that
simulate cortical input to spinal motor circuits via Poisson processes.
"""

from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype

from .base import _Pool


@beartowertype
class DescendingDrive__Pool(_Pool):
    """
    Container for a population of descending drive neurons.

    Manages a collection of DD (descending drive) cells that generate
    Poisson random processes for cortical input to spinal circuits.

    Parameters
    ----------
    n : int
        Number of descending drive neurons to create.
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    """

    def __init__(self, n: int, poisson_random_process_order: int, timestep__ms: float):
        self.n = n
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms

        _cells = []

        super().__init__(
            cells=[
                cells.DD(N=poisson_random_process_order, dt=timestep__ms, pool__ID=i)
                for i in range(n)
            ]
        )

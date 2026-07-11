"""
Descending drive neuron populations for cortical input.

This module contains the population classes for descending drive neurons that
simulate cortical input to spinal motor circuits via Poisson and Gamma processes.
"""

from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__ms

from .base import _Pool


@beartowertype
class DescendingDrive__Pool(_Pool):
    """
    Container for a population of descending drive neurons.

    Manages a collection of DD cells that generate spike trains using either
    Poisson or Gamma point processes for cortical input to spinal circuits.

    Parameters
    ----------
    n : int
        Number of descending drive neurons to create.
    timestep__ms : float
        Time step for simulation (ms).
    process_type : str, optional
        Type of point process: "poisson" or "gamma", by default "poisson".
        - "poisson": true Poisson process, irregular firing (CV=1.0)
        - "gamma": More regular firing with CV controlled by shape parameter
    shape : float, optional
        Shape parameter for Gamma process (only used when process_type="gamma"),
        by default 3.0. Controls spike regularity:
        - shape=1: Poisson-like (CV=1.0)
        - shape=2-5: Typical cortical neuron regularity (CV=0.45-0.71)
        - Higher values: More regular firing (CV=1/sqrt(shape))
    """

    def __init__(
        self,
        n: int,
        timestep__ms: Quantity__ms | None = None,
        process_type: str = "poisson",
        shape: float = 3.0,
    ):
        if timestep__ms is None:
            raise ValueError("timestep__ms is required")

        self.n = n
        self.timestep__ms = timestep__ms
        self.process_type = process_type
        self.shape = shape

        if process_type.lower() == "gamma":
            _cells = [
                cells.DD_Gamma(
                    timestep__ms=timestep__ms,
                    shape=shape,
                    pool__ID=i,
                )
                for i in range(n)
            ]
        elif process_type.lower() == "poisson":
            _cells = [cells.DD(dt=timestep__ms, pool__ID=i) for i in range(n)]
        else:
            raise ValueError(
                f"Invalid process_type '{process_type}'. Must be 'poisson' or 'gamma'."
            )

        super().__init__(cells=_cells)


@beartowertype
class DescendingDrive_Gamma__Pool(_Pool):
    """
    Container for a population of descending drive neurons using Gamma process.

    Manages a collection of DD_Gamma cells that generate Gamma-distributed
    spike trains for more regular cortical input to spinal circuits, typical
    of cortical neuron firing patterns.

    Note: This class is kept for backward compatibility. Consider using
    DescendingDrive__Pool with process_type='gamma' instead.

    Parameters
    ----------
    n : int
        Number of descending drive neurons to create.
    timestep__ms : Quantity__ms
        Time step for simulation as a Quantity with units of milliseconds.
    shape : float, optional
        Shape parameter controlling spike regularity, by default 3.0.
        - shape=1: Poisson-like (irregular) firing
        - shape=2-5: Typical cortical neuron regularity
        - Higher values: More regular, clock-like firing
    """

    def __init__(
        self,
        n: int,
        timestep__ms: Quantity__ms,
        shape: float = 3.0,
    ):
        self.n = n
        self.timestep__ms = timestep__ms
        self.shape = shape

        super().__init__(
            cells=[
                cells.DD_Gamma(
                    timestep__ms=timestep__ms,
                    shape=shape,
                    pool__ID=i,
                )
                for i in range(n)
            ]
        )

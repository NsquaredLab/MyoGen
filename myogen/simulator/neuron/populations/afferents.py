"""
Afferent neuron populations for proprioceptive feedback.

This module contains population classes for different types of afferent neurons
that provide sensory feedback from muscle spindles and Golgi tendon organs.
"""

import numpy as np

from myogen.simulator.neuron import cells
from myogen.utils.decorators import beartowertype

from .base import _Pool


@beartowertype
class AffIa__Pool(_Pool):
    """
    Container for a population of afferent Ia neurons.

    Manages a collection of AffIa (type Ia afferent) cells that provide
    proprioceptive feedback from muscle spindles to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type Ia afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length__m : float
        Length of the axon (m).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 40),
        axon_velocities: tuple[float, float] = (58, 72),
        axon_length__m: float = 0.6,
        poisson_random_process_order: int = 145,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length__m
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for i, (rt_i, vcon_i) in enumerate(zip(rt, vcon)):
            ia = cells.AffIa(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
                pool__ID=i,
            )
            ia.create_axon(length__m=axon_length__m, conduction_velocity__m_per_s=vcon_i)
            _cells.append(ia)

        super().__init__(cells=_cells)


@beartowertype
class AffII__Pool(_Pool):
    """
    Container for a population of afferent II neurons.

    Manages a collection of AffII (type II afferent) cells that provide
    secondary proprioceptive feedback from muscle spindles to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type II afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length__m : float
        Length of the axon (m).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 50),
        axon_velocities: tuple[float, float] = (32, 52),
        axon_length__m: float = 0.6,
        poisson_random_process_order: int = 500,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length__m
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for i, (rt_i, vcon_i) in enumerate(zip(rt, vcon)):
            ii = cells.AffII(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
                pool__ID=i,
            )
            ii.create_axon(length__m=axon_length__m, conduction_velocity__m_per_s=vcon_i)
            _cells.append(ii)

        super().__init__(cells=_cells)


@beartowertype
class AffIb__Pool(_Pool):
    """
    Container for a population of afferent Ib neurons.

    Manages a collection of AffIb (type Ib afferent) cells that provide
    primary proprioceptive feedback from Golgi tendon organs to spinal circuits.

    Parameters
    ----------
    n : int
        Number of type Ib afferent neurons to create.
    recruitment_thresholds : tuple[float, float]
        Min and max recruitment thresholds (Hz).
    axon_velocities : tuple[float, float]
        Min and max axon conduction velocities (m/s).
    axon_length : float
        Length of the axon (mm).
    poisson_random_process_order : int
        Order parameter for the Poisson process generation.
    timestep__ms : float
        Time step for simulation (ms).
    init_order : int
        Initial order parameter for afferent initialization.
    """

    def __init__(
        self,
        n: int,
        timestep__ms: float,
        recruitment_thresholds: tuple[float, float] = (0, 50),
        axon_velocities: tuple[float, float] = (62, 66),
        axon_length: float = 0.6,
        poisson_random_process_order: int = 30,
        init_order: int = 0,
    ):
        self.n = n
        self.recruitment_thresholds = recruitment_thresholds
        self.axon_velocities = axon_velocities
        self.axon_length = axon_length
        self.poisson_random_process_order = poisson_random_process_order
        self.timestep__ms = timestep__ms
        self.init_order = init_order

        rt = np.linspace(*recruitment_thresholds, n)
        vcon = np.linspace(*axon_velocities, n)

        _cells = []
        for i, (rt_i, vcon_i) in enumerate(zip(rt, vcon)):
            ib = cells.AffIb(
                RT=rt_i,
                N=poisson_random_process_order,
                dt=timestep__ms,
                initN=init_order,
                pool__ID=i,
            )
            ib.create_axon(length__m=axon_length, conduction_velocity__m_per_s=vcon_i)
            _cells.append(ib)

        super().__init__(cells=_cells)
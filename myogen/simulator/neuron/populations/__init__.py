"""
Neuron population management for MyoGen spinal circuit simulations.

This module provides population classes for different types of neurons used in
spinal motor circuit modeling, including motor neurons, interneurons, and afferents.

Classes
-------
_Pool
    Base class for all neuron populations providing common functionality.

DescendingDrive__Pool
    Population of descending drive neurons generating cortical input via Poisson or Gamma processes.

AffIa__Pool
    Population of type Ia afferent neurons providing primary proprioceptive feedback
    from muscle spindles.

AffII__Pool
    Population of type II afferent neurons providing secondary proprioceptive feedback
    from muscle spindles.

AffIb__Pool
    Population of type Ib afferent neurons providing feedback from Golgi tendon organs.

GII__Pool
    Population of group II interneurons processing type II afferent input in spinal circuits.

GIb__Pool
    Population of group Ib interneurons processing type Ib afferent input in spinal circuits.

AlphaMN__Pool
    Population of alpha motor neurons forming the final common pathway for motor control.

Usage
-----
>>> from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
>>>
>>> # Create motor neuron population
>>> motor_pool = AlphaMN__Pool(n=10, model="Powers2017", mode="active")
>>>
>>> # Create descending drive population with Poisson process (default)
>>> drive_pool = DescendingDrive__Pool(n=5, poisson_batch_size=16,
...                                     timestep__ms=0.05)
>>>
>>> # Create descending drive with Gamma process for more regular firing
>>> drive_gamma = DescendingDrive__Pool(n=5, timestep__ms=0.05, process_type='gamma',
...                                      shape=3.0)
"""

from .afferents import AffIa__Pool, AffII__Pool, AffIb__Pool
from .base import _Pool
from .descending_drive import DescendingDrive__Pool, DescendingDrive_Gamma__Pool
from .interneurons import GII__Pool, GIb__Pool
from .motor_neurons import AlphaMN__Pool

__all__ = [
    "_Pool",
    "DescendingDrive__Pool",
    "DescendingDrive_Gamma__Pool",
    "AffIa__Pool",
    "AffII__Pool",
    "AffIb__Pool",
    "GII__Pool",
    "GIb__Pool",
    "AlphaMN__Pool",
]

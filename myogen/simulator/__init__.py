"""
MyoGen Simulator Module

This module provides high-level simulation functions for muscle and EMG modeling.
NMODL files are automatically loaded when needed.
"""

from myogen.simulator.core.emg import (
    IntramuscularElectrodeArray,
    IntramuscularEMG,
    SurfaceElectrodeArray,
    SurfaceEMG,
)
from myogen.utils.neo import GridAnalogSignal
from myogen.simulator.core.force import ForceModel
from myogen.simulator.core.muscle import Muscle

# Always import all public APIs (they will fail gracefully if NMODL not loaded)
from myogen.simulator.core.physiological_distribution import RecruitmentThresholds

# Spinal neural population models
# PyNN integration - commented out to avoid auto-loading conflicts
# from myogen.simulator import pyNN

__all__ = [
    "RecruitmentThresholds",
    "Muscle",
    "SurfaceEMG",
    "IntramuscularEMG",
    "SurfaceElectrodeArray",
    "IntramuscularElectrodeArray",
    "ForceModel",
    "GridAnalogSignal",
    # "pyNN",  # commented out to avoid auto-loading conflicts
]

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
from myogen.simulator.core.force import ForceModel
from myogen.simulator.core.muscle import Muscle

# Always import all public APIs (they will fail gracefully if NMODL not loaded)
from myogen.simulator.core.physiological_distribution import RecruitmentThresholds
from myogen.simulator.neuron.muscle import HillModel
from myogen.simulator.neuron.joint_dynamics import JointDynamics
from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.proprioception import GolgiTendonOrganModel, SpindleModel
from myogen.simulator.neuron.simulation_runner import SimulationRunner
from myogen.utils.neo import (
    create_grid_signal,
    signal_to_grid,
    get_electrode,
    get_row,
    get_column,
    GridAnalogSignal,  # Deprecated, kept for backwards compatibility
)

__all__ = [
    "RecruitmentThresholds",
    "Muscle",
    "Network",
    "SimulationRunner",
    "SurfaceEMG",
    "IntramuscularEMG",
    "SurfaceElectrodeArray",
    "IntramuscularElectrodeArray",
    "ForceModel",
    "HillModel",
    "SpindleModel",
    "GolgiTendonOrganModel",
    "JointDynamics",
    # Grid signal utilities (NWB-compatible)
    "create_grid_signal",
    "signal_to_grid",
    "get_electrode",
    "get_row",
    "get_column",
    "GridAnalogSignal",  # Deprecated
]

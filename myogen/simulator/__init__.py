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
from myogen.simulator.core.force import (
    ForceModel,
    ForceModelVectorized,
    JointBiomechanics,
    JointGeometry,
    MuscleGeometry,
)
from myogen.simulator.core.muscle import Muscle

# Always import all public APIs (they will fail gracefully if NMODL not loaded)
from myogen.simulator.core.physiological_distribution import RecruitmentThresholds

# Default simulation backend is Jaxley — importing ``myogen.simulator`` no longer
# requires the NEURON runtime. The NEURON backend remains available (optionally, if
# the ``neuron`` package is installed) via the explicit namespace, e.g.
# ``from myogen.simulator.neuron.muscle import HillModel`` or ``simulator.neuron.*``.
from myogen.simulator.jaxley.muscle import HillModel
from myogen.simulator.jaxley.joint_dynamics import JointDynamics
from myogen.simulator.jaxley.network import Network
from myogen.simulator.jaxley.proprioception import GolgiTendonOrganModel, SpindleModel
from myogen.simulator.jaxley.simulation_runner import SimulationRunner
from myogen.utils.neo import (
    create_grid_signal,
    signal_to_grid,
    get_electrode,
    get_row,
    get_column,
    GridAnalogSignal,  # Deprecated, kept for backwards compatibility
)

def __getattr__(name):
    """Lazily expose the optional NEURON backend subpackage as ``simulator.neuron``.

    Keeps ``import myogen.simulator`` free of the NEURON runtime while still allowing
    ``simulator.neuron.*`` access when the ``neuron`` package is installed.
    """
    if name == "neuron":
        import importlib
        return importlib.import_module("myogen.simulator.neuron")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "ForceModelVectorized",
    "JointBiomechanics",
    "JointGeometry",
    "MuscleGeometry",
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

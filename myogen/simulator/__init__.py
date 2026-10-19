"""Public names loaded on demand.

Top-level myogen retains its existing NEURON mechanism-loading behavior.
"""
from importlib import import_module
from typing import TYPE_CHECKING

# Make lazy public exports visible to static API documentation and type checkers
# without importing the numerical kernels during normal package initialization.
if TYPE_CHECKING:
    from .core.emg import (
        IntramuscularElectrodeArray as IntramuscularElectrodeArray,
        IntramuscularEMG as IntramuscularEMG,
        SurfaceElectrodeArray as SurfaceElectrodeArray,
        SurfaceEMG as SurfaceEMG,
    )
    from .core.emg.runtime import (
        ExecutionPolicy as ExecutionPolicy,
        MemoryBudgetError as MemoryBudgetError,
    )
    from .core.force import (
        ForceModel as ForceModel,
        ForceModelVectorized as ForceModelVectorized,
        JointBiomechanics as JointBiomechanics,
        JointGeometry as JointGeometry,
        MuscleGeometry as MuscleGeometry,
    )
    from .core.muscle import Muscle as Muscle
    from .core.physiological_distribution import RecruitmentThresholds as RecruitmentThresholds
    from .neuron.joint_dynamics import JointDynamics as JointDynamics
    from .neuron.muscle import HillModel as HillModel
    from .neuron.network import Network as Network
    from .neuron.proprioception import (
        GolgiTendonOrganModel as GolgiTendonOrganModel,
        SpindleModel as SpindleModel,
    )
    from .neuron.simulation_runner import SimulationRunner as SimulationRunner
    from myogen.utils.neo import (
        GridAnalogSignal as GridAnalogSignal,
        create_grid_signal as create_grid_signal,
        get_column as get_column,
        get_electrode as get_electrode,
        get_row as get_row,
        signal_to_grid as signal_to_grid,
    )

_EXPORTS = {
    'ExecutionPolicy': 'myogen.simulator.core.emg.runtime',
    'MemoryBudgetError': 'myogen.simulator.core.emg.runtime',
    'RecruitmentThresholds': 'myogen.simulator.core.physiological_distribution',
    'Muscle': 'myogen.simulator.core.muscle',
    'Network': 'myogen.simulator.neuron.network',
    'SimulationRunner': 'myogen.simulator.neuron.simulation_runner',
    'SurfaceEMG': 'myogen.simulator.core.emg',
    'IntramuscularEMG': 'myogen.simulator.core.emg',
    'SurfaceElectrodeArray': 'myogen.simulator.core.emg',
    'IntramuscularElectrodeArray': 'myogen.simulator.core.emg',
    'ForceModel': 'myogen.simulator.core.force',
    'ForceModelVectorized': 'myogen.simulator.core.force',
    'JointBiomechanics': 'myogen.simulator.core.force',
    'JointGeometry': 'myogen.simulator.core.force',
    'MuscleGeometry': 'myogen.simulator.core.force',
    'HillModel': 'myogen.simulator.neuron.muscle',
    'SpindleModel': 'myogen.simulator.neuron.proprioception',
    'GolgiTendonOrganModel': 'myogen.simulator.neuron.proprioception',
    'JointDynamics': 'myogen.simulator.neuron.joint_dynamics',
    'create_grid_signal': 'myogen.utils.neo',
    'signal_to_grid': 'myogen.utils.neo',
    'get_electrode': 'myogen.utils.neo',
    'get_row': 'myogen.utils.neo',
    'get_column': 'myogen.utils.neo',
    'GridAnalogSignal': 'myogen.utils.neo',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __package__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

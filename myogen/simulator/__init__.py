"""Public names loaded on demand.

Top-level myogen retains its existing NEURON mechanism-loading behavior.
"""
from importlib import import_module

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

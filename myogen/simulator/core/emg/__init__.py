"""Public names loaded on demand.

Top-level myogen retains its existing NEURON mechanism-loading behavior.
"""
from importlib import import_module

_EXPORTS = {
    'SurfaceEMG': 'myogen.simulator.core.emg.surface.surface_emg',
    'IntramuscularEMG': 'myogen.simulator.core.emg.intramuscular.intramuscular_emg',
    'SurfaceElectrodeArray': 'myogen.simulator.core.emg.electrodes',
    'IntramuscularElectrodeArray': 'myogen.simulator.core.emg.electrodes',
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

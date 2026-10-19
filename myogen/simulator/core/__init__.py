"""Public names loaded on demand.

Top-level myogen retains its existing NEURON mechanism-loading behavior.
"""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .emg import SurfaceEMG as SurfaceEMG
    from .muscle import Muscle as Muscle
    from .physiological_distribution import RecruitmentThresholds as RecruitmentThresholds

_EXPORTS = {
    'Muscle': '.muscle',
    'RecruitmentThresholds': '.physiological_distribution',
    'SurfaceEMG': '.emg',
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

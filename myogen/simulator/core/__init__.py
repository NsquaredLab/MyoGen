"""
Core components for the simulator package.

This module contains the core classes and functions that are used across
the simulator package, organized to eliminate circular dependencies.
"""

from .emg import SurfaceEMG
from .muscle import Muscle
from .physiological_distribution import RecruitmentThresholds

__all__ = ["Muscle", "RecruitmentThresholds", "SurfaceEMG"]

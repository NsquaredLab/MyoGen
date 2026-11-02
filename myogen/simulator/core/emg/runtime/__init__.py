"""Opt-in CPU execution for the public EMG models, independent of anatomy."""
from .policy import ExecutionPolicy
from .memory import MemoryBudgetError

__all__ = ["ExecutionPolicy", "MemoryBudgetError"]

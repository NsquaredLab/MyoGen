"""Explicit memory inputs for bounded numerical and lifecycle contracts."""

from myogen.simulator.core.emg.runtime.memory import MemorySample


def ample_memory():
    """Keep unrelated host load out of tests that do not exercise pressure."""
    return MemorySample(16 * 1024**3, 12 * 1024**3, 0)

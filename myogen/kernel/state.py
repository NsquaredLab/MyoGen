from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

import numpy as np


@dataclass
class SimState:
    """The shared simulation buffer.

    Single source of truth for a run. Holds named, contiguous SI float arrays.
    Stages claim buffers in ``setup()`` and write them in place in ``step()``.
    No physical units and no neo objects ever live here — SI is implied.
    """

    n_units: int
    dt_s: float
    n_steps: int
    xp: ModuleType = np
    t: int = 0  # current step index, set by the driver each tick
    _buffers: dict = field(default_factory=dict, repr=False)

    def alloc(self, name: str, shape: tuple[int, ...], dtype=None):
        """Allocate a zeroed buffer and return it. Raises if name is taken."""
        if name in self._buffers:
            raise KeyError(f"buffer {name!r} already allocated")
        if dtype is None:
            dtype = self.xp.float64
        buf = self.xp.zeros(shape, dtype=dtype)
        self._buffers[name] = buf
        return buf

    def view(self, name: str):
        """Return the buffer named ``name`` (the same object, zero-copy)."""
        try:
            return self._buffers[name]
        except KeyError:
            raise KeyError(f"no buffer named {name!r}") from None

    def has(self, name: str) -> bool:
        return name in self._buffers

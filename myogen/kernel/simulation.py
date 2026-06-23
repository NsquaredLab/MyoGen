from __future__ import annotations

from collections.abc import Callable

import numpy as np

from myogen.kernel.state import SimState


class Simulation:
    """Backend-agnostic driver. Owns dt and the lockstep tick over the stages.

    Stages run in fixed list order each tick. Closed-loop feedback emerges
    naturally: a stage that reads a buffer a *later* stage writes sees the
    previous tick's value (a one-tick delay where conduction delays live).
    """

    def __init__(
        self,
        *stages,
        n_units: int,
        dt_s: float,
        n_steps: int,
        xp=np,
    ):
        if not stages:
            raise ValueError("Simulation requires at least one stage")
        self.stages = list(stages)
        self.state = SimState(n_units=n_units, dt_s=dt_s, n_steps=n_steps, xp=xp)
        self._callbacks: list[Callable[[SimState, int], None]] = []
        self._setup_done = False

    def on_step(self, fn: Callable[[SimState, int], None]) -> None:
        """Register a per-tick callback (closed-loop hook). Gets raw buffers."""
        self._callbacks.append(fn)

    def setup(self) -> None:
        for stage in self.stages:
            stage.setup(self.state)
        self._setup_done = True

    def run(self) -> SimState:
        if not self._setup_done:
            self.setup()
        s = self.state
        for t in range(s.n_steps):
            s.t = t
            for stage in self.stages:
                stage.step(s, t)
            for cb in self._callbacks:
                cb(s, t)
        return s

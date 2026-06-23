"""Test doubles for the kernel. Tiny, dependency-free Stage/Backend fakes."""
from __future__ import annotations

from myogen.kernel.state import SimState


class RampStage:
    """Writes t into buffer ``name`` at row t each step (a deterministic ramp)."""

    def __init__(self, name: str = "ramp"):
        self.name = name

    def setup(self, state: SimState) -> None:
        state.alloc(self.name, (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view(self.name)[t] = float(t)


class CountingBackend:
    """A Backend double that records lifecycle calls and integrates a counter."""

    def __init__(self):
        self.calls: list[str] = []
        self.advances = 0

    def init(self, state: SimState) -> None:
        self.calls.append("init")
        state.alloc("counter", (state.n_steps,))

    def advance(self, state: SimState, dt_s: float) -> None:
        state.view("counter")[state.t] = self.advances
        self.advances += 1

    def teardown(self) -> None:
        self.calls.append("teardown")


class BackendStage:
    """A Stage that delegates each step to a Backend (the seam pattern)."""

    def __init__(self, backend):
        self.backend = backend

    def setup(self, state: SimState) -> None:
        self.backend.init(state)

    def step(self, state: SimState, t: int) -> None:
        self.backend.advance(state, state.dt_s)

from __future__ import annotations

from myogen.kernel.protocols import Backend, Stage
from myogen.kernel.state import SimState


class _GoodStage:
    def setup(self, state: SimState) -> None:
        state.alloc("x", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view("x")[t] = 1.0


class _GoodBackend:
    def init(self, state: SimState) -> None: ...
    def advance(self, state: SimState, dt_s: float) -> None: ...
    def teardown(self) -> None: ...


class _NotAStage:
    def setup(self, state) -> None: ...
    # missing step()


def test_stage_protocol_is_runtime_checkable():
    assert isinstance(_GoodStage(), Stage)
    assert not isinstance(_NotAStage(), Stage)


def test_backend_protocol_is_runtime_checkable():
    assert isinstance(_GoodBackend(), Backend)
    assert not isinstance(_GoodStage(), Backend)

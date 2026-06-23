from __future__ import annotations

import numpy as np

from myogen.kernel.simulation import Simulation
from myogen.kernel.state import SimState


class _Writer:
    """Stage B: writes 'signal'[t] = t each tick (runs AFTER the reader)."""

    def setup(self, state: SimState) -> None:
        state.alloc("signal", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        state.view("signal")[t] = float(t)


class _Reader:
    """Stage A: reads 'signal' from the PREVIOUS tick into 'fed_back'."""

    def setup(self, state: SimState) -> None:
        state.alloc("fed_back", (state.n_steps,))

    def step(self, state: SimState, t: int) -> None:
        prev = state.view("signal")[t - 1] if t > 0 else 0.0
        state.view("fed_back")[t] = prev


def test_feedback_has_one_tick_delay():
    # Reader runs before Writer, so it sees last tick's signal: a one-tick delay.
    sim = Simulation(_Reader(), _Writer(), n_units=1, dt_s=0.001, n_steps=4)
    sim.run()
    # signal = [0,1,2,3]; fed_back = [0, signal[0], signal[1], signal[2]] = [0,0,1,2]
    assert np.array_equal(sim.state.view("fed_back"), np.array([0.0, 0.0, 1.0, 2.0]))


def test_on_step_callback_runs_each_tick_with_raw_state():
    sim = Simulation(_Writer(), n_units=1, dt_s=0.001, n_steps=3)
    seen: list[float] = []
    sim.on_step(lambda s, t: seen.append(float(s.view("signal")[t])))
    sim.run()
    assert seen == [0.0, 1.0, 2.0]

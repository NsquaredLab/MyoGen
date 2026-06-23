from __future__ import annotations

import numpy as np
import pytest

from myogen.kernel.simulation import Simulation
from tests.kernel._doubles import RampStage


def test_run_steps_each_stage_once_per_tick():
    sim = Simulation(RampStage("a"), RampStage("b"), n_units=2, dt_s=0.001, n_steps=5)
    sim.run()
    assert np.array_equal(sim.state.view("a"), np.arange(5, dtype=float))
    assert np.array_equal(sim.state.view("b"), np.arange(5, dtype=float))


def test_run_calls_setup_before_stepping_and_sets_t():
    sim = Simulation(RampStage(), n_units=1, dt_s=0.001, n_steps=3)
    sim.run()
    assert sim.state.t == 2


def test_simulation_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        Simulation(n_units=1, dt_s=0.001, n_steps=1)

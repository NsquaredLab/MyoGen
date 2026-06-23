from __future__ import annotations

import numpy as np

from myogen.kernel.result import SimResult
from myogen.kernel.state import SimState


def _state_with_spikes_and_force() -> SimState:
    state = SimState(n_units=2, dt_s=0.5, n_steps=4)
    spikes = state.alloc("spikes", (4, 2), dtype=np.int8)
    spikes[1, 0] = 1  # unit 0 spikes at step 1 -> t = 0.5 s
    spikes[3, 1] = 1  # unit 1 spikes at step 3 -> t = 1.5 s
    force = state.alloc("force", (4, 1))
    force[:, 0] = [0.0, 1.0, 2.0, 3.0]
    state.t = 3
    return state


def test_from_state_extracts_spike_times_in_seconds():
    res = SimResult.from_state(_state_with_spikes_and_force())
    assert res.n_units == 2
    assert res.dt_s == 0.5
    assert np.array_equal(res.spike_times_s[0], np.array([0.5]))
    assert np.array_equal(res.spike_times_s[1], np.array([1.5]))


def test_from_state_copies_force_array():
    state = _state_with_spikes_and_force()
    res = SimResult.from_state(state)
    assert np.array_equal(res.force_N[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))
    # the snapshot must NOT alias the live SimState buffer
    assert not np.shares_memory(res.force_N, state.view("force"))


def test_from_state_handles_missing_optional_buffers():
    state = SimState(n_units=3, dt_s=0.001, n_steps=2)
    res = SimResult.from_state(state)
    assert res.force_N is None
    assert res.surface_emg_V is None
    assert len(res.spike_times_s) == 3
    for arr in res.spike_times_s:
        assert arr.size == 0


def test_from_state_infers_grid_shape_from_emg():
    state = SimState(n_units=1, dt_s=0.001, n_steps=2)
    state.alloc("surface_emg", (2, 8, 4))
    res = SimResult.from_state(state)
    assert res.grid_shape == (8, 4)

from __future__ import annotations

import numpy as np
import pytest

from myogen.kernel.state import SimState


def test_alloc_returns_zeroed_array_and_view_returns_same_object():
    state = SimState(n_units=4, dt_s=0.001, n_steps=10)
    buf = state.alloc("force", (10, 2))
    assert buf.shape == (10, 2)
    assert buf.dtype == np.float64
    assert np.all(buf == 0.0)
    # view returns the SAME underlying object (zero-copy contract)
    assert state.view("force") is buf
    assert state.has("force") is True


def test_view_missing_buffer_raises_keyerror():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    assert state.has("nope") is False
    with pytest.raises(KeyError, match="no buffer named 'nope'"):
        state.view("nope")


def test_double_alloc_raises():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    state.alloc("x", (3,))
    with pytest.raises(KeyError, match="already allocated"):
        state.alloc("x", (3,))


def test_alloc_respects_dtype_and_xp_default_is_numpy():
    state = SimState(n_units=1, dt_s=0.001, n_steps=1)
    assert state.xp is np
    buf = state.alloc("spikes", (5, 1), dtype=np.int8)
    assert buf.dtype == np.int8

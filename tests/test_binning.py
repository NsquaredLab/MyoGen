"""Prove myogen.utils.binning.bin_spike_trains matches elephant's
BinnedSpikeTrain across the cases MyoGen relies on (issue #16)."""

import numpy as np
import pytest
import quantities as pq
from neo import SpikeTrain

from myogen.utils.binning import bin_spike_trains

elephant = pytest.importorskip("elephant")
from elephant.conversion import BinnedSpikeTrain  # noqa: E402


def _make_trains(spike_lists, t_start, t_stop, sampling_period):
    trains = []
    for spikes in spike_lists:
        st = SpikeTrain(
            np.asarray(spikes) * pq.ms, t_start=t_start * pq.ms, t_stop=t_stop * pq.ms
        )
        st.sampling_period = sampling_period * pq.ms
        trains.append(st)
    return trains


# (spike_lists, t_start_ms, t_stop_ms, dt_ms)
CASES = [
    # Regular grid, the common MyoGen case.
    ([[0, 1, 2, 5, 10]], 0, 11, 1.0),
    # Multiple trains in one pool.
    ([[5, 7], [0, 5, 10], []], 0, 11, 1.0),
    # IEEE-754 awkward length (cf. issue #12): N=1001 bins at dt=1 ms.
    ([list(range(0, 1001, 3))], 0, 1001, 1.0),
    # Spike exactly on t_stop must be discarded (right edge).
    ([[0, 5, 11]], 0, 11, 1.0),
    # Sub-ms bins / fractional timestep.
    ([[0.0, 0.1, 0.25, 0.9, 1.0]], 0, 2, 0.1),
    # Non-zero t_start.
    ([[3, 4, 7, 9]], 3, 10, 1.0),
    # Dense, irregular spike times.
    ([sorted(np.random.RandomState(0).uniform(0, 50, 40).tolist())], 0, 50, 0.5),
    # Duplicate spikes in the same bin -> occupancy stays boolean (one True).
    ([[0, 0, 1, 5, 5, 5]], 0, 11, 1.0),
    # Every train empty.
    ([[], []], 0, 11, 1.0),
]


@pytest.mark.parametrize("spike_lists,t0,t1,dt", CASES)
def test_dense_matches_elephant_default_bounds(spike_lists, t0, t1, dt):
    trains = _make_trains(spike_lists, t0, t1, dt)
    expected = BinnedSpikeTrain(trains, bin_size=dt * pq.ms).to_array().astype(bool)
    observed = bin_spike_trains(trains, bin_size=dt * pq.ms)
    assert observed.dtype == bool
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("spike_lists,t0,t1,dt", CASES)
def test_sparse_matches_elephant_explicit_bounds(spike_lists, t0, t1, dt):
    # Mirrors the force-model usage: explicit t_start/t_stop + sparse bool.
    trains = _make_trains(spike_lists, t0, t1, dt)
    expected = BinnedSpikeTrain(
        trains, bin_size=dt * pq.ms, t_start=t0 * pq.ms, t_stop=t1 * pq.ms
    ).to_sparse_bool_array()
    observed = bin_spike_trains(
        trains, bin_size=dt * pq.ms, t_start=t0 * pq.ms, t_stop=t1 * pq.ms, sparse=True
    )
    assert observed.shape == expected.shape
    assert (observed != expected).nnz == 0


def test_explicit_subrange_bounds_match_elephant():
    # Bin over a window strictly inside the train's [t_start, t_stop] (allowed
    # by elephant), exercising the spike filtering against custom bounds.
    st = SpikeTrain(
        [1, 3, 5, 7, 9] * pq.ms, t_start=0 * pq.ms, t_stop=20 * pq.ms
    )
    st.sampling_period = 1 * pq.ms
    expected = (
        BinnedSpikeTrain([st], bin_size=1 * pq.ms, t_start=2 * pq.ms, t_stop=8 * pq.ms)
        .to_array()
        .astype(bool)
    )
    observed = bin_spike_trains(
        [st], bin_size=1 * pq.ms, t_start=2 * pq.ms, t_stop=8 * pq.ms
    )
    np.testing.assert_array_equal(observed, expected)


def test_tolerance_shifts_edge_spikes_like_elephant():
    # A spike a hair below a bin edge should snap into the next bin (elephant's
    # tolerance behaviour), not stay in the lower bin.
    edge = 5.0 - 1e-12
    trains = _make_trains([[edge]], 0, 11, 1.0)
    expected = BinnedSpikeTrain(trains, bin_size=1.0 * pq.ms).to_array().astype(bool)
    observed = bin_spike_trains(trains, bin_size=1.0 * pq.ms)
    np.testing.assert_array_equal(observed, expected)
    assert observed[0, 5]  # snapped up to bin 5, not 4

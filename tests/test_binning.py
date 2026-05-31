"""Tests for myogen.utils.binning.bin_spike_trains (issue #16).

Two layers:

* **Golden tests** assert the binner against hand-verified expected occupancy.
  They are self-contained and require no optional dependencies, so they give
  CI coverage even when ``elephant`` is not installed.
* **Equivalence tests** assert bit-identical output to
  ``elephant.conversion.BinnedSpikeTrain`` (the implementation it replaces).
  They run only where ``elephant`` is available.
"""

import numpy as np
import pytest
import quantities as pq
from neo import SpikeTrain

from myogen.utils.binning import bin_spike_trains

try:
    from elephant.conversion import BinnedSpikeTrain

    HAS_ELEPHANT = True
except ImportError:
    HAS_ELEPHANT = False


def _make_trains(spike_lists, t_start, t_stop, sampling_period):
    trains = []
    for spikes in spike_lists:
        st = SpikeTrain(
            np.asarray(spikes) * pq.ms, t_start=t_start * pq.ms, t_stop=t_stop * pq.ms
        )
        st.sampling_period = sampling_period * pq.ms
        trains.append(st)
    return trains


def _occupancy(n_trains, n_bins, true_indices):
    """Build a boolean (n_trains, n_bins) matrix from per-train True-bin lists."""
    out = np.zeros((n_trains, n_bins), dtype=bool)
    for row, idxs in enumerate(true_indices):
        for j in idxs:
            out[row, j] = True
    return out


# --------------------------------------------------------------------------- #
# Golden tests (no elephant required)
# --------------------------------------------------------------------------- #

# (spike_lists, t_start, t_stop, dt, n_bins, expected True bins per train)
GOLDEN = [
    ("grid", [[0, 1, 2, 5, 10]], 0, 11, 1.0, 11, [[0, 1, 2, 5, 10]]),
    ("multi_train", [[5, 7], [0, 5, 10], []], 0, 11, 1.0, 11, [[5, 7], [0, 5, 10], []]),
    # Spike on t_stop (11) is discarded (bins cover [t_start, t_stop)).
    ("tstop_discard", [[0, 5, 11]], 0, 11, 1.0, 11, [[0, 5]]),
    # Fractional bins: 0.25/0.1 -> bin 2, 1.0/0.1 -> bin 10.
    ("fractional", [[0, 0.1, 0.25, 0.9, 1.0]], 0, 2, 0.1, 20, [[0, 1, 2, 9, 10]]),
    # Duplicate spikes collapse to one occupied bin.
    ("duplicates", [[0, 0, 1, 5, 5, 5]], 0, 11, 1.0, 11, [[0, 1, 5]]),
    # Every train empty.
    ("empty", [[], []], 0, 11, 1.0, 11, [[], []]),
]


@pytest.mark.parametrize(
    "name,spike_lists,t0,t1,dt,n_bins,true_idx",
    GOLDEN,
    ids=[g[0] for g in GOLDEN],
)
def test_golden_dense(name, spike_lists, t0, t1, dt, n_bins, true_idx):
    trains = _make_trains(spike_lists, t0, t1, dt)
    expected = _occupancy(len(spike_lists), n_bins, true_idx)
    observed = bin_spike_trains(trains, bin_size=dt * pq.ms)
    assert observed.dtype == bool
    np.testing.assert_array_equal(observed, expected)


def test_golden_sparse_matches_dense():
    trains = _make_trains([[5, 7], [0, 5, 10]], 0, 11, 1.0)
    dense = bin_spike_trains(trains, bin_size=1.0 * pq.ms)
    sparse = bin_spike_trains(trains, bin_size=1.0 * pq.ms, sparse=True)
    assert sparse.shape == dense.shape
    np.testing.assert_array_equal(sparse.toarray().astype(bool), dense)


def test_golden_tolerance_snaps_edge_spike_up():
    # A spike a hair below bin 5's edge snaps up into bin 5, not 4.
    trains = _make_trains([[5.0 - 1e-12]], 0, 11, 1.0)
    observed = bin_spike_trains(trains, bin_size=1.0 * pq.ms)
    expected = _occupancy(1, 11, [[5]])
    np.testing.assert_array_equal(observed, expected)


def test_golden_explicit_subrange_bounds():
    # Bin over [2, 8) of a train spanning [0, 20]; spikes 3,5,7 land in bins
    # 1,3,5 relative to t_start=2 (1 and 9 are outside the window).
    st = SpikeTrain([1, 3, 5, 7, 9] * pq.ms, t_start=0 * pq.ms, t_stop=20 * pq.ms)
    st.sampling_period = 1 * pq.ms
    observed = bin_spike_trains(
        [st], bin_size=1 * pq.ms, t_start=2 * pq.ms, t_stop=8 * pq.ms
    )
    expected = _occupancy(1, 6, [[1, 3, 5]])
    np.testing.assert_array_equal(observed, expected)


# --------------------------------------------------------------------------- #
# Equivalence tests vs elephant (only where elephant is installed)
# --------------------------------------------------------------------------- #

CASES = [
    ([[0, 1, 2, 5, 10]], 0, 11, 1.0),
    ([[5, 7], [0, 5, 10], []], 0, 11, 1.0),
    ([list(range(0, 1001, 3))], 0, 1001, 1.0),  # IEEE-754 awkward length (cf. #12)
    ([[0, 5, 11]], 0, 11, 1.0),
    ([[0.0, 0.1, 0.25, 0.9, 1.0]], 0, 2, 0.1),
    ([[3, 4, 7, 9]], 3, 10, 1.0),
    ([sorted(np.random.RandomState(0).uniform(0, 50, 40).tolist())], 0, 50, 0.5),
    ([[0, 0, 1, 5, 5, 5]], 0, 11, 1.0),
    ([[], []], 0, 11, 1.0),
]


@pytest.mark.skipif(not HAS_ELEPHANT, reason="elephant not installed")
@pytest.mark.parametrize("spike_lists,t0,t1,dt", CASES)
def test_dense_matches_elephant(spike_lists, t0, t1, dt):
    trains = _make_trains(spike_lists, t0, t1, dt)
    expected = BinnedSpikeTrain(trains, bin_size=dt * pq.ms).to_array().astype(bool)
    np.testing.assert_array_equal(bin_spike_trains(trains, bin_size=dt * pq.ms), expected)


@pytest.mark.skipif(not HAS_ELEPHANT, reason="elephant not installed")
@pytest.mark.parametrize("spike_lists,t0,t1,dt", CASES)
def test_sparse_matches_elephant_explicit_bounds(spike_lists, t0, t1, dt):
    trains = _make_trains(spike_lists, t0, t1, dt)
    expected = BinnedSpikeTrain(
        trains, bin_size=dt * pq.ms, t_start=t0 * pq.ms, t_stop=t1 * pq.ms
    ).to_sparse_bool_array()
    observed = bin_spike_trains(
        trains, bin_size=dt * pq.ms, t_start=t0 * pq.ms, t_stop=t1 * pq.ms, sparse=True
    )
    assert observed.shape == expected.shape
    assert (observed != expected).nnz == 0

"""Regression test for R1 Code Review 3: FR_std must not be NaN when only a single unit is active."""

from __future__ import annotations

import math

import neo
import numpy as np
import pytest
import quantities as pq

from myogen.utils.helper import calculate_firing_rate_statistics


def _build_spiketrain(times_ms, t_stop_ms=2000.0):
    """Construct a ``neo.SpikeTrain`` in milliseconds."""
    return neo.SpikeTrain(np.asarray(times_ms) * pq.ms, t_stop=t_stop_ms * pq.ms)


def test_fr_std_is_zero_not_nan_for_single_active_unit():
    """When only one unit passes the firing-rate filter, ``FR_std`` must be 0.0, never NaN — otherwise downstream analyses silently corrupt (CR3)."""
    active = _build_spiketrain(np.linspace(100.0, 1900.0, num=30))
    silent = _build_spiketrain([])

    stats = calculate_firing_rate_statistics(
        [active, silent],
        plateau_start_ms=100.0,
        plateau_end_ms=1900.0,
    )

    assert stats["n_active"] == 1
    assert not math.isnan(stats["FR_std"]), "FR_std must not be NaN when n_active == 1"
    assert stats["FR_std"] == 0.0


def test_fr_std_is_zero_for_empty_population():
    """Empty-population branch already returned 0.0 and must keep doing so."""
    silent = _build_spiketrain([])
    stats = calculate_firing_rate_statistics(
        [silent, silent],
        plateau_start_ms=0.0,
        plateau_end_ms=2000.0,
    )
    assert stats["n_active"] == 0
    assert stats["FR_std"] == 0.0


def test_fr_std_is_finite_and_positive_for_multiple_active_units():
    """Two active units with distinct rates must produce a positive, finite FR_std."""
    fast = _build_spiketrain(np.linspace(100.0, 1900.0, num=40))  # ~22 Hz
    slow = _build_spiketrain(np.linspace(100.0, 1900.0, num=15))  # ~7.5 Hz

    stats = calculate_firing_rate_statistics(
        [fast, slow],
        plateau_start_ms=100.0,
        plateau_end_ms=1900.0,
    )

    assert stats["n_active"] == 2
    assert math.isfinite(stats["FR_std"])
    assert stats["FR_std"] > 0.0

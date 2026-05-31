"""Dependency-free binning of spike trains into boolean occupancy matrices.

This reproduces the subset of :class:`elephant.conversion.BinnedSpikeTrain`
that MyoGen relies on -- turning a list of :class:`neo.SpikeTrain` objects into
a boolean ``(n_trains, n_bins)`` occupancy matrix -- so the simulator no longer
needs ``elephant`` just to bin spikes. The bin assignment matches elephant
exactly, including its floating-point tolerance handling (spikes within
``tolerance`` of the next bin edge are shifted into it) and its right-edge
discard, but without emitting the "shifting spikes to the next bin" warnings
that elephant logs (so no logger suppression is needed at the call sites).
"""

from __future__ import annotations

import numpy as np
import quantities as pq

__all__ = ["bin_spike_trains"]


def _round_binning_errors(values, tolerance: float | None = 1e-8):
    """Floor ``values`` to ints, nudging values within ``tolerance`` of the
    next integer up first.

    Mirrors :func:`elephant.utils.round_binning_errors` (array and scalar
    forms) but does not log a warning.
    """
    scalar = np.ndim(values) == 0
    if not tolerance:
        return int(values) if scalar else np.asarray(values).astype(np.int32)
    if scalar:
        v = float(values)
        if (1.0 - tolerance) <= (v % 1.0):
            v += 0.5
        return int(v)
    v = np.array(values, dtype=float, copy=True)
    v[(1.0 - tolerance) <= (v % 1.0)] += 0.5
    return v.astype(np.int32)


def bin_spike_trains(
    spiketrains,
    bin_size,
    *,
    t_start=None,
    t_stop=None,
    tolerance: float | None = 1e-8,
    sparse: bool = False,
):
    """Bin spike trains into a boolean occupancy matrix.

    Drop-in replacement for::

        elephant.conversion.BinnedSpikeTrain(
            spiketrains, bin_size, t_start=t_start, t_stop=t_stop
        ).to_array().astype(bool)        # sparse=False
        # ... or ...
        ).to_sparse_bool_array()         # sparse=True

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        Spike trains to bin. They must share the same units and (unless
        ``t_start`` / ``t_stop`` are given) the same ``t_start`` / ``t_stop``.
    bin_size : pq.Quantity
        Width of each bin.
    t_start, t_stop : pq.Quantity, optional
        Binning interval. Default to the first spike train's ``t_start`` /
        ``t_stop`` (elephant's default).
    tolerance : float or None, default 1e-8
        Absolute tolerance for snapping spikes that sit within ``tolerance`` of
        the next bin edge into it. ``None`` disables the correction.
    sparse : bool, default False
        If ``True`` return a :class:`scipy.sparse.csr_matrix`; otherwise a dense
        boolean ``np.ndarray``.

    Returns
    -------
    np.ndarray or scipy.sparse.csr_matrix
        Boolean occupancy matrix of shape ``(n_trains, n_bins)``. ``True`` marks
        bins that contain at least one spike.
    """
    n_trains = len(spiketrains)
    units = spiketrains[0].units

    if t_start is None:
        t_start = spiketrains[0].t_start
    if t_stop is None:
        t_stop = spiketrains[0].t_stop

    t_start = float(pq.Quantity(t_start).rescale(units).magnitude)
    t_stop = float(pq.Quantity(t_stop).rescale(units).magnitude)
    bin_size = float(pq.Quantity(bin_size).rescale(units).magnitude)

    n_bins = _round_binning_errors((t_stop - t_start) / bin_size, tolerance)
    n_bins = max(0, int(n_bins))
    scale = 1.0 / bin_size

    rows, cols = [], []
    for idx, train in enumerate(spiketrains):
        times = np.asarray(train.rescale(units).magnitude, dtype=float)
        times = times[(times >= t_start) & (times <= t_stop)] - t_start
        bins = _round_binning_errors(times * scale, tolerance)
        bins = np.unique(bins[bins < n_bins])
        if bins.size:
            rows.append(np.full(bins.size, idx, dtype=np.int64))
            cols.append(bins.astype(np.int64))

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)

    if sparse:
        import scipy.sparse as sps

        data = np.ones(rows.size, dtype=bool)
        return sps.csr_matrix((data, (rows, cols)), shape=(n_trains, n_bins), dtype=bool)

    occupancy = np.zeros((n_trains, n_bins), dtype=bool)
    occupancy[rows, cols] = True
    return occupancy

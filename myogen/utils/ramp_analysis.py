# myogen/utils/ramp_analysis.py
"""
Extract recruitment curves from trapezoidal ramp simulations.

Analyzes spike trains produced during a time-varying drive signal
to count active MUs and compute mean firing rates at %MVC checkpoints.
"""

import numpy as np
from neo import Block


def extract_ramp_recruitment(
    spike_block: Block,
    drive_signal: np.ndarray,
    threshold_hz: float,
    max_drive_hz: float,
    timestep_ms: float = 0.1,
    checkpoints_pct: list[float] | None = None,
    window_ms: float = 500.0,
    min_spikes: int = 3,
) -> dict[float, dict]:
    """Extract recruitment data at %MVC checkpoints during ramp.

    Parameters
    ----------
    spike_block : neo.Block
        Block with one segment containing spike trains from the ramp sim.
    drive_signal : ndarray
        The time-varying drive signal (Hz) used in the simulation.
    threshold_hz : float
        DD recruitment threshold (Hz).
    max_drive_hz : float
        Maximum DD drive (Hz) corresponding to 100% MVC.
    timestep_ms : float
        Simulation timestep in ms.
    checkpoints_pct : list of float
        %MVC levels to extract data at. Default [5, 10, 20, 30].
    window_ms : float
        Sliding window width for counting spikes (ms).
    min_spikes : int
        Minimum spikes in window to count a MU as "active".

    Returns
    -------
    dict
        Mapping mvc_pct -> {"n_active": int, "mean_fr_hz": float}
    """
    if checkpoints_pct is None:
        checkpoints_pct = [5, 10, 20, 30]

    spiketrains = spike_block.segments[0].spiketrains
    window_s = window_ms / 1000.0

    def mvc_to_hz(pct):
        return threshold_hz + (pct / 100.0) * (max_drive_hz - threshold_hz)

    results = {}
    for mvc_pct in checkpoints_pct:
        target_hz = mvc_to_hz(mvc_pct)

        # Find the time window where drive is closest to target_hz
        # (during the ramp-up phase — first occurrence)
        drive_diff = np.abs(drive_signal - target_hz)
        midpoint = len(drive_signal) // 2
        ramp_up_diff = drive_diff[:midpoint]
        best_step = int(np.argmin(ramp_up_diff))
        center_s = best_step * timestep_ms / 1000.0

        win_start = center_s - window_s / 2
        win_end = center_s + window_s / 2

        n_active = 0
        firing_rates = []
        for st in spiketrains:
            spike_times = np.asarray(st.magnitude)
            in_window = spike_times[(spike_times >= win_start) & (spike_times < win_end)]
            if len(in_window) >= min_spikes:
                n_active += 1
                if len(in_window) >= 2:
                    isis = np.diff(in_window)
                    fr = 1.0 / np.mean(isis)
                    firing_rates.append(fr)

        mean_fr = float(np.mean(firing_rates)) if firing_rates else 0.0
        results[mvc_pct] = {"n_active": n_active, "mean_fr_hz": mean_fr}

    return results

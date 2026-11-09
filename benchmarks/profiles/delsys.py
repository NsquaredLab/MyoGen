"""Delsys 5-pin scorer: plateau RMS and median frequency of the corner-centre (cc) and adjacent-corner (adj) channels.

The scorer the fdi_semg validation limits were frozen with (arm F of benchmarks/semg_dispersion, 2026-11-03).
``mono_emg`` is monopolar EMG shaped (time, sensors, 3, 3): each sensor a 3 x 3 grid whose corners and centre are
the Delsys pins. Each channel is band-passed (4th-order Butterworth, 20-2000 Hz, zero phase), cut to the plateau
(0.5 + ramp + 0.5 s to 0.5 + ramp + hold - 0.5 s), and scored: RMS, and the median frequency of the Welch spectrum
(nperseg 4096) over 20-2000 Hz. The result is the median over sensors x channels.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

CORNERS = [(0, 0), (0, 2), (2, 2), (2, 0)]
PAIRS = {"cc": [(c, (1, 1)) for c in CORNERS], "adj": [(CORNERS[i], CORNERS[(i + 1) % 4]) for i in range(4)]}
BAND__HZ = (20.0, 2000.0)


def score(mono_emg: np.ndarray, fs: float, ramp_s: float, hold_s: float) -> dict[str, tuple[float, float]]:
    """``{"cc": (median plateau RMS, median MDF Hz), "adj": (...)}``; RMS in the units of ``mono_emg``."""
    x = np.asarray(mono_emg, dtype=float)
    if x.ndim != 4 or x.shape[2:] != (3, 3):
        raise ValueError(f"mono_emg must be shaped (time, sensors, 3, 3), got {x.shape}")
    sos = butter(4, BAND__HZ, btype="bandpass", fs=fs, output="sos")
    t = np.arange(x.shape[0]) / fs
    plateau = (t > 0.5 + ramp_s + 0.5) & (t < 0.5 + ramp_s + hold_s - 0.5)
    out = {}
    for kind, pairs in PAIRS.items():
        rms, mdf = [], []
        for k in range(x.shape[1]):
            for p, q in pairs:
                y = sosfiltfilt(sos, x[:, k, p[0], p[1]] - x[:, k, q[0], q[1]])[plateau]
                f, P = welch(y, fs=fs, nperseg=4096)
                band = (f >= BAND__HZ[0]) & (f <= BAND__HZ[1])
                rms.append(np.sqrt(np.mean(y**2)))
                mdf.append(f[band][np.searchsorted(np.cumsum(P[band]), P[band].sum() / 2)])
        out[kind] = (float(np.median(rms)), float(np.median(mdf)))
    return out

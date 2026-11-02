"""Stimulation protocols for digital-twin experiments.

Each builder returns a scalar command time-series ``(n_steps,)`` (NumPy,
float32) — typically the per-neuron descending-drive rate in Hz — plus the time
vector. Pool-specific assembly into ``run_jax`` input dicts happens in
``poolbuild.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["time_vector", "triangular_ramp", "trapezoid", "step", "hold"]


def time_vector(t_total_s: float, dt_ms: float) -> np.ndarray:
    n = int(round(t_total_s * 1000.0 / dt_ms))
    return np.arange(n) * dt_ms * 1e-3


def triangular_ramp(
    dt_ms: float,
    t_rise_s: float = 10.0,
    t_fall_s: float | None = None,
    t_pre_s: float = 2.0,
    t_post_s: float = 2.0,
    baseline: float = 0.0,
    peak: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric (or asymmetric) triangular ramp — the ΔF/PIC protocol.

    Returns ``(t, command)`` with command in the same units as ``peak``.
    """
    t_fall_s = t_rise_s if t_fall_s is None else t_fall_s
    seg = lambda dur: int(round(dur * 1000.0 / dt_ms))
    pre = np.full(seg(t_pre_s), baseline)
    up = np.linspace(baseline, peak, seg(t_rise_s), endpoint=False)
    down = np.linspace(peak, baseline, seg(t_fall_s), endpoint=False)
    post = np.full(seg(t_post_s), baseline)
    cmd = np.concatenate([pre, up, down, post]).astype(np.float32)
    t = np.arange(len(cmd)) * dt_ms * 1e-3
    return t, cmd


def trapezoid(
    dt_ms: float,
    t_rise_s: float = 5.0,
    t_hold_s: float = 10.0,
    t_fall_s: float = 5.0,
    t_pre_s: float = 2.0,
    t_post_s: float = 2.0,
    baseline: float = 0.0,
    peak: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal command — recruitment/derecruitment + steady-state rates."""
    seg = lambda dur: int(round(dur * 1000.0 / dt_ms))
    cmd = np.concatenate(
        [
            np.full(seg(t_pre_s), baseline),
            np.linspace(baseline, peak, seg(t_rise_s), endpoint=False),
            np.full(seg(t_hold_s), peak),
            np.linspace(peak, baseline, seg(t_fall_s), endpoint=False),
            np.full(seg(t_post_s), baseline),
        ]
    ).astype(np.float32)
    t = np.arange(len(cmd)) * dt_ms * 1e-3
    return t, cmd


def step(
    dt_ms: float,
    t_pre_s: float = 2.0,
    t_on_s: float = 8.0,
    t_post_s: float = 2.0,
    baseline: float = 0.0,
    level: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Step command — F–I style responses and self-sustained-firing tests."""
    seg = lambda dur: int(round(dur * 1000.0 / dt_ms))
    cmd = np.concatenate(
        [
            np.full(seg(t_pre_s), baseline),
            np.full(seg(t_on_s), level),
            np.full(seg(t_post_s), baseline),
        ]
    ).astype(np.float32)
    t = np.arange(len(cmd)) * dt_ms * 1e-3
    return t, cmd


def hold(dt_ms: float, t_total_s: float, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant command."""
    t = time_vector(t_total_s, dt_ms)
    return t, np.full(len(t), level, dtype=np.float32)

"""Per-seed cache for descending-drive Poisson spike trains.

The per-step Python loop that ticks every DD cell once per timestep is the
single dominant cost of the ex03 scripts on long simulations
(approximately ~10 of 15 min on a 32 s run). It's also the same loop on
both backends, so for paired NEURON / Jaxley validation runs it's wasted
work to regenerate it twice for the same seed.

This module provides a tiny load-or-save cache keyed by the parameters
that actually change the realised spike pattern:

    (seed, drive_shape, simulation_time_ms, n_dd, dt_ms)

If two runs share all five values, they share the cached spike-time list.
The cache file is a single ``joblib.dump`` of ``dd_spike_times``: a list
with one entry per DD cell, each entry a Python list of spike times in
milliseconds.

Typical usage at the top of an ex03 script::

    from myogen.utils.dd_cache import dd_cache_path, load_dd_cache, save_dd_cache

    dd_spike_times = None
    if args.seed is not None:
        dd_spike_times = load_dd_cache(
            seed=args.seed, drive_shape=DRIVE_SHAPE,
            simulation_time_ms=simulation_time_ms,
            n_dd=len(descending_drive_pool), dt_ms=dt_ms,
            cache_dir=save_path / "dd_cache",
        )
    if dd_spike_times is None:
        dd_spike_times = [[] for _ in range(len(descending_drive_pool))]
        ...  # original Python loop
        if args.seed is not None:
            save_dd_cache(
                dd_spike_times, seed=args.seed, drive_shape=DRIVE_SHAPE,
                simulation_time_ms=simulation_time_ms,
                n_dd=len(descending_drive_pool), dt_ms=dt_ms,
                cache_dir=save_path / "dd_cache",
            )

The same seed and the same parameters produce the same spike train on both
backends, so for the validation runner the NEURON pass primes the cache and
the Jaxley pass reads it — pairing is guaranteed by construction, not by
hoping the two backends share an RNG path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib


def _fmt_dt(dt_ms: float) -> str:
    """Render dt_ms safely as a filename fragment (e.g. 0.1 -> '0p1')."""
    return f"{dt_ms:g}".replace(".", "p")


def dd_cache_path(
    *,
    seed: int,
    drive_shape: str,
    simulation_time_ms: float,
    n_dd: int,
    dt_ms: float,
    cache_dir: Path,
) -> Path:
    """Deterministic cache filename for one (seed, params) combination.

    A different drive_shape, duration, n_dd or dt_ms yields a different
    filename, so changing any of those invalidates the cache automatically.
    """
    fname = (
        f"dd_cache_seed{int(seed):02d}"
        f"_{drive_shape}"
        f"_dur{int(round(simulation_time_ms))}ms"
        f"_n{int(n_dd)}"
        f"_dt{_fmt_dt(float(dt_ms))}ms.pkl"
    )
    return Path(cache_dir) / fname


def load_dd_cache(
    *,
    seed: int,
    drive_shape: str,
    simulation_time_ms: float,
    n_dd: int,
    dt_ms: float,
    cache_dir: Path,
) -> Optional[list]:
    """Return cached ``dd_spike_times`` or ``None`` if absent.

    Returns ``None`` rather than raising so the caller can fall back to
    generating fresh.
    """
    p = dd_cache_path(
        seed=seed, drive_shape=drive_shape,
        simulation_time_ms=simulation_time_ms,
        n_dd=n_dd, dt_ms=dt_ms, cache_dir=cache_dir,
    )
    if not p.exists():
        return None
    try:
        dd_spike_times = joblib.load(p)
    except Exception as exc:
        print(f"[dd_cache] WARNING: failed to load {p}: {exc}; regenerating.")
        return None
    if not isinstance(dd_spike_times, list) or len(dd_spike_times) != n_dd:
        print(
            f"[dd_cache] WARNING: cache at {p} has {len(dd_spike_times) if isinstance(dd_spike_times, list) else '?'} entries, "
            f"expected {n_dd}; regenerating."
        )
        return None
    print(f"[dd_cache] HIT: loaded DD spikes from {p.name}")
    return dd_spike_times


def save_dd_cache(
    dd_spike_times: list,
    *,
    seed: int,
    drive_shape: str,
    simulation_time_ms: float,
    n_dd: int,
    dt_ms: float,
    cache_dir: Path,
) -> Path:
    """Persist ``dd_spike_times`` under the canonical cache filename."""
    p = dd_cache_path(
        seed=seed, drive_shape=drive_shape,
        simulation_time_ms=simulation_time_ms,
        n_dd=n_dd, dt_ms=dt_ms, cache_dir=cache_dir,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dd_spike_times, p)
    print(f"[dd_cache] MISS: saved DD spikes to {p.name}")
    return p

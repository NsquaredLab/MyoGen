"""Regenerate the NEURON-vs-Jaxley descending-drive validation figure
(poster Figure 3: panels A–D) from the archived data in
``results/dd_validation_seeds/``.

Reconstructed 2026-09-26 — the original generator script was lost; this one
recomputes everything from the saved per-seed pickles:

* ``{neuron,jaxley}_spike_trains_seedNN.pkl`` — neo Block, 100 MN spike trains,
  0–15 s (joblib format).
* ``{neuron,jaxley}_drive_pattern_seedNN.pkl`` — neo AnalogSignal, 10 kHz, pps.

Panels
------
A  Paired per-seed summary (active MNs, total spikes, mean rate per active MN)
   with paired t-tests over seeds. Metrics are recomputed from the raw spike
   trains and cross-checked against the archived ``panel_a_metrics.csv``.
B  Direct spike-time matching (% spikes matched vs coincidence tolerance,
   greedy one-to-one within each MN) for one seed.
C  Smoothed population discharge-rate trajectory (50 ms bins, 300 ms moving
   average) for the same seed, with Pearson r.
D  Population DR correlation scatter (same bins), with r.

Panels B–D use the poster's ORIGINAL single-run dataset — the 32 s triangular
ramp (0 → 65 pps at 16 s → 0) found in the top-level ``../results/`` folder:
``trapezoid_dd_spike_trains.pkl`` (NEURON: 76/100 active, 13 768 spikes) and
``trapezoid_dd_spike_trains_jaxley.pkl`` (Jaxley: 88/100, 14 287) — exactly the
numbers annotated on the poster. (Careful: ``MyoGen/results/`` holds a stale
15 s copy of the Jaxley file; the 32 s pair lives in the repo-parent
``results/``.) If those files are missing, panels B–D fall back to an archived
seed from ``dd_validation_seeds/`` (``--match-seed``).

Usage::

    python scripts/replot_dd_validation.py                    # defaults
    python scripts/replot_dd_validation.py --seeds 0-19       # n=20 panel A
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

HERE = Path(__file__).resolve().parent
if (HERE / "data" / "panel_a_metrics.csv").exists():
    # Self-contained package layout: script + data/ + figures/ in one folder.
    DATA_DIR = HERE / "data"
    ROOT_RESULTS = HERE / "data"
    OUT_DIR = HERE / "figures"
else:
    # Repo layout: script in MyoGen/scripts/.
    REPO = HERE.parent
    DATA_DIR = REPO / "results" / "dd_validation_seeds"
    OUT_DIR = REPO / "results" / "poster_figures_replot"
    # The poster's original triangular-ramp run lives in the repo-parent results/.
    ROOT_RESULTS = REPO.parent / "results"

NEURON_C = "#2CB5B5"  # teal
JAXLEY_C = "#5A64C8"  # slate blue
BG = "#F8F5FB"
INK, MUTED = "#1a1a1a", "#5b6770"

DURATION_S = 15.0
BIN_S = 0.050  # 50 ms bins
SMOOTH_S = 0.300  # 300 ms moving average
TOLERANCES_MS = np.unique(np.round(
    np.append(np.logspace(0, np.log10(50), 30), 5.0), 2))
ANNOT_TOL_MS = 5.0


# --------------------------------------------------------------------------- #
# Data loading / metrics
# --------------------------------------------------------------------------- #

def load_trains_file(path: Path) -> tuple[list[np.ndarray], float]:
    """Per-MN spike-time arrays in seconds + recording duration."""
    blk = joblib.load(path)
    sts = blk.segments[0].spiketrains
    dur = float(sts[0].t_stop.rescale("s").magnitude)
    return [np.asarray(st.rescale("s").magnitude).ravel() for st in sts], dur


def load_trains(backend: str, seed: int, data_dir: Path) -> list[np.ndarray]:
    return load_trains_file(
        data_dir / f"{backend}_spike_trains_seed{seed:02d}.pkl")[0]


def seed_metrics(trains: list[np.ndarray]) -> dict:
    counts = np.array([len(t) for t in trains])
    active = int((counts > 0).sum())
    total = int(counts.sum())
    return {
        "active_MUs": active,
        "total_spikes": total,
        "mean_active_rate": total / active / DURATION_S if active else 0.0,
    }


def compute_metrics(seeds: list[int], data_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for backend in ("neuron", "jaxley"):
            m = seed_metrics(load_trains(backend, seed, data_dir))
            rows.append({"seed": seed, "backend": backend, **m})
    return pd.DataFrame(rows)


def match_spikes(a: np.ndarray, b: np.ndarray, tol_s: float) -> int:
    """Greedy one-to-one matching of two sorted spike-time arrays."""
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol_s:
            n += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n


def matching_curve(neuron: list[np.ndarray], jaxley: list[np.ndarray]):
    tot_n = sum(len(t) for t in neuron)
    tot_j = sum(len(t) for t in jaxley)
    pct_n, pct_j, matches_at = [], [], {}
    for tol in TOLERANCES_MS:
        m = sum(match_spikes(a, b, tol * 1e-3) for a, b in zip(neuron, jaxley))
        pct_n.append(100.0 * m / tot_n)
        pct_j.append(100.0 * m / tot_j)
        matches_at[tol] = m
    return np.array(pct_n), np.array(pct_j), matches_at


def population_dr(
    trains: list[np.ndarray], duration_s: float = DURATION_S
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed population discharge rate (Hz/MN) on 50 ms bins."""
    edges = np.arange(0.0, duration_s + BIN_S, BIN_S)
    counts = np.zeros(len(edges) - 1)
    for t in trains:
        counts += np.histogram(t, bins=edges)[0]
    rate = counts / (BIN_S * len(trains))  # Hz per MN
    w = max(1, int(round(SMOOTH_S / BIN_S)))
    kernel = np.ones(w) / w
    smoothed = np.convolve(rate, kernel, mode="same")
    centers = edges[:-1] + BIN_S / 2
    return centers, smoothed


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color="#c9c4d4", linestyle=":", linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _p_label(p: float) -> str:
    return f"{_stars(p)}  (p < 0.001)" if p < 0.001 else f"{_stars(p)}  (p = {p:.3f})"


def draw_panel_a(fig, subspec, df: pd.DataFrame):
    """Three paired-bar mini-panels with per-seed dots and paired t-tests."""
    metrics = [
        ("active_MUs", "Active MNs", "MUs"),
        ("total_spikes", "Total MN spikes", "Spikes"),
        ("mean_active_rate", "Mean rate / active MN (Hz)", "Hz"),
    ]
    axes = subspec.subgridspec(1, 3, wspace=0.55)
    rng = np.random.default_rng(0)
    for k, (col, title, ylab) in enumerate(metrics):
        ax = fig.add_subplot(axes[0, k])
        _style_ax(ax)
        nv = df[df.backend == "neuron"].sort_values("seed")[col].to_numpy()
        jv = df[df.backend == "jaxley"].sort_values("seed")[col].to_numpy()
        p = stats.ttest_rel(nv, jv).pvalue
        ax.bar([0, 1], [nv.mean(), jv.mean()], width=0.62,
               color=[NEURON_C, JAXLEY_C], zorder=2)
        x0 = 0 + rng.uniform(-0.06, 0.06, len(nv))
        x1 = 1 + rng.uniform(-0.06, 0.06, len(jv))
        for a, b, c, d in zip(x0, nv, x1, jv):
            ax.plot([a, c], [b, d], color="#9aa0a8", lw=0.7, alpha=0.7, zorder=3)
        ax.plot(x0, nv, "o", ms=5, color=NEURON_C, mec="white", mew=0.8, zorder=4)
        ax.plot(x1, jv, "o", ms=5, color=JAXLEY_C, mec="white", mew=0.8, zorder=4)
        top = max(nv.max(), jv.max())
        y = top * 1.06
        ax.plot([0, 0, 1, 1], [y, y * 1.02, y * 1.02, y], color=INK, lw=1.0)
        ax.text(0.5, y * 1.04, _p_label(p), ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.set_ylim(0, top * 1.22)
        ax.set_xticks([0, 1], ["NEURON", "Jaxley"], rotation=20)
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
        ax.set_ylabel(ylab)


def draw_panel_b(ax, pct_n, pct_j, matches_at):
    _style_ax(ax)
    ax.plot(TOLERANCES_MS, pct_n, "-o", ms=4.5, color=NEURON_C,
            label="% NEURON spikes matched")
    ax.plot(TOLERANCES_MS, pct_j, "-o", ms=4.5, color=JAXLEY_C,
            label="% Jaxley spikes matched")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    tol = min(matches_at, key=lambda t: abs(t - ANNOT_TOL_MS))
    ax.text(0.97, 0.15, f"±{tol:g} ms: {matches_at[tol]:,} matches",
            transform=ax.transAxes, ha="right", fontsize=10, color=MUTED,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#d5d0dd", alpha=0.85))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Spike coincidence tolerance (ms)")
    ax.set_ylabel("Matched spikes (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def draw_panel_c(ax, t, dr_n, dr_j, r, drive=None):
    _style_ax(ax)
    if drive is not None:
        # Protocol reference: drive envelope scaled onto the DR axis (shape
        # only — peak pps noted in the label, no second y-scale).
        t_drv, env = drive
        peak = float(np.max(env))
        scaled = env / peak * max(dr_n.max(), dr_j.max()) * 1.04
        ax.plot(t_drv, scaled, color="#b3adc2", lw=1.2, zorder=1,
                label=f"Drive envelope (peak {peak:.0f} pps, scaled)")
    ax.plot(t, dr_n, color=NEURON_C, lw=1.4, label="NEURON", zorder=3)
    ax.plot(t, dr_j, color=JAXLEY_C, lw=1.4, label="Jaxley", zorder=3)
    ax.text(0.03, 0.06,
            f"{BIN_S*1e3:.0f} ms bins, {SMOOTH_S*1e3:.0f} ms smoothing: r = {r:.3f}",
            transform=ax.transAxes, fontsize=10, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#d5d0dd", alpha=0.85))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population DR (Hz/MN)")
    ax.legend(frameon=False, fontsize=9)


def draw_panel_d(ax, dr_n, dr_j, r):
    _style_ax(ax)
    lim = max(dr_n.max(), dr_j.max()) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="#8a93a0", lw=1.0)
    ax.plot(dr_n, dr_j, "o", ms=5, color=JAXLEY_C, alpha=0.55, mec="none")
    ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=13, color=INK, fontweight="bold")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("NEURON population DR (Hz/MN)")
    ax.set_ylabel("Jaxley population DR (Hz/MN)")


def save_all(fig, stem: str):
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, facecolor=fig.get_facecolor())


# --------------------------------------------------------------------------- #

def parse_seeds(spec: str) -> list[int]:
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--seeds", type=str, default="0-9",
                    help="seeds for panel A stats, e.g. '0-9' or '0,3,7'")
    ap.add_argument("--match-seed", type=int, default=0,
                    help="fallback seed for panels B/C/D if the original "
                         "triangular-ramp pickles are absent")
    ap.add_argument("--bcd-neuron", type=Path, default=None)
    ap.add_argument("--bcd-jaxley", type=Path, default=None)
    ap.add_argument("--bcd-drive", type=Path, default=None)
    ap.add_argument("--show-drive", action="store_true",
                    help="overlay the (scaled) drive envelope in panel C")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    OUT_DIR.mkdir(exist_ok=True)

    # ---- metrics (panel A) + cross-check against the archived CSV
    df = compute_metrics(seeds, args.data_dir)
    df.to_csv(OUT_DIR / "panel_a_metrics_recomputed.csv", index=False)
    archived = args.data_dir / "panel_a_metrics.csv"
    if archived.exists():
        old = pd.read_csv(archived)
        merged = df.merge(old, on=["seed", "backend"], suffixes=("", "_old"))
        ok = (
            (merged.active_MUs == merged.active_MUs_old).all()
            and (merged.total_spikes == merged.total_spikes_old).all()
            and np.allclose(merged.mean_active_rate,
                            merged.mean_active_rate_old, atol=1e-3)
        )
        print(f"cross-check vs archived panel_a_metrics.csv: "
              f"{'MATCH' if ok else 'MISMATCH!'} ({len(merged)} rows compared)")

    # ---- panels B/C/D: prefer the poster's original triangular-ramp run
    bcd_n = args.bcd_neuron or (ROOT_RESULTS / "trapezoid_dd_spike_trains.pkl")
    bcd_j = args.bcd_jaxley or (ROOT_RESULTS / "trapezoid_dd_spike_trains_jaxley.pkl")
    bcd_d = args.bcd_drive or (ROOT_RESULTS / "trapezoid_drive_pattern.pkl")
    if bcd_n.exists() and bcd_j.exists():
        tr_n, dur_n = load_trains_file(bcd_n)
        tr_j, dur_j = load_trains_file(bcd_j)
        duration = max(dur_n, dur_j)
        drive_path = bcd_d if bcd_d.exists() else None
        src = f"original triangular-ramp run ({duration:.0f} s)"
    else:
        tr_n = load_trains("neuron", args.match_seed, args.data_dir)
        tr_j = load_trains("jaxley", args.match_seed, args.data_dir)
        duration = DURATION_S
        drive_path = args.data_dir / f"neuron_drive_pattern_seed{args.match_seed:02d}.pkl"
        src = f"archived seed {args.match_seed} (fallback)"
    print(f"panels B–D source: {src} | NEURON "
          f"{sum(1 for x in tr_n if len(x))}/{len(tr_n)} active, "
          f"{sum(len(x) for x in tr_n):,} spikes | Jaxley "
          f"{sum(1 for x in tr_j if len(x))}/{len(tr_j)} active, "
          f"{sum(len(x) for x in tr_j):,} spikes")

    pct_n, pct_j, matches_at = matching_curve(tr_n, tr_j)
    t, dr_n = population_dr(tr_n, duration)
    _, dr_j = population_dr(tr_j, duration)
    r = stats.pearsonr(dr_n, dr_j).statistic

    # drive envelope for the protocol overlay in panel C (off by default,
    # enable with --show-drive; decimated for plotting)
    drive = None
    if args.show_drive and drive_path is not None and drive_path.exists():
        sig = joblib.load(drive_path)
        env = np.asarray(sig.magnitude).ravel()
        fs = float(sig.sampling_rate.rescale("Hz").magnitude)
        dec = max(1, int(fs * 0.005))  # ~5 ms resolution
        drive = (np.arange(len(env))[::dec] / fs, env[::dec])
    print(f"panels B–D: "
          f"±5 ms matches = {matches_at[min(matches_at, key=lambda x: abs(x-5))]:,}, "
          f"population-DR r = {r:.3f}")

    # ---- combined 4-panel figure
    fig = plt.figure(figsize=(16, 10.5), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22,
                  left=0.06, right=0.97, top=0.90, bottom=0.07)
    fig.suptitle(
        f"NEURON vs Jaxley — paired descending-drive validation "
        f"(n = {len(seeds)} seeds)",
        fontsize=17, fontweight="bold", color=INK, x=0.06, ha="left", y=0.965,
    )
    draw_panel_a(fig, gs[0, 0], df)
    axB = fig.add_subplot(gs[0, 1])
    draw_panel_b(axB, pct_n, pct_j, matches_at)
    axB.set_title("Direct spike-time matching", fontsize=12, color=INK,
                  fontweight="bold")
    axC = fig.add_subplot(gs[1, 0])
    draw_panel_c(axC, t, dr_n, dr_j, r, drive=drive)
    axC.set_title("Smoothed population discharge-rate trajectory",
                  fontsize=12, color=INK, fontweight="bold")
    axD = fig.add_subplot(gs[1, 1])
    draw_panel_d(axD, dr_n, dr_j, r)
    axD.set_title("Population DR correlation", fontsize=12, color=INK,
                  fontweight="bold")
    save_all(fig, "figure_dd_validation_4panel")
    plt.close(fig)

    # ---- individual panels (for poster tools)
    figA = plt.figure(figsize=(10.5, 3.6), facecolor=BG)
    gsA = GridSpec(1, 1, figure=figA, left=0.07, right=0.97, top=0.80, bottom=0.18)
    figA.suptitle(f"A. Paired descending-drive summary (n = {len(seeds)} seeds)",
                  fontsize=13, fontweight="bold", color=INK, x=0.07, ha="left")
    draw_panel_a(figA, gsA[0, 0], df)
    save_all(figA, "panel_a_summary")
    plt.close(figA)

    for stem, drawer in [
        ("panel_b_spike_matching", lambda ax: draw_panel_b(ax, pct_n, pct_j, matches_at)),
        ("panel_c_rate_trajectory", lambda ax: draw_panel_c(ax, t, dr_n, dr_j, r, drive=drive)),
        ("panel_d_rate_correlation", lambda ax: draw_panel_d(ax, dr_n, dr_j, r)),
    ]:
        f, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=BG)
        f.subplots_adjust(left=0.13, right=0.96, top=0.94, bottom=0.13)
        drawer(ax)
        save_all(f, stem)
        plt.close(f)

    print(f"saved figures → {OUT_DIR}")


if __name__ == "__main__":
    main()

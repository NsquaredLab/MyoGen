#!/usr/bin/env python3
"""Build Paper Figure 1: cross-simulator validation of the forward twin.

This figure deliberately uses the scientific *hard-spike forward* outputs.
Gradient qualification belongs to the inverse-problem figure, not to the
NEURON-versus-Jaxley forward-validation figure.

Inputs are the immutable paired-validation artifacts and the confirmatory
Phase-1C CaL x NaP grid already saved under ``experiments/twin/results``.
No simulation is rerun by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TWIN_RESULTS = ROOT / "experiments" / "twin" / "results"
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "figure1_forward_twin_validation"

JAXLEY = "#0072B2"
NEURON = "#E69F00"
GNAP1 = "#009E73"
GNAP5 = "#CC79A7"
DRIVE = "#777777"
GRID = "#D9D9D9"
VALID = "#4DAF7C"
UNRESOLVED = "#F0B35A"


def smooth_population_rate(
    spikes: np.ndarray,
    dt_ms: float,
    bin_ms: float = 100.0,
    smooth_ms: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time and smoothed mean firing rate across motor units."""
    n_per_bin = int(round(bin_ms / dt_ms))
    n_bins = spikes.shape[0] // n_per_bin
    trimmed = spikes[: n_bins * n_per_bin]
    counts = trimmed.reshape(n_bins, n_per_bin, spikes.shape[1]).sum(axis=1)
    rate = counts.mean(axis=1) / (bin_ms * 1e-3)

    sigma = max(smooth_ms / bin_ms, 1.0)
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    rate = np.convolve(rate, kernel, mode="same")
    time_s = (np.arange(n_bins) + 0.5) * bin_ms * 1e-3
    return time_s, rate


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13) -> None:
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def mean_metric(points: list[dict], gcal: float, gnap: float, key: str) -> float:
    vals = [
        p["metrics"][key]
        for p in points
        if p["gcal"] == gcal and p["gnap"] == gnap
    ]
    return float(np.mean(vals))


def build_figure(
    firing_artifact: Path,
    jaxley_result: Path,
    neuron_result: Path,
    comparison_json: Path,
    grid_json: Path,
    output_prefix: Path,
) -> dict:
    artifact = joblib.load(firing_artifact)
    jax = joblib.load(jaxley_result)
    nrn = joblib.load(neuron_result)
    comparison = json.loads(comparison_json.read_text())
    grid = json.loads(grid_json.read_text())

    expected_hash = artifact["content_hash"]
    if jax["content_hash"] != expected_hash or nrn["content_hash"] != expected_hash:
        raise ValueError("Paired-validation files do not share one content hash")
    if comparison["comparison"]["content_hash"] != expected_hash:
        raise ValueError("Comparison JSON does not match paired raw outputs")

    dt_ms = float(artifact["dt_ms"])
    dt_s = dt_ms * 1e-3
    pj, pn = jax["pool"], nrn["pool"]
    sj, sn = np.asarray(pj["spikes"]), np.asarray(pn["spikes"])
    points = grid["points"]
    acceptance = grid["acceptance"]

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.35, 8.45), constrained_layout=True)
    outer = fig.add_gridspec(3, 2, height_ratios=(1.04, 1.0, 0.94))

    # ------------------------------------------------------------------ A
    ax_a = fig.add_subplot(outer[0, 0])
    t = np.arange(pj["v"].shape[0]) * dt_s
    zoom = (t >= 7.0) & (t <= 9.0)
    ax_a.plot(t[zoom], np.asarray(pj["v"])[zoom, 0], color=JAXLEY, label="Jaxley")
    ax_a.plot(
        t[zoom],
        np.asarray(pn["v"])[zoom, 0],
        color=NEURON,
        linestyle="--",
        dashes=(3, 1.5),
        label="NEURON",
    )
    ax_a.set(xlabel="Time (s)", ylabel="Soma voltage (mV; 1952 HH frame,\n" r"$V_\mathrm{rest} \approx 0$ mV)")
    ax_a.set_title("Matched firing-regime voltage")
    ax_a.legend(frameon=False, loc="upper left", ncol=2, handlelength=2.5)
    ax_a.grid(True, color=GRID, linewidth=0.45)
    ax_a.set_axisbelow(True)
    pool_cmp = comparison["comparison"]["pool"]
    match = pool_cmp["spike_match_pm5ms"]
    ax_a.text(
        0.98,
        0.04,
        f"{match['jaxley_total']} vs {match['neuron_total']} spikes\n"
        f"{100 * match['matched'] / match['jaxley_total']:.0f}% matched within ±5 ms",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#BBBBBB", "alpha": 0.9},
    )
    panel_label(ax_a, "A")

    # ------------------------------------------------------------------ B
    sub_b = outer[0, 1].subgridspec(2, 1, height_ratios=(0.55, 0.45), hspace=0.05)
    ax_b_rate = fig.add_subplot(sub_b[0])
    ax_b_raster = fig.add_subplot(sub_b[1], sharex=ax_b_rate)
    trj, rj = smooth_population_rate(sj, dt_ms)
    trn, rn = smooth_population_rate(sn, dt_ms)
    ax_b_rate.plot(trj, rj, color=JAXLEY, label="Jaxley")
    ax_b_rate.plot(trn, rn, color=NEURON, linestyle="--", label="NEURON")
    ax_b_rate.set_ylabel("Pool rate\n(Hz/MN)")
    ax_b_rate.set_title("Population trajectory and recruitment")
    ax_b_rate.grid(True, color=GRID, linewidth=0.45)
    ax_b_rate.set_axisbelow(True)
    ax_drive = ax_b_rate.twinx()
    command = np.asarray(artifact["ramp_command_hz"])
    td = np.arange(len(command)) * dt_s
    ax_drive.plot(td[::100], command[::100], color=DRIVE, linewidth=0.8, alpha=0.55)
    ax_drive.set_ylabel("Drive (Hz)", color=DRIVE, labelpad=2)
    ax_drive.tick_params(axis="y", colors=DRIVE)
    ax_drive.set_ylim(0, max(command) * 1.15)
    plt.setp(ax_b_rate.get_xticklabels(), visible=False)

    n_units = sj.shape[1]
    for i in range(n_units):
        tj = np.flatnonzero(sj[:, i]) * dt_s
        tn = np.flatnonzero(sn[:, i]) * dt_s
        ax_b_raster.vlines(tj, i + 0.10, i + 0.35, color=JAXLEY, linewidth=0.55)
        ax_b_raster.vlines(tn, i - 0.35, i - 0.10, color=NEURON, linewidth=0.55)
    ax_b_raster.set(
        xlabel="Time (s)",
        ylabel="MN index",
        yticks=np.arange(n_units),
        ylim=(-0.65, n_units - 0.35),
        xlim=(0, t[-1]),
    )
    ax_b_raster.grid(True, axis="x", color=GRID, linewidth=0.45)
    ax_b_raster.set_axisbelow(True)
    both_active = (sj.sum(axis=0) > 0) & (sn.sum(axis=0) > 0)
    first_j = [np.flatnonzero(sj[:, i])[0] for i in np.flatnonzero(both_active)]
    first_n = [np.flatnonzero(sn[:, i])[0] for i in np.flatnonzero(both_active)]
    same_order = bool(np.array_equal(np.argsort(first_j), np.argsort(first_n)))
    order_text = "identical" if same_order else "DIFFERING"
    ax_b_raster.text(
        0.98,
        0.08,
        f"{int(both_active.sum())}/{n_units} active in both; "
        f"{order_text} recruitment order",
        transform=ax_b_raster.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    ax_b_rate.legend(frameon=False, loc="upper left", ncol=2)
    panel_label(ax_b_rate, "B")

    # ------------------------------------------------------------------ C
    ax_c = fig.add_subplot(outer[1, 0])
    totals_n = np.array([p["metrics"]["total_neuron"] for p in points])
    totals_j = np.array([p["metrics"]["total_jaxley"] for p in points])
    timing = np.array([p["metrics"]["spike_match_pm5ms_fraction"] for p in points])
    gnap_values = np.array([p["gnap"] for p in points])
    lo, hi = 0.0, float(max(totals_n.max(), totals_j.max()) * 1.04)
    ax_c.plot([lo, hi], [lo, hi], color="#555555", linewidth=0.8, zorder=0)
    scatter = None
    for gnap, marker in ((1.0, "o"), (5.0, "s")):
        keep = gnap_values == gnap
        scatter = ax_c.scatter(
            totals_n[keep],
            totals_j[keep],
            c=timing[keep],
            cmap="viridis",
            vmin=0.45,
            vmax=0.95,
            marker=marker,
            s=27,
            edgecolor="white",
            linewidth=0.4,
            label=f"NaP ×{gnap:g}",
            zorder=2,
        )
    ax_c.set(
        xlabel="NEURON total spikes",
        ylabel="Jaxley total spikes",
        xlim=(lo, hi),
        ylim=(lo, hi),
    )
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.set_title("Spike-count and timing agreement across the PIC grid")
    ax_c.grid(True, color=GRID, linewidth=0.45)
    ax_c.set_axisbelow(True)
    ax_c.legend(frameon=False, loc="upper left")
    cb = fig.colorbar(scatter, ax=ax_c, orientation="horizontal", pad=0.12, fraction=0.065)
    cb.set_label("Fraction of spikes matched within ±5 ms", labelpad=1)
    rel = np.abs(totals_j - totals_n) / np.maximum(totals_j, totals_n)
    ax_c.text(
        0.98,
        0.04,
        f"Median count error {100*np.median(rel):.2f}%\n"
        "Timing match declines with PIC strength;\ncounts remain matched",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
    )
    panel_label(ax_c, "C")

    # ------------------------------------------------------------------ D
    ax_d = fig.add_subplot(outer[1, 1])
    gcals = sorted({float(p["gcal"]) for p in points})
    for gnap, color, marker in ((1.0, GNAP1, "o"), (5.0, GNAP5, "s")):
        for backend, ls, fill in (("jaxley", "-", color), ("neuron", "--", "white")):
            vals = [
                mean_metric(points, g, gnap, f"hysteresis_{backend}") for g in gcals
            ]
            ax_d.plot(
                gcals,
                vals,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=3.8,
                markerfacecolor=fill,
                markeredgecolor=color,
                label=f"NaP ×{gnap:g}, {backend.capitalize()}",
            )
    ax_d.axhline(0, color="#777777", linewidth=0.7)
    ax_d.set(
        xlabel="CaL conductance scale",
        ylabel="Derecruitment − recruitment\ndrive (Hz)",
    )
    ax_d.text(
        0.98,
        0.97,
        "negative = PIC hysteresis",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#555555",
    )
    ax_d.set_title("PIC-dependent hysteresis")
    ax_d.grid(True, color=GRID, linewidth=0.45)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.8, handlelength=2)
    panel_label(ax_d, "D")

    # ------------------------------------------------------------------ E
    ax_e = fig.add_subplot(outer[2, 0])
    for gnap, color, marker in ((1.0, GNAP1, "o"), (5.0, GNAP5, "s")):
        for backend, ls, fill in (("jaxley", "-", color), ("neuron", "--", "white")):
            vals = [
                mean_metric(points, g, gnap, f"self_sustained_{backend}") for g in gcals
            ]
            ax_e.plot(
                gcals,
                vals,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=4,
                markerfacecolor=fill,
                markeredgecolor=color,
            )
    ax_e.axvline(6.5, color=GNAP5, linestyle=":", linewidth=0.9)
    ax_e.axvline(7.5, color=GNAP1, linestyle=":", linewidth=0.9)
    ax_e.text(6.45, 8.25, "NaP×5 boundary = 6.5", color=GNAP5, ha="right", va="bottom", fontsize=6.2)
    ax_e.text(7.45, 6.25, "NaP×1 boundary = 7.5", color=GNAP1, ha="right", va="bottom", fontsize=6.2)
    ax_e.set(
        xlabel="CaL conductance scale",
        ylabel="Self-sustained units\n(mean across seeds)",
        ylim=(-0.35, 8.75),
    )
    ax_e.set_title("Matched bistability boundaries")
    ax_e.grid(True, color=GRID, linewidth=0.45)
    ax_e.set_axisbelow(True)
    style_legend = [
        Line2D([0], [0], color="#444444", linestyle="-", label="Jaxley"),
        Line2D([0], [0], color="#444444", linestyle="--", label="NEURON"),
        Line2D([0], [0], color=GNAP1, marker="o", linestyle="", label="NaP ×1"),
        Line2D([0], [0], color=GNAP5, marker="s", linestyle="", label="NaP ×5"),
    ]
    ax_e.legend(handles=style_legend, frameon=False, ncol=2, loc="upper left")
    panel_label(ax_e, "E")

    # ------------------------------------------------------------------ F
    ax_f = fig.add_subplot(outer[2, 1])
    gnaps = [1.0, 5.0]
    validated = np.zeros((len(gnaps), len(gcals)), dtype=int)
    failures_by_condition: dict[tuple[float, float], int] = {}
    failing_cells: list[dict] = []
    for iy, gnap in enumerate(gnaps):
        for ix, gcal in enumerate(gcals):
            cells = [
                a
                for a in grid["acceptance_points"]
                if a["gcal"] == gcal and a["gnap"] == gnap
            ]
            n_fail = 0
            for cell in cells:
                for metric, v in cell["verdicts"].items():
                    if v["class"] == "FAIL":
                        n_fail += 1
                        failing_cells.append(
                            {
                                "gcal": gcal,
                                "gnap": gnap,
                                "seed": cell.get("seed"),
                                "metric": metric,
                                "value": v.get("value"),
                            }
                        )
            failures_by_condition[(gcal, gnap)] = int(n_fail)
            validated[iy, ix] = int(n_fail == 0)
    ax_f.imshow(
        validated,
        cmap=ListedColormap([UNRESOLVED, VALID]),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    for iy, gnap in enumerate(gnaps):
        for ix, gcal in enumerate(gcals):
            ok = bool(validated[iy, ix])
            text = "OK" if ok else f"{failures_by_condition[(gcal, gnap)]}×"
            ax_f.text(ix, iy, text, ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    ax_f.set(
        xticks=np.arange(len(gcals)),
        xticklabels=[f"{g:g}" for g in gcals],
        yticks=np.arange(len(gnaps)),
        yticklabels=[f"NaP ×{g:g}" for g in gnaps],
        xlabel="CaL conductance scale",
    )
    ax_f.set_title(
        "Frozen Phase-1C validated envelope\n"
        f"{acceptance['n_evaluated_metric_cells'] - acceptance['n_failed_metric_cells']} / "
        f"{acceptance['n_evaluated_metric_cells']} metric cells passed "
        f"({100*(1-acceptance['fail_fraction']):.1f}%)"
    )
    ax_f.tick_params(length=0)
    ax_f.set_xticks(np.arange(-0.5, len(gcals), 1), minor=True)
    ax_f.set_yticks(np.arange(-0.5, len(gnaps), 1), minor=True)
    ax_f.grid(which="minor", color="white", linewidth=1.1)
    ax_f.tick_params(which="minor", bottom=False, left=False)
    metric_short = {
        "spike_match_pm5ms_fraction_min": "±5 ms timing match",
        "binned_rate_rmse_hz2": "binned-rate RMSE",
    }
    fail_counts: dict[str, int] = {}
    for cell in failing_cells:
        name = metric_short.get(cell["metric"], cell["metric"])
        fail_counts[name] = fail_counts.get(name, 0) + 1
    breakdown = ", ".join(
        f"{name} ×{n}"
        for name, n in sorted(fail_counts.items(), key=lambda kv: -kv[1])
    )
    ax_f.legend(
        handles=[
            mpl.patches.Patch(color=VALID, label="validated at spec v1"),
            mpl.patches.Patch(
                color=UNRESOLVED,
                label=f"unresolved at spec v1 tolerance\n({breakdown})",
            ),
        ],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=1,
    )
    panel_label(ax_f, "F")

    # Consistent cosmetic cleanup.
    for ax in (ax_a, ax_b_rate, ax_b_raster, ax_c, ax_d, ax_e, ax_f):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    source_summary = {
        "figure": "Figure 1 — Cross-simulator validation of the forward twin",
        "scope": "Hard-spike forward validation only; no surrogate gradients are plotted.",
        "inputs": {
            "paired_artifact": str(firing_artifact.relative_to(ROOT)),
            "paired_content_hash": expected_hash,
            "comparison_json": str(comparison_json.relative_to(ROOT)),
            "pic_grid_json": str(grid_json.relative_to(ROOT)),
            "acceptance_spec_version": acceptance["spec_version"],
        },
        "paired_firing": {
            "jaxley_spikes": match["jaxley_total"],
            "neuron_spikes": match["neuron_total"],
            "matched_pm5ms": match["matched"],
            "matched_fraction_of_jaxley": match["matched"] / match["jaxley_total"],
            "active_jaxley": int((sj.sum(axis=0) > 0).sum()),
            "active_neuron": int((sn.sum(axis=0) > 0).sum()),
            "active_in_both": int(both_active.sum()),
            "recruitment_order_identical": same_order,
        },
        "confirmatory_grid": {
            "n_conditions_x_seeds": len(points),
            "n_evaluated_metric_cells": acceptance["n_evaluated_metric_cells"],
            "n_failed_metric_cells": acceptance["n_failed_metric_cells"],
            "fail_fraction": acceptance["fail_fraction"],
            "failure_policy_ok": acceptance["failure_policy_ok"],
            "bistability_boundary": acceptance["bistability_boundary"],
            "validated_envelope_gcal_by_gnap": acceptance[
                "validated_envelope_gcal_by_gnap"
            ],
            "median_total_spike_relative_error": float(np.median(rel)),
            "failing_metric_cells": failing_cells,
        },
        "conventions": {
            "voltage_frame": "1952 HH convention (V_rest ~ 0 mV, ENa = +120 mV)",
            "hysteresis_sign": (
                "derecruitment drive minus recruitment drive; "
                "negative means derecruitment at lower drive (PIC hysteresis)"
            ),
        },
    }
    output_prefix.with_name(output_prefix.name + "_source_data.json").write_text(
        json.dumps(source_summary, indent=2)
    )
    return source_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path without extension",
    )
    args = parser.parse_args()

    summary = build_figure(
        firing_artifact=TWIN_RESULTS / "val_protocol_firing.pkl",
        jaxley_result=TWIN_RESULTS / "val_jaxley_bac411989ee8d42b.pkl",
        neuron_result=TWIN_RESULTS / "val_neuron_bac411989ee8d42b.pkl",
        comparison_json=TWIN_RESULTS / "val_comparison_bac411989ee8d42b.json",
        grid_json=TWIN_RESULTS / "pic_grid_full.json",
        output_prefix=args.output_prefix,
    )
    print(f"Saved {args.output_prefix.with_suffix('.pdf')}")
    print(f"Saved {args.output_prefix.with_suffix('.png')}")
    print(
        "Evidence: "
        f"{summary['paired_firing']['jaxley_spikes']} vs "
        f"{summary['paired_firing']['neuron_spikes']} paired spikes; "
        f"{summary['confirmatory_grid']['n_failed_metric_cells']} / "
        f"{summary['confirmatory_grid']['n_evaluated_metric_cells']} "
        "confirmatory metric cells failed."
    )


if __name__ == "__main__":
    main()

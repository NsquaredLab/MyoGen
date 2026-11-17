#!/usr/bin/env python3
"""Draft figure: computational scaling and gradient qualification.

Reads ONLY saved evidence (no simulation, no JAX):

* A  ``gpu_benchmark.py`` output: time per call of the hard-spike forward and the
  surrogate value-and-grad, CPU vs GPU, over pool size (GPU stress sizes
  included) and over batch size.
* B  ``gradient_repair.py --tests g2`` (with and without ``--x64``, per device):
  sign of the surrogate gradient against the common-random-number (CRN)
  finite-difference control direction, every variant and coordinate.
* C  ``gradient_repair.py --tests g1 --x64 --regime b_config``: AD of the
  hard-spike mean-voltage objective against a local finite-difference
  convergence region, per timestep x horizon.
* D, E  ``g4_descent.py`` Stage A: fraction of proposed steps the hard-spike
  forward loss accepts, per start and overall, and the surrogate-vs-CRN
  direction cosines.

Writes ``results/figures/figure_computational_gradient_scaling.{png,pdf}`` and
``..._source_data.json`` (every plotted value plus input hashes and manifests).
Uses ``simpetru_figstyle`` when installed, plain Matplotlib otherwise (draft).

    python scripts/plot_computational_gradient_scaling.py \\
        --benchmark .../bench/benchmark.json .../stress/benchmark.json \\
        --g2 "Float32, CPU=..." "Float32, GPU=..." "Float64 (Mixed), CPU=..." ... \\
        --g1 .../gradient_repair_x64_*.json --g4 .../g4_descent_full_*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "figure_computational_gradient_scaling"
CLASSES = ("agree", "wrong", "unresolved")
VERDICT = {"PASS": "Pass", "FAIL": "Failed", "FD_UNCONVERGED": "Unresolved"}
COORDS = ("gCaL", "w_exc")
STARTS = ("S1_both_wrong", "S2_gcal_wrong_only", "S3_w_wrong_only")

#  name: (sweeps, op, x key); the GPU-only stress sizes extend the pool sweep
SCALING = {"pool_forward": (("pool", "stress"), "hard_forward", "n_mn"),
           "pool_gradient": (("pool", "stress"), "surrogate_value_and_grad", "n_mn"),
           "batch_forward": (("batch",), "hard_forward", "batch_size"),
           "batch_gradient": (("batch",), "surrogate_value_and_grad", "batch_size")}


def _load(path: Path) -> tuple[dict, dict]:
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    m = d.get("manifest", {})
    src = m.get("source", m)
    info = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_revision": m.get("git", {}).get("sha") or src.get("git_revision"),
            "git_dirty": m.get("git", {}).get("dirty", src.get("git_dirty")),
            "device": src.get("device"), "precision": src.get("precision")}
    return d, info


def build_source_data(benchmarks, g2: dict, g1, g4) -> dict:
    """Every plotted number, parsed from the evidence files."""
    inputs, panels = {}, {}

    rows = []
    for i, path in enumerate(benchmarks):
        bench, inputs[f"benchmark_{i}"] = _load(path)
        rows += [r for r in bench["rows"] if r["workload"] == "step"]
    panels["A"] = {}
    for name, (sweeps, op, x_key) in SCALING.items():
        panels["A"][name] = {"op": op, "sweeps": sweeps, "x": x_key, "y": "median_s",
                             "series": {dev: sorted((
                                 {"x": r[x_key], "y": r.get("median_s"),
                                  "status": r["status"], "sweep": r["sweep"],
                                  "iqr_s": r.get("iqr_s"),
                                  "first_call_s": r.get("first_call_s"),
                                  "sims_per_s": r.get("sims_per_s"),
                                  "mn_steps_per_s": r.get("mn_steps_per_s"),
                                  "precision_label": r.get("precision_label")}
                                 for r in rows if r["sweep"] in sweeps and r["op"] == op
                                 and r["device"] == dev), key=lambda p: p["x"])
                                 for dev in ("cpu", "gpu")}}

    panels["B"] = {}
    for label, path in g2.items():
        d, inputs[f"g2_{label}"] = _load(path)
        inputs[f"g2_{label}"]["x64"] = d.get("x64")
        counts = d["g2"]["per_coordinate"]
        panels["B"][label] = {
            "x64": d.get("x64"), "n": sum(sum(c.values()) for c in counts.values()),
            "per_coordinate": counts,
            **{c: sum(v[c] for v in counts.values()) for c in CLASSES},
            "variants": [r["variant"] for r in d["g2"]["rows"]],
            "classes": {c: [r["classification"][i] for r in d["g2"]["rows"]]
                        for i, c in enumerate(COORDS)},
            "leaf_dtypes": sorted({r["leaf_dtype"] for r in d["g2"]["rows"]}),
            "nonfinite": sum(not r["grad_finite"] for r in d["g2"]["rows"])}

    d, inputs["g1"] = _load(g1)
    if d.get("regime") != "b_config" or not d.get("x64"):
        raise ValueError(f"{g1} is not the float64 b_config G1 run")
    panels["C"] = [{"dt_ms": r["dt_ms"], "horizon_s": r["horizon_s"],
                    "n_spikes": r["n_spikes"], "verdict": r["verdict"],
                    "class": VERDICT[r["verdict"]], "rel_error": r["rel_rev_vs_fdconv"],
                    "cosine": r["cosine"]} for r in d["g1"]["rows"]]

    d, inputs["g4"] = _load(g4)
    panels["D"] = {"gate": 0.8, "estimators": {}}
    for est, g in d["stageA"]["gate"].items():
        per_start = {}
        for s, v in g["per_start"].items():
            per_start[s] = {"joint": v["joint_acc"] / v["joint_tot"],
                            **{c: v["coord_acc"][c] / v["coord_tot"][c]
                               for c in ("gCaL", "w")},
                            "counts": {"joint": [v["joint_acc"], v["joint_tot"]],
                                       **{c: [v["coord_acc"][c], v["coord_tot"][c]]
                                          for c in ("gCaL", "w")}}}
        panels["D"]["estimators"][est] = {
            "joint": g["joint_accepted_fraction"],
            "gCaL": g["coord_accepted_fraction"]["gCaL"],
            "w": g["coord_accepted_fraction"]["w"],
            "per_start": per_start, "nonfinite": g["nonfinite"],
            "joint_pass": g["joint_pass"]}
    panels["E"] = [{"case": r["case"], "start": s, "cosine": v["cosine_sur_vs_fd"]}
                   for r in d["stageA"]["rows"] for s, v in r["starts"].items()
                   if "cosine_sur_vs_fd" in v]
    return {"inputs": inputs, "panels": panels}


# --------------------------------------------------------------------------- #

def draw(src: dict):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FixedLocator, NullFormatter

    try:
        import simpetru_figstyle as sf
        plt.style.use(["science", "ieee", "simpetru"])
        pal, grey, width = sf.PALETTE, sf.GREYS, sf.COL_2
        letter_kw = dict(fontsize=sf.PANEL_LETTER_SIZE, gid=sf.PANEL_LETTER_GID)
        trim = sf.trim_spines
    except ImportError:                                    # draft fallback
        sf = None
        pal = ("#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")
        grey, width, letter_kw = ("#6d6f72", "#939699"), 7.16, dict(fontsize=12)

        def trim(ax, which=("left", "bottom"), offset=5):
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    P = src["panels"]
    dev_c = {"gpu": pal[0], "cpu": pal[1]}
    cls_c = {"agree": "black", "wrong": pal[2], "unresolved": "white"}
    est_c = {"surrogate": pal[5], "crn_fd": pal[4]}

    def heading(ax, letter, title, dx=-30, dy=12):
        """Letter and title on one line, left-aligned (Affinity layout)."""
        ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(dx, dy),
                    textcoords="offset points", ha="left", va="baseline", **letter_kw)
        ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(dx + 14, dy),
                    textcoords="offset points", ha="left",
                    va="baseline").set_in_layout(False)

    def ends(ax, axis, lo, hi, fmt="{:g}"):
        """End-point ticks only."""
        getattr(ax, f"set_{axis}ticks")([lo, hi], [fmt.format(lo), fmt.format(hi)])

    fig = plt.figure(figsize=(width, 6.4), layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1.15, 1.05])
    row_a = outer[0].subgridspec(1, 4)
    row_b = outer[1].subgridspec(1, 2, width_ratios=[1.55, 1])
    row_c = outer[2].subgridspec(1, 2, width_ratios=[1.9, 1])

    # A -- time per call; failed runs sit on the top edge as crosses
    ylim = (0.15, 300)
    names = [("pool_forward", "Forward, One Pool", "Motor Neurons"),
             ("pool_gradient", "Gradient, One Pool", "Motor Neurons"),
             ("batch_forward", "Forward, Batched", "Pools of 10 MN"),
             ("batch_gradient", "Gradient, Batched", "Pools of 10 MN")]
    axA = []
    for i, (name, title, xlabel) in enumerate(names):
        a = fig.add_subplot(row_a[0, i], sharey=axA[0] if axA else None)
        axA.append(a)
        for dev in ("cpu", "gpu"):
            pts = P["A"][name]["series"][dev]
            ok = [p for p in pts if p["status"] == "ok"]
            a.plot([p["x"] for p in ok], [p["y"] for p in ok], color=dev_c[dev],
                   marker="o", markersize=2.5, lw=1.2)
            bad = [p for p in pts if p["status"] != "ok"]
            if bad:  # an empty unclipped line spans the figure in the tight bbox
                a.plot([p["x"] for p in bad], [ylim[1]] * len(bad), "x",
                       color=dev_c[dev], markersize=4, mew=1.2, clip_on=False)
                first = bad[0]
                a.annotate("OOM" if first["status"] == "oom" else first["status"],
                           (first["x"], ylim[1]), xytext=(-4, 0),
                           textcoords="offset points", ha="right", va="center",
                           color=dev_c[dev])
        a.set(xscale="log", yscale="log", ylim=ylim, xlabel=xlabel)
        a.set_title(title)
        hi = max(p["x"] for s in P["A"][name]["series"].values() for p in s)
        a.xaxis.set_major_locator(FixedLocator([1, hi]))
        a.set_xticklabels(["1", f"{hi:g}"])
        a.xaxis.set_minor_formatter(NullFormatter())
        a.minorticks_off()
        a.yaxis.set_major_locator(FixedLocator([1, 10, 100]))
        a.set_yticklabels(["1", "10", "100"])
        if i:
            a.tick_params(labelleft=False)
        trim(a)
    axA[0].set_ylabel("Time per Call (s)")
    axA[0].text(1.3, 6, "GPU", color=dev_c["gpu"])
    axA[0].text(1.3, 0.24, "CPU", color=dev_c["cpu"], va="center")
    heading(axA[0], "A", "Wall Time per Simulation Call, Float32, Median of 7",
            dy=24)

    # B -- every G2 variant and coordinate, per precision and device
    a = fig.add_subplot(row_b[0, 0])
    runs = list(P["B"])
    variants = P["B"][runs[0]]["variants"]
    groups = [v.split("=")[0] for v in variants]
    xs, x = [], 0.0
    for j, g in enumerate(groups):
        x += (1.6 if g == "dt" else 1.2) if j and g == groups[j - 1] else (1.9 if j else 0.0)
        xs.append(x)
    ys = {}
    for r, run in enumerate(runs):
        for c, coord in enumerate(COORDS):
            y = r * 2.8 + c
            ys[(run, coord)] = y
            for xv, k in zip(xs, P["B"][run]["classes"][coord]):
                a.scatter(xv, y, s=26, color=cls_c[k],
                          edgecolors=grey[0] if k == "unresolved" else cls_c[k],
                          linewidths=0.8, zorder=3)
        e = P["B"][run]
        a.text(xs[-1] + 1.2, r * 2.8 + 0.5, f"{e['agree']} of {e['n']}", va="center")
        a.text(-3.0, r * 2.8 + 0.5, run, ha="right", va="center")
    a.set_yticks([ys[k] for k in ys], ["gCaL" if k[1] == "gCaL" else "w" for k in ys])
    names_short = {"beta": "Surrogate\nSlope", "vth": "Threshold\n(mV)",
                   "dt": "Step\n(ms)", "evshift": "Event\nShift"}
    a.set_xticks(xs, [v.split("=")[1].removesuffix(".0") for v in variants])
    for g in dict.fromkeys(groups):
        gx = [xv for xv, gg in zip(xs, groups) if gg == g]
        a.annotate(names_short[g], (np.mean(gx), 0), xycoords=("data", "axes fraction"),
                   xytext=(0, -24), textcoords="offset points", ha="center", va="top")
    a.set(xlim=(-0.6, xs[-1] + 0.6), ylim=(max(ys.values()) + 0.6, -0.6))
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    a.minorticks_off()
    key = [("Agree", "agree"), ("Wrong", "wrong"), ("Unresolved", "unresolved")]
    for i, (lab, k) in enumerate(key):
        a.scatter([], [], s=26, color=cls_c[k], label=lab,
                  edgecolors=grey[0] if k == "unresolved" else cls_c[k], linewidths=0.8)
    a.legend(loc="upper left", bbox_to_anchor=(1.0, -0.02), frameon=False,
             handletextpad=0.2, borderaxespad=0.0, labelspacing=0.3)
    heading(a, "B", "Surrogate Sign Against the CRN Control", dx=-118)

    # C -- timestep x horizon verdicts, same encoding as B
    a = fig.add_subplot(row_b[0, 1])
    dts = sorted({r["dt_ms"] for r in P["C"]})
    hs = sorted({r["horizon_s"] for r in P["C"]})
    fill = {"Pass": "black", "Failed": pal[2], "Unresolved": "white"}
    for r in P["C"]:
        i, j = dts.index(r["dt_ms"]), hs.index(r["horizon_s"])
        a.add_patch(FancyBboxPatch((j - 0.44, i - 0.42), 0.88, 0.84,
                                   boxstyle="round,pad=0,rounding_size=0.06",
                                   fc=fill[r["class"]],
                                   ec=grey[0] if r["class"] == "Unresolved" else "none",
                                   lw=0.8))
        a.text(j, i, r["class"], ha="center", va="center",
               color="white" if r["class"] != "Unresolved" else "black")
    spikes = {r["horizon_s"]: r["n_spikes"] for r in P["C"]}  # set by the horizon
    a.set(xlim=(-0.5, len(hs) - 0.5), ylim=(len(dts) - 0.5, -0.5),
          xticks=range(len(hs)),
          xticklabels=[f"{h:g}\n{spikes[h]} Spikes" for h in hs],
          yticks=range(len(dts)), yticklabels=[f"{d:g}" for d in dts],
          xlabel="Horizon (s)", ylabel="Step (ms)")
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    a.minorticks_off()
    heading(a, "C", "AD Against Local FD, Mixed x64", dx=-42)

    # D -- accepted fractions: one dot per start, bar = all steps
    a = fig.add_subplot(row_c[0, 0])
    steps = [("joint", "Joint"), ("gCaL", "gCaL Only"), ("w", "w Only")]
    off = {"surrogate": -0.17, "crn_fd": 0.17}
    for gi, (s, _) in enumerate(steps):
        for est, e in P["D"]["estimators"].items():
            x0 = gi + off[est]
            for k, v in enumerate(e["per_start"].values()):
                a.scatter(x0 + (k - 1) * 0.06, v[s], s=12, color=est_c[est],
                          alpha=0.85, linewidths=0, zorder=3)
            a.plot([x0 - 0.12, x0 + 0.12], [e[s]] * 2, color=est_c[est], lw=2.4,
                   solid_capstyle="butt", zorder=4)
    a.axhline(P["D"]["gate"], color=grey[1], ls="--", lw=0.8, zorder=1)
    a.text(len(steps) - 0.45, P["D"]["gate"], "Gate", va="center", ha="left")
    s2 = P["D"]["estimators"]["surrogate"]["per_start"].get(STARTS[1])
    if s2:  # S2 starts with w at its true value, so w-only steps cannot help
        a.annotate("Start S2: w Already True", (2, s2["w"]), xytext=(0, 7),
                   textcoords="offset points", ha="center", va="bottom")
    a.set(xlim=(-0.5, len(steps) - 0.5), ylim=(-0.05, 1.05))
    a.set_xticks(range(len(steps)), [lab for _, lab in steps])
    ends(a, "y", 0, 1)
    a.set_ylabel("Accepted Fraction")
    a.minorticks_off()
    nf = sum(e["nonfinite"] for e in P["D"]["estimators"].values())
    a.text(0.62, 0.50, "Surrogate", color=est_c["surrogate"])
    a.text(0.62, 0.36, "CRN Control", color=est_c["crn_fd"])
    a.text(0.62, 0.18, f"{nf} Non-Finite", color=grey[0])
    trim(a)
    heading(a, "D", "Steps the Hard-Spike Loss Accepts, 3 Starts, 8 Cases", dx=-36)

    # E -- direction cosines, every start shown, median bar
    a = fig.add_subplot(row_c[0, 1])
    cos = np.array([r["cosine"] for r in P["E"]])
    order = np.argsort(cos)
    xj = np.zeros(len(cos))
    for rank, i in enumerate(order):               # small beeswarm by rank
        xj[i] = ((rank % 5) - 2) * 0.07
    a.scatter(xj, cos, s=12, color=est_c["surrogate"], alpha=0.85, linewidths=0,
              zorder=3)
    med = float(np.median(cos))
    a.plot([-0.25, 0.25], [med] * 2, color="black", lw=2.4, solid_capstyle="butt")
    a.text(0.3, med, f"Median {med:.3f}", va="center")
    i = int(np.argmin(cos))
    a.annotate(f"{cos[i]:.2f}", (xj[i], cos[i]), xytext=(8, 0),
               textcoords="offset points", va="center")
    a.axhline(0, color=grey[1], ls="--", lw=0.8, zorder=1)
    a.set(xlim=(-0.5, 1.2), ylim=(-1.08, 1.08), xticks=[])
    ends(a, "y", -1, 1)
    a.set_ylabel("Cosine to CRN Control")
    a.spines["bottom"].set_visible(False)
    a.minorticks_off()
    trim(a, which=("left",))
    heading(a, "E", f"Surrogate Direction, n = {len(cos)}", dx=-36)

    if sf is None:
        fig.suptitle("DRAFT (plain Matplotlib, simpetru_figstyle unavailable)")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", type=Path, nargs="+", required=True,
                    help="gpu_benchmark.py outputs (main run, stress run)")
    ap.add_argument("--g2", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--g1", type=Path, required=True)
    ap.add_argument("--g4", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    g2 = dict(s.split("=", 1) for s in args.g2)
    src = build_source_data(args.benchmark, g2, args.g1, args.g4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}_source_data.json").write_text(json.dumps(src, indent=1))
    fig = draw(src)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}")
    print(f"saved -> {args.out}.{{png,pdf}} + _source_data.json")


if __name__ == "__main__":
    main()

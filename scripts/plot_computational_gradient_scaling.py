#!/usr/bin/env python3
"""Draft figure: computational scaling and gradient qualification.

Reads ONLY saved evidence (no simulation, no JAX):

* A, B  ``gpu_benchmark.py`` output: hard-spike forward and surrogate
  value-and-grad throughput, CPU vs GPU, over pool size (or batch size).
* C     ``gradient_repair.py --tests g2`` (with and without ``--x64``): sign of the
  surrogate gradient against the common-random-number (CRN) finite-difference
  control direction, per coordinate variant: agree / wrong / unresolved.
* D     ``gradient_repair.py --tests g1 --x64 --regime b_config``: AD of the
  hard-spike mean-voltage objective against a local finite-difference
  convergence region, per timestep x horizon: pass / failed / unresolved.
* E, F  ``g4_descent.py`` Stage A: fraction of proposed steps the hard-spike
  forward loss accepts, and the surrogate-vs-CRN direction cosines.

Writes ``results/figures/figure_computational_gradient_scaling.{png,pdf}`` and
``..._source_data.json`` (every plotted value plus input hashes and manifests).
Uses ``simpetru_figstyle`` when installed, plain Matplotlib otherwise (draft).

    python scripts/plot_computational_gradient_scaling.py \\
        --benchmark .../benchmark.json \\
        --g2 Float32=.../gradient_repair_f32_*.json Float64=.../gradient_repair_x64_*.json \\
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


def build_source_data(benchmark, g2: dict, g1, g4, workload="pool") -> dict:
    """Every plotted number, parsed from the evidence files."""
    inputs, panels = {}, {}

    bench, inputs["benchmark"] = _load(benchmark)
    rows = [r for r in bench["rows"] if r["sweep"] == workload and r["workload"] == "step"]
    x_key = "n_mn" if workload == "pool" else "batch_size"
    y_key = "mn_steps_per_s" if workload == "pool" else "sims_per_s"
    for panel, op in (("A", "hard_forward"), ("B", "surrogate_value_and_grad")):
        panels[panel] = {"op": op, "x": x_key, "y": y_key, "series": {
            dev: [{"x": r[x_key], "y": r.get(y_key), "status": r["status"],
                   "median_s": r.get("median_s"), "iqr_s": r.get("iqr_s"),
                   "first_call_s": r.get("first_call_s"),
                   "precision_label": r.get("precision_label")}
                  for r in rows if r["op"] == op and r["device"] == dev]
            for dev in ("cpu", "gpu")}}

    panels["C"] = {}
    for label, path in g2.items():
        d, inputs[f"g2_{label}"] = _load(path)
        inputs[f"g2_{label}"]["x64"] = d.get("x64")
        counts = d["g2"]["per_coordinate"]
        panels["C"][label] = {
            "x64": d.get("x64"), "n": sum(sum(c.values()) for c in counts.values()),
            "per_coordinate": counts,
            **{c: sum(v[c] for v in counts.values()) for c in CLASSES},
            "leaf_dtypes": sorted({r["leaf_dtype"] for r in d["g2"]["rows"]}),
            "nonfinite": sum(not r["grad_finite"] for r in d["g2"]["rows"])}

    d, inputs["g1"] = _load(g1)
    if d.get("regime") != "b_config" or not d.get("x64"):
        raise ValueError(f"{g1} is not the float64 b_config G1 run")
    panels["D"] = [{"dt_ms": r["dt_ms"], "horizon_s": r["horizon_s"],
                    "n_spikes": r["n_spikes"], "verdict": r["verdict"],
                    "class": VERDICT[r["verdict"]], "rel_error": r["rel_rev_vs_fdconv"],
                    "cosine": r["cosine"]} for r in d["g1"]["rows"]]

    d, inputs["g4"] = _load(g4)
    gate = d["stageA"]["gate"]
    panels["E"] = {"gate": 0.8, "estimators": {}}
    for est, g in gate.items():
        tot = {"joint": [0, 0], "gCaL": [0, 0], "w": [0, 0]}
        for s in g["per_start"].values():
            tot["joint"][0] += s["joint_acc"]
            tot["joint"][1] += s["joint_tot"]
            for c in ("gCaL", "w"):
                tot[c][0] += s["coord_acc"][c]
                tot[c][1] += s["coord_tot"][c]
        panels["E"]["estimators"][est] = {
            "joint": g["joint_accepted_fraction"],
            "gCaL": g["coord_accepted_fraction"]["gCaL"],
            "w": g["coord_accepted_fraction"]["w"],
            "counts": tot, "nonfinite": g["nonfinite"],
            "joint_pass": g["joint_pass"]}
    panels["F"] = [{"case": r["case"], "start": s, "cosine": v["cosine_sur_vs_fd"]}
                   for r in d["stageA"]["rows"] for s, v in r["starts"].items()
                   if "cosine_sur_vs_fd" in v]
    return {"inputs": inputs, "workload": workload, "panels": panels}


# --------------------------------------------------------------------------- #

def draw(src: dict):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    try:
        import simpetru_figstyle as sf
        plt.style.use(["science", "ieee", "simpetru"])
        pal, grey, width = sf.PALETTE, sf.GREYS, sf.COL_2
        letter = dict(fontsize=sf.PANEL_LETTER_SIZE, gid=sf.PANEL_LETTER_GID)
        legend, trim = sf.place_legend, sf.trim_spines
    except ImportError:                                    # draft fallback
        sf = None
        pal = ("#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd")
        grey, width, letter = ("#6d6f72", "#939699"), 7.16, dict(fontsize=12)

        def legend(ax, groups=None):
            ax.legend(**({"handles": groups[0]} if groups else {}), frameon=False)

        def trim(ax, which=None):
            pass

    P = src["panels"]
    dev_style = {"gpu": (pal[0], "o", "GPU"), "cpu": (pal[1], "s", "CPU")}
    fig = plt.figure(figsize=(width, 7.2), layout="constrained")
    gs = fig.add_gridspec(3, 2)
    ax = {k: fig.add_subplot(gs[i // 2, i % 2]) for i, k in enumerate("ABCDEF")}

    # A, B -- throughput, shared y range
    ys = [p["y"] for k in "AB" for s in P[k]["series"].values() for p in s if p["y"]]
    lim = (10 ** np.floor(np.log10(min(ys))), 10 ** np.ceil(np.log10(max(ys))))
    titles = {"A": "Hard-Spike Forward", "B": "Surrogate Value and Gradient"}
    for k in "AB":
        a = ax[k]
        for dev, pts in P[k]["series"].items():
            c, m, name = dev_style[dev]
            ok = [p for p in pts if p["status"] == "ok"]
            a.plot([p["x"] for p in ok], [p["y"] for p in ok], marker=m, color=c,
                   markersize=3.5, label=name)
            bad = [p for p in pts if p["status"] != "ok"]
            if bad:  # an empty unclipped line spans the figure in the tight bbox
                a.plot([p["x"] for p in bad], [lim[0]] * len(bad), "x", color=c,
                       markersize=5, clip_on=False)
        a.set(xscale="log", yscale="log", ylim=lim, title=titles[k],
              xlabel="Motor Neurons" if src["workload"] == "pool" else "Batch Size")
        a.set_ylabel("Throughput (MN Steps/s, Float32)" if src["workload"] == "pool"
                     else "Simulations (1/s, Float32)")
        a.minorticks_off()
    handles = [Line2D([], [], color=dev_style[d][0], marker=dev_style[d][1],
                      markersize=3.5, label=dev_style[d][2]) for d in ("gpu", "cpu")]
    handles.append(Line2D([], [], color=grey[0], marker="x", ls="", label="Not Completed"))

    # C -- stacked agree / wrong / unresolved
    a = ax["C"]
    labels = list(P["C"])
    colors = {"agree": pal[0], "wrong": pal[2], "unresolved": grey[1]}
    left = np.zeros(len(labels))
    for c in CLASSES:
        v = np.array([P["C"][l][c] for l in labels])
        a.barh(labels, v, left=left, color=colors[c], label=c.capitalize(), height=0.6)
        for i, (x0, n) in enumerate(zip(left, v)):
            if n >= 3:
                a.text(x0 + n / 2, i, str(n), ha="center", va="center", color="white")
            elif n:
                a.text(x0 + n / 2, i - 0.32, str(n), ha="center", va="bottom")
        left += v
    a.set(xlabel="Coordinate Variants", title="Surrogate Sign vs CRN Control",
          xlim=(0, max(left)))
    a.invert_yaxis()

    # D -- timestep x horizon verdicts
    a = ax["D"]
    dts = sorted({r["dt_ms"] for r in P["D"]})
    hs = sorted({r["horizon_s"] for r in P["D"]})
    code = {"Pass": 0, "Failed": 1, "Unresolved": 2}
    grid = np.full((len(dts), len(hs)), np.nan)
    for r in P["D"]:
        i, j = dts.index(r["dt_ms"]), hs.index(r["horizon_s"])
        grid[i, j] = code[r["class"]]
        a.text(j, i, f"{r['class']}\n{r['n_spikes']} Spikes", ha="center",
               va="center", color="white")
    a.imshow(grid, cmap=ListedColormap([pal[0], pal[2], grey[1]]), vmin=-0.5,
             vmax=2.5, aspect="auto")
    a.set(xticks=range(len(hs)), xticklabels=[f"{h:g}" for h in hs],
          yticks=range(len(dts)), yticklabels=[f"{d:g}" for d in dts],
          xlabel="Horizon (s)", ylabel="Timestep (ms)",
          title="AD vs Local FD (x64 Flag, Mixed)")
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)

    # E -- accepted-step fractions, hard-spike forward judges
    a = ax["E"]
    est_style = {"surrogate": (pal[0], "Surrogate"), "crn_fd": (grey[0], "CRN Control")}
    steps = [("joint", "Joint"), ("gCaL", "gCaL Only"), ("w", "w Only")]
    y = np.arange(len(steps))
    for n, (est, (c, name)) in enumerate(est_style.items()):
        e = P["E"]["estimators"][est]
        a.barh(y + (n - 0.5) * 0.38, [e[s] for s, _ in steps], height=0.36, color=c,
               label=f"{name} ({e['nonfinite']} Non-Finite)")
    a.axvline(P["E"]["gate"], color=grey[1], ls="--", lw=1)
    a.text(P["E"]["gate"] + 0.02, 2.72, f"Gate {P['E']['gate']:g}", ha="left",
           va="center")
    a.set(yticks=y, yticklabels=[s for _, s in steps], xlim=(0, 1), ylim=(2.9, -0.5),
          xlabel="Accepted Step Fraction", title="Steps Accepted by Hard-Spike Loss")

    # F -- direction cosines, every start shown
    a = ax["F"]
    cos = np.array([r["cosine"] for r in P["F"]])
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(cos))
    a.scatter(cos, jitter, s=12, facecolors="none", edgecolors=pal[0], linewidths=0.8)
    a.plot([np.median(cos)] * 2, [-0.36, -0.2], color=grey[0], lw=1.6)
    a.text(np.median(cos), -0.28, f"Median {np.median(cos):.2f}  ", ha="right",
           va="center")
    i = int(np.argmin(cos))
    a.annotate(f"{cos[i]:.2f}", (cos[i], jitter[i]), xytext=(0, 8),
               textcoords="offset points", ha="center")
    a.set(xlim=(-1.05, 1.05), ylim=(-0.4, 0.4), yticks=[], xlabel="Direction Cosine",
          title=f"Surrogate vs CRN Control (n = {len(cos)})")
    a.spines["left"].set_visible(False)

    for k, a in ax.items():
        a.text(-0.2, 1.04, k, transform=a.transAxes, va="bottom", ha="left", **letter)
    legend(ax["A"], groups=[handles])
    legend(ax["C"])
    legend(ax["E"])
    for k in "ABCE":
        trim(ax[k])
    trim(ax["F"], which=("bottom",))
    if sf is None:
        fig.suptitle("DRAFT (plain Matplotlib, simpetru_figstyle unavailable)")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--g2", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--g1", type=Path, required=True)
    ap.add_argument("--g4", type=Path, required=True)
    ap.add_argument("--workload", choices=["pool", "batch"], default="pool")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    g2 = dict(s.split("=", 1) for s in args.g2)
    src = build_source_data(args.benchmark, g2, args.g1, args.g4, args.workload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}_source_data.json").write_text(json.dumps(src, indent=1))
    fig = draw(src)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"saved -> {args.out}.{{png,pdf}} + _source_data.json")


if __name__ == "__main__":
    main()

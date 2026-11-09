"""Confirmatory NEURON–Jaxley PIC validation grid (Phase 1, work package 2).

Grid: gCaL × {0, 0.5, 1, 2, 4, 6.5, 7.5} crossed with gNaP × {1, 5}, on
PREVIOUSLY UNUSED seeds (confirmatory — the acceptance thresholds in
``configs/phase1_acceptance_v1.yaml`` were frozen from exploratory data
before this script ran). Identical frozen artifacts feed both simulators:
same cells, thresholds, connectivity, event matrices, synaptic site/kinetics
/reversals/weights (0.15 µS firing regime), zero delays, backward Euler.

Protocol: slow triangular ramp (7.5 s rise/fall, 60 Hz peak) — the ΔF/PIC
paradigm. Per (condition × seed) the paired metrics are computed and the
FROZEN acceptance spec classifies every metric PASS / FAIL /
OUTSIDE_VALIDATED_ENVELOPE / NOT_EVALUABLE; the maximal passing CaL range
per NaP level is the candidate validated envelope.

Timestep-refinement references (dt 0.1 → 0.05, Jaxley) are run for two
sentinel conditions to anchor E_numerical.

Process isolation: Jaxley runs in-parent; all NEURON conditions for a seed
run in ONE subprocess (JAX and NEURON cannot share a process), reusing one
pool and rescaling conductances between conditions.

Usage::

    python pic_grid.py                 # full confirmatory run (~30 min CPU)
    python pic_grid.py --seeds 101 --gcal 0 1 6.5 --dur-scale 0.5   # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import observables as obs
import validation_protocol as vp
from acceptance import evaluate_grid, load_spec

RESULTS = Path(__file__).parent / "results"
EVIDENCE = Path(__file__).parent / "evidence"

GCAL_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 6.5, 7.5]
GNAP_GRID = [1.0, 5.0]
DT_REF_CONDITIONS = [(1.0, 1.0), (6.5, 5.0)]

_NEURON_CODE = r"""
import sys, joblib
import numpy as np
import validation_protocol as vp

art_path, out_path = sys.argv[1], sys.argv[2]
conds = [tuple(map(float, c.split(","))) for c in sys.argv[3].split(";")]
art = vp.load_protocol(art_path)
dt = art["dt_ms"]; n_mn = art["n_mn"]; syn_p = art["synapse"]

h, pool = vp._neuron_pool(art)   # art cell scales are 1,1 — bases below
base = [(c.dend[0].gcaLbar_caL, c.soma.gnapbar_napp) for c in pool]

keep, vstims, _ = [], [], None
for events, conn, syn_idx, w, e in (
        (art["exc_events"], art["exc_conn"], 0, syn_p["w_exc_uS"], syn_p["e_exc_mV"]),
        (art["inh_events"], art["inh_conn"], 1, syn_p["w_inh_uS"], syn_p["e_inh_mV"])):
    stims = []
    for s in range(events.shape[1]):
        tv = h.Vector((np.flatnonzero(events[:, s]) * dt).astype(float))
        vs = h.VecStim(); vs.play(tv); stims.append((vs, tv))
    vstims.append(stims)
    for i, c in enumerate(pool):
        syn = c.synapse__list[syn_idx]
        syn.tau1 = syn_p["tau_rise_ms"]; syn.tau2 = syn_p["tau_decay_ms"]; syn.e = e
        for s in np.flatnonzero(conn[i]):
            nc = h.NetCon(stims[s][0], syn); nc.weight[0] = w; nc.delay = 0.0
            keep.append(nc)
rec_keep, vrec, dvrec, spk = vp._neuron_record(h, pool)
n_steps = art["exc_events"].shape[0]

out = {}
for gcal, gnap in conds:
    for c, (b_cal, b_nap) in zip(pool, base):
        c.dend[0].gcaLbar_caL = b_cal * gcal
        c.soma.gnapbar_napp = b_nap * gnap
    for sv in spk:
        sv.resize(0)
    vp._neuron_advance(h, pool, dt, n_steps)
    res = vp._collect(vrec, dvrec, spk, dt, n_steps, n_mn)
    out[(gcal, gnap)] = res["spikes"]
    print(f"NEURON ({gcal},{gnap}): {int(res['spikes'].sum())} spikes", flush=True)
joblib.dump(out, out_path)
"""


def jaxley_spikes(twin, art, gcal, gnap):
    import jax.numpy as jnp

    th = twin.default_theta()
    th["gCaL_scale"] = jnp.float32(gcal)
    th["gNaP_scale"] = jnp.float32(gnap)
    out = twin.simulate_events(
        th, art["exc_events"], inh_events=art["inh_events"],
        exc_connectivity=art["exc_conn"], inh_connectivity=art["inh_conn"],
        spike_mode="hard")
    return np.asarray(out["spikes"])


def paired_metrics(sj, sn, cmd, dt_ms, t_ramp_end_s):
    from deliverable_b import binned_rates_jnp
    import jax.numpy as jnp

    rj = np.asarray(binned_rates_jnp(jnp.asarray(sj), dt_ms))
    rn = np.asarray(binned_rates_jnp(jnp.asarray(sn), dt_ms))
    aj = set(np.flatnonzero(sj.sum(axis=0) > 0).tolist())
    an = set(np.flatnonzero(sn.sum(axis=0) > 0).tolist())
    tj, tn = int(sj.sum()), int(sn.sum())
    matched = sum(vp._match_spikes(
        np.flatnonzero(sj[:, i]) * dt_ms * 1e-3,
        np.flatnonzero(sn[:, i]) * dt_ms * 1e-3, 5e-3)
        for i in range(sj.shape[1]))
    t = np.arange(len(cmd)) * dt_ms * 1e-3
    post = t >= t_ramp_end_s
    last = t >= (t[-1] - 0.5)

    def side_stats(sp):
        hyst = obs.hysteresis_index(cmd, sp, dt_ms)
        hmean = float(np.nanmean(hyst)) if np.any(~np.isnan(hyst)) else None
        df = obs.delta_f_pairs(sp, dt_ms)
        n_act = int((sp.sum(axis=0) > 0).sum())
        possible = max(n_act * (n_act - 1) // 2, 1)
        return {
            "hyst": hmean,
            "df_mean": df.mean,
            "df_pairs": int(len(df.delta_f)),
            "df_fraction": len(df.delta_f) / possible,
            "self_sustained": int((sp[post].sum(axis=0) >= 5).sum()),
            "censored": int((sp[last].sum(axis=0) > 0).sum()),
        }

    j, n = side_stats(sj), side_stats(sn)
    hyst_diff = (abs(j["hyst"] - n["hyst"])
                 if j["hyst"] is not None and n["hyst"] is not None else None)
    df_diff = (abs(j["df_mean"] - n["df_mean"])
               if np.isfinite(j["df_mean"]) and np.isfinite(n["df_mean"])
               else None)
    return {
        "binned_rate_rmse_hz2": float(((rj - rn) ** 2).mean()),
        "active_unit_mismatch": len(aj.symmetric_difference(an)),
        "active_jaxley": len(aj), "active_neuron": len(an),
        "total_spike_rel_diff": abs(tj - tn) / max(tj, tn, 1),
        "total_jaxley": tj, "total_neuron": tn,
        "spike_match_pm5ms_fraction": matched / max(tj, tn, 1),
        "spike_total_max": max(tj, tn),
        "hysteresis_mean_diff_hz": hyst_diff,
        "hysteresis_jaxley": j["hyst"], "hysteresis_neuron": n["hyst"],
        "self_sustained_mismatch": abs(j["self_sustained"] - n["self_sustained"]),
        "self_sustained_jaxley": j["self_sustained"],
        "self_sustained_neuron": n["self_sustained"],
        "after_discharge_censored_jaxley": j["censored"],
        "after_discharge_censored_neuron": n["censored"],
        "delta_f_mean_diff_hz": df_diff,
        "delta_f_pairs_jaxley": j["df_pairs"],
        "delta_f_pairs_neuron": n["df_pairs"],
        "delta_f_eligible_fraction_diff": abs(j["df_fraction"] - n["df_fraction"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 202],
                    help="confirmatory seeds — must be previously unused")
    ap.add_argument("--gcal", type=float, nargs="+", default=GCAL_GRID)
    ap.add_argument("--gnap", type=float, nargs="+", default=GNAP_GRID)
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--dur-scale", type=float, default=1.0)
    ap.add_argument("--skip-dt-ref", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    tag = ("full" if (args.dur_scale >= 1.0 and len(args.seeds) >= 2
                      and len(args.gcal) >= 7) else "smoke")
    stem = f"pic_grid_{tag}"
    t0_all = time.time()
    spec = load_spec(1)
    conds = [(g, n) for n in args.gnap for g in args.gcal]
    cond_str = ";".join(f"{g},{n}" for g, n in conds)

    points = []
    dt_refs = {}
    for seed in args.seeds:
        t0 = time.time()
        art = vp.make_protocol(
            n_mn=args.n_mn, t_rise_s=7.5 * args.dur_scale,
            t_pre_s=1.0, t_post_s=3.0, peak_hz=60.0,
            w_exc_uS=0.15, step_currents_nA=(), seed=seed)
        art_path = RESULTS / f"pic_grid_art_seed{seed}.pkl"
        vp.save_protocol(art, art_path)
        n = len(art["ramp_command_hz"])
        t_ramp_end = 1.0 + 2 * 7.5 * args.dur_scale
        print(f"[seed {seed}] artifact {art['content_hash']} "
              f"({time.time()-t0:.0f}s)", flush=True)

        # NEURON: all conditions in one subprocess
        t0 = time.time()
        npkl = RESULTS / f"pic_grid_neuron_seed{seed}.pkl"
        proc = subprocess.run(
            [sys.executable, "-u", "-c", _NEURON_CODE, str(art_path),
             str(npkl), cond_str],
            cwd=Path(__file__).parent, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"neuron subprocess failed:\n{proc.stdout}"
                               f"\n{proc.stderr}")
        neuron_out = joblib.load(npkl)
        print(f"[seed {seed}] NEURON {len(conds)} conditions "
              f"({time.time()-t0:.0f}s)", flush=True)

        # Jaxley in-parent, one twin per seed
        twin = vp._twin_from_artifact(art)
        for gcal, gnap in conds:
            t0 = time.time()
            sj = jaxley_spikes(twin, art, gcal, gnap)
            sn = neuron_out[(gcal, gnap)]
            m = paired_metrics(sj, sn, art["ramp_command_hz"], art["dt_ms"],
                               t_ramp_end)
            points.append({"gcal": gcal, "gnap": gnap, "seed": seed,
                           "metrics": m})
            print(f"  ({gcal:4.1f},{gnap:3.1f}) seed{seed}: "
                  f"spk j{m['total_jaxley']}/n{m['total_neuron']} "
                  f"match {m['spike_match_pm5ms_fraction']:.2f} "
                  f"rateMSE {m['binned_rate_rmse_hz2']:.2f} "
                  f"selfsust j{m['self_sustained_jaxley']}/"
                  f"n{m['self_sustained_neuron']} ({time.time()-t0:.0f}s)",
                  flush=True)

        if not args.skip_dt_ref and seed == args.seeds[0]:
            from deliverable_b import binned_rates_jnp
            import jax.numpy as jnp
            from poolbuild import PoolTwin, TwinSpec

            for gcal, gnap in DT_REF_CONDITIONS:
                if (gcal, gnap) not in conds:
                    continue
                twin_h = PoolTwin(TwinSpec(
                    n_mn=args.n_mn, n_dd=100,
                    n_inh=art["inh_events"].shape[1], dt_ms=0.05, seed=seed,
                    syn_site=art["synapse"]["site"],
                    live_driving_force=True,
                    syn_tau_rise_ms=art["synapse"]["tau_rise_ms"],
                    tau_syn_ms=art["synapse"]["tau_decay_ms"],
                    e_exc_mV=art["synapse"]["e_exc_mV"],
                    e_inh_mV=art["synapse"]["e_inh_mV"],
                    w_exc_uS=art["synapse"]["w_exc_uS"],
                    w_inh_uS=art["synapse"]["w_inh_uS"]))
                fine = np.zeros((n * 2, art["exc_events"].shape[1]),
                                dtype=np.float32)
                fine[::2] = np.asarray(art["exc_events"])
                th = twin_h.default_theta()
                th["gCaL_scale"] = jnp.float32(gcal)
                th["gNaP_scale"] = jnp.float32(gnap)
                sp_f = np.asarray(twin_h.simulate_events(
                    th, fine, exc_connectivity=art["exc_conn"],
                    spike_mode="hard")["spikes"])
                r_f = np.asarray(binned_rates_jnp(jnp.asarray(sp_f), 0.05))
                sj = jaxley_spikes(twin, art, gcal, gnap)
                r_c = np.asarray(binned_rates_jnp(jnp.asarray(sj), 0.1))
                m = min(len(r_f), len(r_c))
                dt_refs[f"{gcal},{gnap}"] = float(
                    ((r_f[:m] - r_c[:m]) ** 2).mean())
                print(f"  dt-ref ({gcal},{gnap}): E_numerical "
                      f"{dt_refs[f'{gcal},{gnap}']:.3f} Hz²", flush=True)

    # --------------------------------------------------- acceptance + report
    accept = evaluate_grid(points, spec)
    report = {
        "manifest": {
            **__import__("sensitivity_checks")._manifest(),
            "pic_grid_sha256_16": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest()[:16],
            "args": vars(args), "run_tag": tag,
            "acceptance_spec_version": spec["version"],
        },
        "dt_refinement_E_numerical_hz2": dt_refs,
        "points": points,
        "acceptance": {k: v for k, v in accept.items() if k != "points"},
        "acceptance_points": accept["points"],
        "wall_s": round(time.time() - t0_all, 1),
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"{stem}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"{stem}_{stamp}.json").write_text(payload)

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, gnap in zip(axes, sorted({p["gnap"] for p in points})):
        rows = [p for p in points if p["gnap"] == gnap]
        gcals = sorted({p["gcal"] for p in rows})
        for backend, c, mk in (("jaxley", "#0072B2", "o"),
                               ("neuron", "#E69F00", "s")):
            ys = [np.mean([p["metrics"][f"self_sustained_{backend}"]
                           for p in rows if p["gcal"] == g]) for g in gcals]
            ax.plot(gcals, ys, mk + "-", color=c, label=backend, mfc="none")
        b = accept["bistability_boundary"][gnap]
        ax.set_title(f"gNaP ×{gnap:g} — self-sustained units; boundary "
                     f"j:{b['jaxley_first_bistable_gcal']} "
                     f"n:{b['neuron_first_bistable_gcal']} [{b['class']}]",
                     fontsize=9)
        ax.set_xlabel("gCaL scale")
        ax.set_ylabel("self-sustained units (mean over seeds)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, color="#e3e3e3", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}_boundary.png", dpi=150)

    print("\n=== acceptance summary (spec v%d, FROZEN) ===" % spec["version"])
    for k in ("n_points", "n_evaluated_metric_cells", "n_failed_metric_cells",
              "fail_fraction", "failure_policy_ok"):
        print(f"  {k}: {accept[k]}")
    print(f"  bistability boundaries: "
          f"{json.dumps(accept['bistability_boundary'], default=float)}")
    print(f"  validated envelope (gCaL range per gNaP): "
          f"{accept['validated_envelope_gcal_by_gnap']}")
    print(f"saved → {RESULTS / (stem + '.json')} (+ evidence copy, figure)")


if __name__ == "__main__":
    main()

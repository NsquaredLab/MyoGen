"""Phase-3 ladder step 2 — CaL + NaP profiles (before any R2 optimizer).

Revised roadmap §7 / review work-package 3: expand identifiability ONE
parameter at a time. This adds gNaP to the Deliverable-B problem:

* Unknowns under profiling: gCaL_scale, gNaP_scale.
* Nuisance reoptimized everywhere: w_exc (effective excitatory amplitude).
* Fixed and known: inhibition (absent), connectivity, drive shape AND the
  frozen event realizations (conditional analysis — same scope as B).
* Observable: binned rates (50 ms / 300 ms); hard forward only (gradients
  remain unqualified).
* EPS per replicate from dt-halving at the truth (same rule as B/C).

PREDECLARED RESOLUTION (per the frozen acceptance spec
``downstream_resolution_declaration``): a parameter "advances" to the R2
optimizer stage only if its sampled compatibility interval lies within
±20 % of the truth in every replicate. The question is NOT whether the
minimum sits at the exact truth (it does by construction — same-model
noiseless data with truth on-axis) but whether the ≤EPS level set is
narrower than the declared resolution. If CaL and NaP are inseparable, the
honest fallback is a combined "PIC strength" parameter — reported here via
the 2-D (CaL × NaP) joint compatibility set at the profiled w_exc.

Grid: 3-D dense (g × nap × w), truth inserted on every axis, profile of
each parameter = min over the other two (shared candidate grid — the
nesting lesson from Deliverable C applies: all minimizations use the same
evaluated set).

Usage::

    python r2a_profiles.py                 # full (~25 min CPU)
    python r2a_profiles.py --dense-g 5 --dense-nap 4 --dense-w 5 \\
        --replicates 1 --dur-scale 0.5     # smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from sensitivity_checks import _manifest, _smoothed_rates_jax
from deliverable_b import (
    EPS_FLOOR, RESULTS, EVIDENCE, axis_with_truth, make_commands,
    build_twin_and_events, numerical_epsilon_spikes, binned_rates_jnp,
    _sha16,
)

OKABE = {"cal": "#0072B2", "nap": "#E69F00", "truth": "#000000"}
GRIDC = "#e3e3e3"
RESOLUTION_REL = 0.20  # predeclared (acceptance spec v1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--true", type=float, nargs=3, default=[1.6, 0.7, 0.0105],
                    metavar=("GCAL", "GNAP", "WEXC"))
    ap.add_argument("--dense-g", type=int, default=13)
    ap.add_argument("--dense-nap", type=int, default=9)
    ap.add_argument("--dense-w", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--dur-scale", type=float, default=1.0)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    tag = ("full" if (args.dense_g >= 11 and args.replicates >= 2
                      and args.dur_scale >= 1.0 and args.n_mn >= 8)
           else "smoke")
    stem = f"r2a_profiles_{tag}"
    t_wall0 = time.time()
    dt_ms = 0.1
    theta_true = np.array(args.true)
    fit_protos = ["fit_ramp", "fit_trap"]
    commands = make_commands(dt_ms, args.dur_scale)

    g_axis = axis_with_truth(0.25, 4.0, args.dense_g, theta_true[0])
    nap_axis = axis_with_truth(0.25, 4.0, args.dense_nap, theta_true[1])
    w_axis = axis_with_truth(0.0045, 0.018, args.dense_w, theta_true[2])
    gi = int(np.argmin(np.abs(g_axis - theta_true[0])))
    ni = int(np.argmin(np.abs(nap_axis - theta_true[1])))

    def observable_fn_3(twin, events):
        def fn(gcal, gnap, w, proto):
            th = twin.default_theta()
            th["gCaL_scale"] = jnp.float32(gcal)
            th["gNaP_scale"] = jnp.float32(gnap)
            th["w_exc"] = jnp.float32(w)
            out = twin.simulate_events(th, events[proto], spike_mode="hard")
            return binned_rates_jnp(out["spikes"], dt_ms)
        return fn

    reps = []
    for r in range(args.replicates):
        conn_seed, ev_off = r, 10 * r
        twin, events = build_twin_and_events(
            args.n_mn, conn_seed, ev_off, commands, dt_ms)
        fn = observable_fn_3(twin, events)
        obs = {p: fn(*theta_true, p) for p in commands}

        def loss(gcal, gnap, w):
            tot = 0.0
            for p in fit_protos:
                tot += float(jnp.mean((fn(gcal, gnap, w, p) - obs[p]) ** 2))
            return tot / len(fit_protos)

        assert loss(*theta_true) == 0.0, "truth loss must be exactly 0"

        t0 = time.time()
        L = np.zeros((len(g_axis), len(nap_axis), len(w_axis)))
        for i, g in enumerate(g_axis):
            for j, nap in enumerate(nap_axis):
                for k, w in enumerate(w_axis):
                    L[i, j, k] = loss(g, nap, w)
            print(f"[rep{r}] g-slice {i+1}/{len(g_axis)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # EPS per replicate (dt-halving at truth, all units). Computed inline
        # because the shared helper applies only (gCaL, w) — here the truth
        # also sets gNaP.
        terms = []
        from poolbuild import PoolTwin, TwinSpec
        twin_h = PoolTwin(TwinSpec(n_mn=args.n_mn, n_dd=100, n_inh=5,
                                   dt_ms=0.05, seed=conn_seed))
        for p in fit_protos:
            ev = np.asarray(events[p])
            fine_ev = np.zeros((ev.shape[0] * 2, ev.shape[1]),
                               dtype=np.float32)
            fine_ev[::2] = ev
            th = twin_h.default_theta()
            th["gCaL_scale"] = jnp.float32(theta_true[0])
            th["gNaP_scale"] = jnp.float32(theta_true[1])
            th["w_exc"] = jnp.float32(theta_true[2])
            sp_f = twin_h.simulate_events(
                th, fine_ev, exc_connectivity=twin._exc_mat,
                spike_mode="hard")["spikes"]
            rf = np.asarray(binned_rates_jnp(sp_f, 0.05))
            rc = np.asarray(obs[p])
            m = min(len(rf), len(rc))
            terms.append(float(np.mean((rf[:m] - rc[:m]) ** 2)))
        EPS = max(float(np.mean(terms)), EPS_FLOOR)

        # profiles: min over the other two axes (one shared evaluated grid)
        prof_g = L.min(axis=(1, 2))
        prof_nap = L.min(axis=(0, 2))
        assert prof_g[gi] == 0.0 and prof_nap[ni] == 0.0

        def interval(curve, axis):
            within = curve <= EPS
            return ([float(axis[within].min()), float(axis[within].max())]
                    if within.any() else None)

        iv_g = interval(prof_g, g_axis)
        iv_nap = interval(prof_nap, nap_axis)

        def advances(iv, truth):
            if iv is None:
                return False
            return (iv[0] >= truth * (1 - RESOLUTION_REL)
                    and iv[1] <= truth * (1 + RESOLUTION_REL))

        # 2-D joint compatibility set at profiled w (PIC-strength fallback)
        L2 = L.min(axis=2)
        joint = [[float(g_axis[i]), float(nap_axis[j])]
                 for i in range(len(g_axis)) for j in range(len(nap_axis))
                 if L2[i, j] <= EPS]

        reps.append({
            "conn_seed": conn_seed, "event_offset": ev_off, "eps": EPS,
            "profile_gcal": prof_g.tolist(),
            "profile_gnap": prof_nap.tolist(),
            "gcal_interval": iv_g, "gnap_interval": iv_nap,
            "gcal_advances_at_pm20pct": advances(iv_g, theta_true[0]),
            "gnap_advances_at_pm20pct": advances(iv_nap, theta_true[1]),
            "joint_compat_set_gcal_gnap": joint,
            "event_hashes": {p: _sha16(np.asarray(events[p]))
                             for p in fit_protos},
        })
        print(f"[rep{r}] EPS {EPS:.4f} | gCaL interval {iv_g} "
              f"(±20% pass: {reps[-1]['gcal_advances_at_pm20pct']}) | "
              f"gNaP interval {iv_nap} "
              f"(±20% pass: {reps[-1]['gnap_advances_at_pm20pct']}) | "
              f"joint set size {len(joint)}", flush=True)

    # ---------------------------------------------------------------- figure
    clip = EPS_FLOOR / 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, pname, axis, key, ti in (
            (axes[0], "gCaL_scale", g_axis, "profile_gcal", theta_true[0]),
            (axes[1], "gNaP_scale", nap_axis, "profile_gnap", theta_true[1])):
        for k, rep in enumerate(reps):
            ax.plot(axis, np.maximum(rep[key], clip),
                    color=OKABE["cal" if pname == "gCaL_scale" else "nap"],
                    lw=1.4, alpha=1.0 if k == 0 else 0.5,
                    label=f"rep{k}" if k < 2 else None)
            ax.axhline(rep["eps"], color="#111111", lw=0.8, ls="--",
                       alpha=1.0 if k == 0 else 0.5)
        ax.axvline(ti, color=OKABE["truth"], lw=1.0)
        ax.axvspan(ti * (1 - RESOLUTION_REL), ti * (1 + RESOLUTION_REL),
                   color="#000000", alpha=0.06,
                   label="±20 % declared resolution")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(pname)
        ax.set_ylabel("profile-min fit loss (Hz²)")
        ax.set_title(f"Profile over {pname} (min over other two params)",
                     fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, color=GRIDC, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}.png", dpi=150)

    here = Path(__file__)
    report = {
        "manifest": {**_manifest(),
                     "r2a_sha256_16": hashlib.sha256(
                         here.read_bytes()).hexdigest()[:16],
                     "args": vars(args), "run_tag": tag},
        "declared": {
            "unknowns_profiled": ["gCaL_scale", "gNaP_scale"],
            "nuisance_reoptimized": ["w_exc"],
            "fixed_known": ["inhibition absent", "connectivity",
                            "frozen event realizations"],
            "observable": "binned (50 ms / 300 ms)",
            "resolution_rel_predeclared": RESOLUTION_REL,
            "theta_true": theta_true.tolist(),
            "interval_terminology": "sampled compatibility intervals "
                                    "(deterministic level sets)",
        },
        "axes": {"gcal": g_axis.tolist(), "gnap": nap_axis.tolist(),
                 "w_exc": w_axis.tolist()},
        "replicates": reps,
        "wall_s": round(time.time() - t_wall0, 1),
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"{stem}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"{stem}_{stamp}.json").write_text(payload)

    print(f"\n=== {stem} summary ===")
    for k, rep in enumerate(reps):
        print(f"rep{k}: gCaL {rep['gcal_interval']} "
              f"(advances: {rep['gcal_advances_at_pm20pct']}) | "
              f"gNaP {rep['gnap_interval']} "
              f"(advances: {rep['gnap_advances_at_pm20pct']})")
    print(f"saved → {RESULTS / (stem + '.json')} (+ evidence copy, figure)")


if __name__ == "__main__":
    main()

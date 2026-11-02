"""Deliverable C (v2) — unknown-input stress test.

Revised roadmap §12C: *"Release a small number of neural-drive nuisance
coefficients... This reveals whether the apparent PIC estimate survived only
because drive was treated as known."*

Setup identical to Deliverable B (frozen event realizations, observable via
``--observable``, per-replicate EPS from dt-halving at truth, axes contain
the truth) EXCEPT the input amplitude is unknown to the fitter. The nuisance
is a per-protocol EFFECTIVE EXCITATORY AMPLITUDE ``effective_w_exc_p`` in µS
(mechanistically the synaptic weight — w_exc and any dimensionless input
gain b_p enter only as their product, an EXACT model symmetry declared up
front). Because of that symmetry, w_exc itself is structurally
non-identifiable here BY CONSTRUCTION; this deliverable can only answer
whether **gCaL remains constrained** after releasing the effective input
amplitude:

* known input   : one amplitude SHARED across fit protocols →
      L_known(g)   = min_b  mean_p  loss_p(g, b)
* unknown input : amplitudes FREE and independent per protocol →
      L_unknown(g) = mean_p  min_{b_p}  loss_p(g, b_p)

Both profiles contain the truth on-axis and are asserted to be exactly 0
there (same model, same events). Minima over amplitude use local log
refinement around the coarse argmin (hard-spike loss is discontinuous).

Observation model: all units (full_obs) only — declared; partial-observation
stress belongs to a later rung of the observation ladder. Held-out
prediction is NOT evaluated here: under unknown input it would require a
declared amplitude rule for the unseen protocol, which is a separate
experiment.

Usage::

    python deliverable_c.py                      # full run
    python deliverable_c.py --dense-n 7 --replicates 1 --dur-scale 0.5  # smoke
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
    build_twin_and_events, make_observable_fn, numerical_epsilon_spikes,
    binned_rates_jnp, run_tag, _sha16,
)

OKABE = {"known": "#0072B2", "unknown": "#E69F00", "truth": "#000000"}
GRIDC = "#e3e3e3"

NESTING_TOL = 1e-9


def compute_nested_profiles(losses_at, g_axis, b_axis, n_protos: int,
                            refine_pts: int = 5):
    """Known- and unknown-input gCaL profiles over ONE shared candidate set.

    ``losses_at(g, b)`` returns a tuple of per-protocol losses. For each g the
    amplitude candidate set is the union of the coarse axis and the local
    log-refinement points around BOTH the shared-amplitude argmin and every
    per-protocol argmin — evaluated once and reused by both models, so the
    nesting property

        unknown(g) = mean_p min_b L_p(g, b)  ≤  min_b mean_p L_p(g, b) = known(g)

    holds by construction (independently optimized amplitudes include the
    shared solution). It is ASSERTED per point: a violation means the two
    models were minimized over different candidate sets (the v1 bug).
    """
    known = np.zeros(len(g_axis))
    unknown = np.zeros(len(g_axis))
    for i, g in enumerate(g_axis):
        rows = {float(b): losses_at(g, b) for b in b_axis}

        def local_candidates(j):
            lo = b_axis[max(j - 1, 0)]
            hi = b_axis[min(j + 1, len(b_axis) - 1)]
            return np.geomspace(lo, hi, refine_pts)[1:-1]

        coarse = list(b_axis)
        argmins = {int(np.argmin([float(np.mean(v)) for v in rows.values()]))}
        for pi in range(n_protos):
            argmins.add(int(np.argmin([v[pi] for v in rows.values()])))
        candidates = set(float(b) for b in coarse)
        for j in argmins:
            candidates.update(float(b) for b in local_candidates(j))
        vals = {b: (rows[b] if b in rows else losses_at(g, b))
                for b in candidates}

        known[i] = min(float(np.mean(v)) for v in vals.values())
        unknown[i] = float(np.mean(
            [min(v[pi] for v in vals.values()) for pi in range(n_protos)]))
        assert unknown[i] <= known[i] + NESTING_TOL, (
            f"nesting violated at g={g}: unknown {unknown[i]} > "
            f"known {known[i]} — candidate sets differ (bug)")
    return known, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--true", type=float, nargs=2, default=[1.6, 0.0105])
    ap.add_argument("--dense-n", type=int, default=21)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--dur-scale", type=float, default=1.0)
    ap.add_argument("--observable", choices=["binned", "pointwise"],
                    default="binned")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    tag = run_tag(args)
    stem = f"deliverable_c_{tag}_{args.observable}"
    t_wall0 = time.time()
    dt_ms = 0.1
    theta_true = np.array(args.true)
    fit_protos = ["fit_ramp", "fit_trap"]
    commands = make_commands(dt_ms, args.dur_scale)

    g_dense = axis_with_truth(0.25, 4.0, args.dense_n, theta_true[0])
    b_dense = axis_with_truth(0.0045, 0.018, args.dense_n, theta_true[1])
    gi_true = int(np.argmin(np.abs(g_dense - theta_true[0])))

    reps = []
    for r in range(args.replicates):
        conn_seed, ev_off = r, 10 * r
        twin, events = build_twin_and_events(
            args.n_mn, conn_seed, ev_off, commands, dt_ms)
        obs_fn = make_observable_fn(twin, events, args.observable)
        obs = {p: obs_fn(theta_true, p) for p in commands}

        cache: dict = {}

        def losses_at(g, b):
            key = (float(g), float(b))
            if key not in cache:
                cache[key] = tuple(
                    float(jnp.mean((obs_fn(np.array([g, b]), p) - obs[p]) ** 2))
                    for p in fit_protos)
            return cache[key]

        assert sum(losses_at(*theta_true)) == 0.0, "truth loss must be 0"

        t0 = time.time()
        # dense per-protocol matrices
        Lp = {p: np.zeros((len(g_dense), len(b_dense))) for p in fit_protos}
        for i, g in enumerate(g_dense):
            for j, b in enumerate(b_dense):
                vals = losses_at(g, b)
                for p, v in zip(fit_protos, vals):
                    Lp[p][i, j] = v

        # per-replicate EPS (full_obs = all units)
        fine = numerical_epsilon_spikes(args.n_mn, conn_seed, events,
                                        theta_true, fit_protos)
        terms = []
        for p, sp_f in zip(fit_protos, fine):
            if args.observable == "pointwise":
                rf = np.asarray(_smoothed_rates_jax(
                    jnp.asarray(sp_f), 0.05, 100.0))[1::2]
            else:
                rf = np.asarray(binned_rates_jnp(jnp.asarray(sp_f), 0.05))
            rc = np.asarray(obs[p])
            m = min(len(rf), len(rc))
            terms.append(float(np.mean((rf[:m] - rc[:m]) ** 2)))
        EPS = max(float(np.mean(terms)), EPS_FLOOR)

        known, unknown = compute_nested_profiles(
            losses_at, g_dense, b_dense, n_protos=len(fit_protos))
        assert known[gi_true] == 0.0 and unknown[gi_true] == 0.0, (
            "profile at truth must be exactly 0")

        def interval(curve):
            w = curve <= EPS
            return ([float(g_dense[w].min()), float(g_dense[w].max())]
                    if w.any() else None)

        reps.append({
            "conn_seed": conn_seed, "event_offset": ev_off,
            "eps_full_obs": EPS,
            "profile_known": known.tolist(),
            "profile_unknown": unknown.tolist(),
            "gcal_interval_known": interval(known),
            "gcal_interval_unknown": interval(unknown),
            "interval_terminology": "sampled compatibility interval "
                                    "(deterministic level set, not a "
                                    "statistical confidence interval)",
            "n_evals": len(cache),
            "event_hashes": {p: _sha16(np.asarray(events[p]))
                             for p in fit_protos},
        })
        print(f"[rep{r}] {len(cache)} evals in {time.time()-t0:.0f}s, "
              f"EPS {EPS:.4f}; truth=0 OK")
        print(f"    gCaL sampled compatibility interval, amplitude known:   "
              f"{reps[-1]['gcal_interval_known']}")
        print(f"    gCaL sampled compatibility interval, amplitude RELEASED: "
              f"{reps[-1]['gcal_interval_unknown']}")

    # ---------------------------------------------------------------- figure
    clip = EPS_FLOOR / 10
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for k, rep in enumerate(reps):
        ax.plot(g_dense, np.maximum(rep["profile_known"], clip),
                color=OKABE["known"], lw=1.4,
                alpha=1.0 if k == 0 else 0.45,
                label="amplitude known (shared)" if k == 0 else None)
        ax.plot(g_dense, np.maximum(rep["profile_unknown"], clip),
                color=OKABE["unknown"], lw=1.4,
                alpha=1.0 if k == 0 else 0.45,
                label="amplitude FREE per protocol" if k == 0 else None)
        ax.axhline(rep["eps_full_obs"], color="#111111", lw=0.8, ls="--",
                   alpha=1.0 if k == 0 else 0.45,
                   label=f"EPS per replicate" if k == 0 else None)
    ax.axvline(theta_true[0], color=OKABE["truth"], lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gCaL_scale")
    ax.set_ylabel(f"profile-min fit loss (Hz², display-clipped at {clip:g})")
    ax.set_title("Deliverable C — does gCaL stay constrained when input "
                 f"amplitude is released? ({args.replicates} replicate(s), "
                 f"{args.observable})", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig_path = RESULTS / f"{stem}_gcal_profiles.png"
    fig.savefig(fig_path, dpi=150)

    # ---------------------------------------------------------------- report
    here = Path(__file__)
    report = {
        "manifest": {**_manifest(),
                     "deliverable_c_sha256_16": hashlib.sha256(
                         here.read_bytes()).hexdigest()[:16],
                     "args": vars(args), "run_tag": tag},
        "declared": {
            "nuisance": "per-protocol effective_w_exc_p (µS); exact "
                        "multiplicative symmetry with w_exc ⇒ w_exc is "
                        "structurally non-identifiable here by construction; "
                        "the question is whether gCaL stays constrained",
            "observable": args.observable,
            "observation_model": "full_obs only (declared)",
            "eps_rule": "per replicate, dt-halving at truth, all units",
            "theta_true": theta_true.tolist(),
            "axes_contain_truth": True,
            "heldout": "not evaluated — would require a declared amplitude "
                       "rule for the unseen protocol (separate experiment)",
        },
        "g_dense": g_dense.tolist(),
        "b_dense_effective_w_exc": b_dense.tolist(),
        "replicates": reps,
        "wall_s": round(time.time() - t_wall0, 1),
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"{stem}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"{stem}_{stamp}.json").write_text(payload)

    print(f"\n=== {stem} summary ===")
    for k, rep in enumerate(reps):
        print(f"rep{k} (EPS {rep['eps_full_obs']:.4f}): known "
              f"{rep['gcal_interval_known']} | unknown "
              f"{rep['gcal_interval_unknown']}")
    print(f"figure → {fig_path}")
    print(f"report → {RESULTS / (stem + '.json')} (+ evidence copy)")


if __name__ == "__main__":
    main()

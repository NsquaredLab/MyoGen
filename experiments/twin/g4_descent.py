"""G4 v2 — descent benchmark with the corrected full-trace objective.

v2 fixes from review of the v1 run:

* **Full-trace observation operator**: windows are simulated with preserved
  state and boundary-stopped gradients, but the per-window SPIKE outputs are
  CONCATENATED and ``binned_rates_jnp`` is applied ONCE to the complete
  sequence — identical to Deliverable B's objective. (v1 restarted the
  300 ms smoothing at each boundary, silently changing the objective by
  ~2.3 Hz² at the v1 operating point.) A regression test asserts the
  chained hard spike train equals the uninterrupted run exactly.
* **Coordinate-resolved gating**: besides the joint direction, Stage A now
  probes gCaL-only and w_exc-only steps from three start types —
  S1 both-wrong, S2 gCaL-wrong-only (w at truth), S3 w-wrong-only — and
  records the directional cosine against the CRN-FD reference. 16/16 joint
  acceptances alone cannot certify the gCaL coordinate (v1 directions were
  ~[0.07, 0.998]).
* **Stage B across all cases and two starts** with success/failure counts,
  parameter and prediction error separated, and the best loss compared to
  the per-case dt-halving EPS of THIS objective (truth loss is exactly 0,
  so any plateau needs that reference).
* **Profile of the exact G4 objective** (case 0): dense gCaL profile with
  w re-optimized, truth on-axis — required before attributing any gCaL
  stall to identifiability.
* Event/connectivity matrices are float64 under x64 (v1 left them f32).

Scoped claims: joint-descent reliability, per-coordinate direction quality,
and recovery statistics for THIS configuration; nothing about other
objectives or regimes.

Usage::

    python g4_descent.py            # full (CPU ~1.5 h)
    python g4_descent.py --smoke
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

RESULTS = Path(__file__).parent / "results"
EVIDENCE = Path(__file__).parent / "evidence"

DT = 0.025
BASE_DT = 0.1
FACTOR = int(round(BASE_DT / DT))
WINDOW_S = 2.0
HAZARD_SIGMA_MV = 2.0
MAX_SPK = 64
LAMBDAS = (0.03, 0.1)
EPS_FLOOR = 1e-3
OKABE = {"surrogate": "#0072B2", "crn_fd": "#CC79A7"}


def build_case(conn_seed, ev_off, dur_scale):
    import jax.numpy as jnp
    from deliverable_b import make_commands, build_twin_and_events
    from gradient_repair import zero_insert_events
    from poolbuild import PoolTwin, TwinSpec

    commands = make_commands(BASE_DT, dur_scale)
    _, events0 = build_twin_and_events(8, conn_seed, ev_off, commands, BASE_DT)
    twin = PoolTwin(TwinSpec(n_mn=8, n_dd=100, n_inh=5, dt_ms=DT,
                             seed=conn_seed))
    ev = {}
    for proto in ("fit_ramp", "heldout_slowramp"):
        fine = zero_insert_events(
            np.asarray(events0[proto], dtype=np.float64), FACTOR)
        ev[proto] = {
            "exc": jnp.asarray(fine),
            "inh": jnp.zeros((fine.shape[0], 5), dtype=fine.dtype),
            "exc_conn": jnp.asarray(np.asarray(twin._exc_mat,
                                               dtype=np.float64)),
            "inh_conn": jnp.asarray(np.asarray(twin._inh_mat,
                                               dtype=np.float64)),
        }
    return twin, ev


def split_windows(ev, n_win):
    n = ev["exc"].shape[0]
    w = n // n_win
    return [{k: (v[i * w:(i + 1) * w] if k in ("exc", "inh") else v)
             for k, v in ev.items()} for i in range(n_win)]


def binned(spikes, dt_ms=DT, bin_ms=50.0, smooth_ms=300.0):
    from deliverable_b import binned_rates_jnp

    return binned_rates_jnp(spikes, dt_ms, bin_ms, smooth_ms)


def chained_spikes(twin, theta, windows, spike_mode, stop_boundary=True):
    """State-preserving windowed run; returns FULL concatenated spike train.
    Gradients are stopped at boundaries iff stop_boundary."""
    import jax

    carry, segs = None, []
    for w in windows:
        out, carry = twin._run_events_carry(theta, w, carry, spike_mode)
        if stop_boundary:
            carry = jax.tree_util.tree_map(jax.lax.stop_gradient, carry)
        segs.append(out["spikes"])
    import jax.numpy as jnp

    return jnp.concatenate(segs, axis=0)


class Case:
    def __init__(self, conn_seed, ev_off, theta_true, dur_scale=1.0):
        import jax
        import jax.numpy as jnp

        self.meta = {"conn_seed": conn_seed, "event_offset": ev_off,
                     "theta_true": list(theta_true)}
        twin, ev = build_case(conn_seed, ev_off, dur_scale)
        self.twin = twin
        n_win = max(1, int(round(
            ev["fit_ramp"]["exc"].shape[0] * DT * 1e-3 / WINDOW_S)))
        self.windows = split_windows(ev["fit_ramp"], n_win)
        self.heldout = ev["heldout_slowramp"]
        self.theta_true = np.asarray(theta_true, dtype=np.float64)

        def theta_of(vec):
            th = twin.default_theta()
            th["gCaL_scale"] = vec[0]
            th["w_exc"] = vec[1]
            return th

        self.theta_of = theta_of

        # observations: full-trace operator on the chained hard run
        tt = jnp.asarray(self.theta_true)
        obs_sp = chained_spikes(twin, theta_of(tt), self.windows, "hard")
        self.obs_rates = jax.lax.stop_gradient(binned(obs_sp))
        self.obs_spikes = np.asarray(obs_sp)
        self.n_obs_spikes = int(self.obs_spikes.sum())
        out_h, _ = twin._run_events_carry(theta_of(tt), self.heldout, None,
                                          "hard")
        self.obs_heldout = jax.lax.stop_gradient(binned(out_h["spikes"]))

        # observed spike indices (hazard NLL), padded, full-trace
        idx = np.zeros((self.obs_spikes.shape[1], MAX_SPK * n_win),
                       dtype=np.int32)
        msk = np.zeros_like(idx, dtype=np.float64)
        for i in range(self.obs_spikes.shape[1]):
            ks = np.flatnonzero(self.obs_spikes[:, i] > 0.5)[:idx.shape[1]]
            idx[i, :len(ks)] = ks
            msk[i, :len(ks)] = 1.0
        self.obs_idx, self.obs_msk = jnp.asarray(idx), jnp.asarray(msk)

        def hard_loss(vec):
            sp = chained_spikes(twin, theta_of(vec), self.windows, "hard")
            return jnp.mean((binned(sp) - self.obs_rates) ** 2)

        def sur_loss(vec):
            sp = chained_spikes(twin, theta_of(vec), self.windows, "surrogate")
            return jnp.mean((binned(sp) - self.obs_rates) ** 2)

        self.hard_loss = jax.jit(hard_loss)
        self._sur_grad = jax.jit(jax.grad(sur_loss))
        self._eval_count = 0

    def hard(self, vec):
        self._eval_count += 1
        return float(self.hard_loss(np.asarray(vec, dtype=np.float64)))

    def grad(self, name, vec):
        import jax.numpy as jnp

        if name == "crn_fd":
            g = np.zeros(2)
            for j in range(2):
                h = 0.05 * float(vec[j])
                e = np.zeros(2)
                e[j] = h
                g[j] = (self.hard(vec + e) - self.hard(vec - e)) / (2 * h)
            return g
        self._eval_count += 3
        return np.asarray(self._sur_grad(jnp.asarray(vec, dtype=jnp.float64)),
                          dtype=np.float64)

    def direction(self, name, vec):
        g = self.grad(name, vec)
        if not np.all(np.isfinite(g)):
            return None, g
        gl = g * np.asarray(vec)
        n = np.linalg.norm(gl)
        return (-gl / n if n > 0 else None), g

    def heldout_mse(self, vec):
        import jax.numpy as jnp

        out, _ = self.twin._run_events_carry(
            self.theta_of(jnp.asarray(vec, dtype=jnp.float64)),
            self.heldout, None, "hard")
        return float(jnp.mean((binned(out["spikes"]) - self.obs_heldout) ** 2))

    def eps_dt_halving(self):
        """Numerical tolerance of THIS objective: rerun truth at dt/2."""
        import jax.numpy as jnp
        from gradient_repair import zero_insert_events
        from poolbuild import PoolTwin, TwinSpec

        twin_h = PoolTwin(TwinSpec(n_mn=8, n_dd=100, n_inh=5, dt_ms=DT / 2,
                                   seed=self.meta["conn_seed"]))
        ev_f = {
            "exc": jnp.asarray(zero_insert_events(
                np.concatenate([np.asarray(w["exc"]) for w in self.windows]),
                2)),
            "inh": None, "exc_conn": None, "inh_conn": None,
        }
        ev_f["inh"] = jnp.zeros((ev_f["exc"].shape[0], 5),
                                dtype=ev_f["exc"].dtype)
        ev_f["exc_conn"] = self.windows[0]["exc_conn"]
        ev_f["inh_conn"] = self.windows[0]["inh_conn"]
        th = self.theta_of(jnp.asarray(self.theta_true))
        out, _ = twin_h._run_events_carry(th, ev_f, None, "hard")
        r_f = np.asarray(binned(out["spikes"], DT / 2))
        r_c = np.asarray(self.obs_rates)
        m = min(len(r_f), len(r_c))
        return max(float(((r_f[:m] - r_c[:m]) ** 2).mean()), EPS_FLOOR)


def main():
    import jax

    jax.config.update("jax_enable_x64", True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    from sensitivity_checks import _manifest

    t0_all = time.time()
    dur_scale = 0.5 if args.smoke else 1.0
    case_specs = ([(0, 0, (1.5, 0.011))] if args.smoke else
                  [(c, e, t) for c in (0, 1) for e in (0, 10)
                   for t in ((1.5, 0.011), (0.7, 0.008))])
    estimators = ["surrogate", "crn_fd"]

    def starts_for(tt):
        return {"S1_both_wrong": np.array([1.0, 0.009]),
                "S2_gcal_wrong_only": np.array([1.0, tt[1]]),
                "S3_w_wrong_only": np.array([tt[0], 0.009])}

    # ------------------------------------------------- Stage A (coordinate-resolved)
    rowsA = []
    tallies = {e: {s: {"joint_acc": 0, "joint_tot": 0,
                       "coord_acc": {"gCaL": 0, "w": 0},
                       "coord_tot": {"gCaL": 0, "w": 0},
                       "nonfinite": 0}
                   for s in ("S1_both_wrong", "S2_gcal_wrong_only",
                             "S3_w_wrong_only")}
               for e in estimators}
    cosines = []
    case_objs = {}
    for ci, (cs, eo, tt) in enumerate(case_specs):
        t0 = time.time()
        case = Case(cs, eo, tt, dur_scale)
        case_objs[ci] = case
        row = {"case": case.meta, "n_obs_spikes": case.n_obs_spikes,
               "starts": {}}
        for sname, s0 in starts_for(tt).items():
            L0 = case.hard(s0)
            srow = {"L0": L0, "estimators": {}}
            dirs = {}
            for est in estimators:
                d, g = case.direction(est, s0)
                if d is None:
                    tallies[est][sname]["nonfinite"] += 1
                    srow["estimators"][est] = {"nonfinite": True}
                    continue
                dirs[est] = d
                erow = {"grad": g.tolist(), "direction": d.tolist(),
                        "joint": {}, "coord": {}}
                for lam in LAMBDAS:  # joint steps
                    Ls = case.hard(s0 * np.exp(lam * d))
                    ok = bool(Ls < L0)
                    erow["joint"][lam] = {"loss": Ls, "accepted": ok}
                    tallies[est][sname]["joint_acc"] += ok
                    tallies[est][sname]["joint_tot"] += 1
                # coordinate-only steps (λ=0.1, sign from that coordinate)
                for j, cname in ((0, "gCaL"), (1, "w")):
                    if d[j] == 0:
                        continue
                    dc = np.zeros(2)
                    dc[j] = np.sign(d[j])
                    Ls = case.hard(s0 * np.exp(0.1 * dc))
                    ok = bool(Ls < L0)
                    erow["coord"][cname] = {"loss": Ls, "accepted": ok}
                    tallies[est][sname]["coord_acc"][cname] += ok
                    tallies[est][sname]["coord_tot"][cname] += 1
                srow["estimators"][est] = erow
            if "surrogate" in dirs and "crn_fd" in dirs:
                c = float(dirs["surrogate"] @ dirs["crn_fd"])
                srow["cosine_sur_vs_fd"] = c
                cosines.append(c)
            row["starts"][sname] = srow
        rowsA.append(row)
        print(f"[case {ci}] conn{cs}/ev{eo}/true{tt} spk {case.n_obs_spikes} "
              f"({time.time()-t0:.0f}s)", flush=True)

    gate = {}
    for est in estimators:
        agg = {"joint_acc": 0, "joint_tot": 0, "nonfinite": 0,
               "coord": {"gCaL": [0, 0], "w": [0, 0]}}
        for s in tallies[est].values():
            agg["joint_acc"] += s["joint_acc"]
            agg["joint_tot"] += s["joint_tot"]
            agg["nonfinite"] += s["nonfinite"]
            for cname in ("gCaL", "w"):
                agg["coord"][cname][0] += s["coord_acc"][cname]
                agg["coord"][cname][1] += s["coord_tot"][cname]
        jf = agg["joint_acc"] / max(agg["joint_tot"], 1)
        cf = {c: (a / max(t, 1)) for c, (a, t) in agg["coord"].items()}
        gate[est] = {
            "joint_accepted_fraction": jf,
            "coord_accepted_fraction": cf,
            "nonfinite": agg["nonfinite"],
            "per_start": tallies[est],
            "joint_pass": bool(jf >= 0.8 and agg["nonfinite"] == 0),
            "gcal_pass": bool(cf["gCaL"] >= 0.8 and agg["nonfinite"] == 0),
        }
        print(f"Stage A — {est:>9}: joint {jf:.2f} | gCaL-coord "
              f"{cf['gCaL']:.2f} | w-coord {cf['w']:.2f} | nonfinite "
              f"{agg['nonfinite']} → joint "
              f"{'PASS' if gate[est]['joint_pass'] else 'fail'}, gCaL "
              f"{'PASS' if gate[est]['gcal_pass'] else 'fail'}", flush=True)
    print(f"Stage A — cosine(surrogate, CRN-FD): mean "
          f"{np.mean(cosines):.3f} min {np.min(cosines):.3f}", flush=True)

    # ------------------------------------------------- Stage B (all cases, 2 starts)
    stageB = []
    for ci, (cs, eo, tt) in enumerate(case_specs):
        case = case_objs[ci]
        eps = case.eps_dt_halving()
        for sname in ("S1_both_wrong", "S2_gcal_wrong_only"):
            s0 = starts_for(np.asarray(tt))[sname]
            for est in estimators:
                case._eval_count = 0
                t0 = time.time()
                vec = s0.copy()
                L = case.hard(vec)
                best = {"loss": L, "theta": vec.tolist(), "it": 0}
                lam, nonfinite = 0.1, 0
                for it in range(1, args.iters + 1):
                    d, _ = case.direction(est, vec)
                    if d is None:
                        nonfinite += 1
                        break
                    for trial in (lam, lam / 3, lam / 9):
                        cand = vec * np.exp(trial * d)
                        Lc = case.hard(cand)
                        if Lc < L:
                            vec, L, lam = cand, Lc, trial
                            break
                    if L < best["loss"]:
                        best = {"loss": L, "theta": vec.tolist(), "it": it}
                rel = (np.abs(np.asarray(best["theta"])
                              - np.asarray(tt)) / np.asarray(tt))
                stageB.append({
                    "case": ci, "start": sname, "estimator": est,
                    "eps_dt_halving": eps,
                    "best_loss": best["loss"],
                    "within_eps": bool(best["loss"] <= eps),
                    "theta_best": best["theta"],
                    "rel_err": rel.tolist(),
                    "gcal_recovered_20pct": bool(rel[0] <= 0.20),
                    "w_recovered_20pct": bool(rel[1] <= 0.20),
                    "heldout_mse": case.heldout_mse(best["theta"]),
                    "nonfinite": nonfinite,
                    "hard_evals": case._eval_count,
                    "wall_s": round(time.time() - t0, 1),
                })
                r = stageB[-1]
                print(f"  B c{ci}/{sname}/{est}: best {r['best_loss']:.3f} "
                      f"(eps {eps:.3f}, within={r['within_eps']}) "
                      f"θ=({r['theta_best'][0]:.3f},{r['theta_best'][1]:.5f}) "
                      f"rel ({rel[0]*100:.0f}%,{rel[1]*100:.0f}%) "
                      f"held-out {r['heldout_mse']:.3f} ({r['wall_s']}s)",
                      flush=True)

    def recov_stats(est):
        rows = [r for r in stageB if r["estimator"] == est]
        return {
            "n_runs": len(rows),
            "gcal_recovered_20pct": sum(r["gcal_recovered_20pct"] for r in rows),
            "w_recovered_20pct": sum(r["w_recovered_20pct"] for r in rows),
            "within_eps": sum(r["within_eps"] for r in rows),
            "nonfinite_runs": sum(r["nonfinite"] > 0 for r in rows),
            "median_heldout": float(np.median([r["heldout_mse"]
                                               for r in rows])),
        }

    summaryB = {e: recov_stats(e) for e in estimators}
    for e, s in summaryB.items():
        print(f"Stage B — {e:>9}: {s}", flush=True)

    # ------------------------------------------- profile of the EXACT objective
    print("profiling exact G4 objective (case 0)...", flush=True)
    from deliverable_b import axis_with_truth

    case = case_objs[0]
    tt = np.asarray(case_specs[0][2])
    g_axis = axis_with_truth(0.25, 4.0, 11, tt[0])
    w_axis = axis_with_truth(0.0045, 0.018, 9, tt[1])
    L2 = np.array([[case.hard(np.array([g, w])) for w in w_axis]
                   for g in g_axis])
    prof_g = L2.min(axis=1)
    gi = int(np.argmin(np.abs(g_axis - tt[0])))
    assert prof_g[gi] == 0.0
    eps0 = [r["eps_dt_halving"] for r in stageB if r["case"] == 0][0]
    within = prof_g <= eps0
    profile = {
        "g_axis": g_axis.tolist(), "profile_gcal": prof_g.tolist(),
        "eps": eps0,
        "gcal_compat_interval": ([float(g_axis[within].min()),
                                  float(g_axis[within].max())]
                                 if within.any() else None),
    }
    print(f"  gCaL compatibility interval (≤ eps {eps0:.3f}): "
          f"{profile['gcal_compat_interval']}", flush=True)

    # -------------------------------------------------------------- report
    tag = "smoke" if args.smoke else "full"
    report = {
        "manifest": {**_manifest(),
                     "g4_sha256_16": hashlib.sha256(
                         Path(__file__).read_bytes()).hexdigest()[:16],
                     "args": vars(args), "run_tag": tag},
        "declared": {
            "dt_ms": DT, "window_s": WINDOW_S,
            "objective": "FULL-TRACE hard binned-rate MSE (50/300 ms) at "
                         "dt 0.025; windows affect gradient flow only",
            "starts": "S1 both-wrong, S2 gCaL-wrong-only, S3 w-wrong-only",
            "lambdas": list(LAMBDAS),
        },
        "stageA": {"rows": rowsA, "gate": gate,
                   "cosine_sur_vs_fd": {"mean": float(np.mean(cosines)),
                                        "min": float(np.min(cosines))}},
        "stageB": {"runs": stageB, "summary": summaryB},
        "profile_case0": profile,
        "wall_s": round(time.time() - t0_all, 1),
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"g4_descent_{tag}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"g4_descent_{tag}_{stamp}.json").write_text(payload)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for k, est in enumerate(estimators):
        vals = [gate[est]["joint_accepted_fraction"],
                gate[est]["coord_accepted_fraction"]["gCaL"],
                gate[est]["coord_accepted_fraction"]["w"]]
        ax.bar(np.arange(3) + 0.4 * k - 0.2, vals, width=0.35,
               color=OKABE[est], label=est)
    ax.set_xticks(range(3), ["joint", "gCaL-only", "w-only"])
    ax.axhline(0.8, color="#111111", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Stage A accepted fractions", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for est in estimators:
        errs = [r["rel_err"][0] for r in stageB if r["estimator"] == est]
        ax.plot(np.full(len(errs), estimators.index(est))
                + np.random.default_rng(0).uniform(-0.08, 0.08, len(errs)),
                errs, "o", color=OKABE[est], alpha=0.6)
    ax.axhline(0.2, color="#111111", ls="--", lw=1, label="±20 %")
    ax.set_xticks(range(len(estimators)), estimators)
    ax.set_ylabel("|gCaL rel err| at best")
    ax.set_title("Stage B gCaL recovery (all runs)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    ax.semilogy(g_axis, np.maximum(prof_g, EPS_FLOOR / 10), "o-",
                color="#0072B2", ms=3)
    ax.axhline(eps0, color="#111111", ls="--", lw=1, label=f"eps {eps0:.3f}")
    ax.axvline(tt[0], color="#000000", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("gCaL_scale")
    ax.set_title("Exact-G4-objective gCaL profile (case 0)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(True, color="#e3e3e3", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / f"g4_descent_{tag}.png", dpi=150)
    print(f"saved → results/g4_descent_{tag}.json (+ evidence, figure)")


if __name__ == "__main__":
    main()

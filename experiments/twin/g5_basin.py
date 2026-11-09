"""G5 — basin characterization and hybrid recovery (post-G4 program).

Answers the three questions G4 left open, per review:

1. **How wide is the gCaL basin really?** Adaptive profile around the
   truth with CONTINUOUS w-optimization at every gCaL (golden-section on
   log-w — no discrete w grid), plus bisection refinement of both
   compatibility boundaries to ~2 % gCaL resolution. Run on ALL cases.
2. **Does the basin survive realistic variability?** Each case reports a
   REPEATABILITY tolerance: the rate-MSE between the truth run and a truth
   run with an independent event realization (the irreducible variability
   a real experiment faces). Compatibility intervals are reported against
   BOTH the numerical EPS and the repeatability tolerance — if the basin
   vanishes under the latter, the exact-fit needle is brittle
   event-sequence matching, not robust physiological information.
3. **Can a hybrid solver recover gCaL where local descent failed?**
   Coarse gCaL scan × continuous w-optimization, retain top-3 candidate
   basins, refine each with the qualified local gradient optimizer, return
   the best — against an equal-budget Nelder–Mead derivative-free baseline.

Objective/configuration identical to G4 v2 (full-trace binned-rate MSE,
dt 0.025, state-preserving windows, float64).

Usage::

    python g5_basin.py            # full (CPU: hours)
    python g5_basin.py --smoke
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

W_LO, W_HI = 0.0045, 0.018
GOLD = 0.381966


def golden_min_w(case, gcal, n_evals=10):
    """Continuous 1-D minimization of the hard loss over log-w at fixed
    gCaL (derivative-free, deterministic). Returns (w*, loss*)."""
    lo, hi = np.log(W_LO), np.log(W_HI)
    x1 = lo + GOLD * (hi - lo)
    x2 = hi - GOLD * (hi - lo)
    f = {x: case.hard(np.array([gcal, np.exp(x)])) for x in (x1, x2)}
    for _ in range(n_evals - 2):
        if f[x1] <= f[x2]:
            hi, x2 = x2, x1
            x1 = lo + GOLD * (hi - lo)
            f[x1] = case.hard(np.array([gcal, np.exp(x1)]))
        else:
            lo, x1 = x1, x2
            x2 = hi - GOLD * (hi - lo)
            f[x2] = case.hard(np.array([gcal, np.exp(x2)]))
    xb = min((x1, x2), key=lambda x: f[x])
    return float(np.exp(xb)), float(f[xb])


def profile_with_boundaries(case, g_truth, tol, span=(0.75, 1.25),
                            n_coarse=7, resolution_rel=0.02):
    """Adaptive gCaL profile: coarse pass around truth, then bisection of
    both compatibility boundaries (profile ≤ tol) to resolution_rel."""
    g_axis = list(np.geomspace(g_truth * span[0], g_truth * span[1],
                               n_coarse))
    if g_truth not in g_axis:
        g_axis = sorted(g_axis + [g_truth])
    prof = {g: golden_min_w(case, g)[1] for g in g_axis}

    def inside(g):
        if g not in prof:
            prof[g] = golden_min_w(case, g)[1]
        return prof[g] <= tol

    gs = sorted(prof)
    ins = [g for g in gs if prof[g] <= tol]
    if not ins:
        return prof, None
    lo_in, hi_in = min(ins), max(ins)
    # bisection on each boundary
    lo_out = max([g for g in gs if g < lo_in], default=lo_in * span[0])
    hi_out = min([g for g in gs if g > hi_in], default=hi_in * span[1])
    for _ in range(12):
        if lo_out / lo_in < 1 + resolution_rel:
            break
        mid = float(np.sqrt(lo_out * lo_in))
        if inside(mid):
            lo_in = mid
        else:
            lo_out = mid
    for _ in range(12):
        if hi_out / hi_in < 1 + resolution_rel:
            break
        mid = float(np.sqrt(hi_out * hi_in))
        if inside(mid):
            hi_in = mid
        else:
            hi_out = mid
    return prof, [lo_in, hi_in]


def hybrid_solver(case, budget_evals=220, n_keep=3, refine_iters=6):
    """Coarse gCaL scan × continuous w-opt → keep top basins → local
    gradient refinement of each → best candidate."""
    case._eval_count = 0
    scan = np.geomspace(0.25, 4.0, 9)
    cands = []
    for g in scan:
        w, L = golden_min_w(case, g, n_evals=8)
        cands.append((L, g, w))
    cands.sort()
    best = None
    for L, g, w in cands[:n_keep]:
        vec = np.array([g, w])
        Lc = L
        lam = 0.1
        for _ in range(refine_iters):
            if case._eval_count > budget_evals:
                break
            d, _ = case.direction("surrogate", vec)
            if d is None:
                break
            for trial in (lam, lam / 3, lam / 9):
                cand = vec * np.exp(trial * d)
                Lt = case.hard(cand)
                if Lt < Lc:
                    vec, Lc, lam = cand, Lt, trial
                    break
        if best is None or Lc < best[0]:
            best = (Lc, vec.copy())
    return {"loss": best[0], "theta": best[1].tolist(),
            "evals": case._eval_count}


def nelder_mead_baseline(case, budget_evals=220):
    from scipy.optimize import minimize

    case._eval_count = 0

    def f(z):
        if case._eval_count > budget_evals:
            return 1e9
        return case.hard(np.exp(z))

    res = minimize(f, np.log([1.0, 0.009]), method="Nelder-Mead",
                   options={"maxfev": budget_evals, "xatol": 1e-3,
                            "fatol": 1e-6})
    return {"loss": float(res.fun), "theta": np.exp(res.x).tolist(),
            "evals": case._eval_count}


def main():
    import jax

    jax.config.update("jax_enable_x64", True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    from sensitivity_checks import _manifest
    from g4_descent import Case, build_case, binned, chained_spikes

    t0_all = time.time()
    dur_scale = 0.5 if args.smoke else 1.0
    specs = ([(0, 0, (1.5, 0.011))] if args.smoke else
             [(c, e, t) for c in (0, 1) for e in (0, 10)
              for t in ((1.5, 0.011), (0.7, 0.008))])

    rows = []
    for ci, (cs, eo, tt) in enumerate(specs):
        t0 = time.time()
        case = Case(cs, eo, tt, dur_scale)
        eps = case.eps_dt_halving()

        # repeatability tolerance: truth with an INDEPENDENT event realization
        import jax.numpy as jnp

        twin_alt, ev_alt = build_case(cs, eo + 1000, dur_scale)
        th_t = case.theta_of(jnp.asarray(np.asarray(tt, dtype=np.float64)))
        out_alt, _ = case.twin._run_events_carry(
            th_t, ev_alt["fit_ramp"], None, "hard")
        t_rep = float(jnp.mean((binned(out_alt["spikes"])
                                - case.obs_rates) ** 2))

        prof_eps, iv_eps = profile_with_boundaries(case, tt[0], eps)
        tol_rep = max(eps, t_rep)
        iv_rep_candidates = {g: L for g, L in prof_eps.items() if L <= tol_rep}
        iv_rep = ([min(iv_rep_candidates), max(iv_rep_candidates)]
                  if iv_rep_candidates else None)

        hyb = hybrid_solver(case)
        nm = nelder_mead_baseline(case)
        rel_h = abs(hyb["theta"][0] - tt[0]) / tt[0]
        rel_n = abs(nm["theta"][0] - tt[0]) / tt[0]
        rows.append({
            "case": case.meta, "eps": eps, "t_repeat": t_rep,
            "profile": {f"{g:.4f}": L for g, L in sorted(prof_eps.items())},
            "gcal_interval_vs_eps": iv_eps,
            "gcal_interval_vs_repeatability": iv_rep,
            "hybrid": {**hyb, "gcal_rel_err": rel_h,
                       "gcal_recovered_20pct": bool(rel_h <= 0.2)},
            "nelder_mead": {**nm, "gcal_rel_err": rel_n,
                            "gcal_recovered_20pct": bool(rel_n <= 0.2)},
        })
        r = rows[-1]
        print(f"[case {ci}] conn{cs}/ev{eo}/true{tt}: eps {eps:.3f} "
              f"T_rep {t_rep:.3f} | interval(eps) {iv_eps} | "
              f"interval(rep) {iv_rep} | hybrid gCaL "
              f"{hyb['theta'][0]:.3f} ({rel_h*100:.0f}%, {hyb['evals']} ev) "
              f"| NM gCaL {nm['theta'][0]:.3f} ({rel_n*100:.0f}%, "
              f"{nm['evals']} ev) ({time.time()-t0:.0f}s)", flush=True)

    summary = {
        "hybrid_gcal_recovered_20pct": sum(
            r["hybrid"]["gcal_recovered_20pct"] for r in rows),
        "nm_gcal_recovered_20pct": sum(
            r["nelder_mead"]["gcal_recovered_20pct"] for r in rows),
        "n_cases": len(rows),
        "basin_survives_repeatability": sum(
            r["gcal_interval_vs_repeatability"] is not None
            and r["gcal_interval_vs_repeatability"][0] > 0 for r in rows),
    }
    print(f"SUMMARY: {summary}", flush=True)

    tag = "smoke" if args.smoke else "full"
    report = {
        "manifest": {**_manifest(),
                     "g5_sha256_16": hashlib.sha256(
                         Path(__file__).read_bytes()).hexdigest()[:16],
                     "args": vars(args), "run_tag": tag},
        "declared": {
            "objective": "G4 v2 full-trace objective (dt 0.025)",
            "profile": "continuous w via golden-section (10 evals), "
                       "boundary bisection to 2 % gCaL resolution",
            "tolerances": "eps = dt-halving at truth; t_repeat = rate-MSE "
                          "of truth vs independent event realization",
            "hybrid": "9-pt gCaL scan × golden w, top-3 basins, surrogate "
                      "refinement, ~220-eval budget",
            "baseline": "Nelder-Mead log-space, same budget",
        },
        "cases": rows, "summary": summary,
        "wall_s": round(time.time() - t0_all, 1),
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"g5_basin_{tag}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"g5_basin_{tag}_{stamp}.json").write_text(payload)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for r in rows:
        gs = sorted(float(g) for g in r["profile"])
        Ls = [r["profile"][f"{g:.4f}"] for g in gs]
        ax.semilogy(np.asarray(gs) / r["case"]["theta_true"][0],
                    np.maximum(Ls, 1e-4), "o-", ms=3, alpha=0.6)
        ax.axhline(r["t_repeat"], color="#E69F00", lw=0.6, alpha=0.5)
        ax.axhline(r["eps"], color="#111111", lw=0.6, ls="--", alpha=0.5)
    ax.axvline(1.0, color="#000000", lw=1)
    ax.set_xlabel("gCaL / truth")
    ax.set_ylabel("profile loss (continuous w-opt)")
    ax.set_title("Adaptive profiles, all cases (black=eps, orange=T_repeat)",
                 fontsize=9)
    ax = axes[1]
    for k, (meth, c) in enumerate((("hybrid", "#0072B2"),
                                   ("nelder_mead", "#CC79A7"))):
        errs = [r[meth]["gcal_rel_err"] for r in rows]
        ax.plot(np.full(len(errs), k)
                + np.random.default_rng(0).uniform(-0.06, 0.06, len(errs)),
                errs, "o", color=c, alpha=0.7)
    ax.axhline(0.2, color="#111111", ls="--", lw=1, label="±20 %")
    ax.set_xticks([0, 1], ["hybrid", "Nelder-Mead"])
    ax.set_ylabel("|gCaL rel err|")
    ax.set_title("Recovery, equal budget", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.grid(True, color="#e3e3e3", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / f"g5_basin_{tag}.png", dpi=150)
    print(f"saved → results/g5_basin_{tag}.json (+ evidence, figure)")


if __name__ == "__main__":
    main()

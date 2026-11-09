"""Gradient-repair diagnostics G1–G3 (v2 — corrected gates).

v2 fixes from review: G1's gate now requires agreement with a LOCAL
FINITE-DIFFERENCE CONVERGENCE REGION (reverse ≡ forward alone is not a
correctness test — both differentiate the same discretization); adds
dt 0.0125 for timestep convergence, directional cosine, and a synaptic-
regime variant. G2 classifies a zero reference slope as "unresolved", not
"wrong". G3 runs at the proposed operating timestep (0.025 ms) across
multiple currents, and its scope is explicit: a frozen-index interpolated
crossing-time derivative — NOT yet a complete event-time estimator (no
spike creation/deletion, matching, or full objective).

Scoped claims this file can support:
- G1: "AD agrees with the local FD slope of the discrete simulation at the
  tested operating points / regimes / timesteps" — nothing broader.
- G2: whether the straight-through direction is wrong / unresolved per
  coordinate under irrelevant perturbations.
- G3: crossing-time AD≡FD on the first K spikes at the tested dt.

Usage::

    python gradient_repair.py --x64                    # all, float64
    python gradient_repair.py --x64 --tests g1 --regime synaptic
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
EVIDENCE = Path(__file__).parent / "evidence"

FD_HS = tuple(10.0 ** -k for k in range(1, 9))  # 1e-1 … 1e-8
FD_PLATEAU_RTOL = 0.05


# --------------------------------------------------------------------------- #
# Testable helpers
# --------------------------------------------------------------------------- #

def detect_plateau(rows, rtol=FD_PLATEAU_RTOL):
    """Convergence plateau from an (h, slope) sweep — INDEPENDENT of any AD
    value: the slope at the smaller h of the last consecutive pair agreeing
    within ``rtol``, or None. Shared by G1 and G3 so neither can select its
    reference by closeness to autodiff."""
    conv = None
    for i in range(len(rows) - 1):
        a, b = rows[i][1], rows[i + 1][1]
        if abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300):
            conv = b
    return conv


def fd_sweep_converged(loss, s0, j, hs=FD_HS, rtol=FD_PLATEAU_RTOL):
    """Centered-FD h-sweep for coordinate j → (rows, plateau_or_None)."""
    rows = []
    for h in hs:
        e = np.zeros(len(s0))
        e[j] = h
        rows.append((float(h),
                     (float(loss(s0 + e)) - float(loss(s0 - e))) / (2 * h)))
    return rows, detect_plateau(rows, rtol)


def zero_insert_events(ev_coarse: np.ndarray, factor: int) -> np.ndarray:
    """One continuous event realization discretized at a finer step: events
    at coarse step k move to fine step k·factor (same physical times)."""
    out = np.zeros((ev_coarse.shape[0] * factor, ev_coarse.shape[1]),
                   dtype=ev_coarse.dtype)
    out[::factor] = ev_coarse
    return out


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def classify_direction(g, ref, zero_tol=1e-12) -> list[str]:
    """Per-coordinate: 'agree' / 'wrong' / 'unresolved' (|ref| ≤ zero_tol)."""
    out = []
    for gj, rj in zip(np.ravel(g), np.ravel(ref)):
        if abs(rj) <= zero_tol:
            out.append("unresolved")
        elif np.sign(gj) == np.sign(rj):
            out.append("agree")
        else:
            out.append("wrong")
    return out


# --------------------------------------------------------------------------- #

_BASE_DT = 0.1  # all event realizations are generated ONCE at this step and
#                 zero-inserted to finer steps so timestep comparisons share
#                 one continuous input realization


def _make_loss(regime, dt, horizon_s, current_nA=10.0):
    """Continuous mean-voltage objective in a given regime; returns
    (loss(scales)->scalar, n_spikes_at_base).

    regimes:
      current  — 1 cell, deterministic current injection.
      synaptic — 1 cell, SYNAPTICALLY DRIVEN FIRING REGIME (w 0.15 µS,
                 frozen renewal events). NOT the Deliverable-B fit setup.
      b_config — the EXACT Deliverable-B configuration: 8 MN, B's fit_ramp
                 command/protocol, B's connectivity + event seed, operating
                 point at B's truth (gCaL 1.6, w_exc 0.0105); mean pool
                 voltage as the continuous objective.
    """
    import jax.numpy as jnp
    from poolbuild import PoolTwin, TwinSpec
    from validation_protocol import _renewal_events

    n = int(horizon_s * 1000 / dt)
    factor = int(round(_BASE_DT / dt))
    if regime == "current":
        twin = PoolTwin(TwinSpec(n_mn=1, n_dd=1, n_inh=1, dt_ms=dt, seed=0))
        i_ext = jnp.full((n, 1), current_nA)

        def loss(scales):
            th = twin.default_theta()
            th["gCaL_scale"], th["gNaP_scale"] = scales[0], scales[1]
            return jnp.mean(twin._run_current(th, i_ext,
                                              spike_mode="hard")["v"])

        sp = twin.simulate_current(twin.default_theta(), np.asarray(i_ext))
        return loss, int(np.asarray(sp["spikes"]).sum())

    if regime == "synaptic":
        twin = PoolTwin(TwinSpec(n_mn=1, n_dd=100, n_inh=1, dt_ms=dt, seed=0,
                                 w_exc_uS=0.15))
        n0 = int(horizon_s * 1000 / _BASE_DT)
        t0_ = np.arange(n0) * _BASE_DT * 1e-3
        cmd = np.interp(t0_, [0, 0.1 * horizon_s, horizon_s],
                        [0, 60, 60]).astype(np.float32)
        ev0 = _renewal_events(np.random.default_rng(0),
                              np.tile(cmd[:, None], (1, 100)), _BASE_DT, 5)
        ev = jnp.asarray(zero_insert_events(ev0, factor))

        def loss(scales):
            th = twin.default_theta()
            th["gCaL_scale"], th["gNaP_scale"] = scales[0], scales[1]
            return jnp.mean(twin.simulate_events(th, ev,
                                                 spike_mode="hard")["v"])

        sp = twin.simulate_events(twin.default_theta(), ev, spike_mode="hard")
        return loss, int(np.asarray(sp["spikes"]).sum())

    # b_config — exact Deliverable-B pool/protocol/operating point
    from deliverable_b import make_commands, build_twin_and_events

    commands = make_commands(_BASE_DT, 1.0)
    twin0, events0 = build_twin_and_events(8, 0, 0, commands, _BASE_DT)
    twin = (twin0 if factor == 1 else
            PoolTwin(TwinSpec(n_mn=8, n_dd=100, n_inh=5, dt_ms=dt, seed=0)))
    ev0 = np.asarray(events0["fit_ramp"])[: int(horizon_s * 1000 / _BASE_DT)]
    ev = jnp.asarray(zero_insert_events(ev0, factor))

    def loss(scales):
        # s0 = (1, 1) is B's TRUTH operating point (gCaL 1.6, w_exc 0.0105)
        th = twin.default_theta()
        th["gCaL_scale"] = scales[0] * 1.6
        th["w_exc"] = scales[1] * 0.0105
        return jnp.mean(twin.simulate_events(th, ev, spike_mode="hard")["v"])

    th_b = twin.default_theta()
    th_b["gCaL_scale"] = jnp.asarray(1.6)
    th_b["w_exc"] = jnp.asarray(0.0105)
    sp = twin.simulate_events(th_b, ev, spike_mode="hard")
    return loss, int(np.asarray(sp["spikes"]).sum())


def g1_continuous_sensitivity(regime="current",
                              dts=(0.1, 0.05, 0.025, 0.0125),
                              horizons=(0.1, 0.5, 1.0)):
    import jax
    import jax.numpy as jnp

    rows = []
    for dt in dts:
        for horizon in horizons:
            t0 = time.time()
            loss, n_spk = _make_loss(regime, dt, horizon)
            s0 = jnp.array([1.0, 1.0])
            g_rev = np.asarray(jax.grad(loss)(s0), dtype=np.float64)
            g_fwd = np.array([jax.jvp(loss, (s0,), (jnp.array([1., 0.]),))[1],
                              jax.jvp(loss, (s0,), (jnp.array([0., 1.]),))[1]],
                             dtype=np.float64)
            fd_rows, fd_conv = {}, []
            for j in range(2):
                r, c = fd_sweep_converged(loss, np.array([1.0, 1.0]), j)
                fd_rows[j] = r
                fd_conv.append(c)
            converged = all(c is not None for c in fd_conv)
            if converged:
                conv = np.array(fd_conv)
                rel = float(np.max(np.abs(g_rev - conv)
                                   / np.maximum(np.abs(conv), 1e-300)))
                cos = cosine(g_rev, conv)
                verdict = "PASS" if (rel < FD_PLATEAU_RTOL and cos > 0.95) \
                    else "FAIL"
            else:
                rel, cos, verdict = None, None, "FD_UNCONVERGED"
            rows.append({
                "regime": regime, "dt_ms": dt, "horizon_s": horizon,
                "n_spikes": n_spk,
                "grad_rev": g_rev.tolist(), "grad_fwd": g_fwd.tolist(),
                "fd_converged_value": [c for c in fd_conv],
                "fd_sweeps": {str(j): fd_rows[j] for j in range(2)},
                "rel_rev_vs_fdconv": rel, "cosine": cos,
                "rev_vs_fwd_rel": float(np.max(
                    np.abs(g_rev - g_fwd)
                    / np.maximum(np.abs(g_fwd), 1e-300))),
                "verdict": verdict,
            })
            r = rows[-1]
            print(f"  [{regime}] dt {dt:6.4f} T {horizon:4.2f}s spk "
                  f"{n_spk:3d} | |g| {np.max(np.abs(g_rev)):.2e} | "
                  f"{verdict}"
                  + (f" rel {rel:.1e} cos {cos:.3f}" if converged else "")
                  + f" ({time.time()-t0:.0f}s)", flush=True)
    passed = [(r["dt_ms"], r["horizon_s"]) for r in rows
              if r["verdict"] == "PASS"]
    failed = [(r["dt_ms"], r["horizon_s"]) for r in rows
              if r["verdict"] == "FAIL"]
    unresolved = [(r["dt_ms"], r["horizon_s"]) for r in rows
                  if r["verdict"] == "FD_UNCONVERGED"]
    # prespecified required cases: every fine-dt (≤0.025) combo must PASS —
    # unresolved required cases block ok (review: the aggregate must not be
    # permissive about requested-but-unresolved configurations)
    required = [(r["dt_ms"], r["horizon_s"]) for r in rows
                if r["dt_ms"] <= 0.025]
    ok = (not failed) and all(c in passed for c in required)
    return {"ok": ok, "rows": rows,
            "passed_cases": passed, "failed_cases": failed,
            "unresolved_cases": unresolved,
            "required_cases_dt_le_0p025": required,
            "qualified_envelope": sorted(set(passed)),
            "scope": f"regime={regime}; claims limited to these operating "
                     "points/timesteps; FD_UNCONVERGED combos are "
                     "unresolvable by local FD, not passes"}


# --------------------------------------------------------------------------- #

def _g2_setup(dt, event_shift=0):
    import jax.numpy as jnp
    from poolbuild import PoolTwin, TwinSpec
    from validation_protocol import _renewal_events
    from deliverable_b import binned_rates_jnp

    twin = PoolTwin(TwinSpec(n_mn=3, n_dd=100, n_inh=5, dt_ms=dt, seed=0))
    dur = 2.0
    n = int(dur * 1000 / dt)
    t = np.arange(n) * dt * 1e-3
    cmd = np.interp(t, [0, 0.3, 1.0, 1.7, 2.0], [0, 0, 60, 0, 0]
                    ).astype(np.float32)
    ev = _renewal_events(np.random.default_rng(0),
                         np.tile(cmd[:, None], (1, 100)), dt, 5)
    if event_shift:
        ev = np.roll(ev, event_shift, axis=0)
    ev = jnp.asarray(ev)

    def rates(theta_vals, spike_mode):
        th = twin.default_theta()
        th["gCaL_scale"] = jnp.asarray(theta_vals[0])
        th["w_exc"] = jnp.asarray(theta_vals[1])
        out = twin.simulate_events(th, ev, spike_mode=spike_mode)
        return binned_rates_jnp(out["spikes"], dt)

    target = rates((1.5, 0.011), "hard")
    return twin, rates, target


def g2_surrogate_dependence():
    import jax
    import jax.numpy as jnp
    import poolbuild
    from myogen.simulator.jaxley import jax_models

    theta0 = (1.0, 0.009)
    variants = ([("beta", b) for b in (5.0, 10.0, 20.0)]
                + [("vth", v) for v in (49.0, 50.0, 51.0)]
                + [("dt", d) for d in (0.1, 0.05)]
                + [("evshift", s) for s in (0, 1)])
    base_beta = jax_models.SURROGATE_BETA
    base_vth = poolbuild.SPIKE_THRESHOLD_MV
    rows = []
    try:
        for kind, val in variants:
            jax_models.SURROGATE_BETA = val if kind == "beta" else base_beta
            poolbuild.SPIKE_THRESHOLD_MV = val if kind == "vth" else base_vth
            dt = val if kind == "dt" else 0.1
            shift = val if kind == "evshift" else 0
            twin, rates, target = _g2_setup(dt, event_shift=int(shift))
            tgt = jax.lax.stop_gradient(target)

            def sur_loss(z):
                r = rates((z[0], z[1]), "surrogate")
                return jnp.mean((r - tgt) ** 2)

            leaf_dtype = (jnp.float64 if jax.config.jax_enable_x64
                          else jnp.float32)
            g = np.asarray(jax.grad(sur_loss)(
                jnp.asarray(theta0, dtype=leaf_dtype)), dtype=np.float64)

            def hard_loss(tv):
                return float(jnp.mean((rates(tv, "hard") - tgt) ** 2))

            ref = np.zeros(2)
            for j, h in enumerate((0.05 * theta0[0], 0.05 * theta0[1])):
                up, dn = list(theta0), list(theta0)
                up[j] += h
                dn[j] -= h
                ref[j] = (hard_loss(tuple(up)) - hard_loss(tuple(dn))) / (2 * h)
            cls = classify_direction(g, ref)
            rows.append({"variant": f"{kind}={val}",
                         "surrogate_grad": g.tolist(),
                         "crn_fd_ref": ref.tolist(),
                         "classification": cls,
                         "leaf_dtype": str(leaf_dtype.__name__),
                         "grad_finite": bool(np.all(np.isfinite(g)))})
            print(f"  {rows[-1]['variant']:>12}: {cls}", flush=True)
    finally:
        jax_models.SURROGATE_BETA = base_beta
        poolbuild.SPIKE_THRESHOLD_MV = base_vth
    counts = {"gCaL": {"agree": 0, "wrong": 0, "unresolved": 0},
              "w_exc": {"agree": 0, "wrong": 0, "unresolved": 0}}
    for r in rows:
        counts["gCaL"][r["classification"][0]] += 1
        counts["w_exc"][r["classification"][1]] += 1
    ok = all(c["wrong"] == 0 for c in counts.values())
    mags = [float(np.max(np.abs(r["surrogate_grad"]))) for r in rows]
    return {"ok": ok, "n_variants": len(rows), "per_coordinate": counts,
            "surrogate_grad_magnitude_range": [min(mags), max(mags)],
            "magnitude_note": "sign agreement does NOT establish usefulness —"
                              " magnitudes remain enormous and must be handled"
                              " by the optimizer (G4 question)",
            "rows": rows}


# --------------------------------------------------------------------------- #

def hard_crossings(v: np.ndarray, v_th: float) -> np.ndarray:
    return np.flatnonzero((v[1:] > v_th) & (v[:-1] <= v_th))


def interp_time(v, k, v_th, dt_ms):
    """Interpolated crossing time (s) at frozen step index k — differentiable
    in the voltages (discrete implicit-function form)."""
    vk, vk1 = v[k], v[k + 1]
    return (k + (v_th - vk) / (vk1 - vk)) * (dt_ms * 1e-3)


def g3_event_time_gradient(dts=(0.025, 0.1), currents=(12.0, 15.0, 30.0),
                           horizon_s=0.3,
                           fd_hs=(3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
                           max_spikes=8):
    import jax
    import jax.numpy as jnp
    from poolbuild import PoolTwin, TwinSpec, SPIKE_THRESHOLD_MV

    cases = []
    for dt in dts:
        twin = PoolTwin(TwinSpec(n_mn=1, n_dd=1, n_inh=1, dt_ms=dt, seed=0))
        n = int(horizon_s * 1000 / dt)
        for amp in currents:
            i_ext = jnp.full((n, 1), amp)

            def v_trace(scales):
                th = twin.default_theta()
                th["gCaL_scale"], th["gNaP_scale"] = scales[0], scales[1]
                return twin._run_current(th, i_ext,
                                         spike_mode="hard")["v"][:, 0]

            def times_concrete(scales):
                v = np.asarray(v_trace(jnp.asarray(scales)))
                ks = hard_crossings(v, SPIKE_THRESHOLD_MV)
                return ([float(interp_time(v, int(k), SPIKE_THRESHOLD_MV, dt))
                         for k in ks], ks)

            s0 = np.array([1.0, 1.0])
            ts0, ks0 = times_concrete(s0)
            n_spk = len(ks0)
            per_spike = []
            for j_spike, k in enumerate(ks0[:max_spikes]):
                k = int(k)

                def tj_ad(scales):
                    return interp_time(v_trace(scales), k,
                                       SPIKE_THRESHOLD_MV, dt)

                g_ad = np.asarray(jax.grad(tj_ad)(jnp.asarray(s0)),
                                  dtype=np.float64)
                # AD-INDEPENDENT reference: per-coordinate FD sweep over h,
                # plateau detected from consecutive-h agreement alone
                # (count-inconsistent h's excluded); AD compared only after.
                plateau = []
                sweeps = {}
                for p in range(2):
                    rows_fd = []
                    for h in fd_hs:
                        e = np.zeros(2)
                        e[p] = h
                        tsp, ksp = times_concrete(s0 + e)
                        tsm, ksm = times_concrete(s0 - e)
                        if len(ksp) != n_spk or len(ksm) != n_spk:
                            continue  # spike creation/deletion — invalid h
                        rows_fd.append(
                            (h, (tsp[j_spike] - tsm[j_spike]) / (2 * h)))
                    sweeps[p] = rows_fd
                    plateau.append(detect_plateau(rows_fd))
                if all(c is not None for c in plateau):
                    conv = np.array(plateau)
                    rel = float(np.max(np.abs(conv - g_ad)
                                       / np.maximum(np.abs(conv), 1e-300)))
                    fd_res = {"plateau": conv.tolist(), "rel": rel,
                              "sweeps": {str(p): sweeps[p] for p in sweeps}}
                else:
                    fd_res = None  # no AD-independent plateau → unresolved
                per_spike.append({"spike": j_spike, "t_s": ts0[j_spike],
                                  "grad_ad": g_ad.tolist(), "fd": fd_res})
            n_pass = 0
            for r in per_spike:
                if r["fd"] is not None and r["fd"]["rel"] < 0.05:
                    n_pass += 1
                else:
                    break  # contiguous validity horizon
            cases.append({"dt_ms": dt, "current_nA": amp, "n_spikes": n_spk,
                          "n_valid_leading_spikes": n_pass,
                          "per_spike": per_spike})
            print(f"  dt {dt:5.3f} I {amp:4.1f} nA: {n_spk} spikes, "
                  f"leading AD≡FD (<5%): {n_pass}", flush=True)
    fine = [c for c in cases if c["dt_ms"] == min(dts) and c["n_spikes"] > 0]
    ok = bool(fine) and all(c["n_valid_leading_spikes"] >= 3 for c in fine)
    return {"ok": ok, "cases": cases,
            "scope": "frozen-index interpolated crossing times only — NOT a "
                     "complete event-time estimator (no creation/deletion, "
                     "matching, or full objective)"}


# --------------------------------------------------------------------------- #

TESTS = {"g1": g1_continuous_sensitivity,
         "g2": g2_surrogate_dependence,
         "g3": g3_event_time_gradient}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", choices=sorted(TESTS),
                    default=sorted(TESTS))
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--regime",
                    choices=["current", "synaptic", "b_config"],
                    default="current")
    args = ap.parse_args()

    import jax
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    from sensitivity_checks import _manifest

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    report = {"manifest": _manifest(), "x64": args.x64,
              "regime": args.regime}
    for tid in args.tests:
        print(f"[{tid}] ...", flush=True)
        t0 = time.time()
        try:
            if tid == "g1" and args.regime == "b_config":
                # horizons must cover the FIRING portion of the
                # ramp (recruitment starts ~1.5 s) — a short
                # window only certifies the subthreshold regime
                res = TESTS[tid](regime=args.regime,
                                 dts=(0.1, 0.025),
                                 horizons=(2.0, 4.0))
            elif tid == "g1":
                res = TESTS[tid](regime=args.regime)
            else:
                res = TESTS[tid]()
        except Exception as e:
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res["runtime_s"] = round(time.time() - t0, 1)
        report[tid] = res
        print(f"[{tid}] {'OK' if res.get('ok') else 'ATTENTION'} "
              f"({res['runtime_s']}s)", flush=True)

    suffix = ("x64" if args.x64 else "f32") + (
        "_syn" if args.regime == "synaptic" else "")
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"gradient_repair_{suffix}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"gradient_repair_{suffix}_{stamp}.json").write_text(payload)
    print(f"saved → results/gradient_repair_{suffix}.json (+ evidence)")


if __name__ == "__main__":
    main()

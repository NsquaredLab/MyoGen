"""Deliverable A — scoped forward and gradient diagnostics.

Implements the gradient-qualification tests of the revised roadmap
(``docs/DIGITAL_TWIN_ROADMAP_REVISED.md`` §6.2B / §12A). Every test prints a
verdict and all raw numbers land in ``results/sensitivity_checks.json`` —
failures are preserved, not hidden.

T1  Parameter plumbing — each theta knob changes exactly the intended jaxley
    leaves (gCaL→caL_gcaLbar only, gNaP→napp_gnapbar only; input knobs touch
    no jaxley leaf), and each knob measurably changes the forward output.
T2  Continuous sensitivities — ONE cell, deterministic SUBTHRESHOLD current
    (smooth voltage trajectory, no events): autodiff dL/dθ vs centered finite
    differences across perturbation sizes; reports the FD convergence region.
    This qualifies the derivative of the continuous dynamics, separate from
    any spike/surrogate issue.
T3  Event-parameter plumbing — with a FIXED PRNG key the pathwise derivative
    w.r.t. drive_gain is a fixed-realization derivative, not the derivative
    of the expected spike count. Reports: surrogate/pathwise gradient sign vs
    the finite-perturbation slope of the hard loss (same key), and the slope
    across independent keys (expected-value sense). Divergence between these
    is a property of the estimator and is recorded, not patched.
T4  Horizon diagnostics — |grad| of a smoothed-rate objective vs simulated
    duration (suprathreshold, spiking): quantifies the reported long-horizon
    explosion instead of asserting it.
T5  Hard-forward descent — surrogate sign-descent directions on the
    Deliverable-B parameter pair (gCaL_scale, w_exc) with FROZEN event input:
    fraction of proposed steps that actually reduce the scientific hard loss.

Usage::

    python sensitivity_checks.py            # all tests, small CPU sizes
    python sensitivity_checks.py --tests t1 t2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from poolbuild import PoolTwin, TwinSpec
from validation_protocol import _renewal_events

RESULTS = Path(__file__).parent / "results"
# Gate evidence is version-controllable (results/ is gitignored by design).
EVIDENCE = Path(__file__).parent / "evidence"


def _safe_norm(g) -> float:
    """L2 norm in float64 — float32 squaring overflows at ~1e19 components,
    which previously misreported finite ~1e28 gradients as Inf."""
    return float(np.linalg.norm(np.asarray(g, dtype=np.float64)))


def _manifest() -> dict:
    """Reproducibility manifest attached to every report."""
    import subprocess

    import hashlib

    repo = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo, capture_output=True, text=True
                             ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    cwd=repo, capture_output=True, text=True
                                    ).stdout.strip())
    except Exception:
        rev, dirty = "unknown", None
    # The diagnostic sources are untracked, so the git revision alone cannot
    # reconstruct this run — hash the actual files used.
    here = Path(__file__).parent
    file_hashes = {}
    for fname in ("sensitivity_checks.py", "poolbuild.py", "observables.py",
                  "validation_protocol.py", "protocols.py",
                  "gradient_repair.py", "deliverable_b.py", "deliverable_c.py",
                  "pic_grid.py", "r2a_profiles.py", "acceptance.py"):
        f = here / fname
        if f.exists():
            file_hashes[fname] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    # simulator internals the experiments depend on (working tree is dirty —
    # the git revision alone cannot reconstruct these)
    for rel in ("myogen/simulator/jaxley/jax_models.py",
                "myogen/simulator/jaxley/channels/nerlab_channels.py",
                "myogen/simulator/jaxley/cells_api.py",
                "myogen/simulator/jaxley/populations/motor_neurons.py"):
        f = repo / rel
        if f.exists():
            file_hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    import jaxley

    return {
        "git_revision": rev or "unknown",
        "git_dirty": dirty,
        "provenance_note": ("experiments/twin sources are untracked by git; "
                            "reconstruction relies on the file hashes below, "
                            "not the revision"),
        "source_sha256_16": file_hashes,
        "versions": {"jax": jax.__version__, "jaxley": jaxley.__version__,
                     "numpy": np.__version__},
        "device": jax.devices()[0].platform,
        "precision": ("float64 (jax_enable_x64=True)"
                      if jax.config.jax_enable_x64
                      else "float32 (JAX default)"),
        "dt_ms": 0.1,
        "voltage_frame": "NERLab 1952-HH (V_rest≈0, spike threshold +50 mV)",
        "seed_taxonomy": {
            "connectivity": "TwinSpec.seed (numpy default_rng)",
            "events": "jax.random.PRNGKey (rate path) / numpy default_rng "
                      "(frozen renewal events)",
            "fd_probes": "deterministic, no sampling",
        },
        "objectives": {
            "t2": "mean soma voltage, subthreshold deterministic current",
            "t3_t4": "mean exponentially-smoothed rate (tau 100 ms)",
            "t5": "MSE of smoothed rates (tau 100 ms) vs frozen-event target",
        },
        "scope": ("this Jaxley harness configuration only — NOT a validated "
                  "NEURON-matched forward model; conclusions are specific to "
                  "the stated parameter points, objectives, and frozen "
                  "input realizations"),
    }


def _smoothed_rates_jax(spikes, dt_ms, tau_ms=100.0):
    a = jnp.exp(-dt_ms / tau_ms)
    inst = spikes / (dt_ms * 1e-3)

    def step(r, x):
        r = a * r + (1.0 - a) * x
        return r, r

    _, rates = jax.lax.scan(step, jnp.zeros(spikes.shape[1]), inst)
    return rates


# --------------------------------------------------------------------------- #

def t1_parameter_plumbing() -> dict:
    """Which jaxley leaves does each theta knob touch; does it move the output?"""
    twin = PoolTwin(TwinSpec(n_mn=2, n_dd=5, n_inh=5, seed=0))
    theta0 = twin.default_theta()
    p0 = twin._scaled_jaxley_params(theta0)

    leaf_map = {}
    for key in theta0:
        th = dict(theta0)
        th[key] = th[key] * 1.5
        p1 = twin._scaled_jaxley_params(th)
        changed = []
        for name in p0:
            a, b = np.asarray(p0[name]), np.asarray(p1[name])
            if a.dtype == object or not np.issubdtype(a.dtype, np.number):
                continue  # non-numeric leaves cannot be theta-dependent
            if np.issubdtype(a.dtype, np.floating):
                both = np.isfinite(a) & np.isfinite(b)
                if np.any(a[both] != b[both]):
                    changed.append(name)
            elif not np.array_equal(a, b):
                changed.append(name)
        leaf_map[key] = changed

    expected = {
        "gCaL_scale": ["caL_gcaLbar"],
        "gNaP_scale": ["napp_gnapbar"],
        "drive_gain": [],
        "w_exc": [],
        "w_inh": [],
    }
    plumbing_ok = {k: leaf_map[k] == expected[k] for k in expected}

    # forward effect: a brief event burst; does each knob move soma voltage?
    n_steps = 3000
    conn = np.ones((2, 1), dtype=np.float32)
    burst = np.zeros((n_steps, 1), dtype=np.float32)
    burst[100:200, 0] = 1.0
    out0 = np.asarray(twin.simulate_events(theta0, burst, exc_connectivity=conn)["v"])
    effects = {}
    for key in theta0:
        th = dict(theta0)
        th[key] = th[key] * 1.5
        kwargs = dict(exc_connectivity=conn)
        if key == "w_inh":  # needs inhibitory input to matter
            kwargs.update(inh_events=burst, inh_connectivity=conn)
        out1 = np.asarray(twin.simulate_events(th, burst, **kwargs)["v"])
        base = out0 if key != "w_inh" else np.asarray(
            twin.simulate_events(theta0, burst, **kwargs)["v"])
        effects[key] = float(np.max(np.abs(out1 - base)))

    # drive_gain scales the stochastic generator rate → no effect in the
    # deterministic events path (by design); check it via the rate path.
    _, cmd = __import__("protocols").step(0.1, t_pre_s=0.05, t_on_s=0.2,
                                          t_post_s=0.05, level=40.0)
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
    key0 = jax.random.PRNGKey(0)
    v_a = np.asarray(twin.simulate(theta0, drives, key0, "hard")["v"])
    th = dict(theta0)
    th["drive_gain"] = th["drive_gain"] * 1.5
    v_b = np.asarray(twin.simulate(th, drives, key0, "hard")["v"])
    effects["drive_gain (rate path)"] = float(np.max(np.abs(v_b - v_a)))

    # Initialization audit (revised roadmap: parameter-dependent vs fixed
    # initialization must be declared). In this harness the initial jaxley
    # state is computed ONCE at build time from the default parameters and is
    # NOT re-equilibrated when theta changes — verify and record that.
    v0 = np.asarray(twin._states0["v"])
    rec = np.concatenate([np.asarray(twin._rec_soma), np.asarray(twin._rec_dend)])
    v_rec = v0[rec]
    v_other = np.delete(v0, rec)
    init_audit = {
        "initial_state_theta_dependent": False,
        "recorded_compartments_at_rest": bool(
            np.allclose(v_rec, 0.0, atol=1e-6)),
        "nonrecorded_init_v_range_mV": (
            [float(v_other.min()), float(v_other.max())]
            if v_other.size else None),
        "observation": ("non-recorded compartment(s) — the axon — initialize "
                        "at −1.0 mV, not the NERLab rest; a jaxley init "
                        "quirk, decays within ms but is part of x0 and is "
                        "recorded here rather than hidden"),
        "caveat": ("theta changes do not re-equilibrate the initial state; "
                   "fits must either discard the initial transient or treat "
                   "x0 explicitly (roadmap x0_sc)"),
    }

    ok = (all(plumbing_ok.values())
          and all(v > 0 for k, v in effects.items() if k != "drive_gain")
          and init_audit["recorded_compartments_at_rest"])
    return {"ok": bool(ok), "leaf_map": leaf_map,
            "plumbing_ok": plumbing_ok, "max_output_effect_mV": effects,
            "initialization_audit": init_audit}


def t2_continuous_sensitivities(current_nA: float = 1.0,
                                duration_s: float = 1.0) -> dict:
    """Single cell, subthreshold deterministic current: autodiff vs FD."""
    twin = PoolTwin(TwinSpec(n_mn=1, n_dd=1, n_inh=1, seed=0))
    n_steps = int(duration_s * 1000 / twin.spec.dt_ms)
    i_ext = np.zeros((n_steps, 1), dtype=np.float32)
    i_ext[int(0.1 * n_steps):, 0] = current_nA
    i_ext = jnp.asarray(i_ext)

    def loss(scales):
        th = twin.default_theta()
        th["gCaL_scale"], th["gNaP_scale"] = scales[0], scales[1]
        out = twin._run_current(th, i_ext, spike_mode="hard")
        return jnp.mean(out["v"])

    s0 = jnp.array([1.0, 1.0])
    v0 = float(loss(s0))
    subthreshold = True
    out0 = twin.simulate_current(twin.default_theta(), np.asarray(i_ext))
    if float(np.max(np.asarray(out0["v"]))) > 50.0:
        subthreshold = False  # recorded, test still runs

    g_ad = np.asarray(jax.grad(loss)(s0))
    report = {"ok": True, "loss0": v0, "subthreshold": subthreshold,
              "autodiff": g_ad.tolist(), "fd": {}}
    for j, name in enumerate(["gCaL_scale", "gNaP_scale"]):
        rows = {}
        for h in (1e-1, 1e-2, 1e-3, 1e-4):
            e = np.zeros(2)
            e[j] = h
            fd = (float(loss(s0 + e)) - float(loss(s0 - e))) / (2 * h)
            rel = abs(fd - g_ad[j]) / max(abs(g_ad[j]), 1e-12)
            rows[h] = {"fd": fd, "rel_err_vs_autodiff": rel}
        report["fd"][name] = rows
        best = min(r["rel_err_vs_autodiff"] for r in rows.values())
        if best > 0.05:
            report["ok"] = False
    return report


def t3_event_parameter_plumbing(level_hz: float = 60.0) -> dict:
    """What does d(loss)/d(drive_gain) mean through the stochastic generator?

    Requires a FIRING regime — in a silent pool the hard slopes are exactly 0
    while the surrogate still reports a nonzero gradient through the
    fast-sigmoid tail (observed: +81.7 with zero spikes). That mismatch is
    itself a finding, but the sign comparison needs spikes.
    """
    import protocols

    twin = PoolTwin(TwinSpec(n_mn=2, n_dd=100, n_inh=5, seed=0))
    _, cmd = protocols.step(twin.spec.dt_ms, t_pre_s=0.1, t_on_s=0.8,
                            t_post_s=0.1, level=level_hz)
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
    key0 = jax.random.PRNGKey(0)

    def loss(gain, key, mode):
        th = twin.default_theta()
        th["drive_gain"] = gain
        out = twin._run(th, drives, key, spike_mode=mode)
        return jnp.mean(_smoothed_rates_jax(out["spikes"], twin.spec.dt_ms))

    out0 = twin.simulate(twin.default_theta(), drives, key0, "hard")
    n_spikes = int(np.asarray(out0["spikes"]).sum())

    g_sur = float(jax.grad(lambda g: loss(g, key0, "surrogate"))(jnp.float32(1.0)))

    # fixed-key finite perturbations of the HARD loss (same realization)
    eps = 0.05
    slope_fixed = (float(loss(jnp.float32(1 + eps), key0, "hard"))
                   - float(loss(jnp.float32(1 - eps), key0, "hard"))) / (2 * eps)
    # expected-value slope across independent keys (per-key slopes preserved)
    keys = [jax.random.PRNGKey(100 + i) for i in range(8)]
    per_key_slopes = [
        (float(loss(jnp.float32(1 + eps), k, "hard"))
         - float(loss(jnp.float32(1 - eps), k, "hard"))) / (2 * eps)
        for k in keys
    ]
    slope_expected = float(np.mean(per_key_slopes))
    slope_sd = float(np.std(per_key_slopes))

    return {
        "ok": bool(n_spikes > 0 and np.sign(g_sur) == np.sign(slope_expected) != 0),
        "n_spikes_at_init": n_spikes,
        "surrogate_grad": g_sur,
        "hard_slope_fixed_key_eps5pct": slope_fixed,
        "hard_slope_expected_8keys_eps5pct": slope_expected,
        "hard_slope_per_key": per_key_slopes,
        "hard_slope_sd_across_keys": slope_sd,
        "note": ("pathwise/fixed-key derivatives are fixed-realization "
                 "quantities; agreement in sign with the expected-value slope "
                 "is required, magnitudes are not expected to match"),
    }


def t4_horizon_diagnostics(current_nA: float = 10.0) -> dict:
    """|grad| vs simulated horizon, suprathreshold (spiking) single cell."""
    twin = PoolTwin(TwinSpec(n_mn=1, n_dd=1, n_inh=1, seed=0))
    rows = {}
    for horizon_s in (0.5, 1.0, 2.0, 4.0):
        n = int(horizon_s * 1000 / twin.spec.dt_ms)
        i_ext = jnp.full((n, 1), current_nA, dtype=jnp.float32)

        def loss(scales):
            th = twin.default_theta()
            th["gCaL_scale"], th["gNaP_scale"] = scales[0], scales[1]
            out = twin._run_current(th, i_ext, spike_mode="surrogate")
            return jnp.mean(_smoothed_rates_jax(out["spikes"], twin.spec.dt_ms))

        g = np.asarray(jax.grad(loss)(jnp.array([1.0, 1.0])))
        out = twin.simulate_current(twin.default_theta(), np.asarray(i_ext),
                                    spike_mode="hard")
        rows[horizon_s] = {
            "n_spikes": int(np.asarray(out["spikes"]).sum()),
            "grad_gCaL": float(g[0]),
            "grad_gNaP": float(g[1]),
            "components_finite": bool(np.all(np.isfinite(g))),
            # float64 norm: float32 squaring overflows for ~1e19 components
            # and previously misreported finite gradients as Inf.
            "grad_norm_f64": _safe_norm(g),
        }
    norms = [r["grad_norm_f64"] for r in rows.values()]
    finite = [r["components_finite"] for r in rows.values()]
    growth = (norms[-1] / max(norms[0], 1e-12)) if all(finite) else float("nan")
    return {
        "ok": bool(all(finite)),
        "per_horizon": rows,
        "norm_growth_shortest_to_longest": float(growth),
        "note": ("distinguish: (a) dynamical explosion of finite components "
                 "(e.g. ~1e28 at 1 s), (b) NaN components (true breakdown), "
                 "(c) diagnostic float32 norm overflow — eliminated here by "
                 "the float64 norm. 'ok' means all components finite at every "
                 "horizon, NOT that magnitudes are usable."),
    }


def t5_hard_forward_descent(
    n_mn: int = 3,
    duration_s: float = 4.0,
    conn_seeds=(0, 1),
    event_seeds=(0, 1, 2, 3),
    true_points=((1.5, 0.011), (0.7, 0.008)),
    lambdas=(0.03, 0.1),
    coord_lambda: float = 0.1,
) -> dict:
    """Do surrogate sign-descent steps reduce the scientific hard loss?

    Deliverable-B parameter pair (gCaL_scale, w_exc), frozen event input,
    evaluated across connectivity seeds × event seeds × true-parameter
    points. For each case:

    * JOINT test — both coordinates stepped along ``-sign(grad)`` (what the
      clipped sign-descent optimizer actually does); accepted-step fraction
      over ``lambdas``.
    * COORDINATE-WISE test — each parameter perturbed alone by
      ``exp(±coord_lambda)``; the hard-loss-descending direction (if resolved
      beyond a tolerance) is compared against that coordinate's surrogate
      sign. A joint direction being wrong does NOT establish each coordinate
      is independently wrong — only this probe can.

    Scope: this harness configuration, objective, and frozen realizations
    only; not a NEURON-validated forward model.
    """
    twins = {cs: PoolTwin(TwinSpec(n_mn=n_mn, n_dd=100, n_inh=5, seed=cs))
             for cs in conn_seeds}
    dt = next(iter(twins.values())).spec.dt_ms
    n = int(duration_s * 1000 / dt)
    t = np.arange(n) * dt * 1e-3
    cmd = np.interp(t, [0, 0.5, duration_s / 2, duration_s - 0.5, duration_s],
                    [0, 0, 60.0, 0, 0]).astype(np.float32)
    rate_mat = np.tile(cmd[:, None], (1, 100))

    cases = []
    n_failures = 0
    for cs, twin in twins.items():
        for es in event_seeds:
            events = _renewal_events(np.random.default_rng(es), rate_mat, dt, 5)
            ev = {"exc": jnp.asarray(events),
                  "inh": jnp.zeros((n, 5), jnp.float32),
                  "exc_conn": twin._exc_mat, "inh_conn": twin._inh_mat}
            for tp in true_points:
                case = {"conn_seed": cs, "event_seed": es,
                        "true_theta": {"gCaL_scale": tp[0], "w_exc": tp[1]}}
                try:
                    theta_true = twin.default_theta()
                    theta_true["gCaL_scale"] = jnp.float32(tp[0])
                    theta_true["w_exc"] = jnp.float32(tp[1])
                    obs = twin._run_events(theta_true, ev, spike_mode="hard")
                    n_obs = int(np.asarray(obs["spikes"]).sum())
                    case["n_obs_spikes"] = n_obs
                    if n_obs == 0:
                        case["skipped"] = "silent observation regime"
                        cases.append(case)
                        continue
                    target = jax.lax.stop_gradient(
                        _smoothed_rates_jax(obs["spikes"], dt))

                    def loss(th, mode):
                        out = twin._run_events(th, ev, spike_mode=mode)
                        return jnp.mean(
                            (_smoothed_rates_jax(out["spikes"], dt) - target) ** 2)

                    theta0 = twin.default_theta()
                    L0 = float(loss(theta0, "hard"))
                    case["loss_at_init_hard"] = L0

                    def loss_fit(vals):
                        th = dict(theta0)
                        th["gCaL_scale"], th["w_exc"] = vals[0], vals[1]
                        return loss(th, "surrogate")

                    g = np.asarray(jax.grad(loss_fit)(jnp.array(
                        [float(theta0["gCaL_scale"]), float(theta0["w_exc"])])))
                    case["surrogate_grad"] = g.tolist()
                    case["grad_finite"] = bool(np.all(np.isfinite(g)))
                    if not case["grad_finite"]:
                        n_failures += 1
                        cases.append(case)
                        continue
                    d = -np.sign(g)
                    case["joint_sign_direction"] = d.tolist()

                    def hard_at(scales):
                        th = dict(theta0)
                        th["gCaL_scale"] = jnp.float32(
                            float(theta0["gCaL_scale"]) * scales[0])
                        th["w_exc"] = jnp.float32(
                            float(theta0["w_exc"]) * scales[1])
                        return float(loss(th, "hard"))

                    joint = {}
                    for lam in lambdas:
                        Ls = hard_at(np.exp(lam * d))
                        joint[lam] = {"hard_loss": Ls, "improved": bool(Ls < L0)}
                    case["joint_steps"] = joint

                    # coordinate-wise ground truth vs surrogate sign.
                    # BASELINE-AWARE: a direction counts as descent only if
                    # it improves on L0 — the smaller of two worse-than-L0
                    # losses is "less bad", not descent.
                    coords = {}
                    tol = 1e-4 * max(L0, 1e-9)
                    for j, name in enumerate(["gCaL_scale", "w_exc"]):
                        up = np.ones(2)
                        up[j] = np.exp(coord_lambda)
                        dn = np.ones(2)
                        dn[j] = np.exp(-coord_lambda)
                        L_up, L_dn = hard_at(up), hard_at(dn)
                        if min(L_up, L_dn) >= L0 - tol:
                            true_dir = 0  # no descent direction at this step
                        elif abs(L_up - L_dn) <= tol:
                            true_dir = 0  # both improve equally — unresolved
                        else:
                            true_dir = 1 if L_up < L_dn else -1
                        sur_dir = int(d[j])
                        coords[name] = {
                            "hard_loss_baseline": L0,
                            "hard_loss_up": L_up, "hard_loss_down": L_dn,
                            "true_descent_dir": true_dir,
                            "surrogate_dir": sur_dir,
                            "agrees": (None if true_dir == 0
                                       else bool(sur_dir == true_dir)),
                        }
                    case["coordinate_probes"] = coords
                except Exception as e:
                    case["error"] = f"{type(e).__name__}: {e}"
                    n_failures += 1
                cases.append(case)

    evaluated = [c for c in cases if "joint_steps" in c]
    all_joint = [s["improved"] for c in evaluated
                 for s in c["joint_steps"].values()]
    accepted_fraction = (float(np.mean(all_joint)) if all_joint else float("nan"))
    coord_agreement = {}
    for name in ("gCaL_scale", "w_exc"):
        votes = [c["coordinate_probes"][name]["agrees"] for c in evaluated
                 if c["coordinate_probes"][name]["agrees"] is not None]
        coord_agreement[name] = {
            "n_resolved": len(votes),
            "n_unresolved": len(evaluated) - len(votes),
            "agreement_rate": (float(np.mean(votes)) if votes
                               else float("nan")),
        }
    # Cases sharing an event seed are CORRELATED — the effective number of
    # independent event realizations is len(event_seeds), not n_cases.
    per_event_seed = {}
    for es in event_seeds:
        sub = [s["improved"] for c in evaluated if c["event_seed"] == es
               for s in c["joint_steps"].values()]
        per_event_seed[es] = {
            "n_steps": len(sub),
            "improved_fraction": (float(np.mean(sub)) if sub
                                  else float("nan")),
        }

    return {
        "n_independent_event_realizations": len(event_seeds),
        "per_event_seed_joint_improvement": per_event_seed,
        # gate: promote only if steps reliably improve and nothing failed
        "ok": bool(all_joint) and accepted_fraction >= 0.8 and n_failures == 0,
        "n_cases": len(cases),
        "n_evaluated": len(evaluated),
        "n_failures": n_failures,
        "accepted_step_fraction": accepted_fraction,
        "coordinate_agreement": coord_agreement,
        "cases": cases,
    }


TESTS = {
    "t1": ("parameter plumbing", t1_parameter_plumbing),
    "t2": ("continuous sensitivities (FD vs autodiff)", t2_continuous_sensitivities),
    "t3": ("event-parameter plumbing", t3_event_parameter_plumbing),
    "t4": ("horizon diagnostics", t4_horizon_diagnostics),
    "t5": ("hard-forward descent", t5_hard_forward_descent),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", choices=sorted(TESTS), default=sorted(TESTS))
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    report = {"manifest": _manifest()}
    for tid in args.tests:
        name, fn = TESTS[tid]
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:  # preserve raw failures
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        res["runtime_s"] = round(time.time() - t0, 1)
        report[tid] = res
        print(f"[{tid}] {name}: {'OK' if res.get('ok') else 'ATTENTION'} "
              f"({res['runtime_s']}s)")

    payload = json.dumps(report, indent=2, default=float)
    out = RESULTS / "sensitivity_checks.json"
    out.write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ev = EVIDENCE / f"sensitivity_checks_{stamp}.json"
    ev.write_text(payload)
    print(f"full report → {out}\nevidence copy (version-controllable) → {ev}")


if __name__ == "__main__":
    main()

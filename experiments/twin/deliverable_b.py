"""Deliverable B (v3) — two-parameter hard-forward inverse experiment.

Revised roadmap §12B / ladder Level B. v3 fixes the v2 profile-grid bug: the
dense axes did NOT contain the true parameter values, so the profile curves
could never reach the known zero-loss optimum and every "profile minimum"
(1.6–3.4 Hz²) was grid-sampling error, not physiology. v3:

* ``axis_with_truth``: both dense axes contain the exact true values; for
  every replicate and observation model the profile at the truth is
  ASSERTED to be exactly 0 (same model, same events — anything else is a
  bug).
* Adaptive nuisance refinement: hard-spike loss is discontinuous, so each
  profile point refines the nuisance axis locally (3 extra log-spaced
  evaluations between the coarse bracketing neighbours, per observation
  model) over a shared evaluation cache.
* EPS is computed PER replicate AND PER observation model (dt-halving at
  the truth, per-unit terms averaged over that model's units), not one
  global value; the conservative maximum is also reported.
* Smoke and canonical outputs are separated: results/figures are tagged
  ``full`` or ``smoke`` from the actual run configuration, so a smoke run
  can never overwrite scientific evidence again.

Declared setup (unchanged from v2): unknowns theta = (gCaL_scale, w_exc);
fixed/known: gNaP ×1, no inhibition, connectivity, drive shape AND frozen
event realizations (conditional fit); fit protocols ramp + ramp–hold–ramp;
untouched held-out slower ramp; observation models all-units vs
lowest-threshold half (active counts reported); observable selected by
``--observable`` (binned = 50 ms bins + 300 ms smoothing, the low-variance
Phase-2A choice; pointwise = exponential tau 100 ms). Estimators: grid +
refinement (primary; does not see the truth) and the unpromoted clipped
sign-descent gradient method with hard-evaluated iterates and non-finite
failure accounting.

Usage::

    python deliverable_b.py                      # full run (tag "full")
    python deliverable_b.py --dense-n 7 --replicates 1 --dur-scale 0.5  # smoke
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
import jax
import jax.numpy as jnp
import optax

from poolbuild import PoolTwin, TwinSpec
from validation_protocol import _renewal_events
from sensitivity_checks import _manifest, _smoothed_rates_jax

RESULTS = Path(__file__).parent / "results"
EVIDENCE = Path(__file__).parent / "evidence"

OKABE = {"truth": "#000000", "grid": "#0072B2", "grad": "#E69F00",
         "grad2": "#CC79A7", "partial": "#56B4E9"}
MUTED, GRIDC = "#666666", "#e3e3e3"

EPS_FLOOR = 1e-3  # Hz² — declared floor for the equivalence tolerance


# --------------------------------------------------------------------------- #
# Axes / protocols / segmentation / observables (module-level, testable)
# --------------------------------------------------------------------------- #

def axis_with_truth(lo: float, hi: float, n: int, truth: float) -> np.ndarray:
    """Log-spaced axis with the true value inserted — profiles must be able
    to reach the known zero-loss optimum (v2 bug: they could not)."""
    axis = np.geomspace(lo, hi, n)
    return np.unique(np.sort(np.append(axis, truth)))


def make_commands(dt_ms: float, dur_scale: float) -> dict:
    def tri(dur, peak, pre=0.5, post=0.5):
        n = int(dur * 1000 / dt_ms)
        t = np.arange(n) * dt_ms * 1e-3
        return np.interp(t, [0, pre, (dur - post + pre) / 2, dur - post, dur],
                         [0, 0, peak, 0, 0]).astype(np.float32)

    def trap(dur, peak, pre=0.4, rise=1.0, hold=1.2, fall=1.0):
        n = int(dur * 1000 / dt_ms)
        t = np.arange(n) * dt_ms * 1e-3
        return np.interp(
            t, [0, pre, pre + rise, pre + rise + hold,
                pre + rise + hold + fall, dur],
            [0, 0, peak, peak, 0, 0]).astype(np.float32)

    s = dur_scale
    return {"fit_ramp": tri(4.0 * s, 60.0), "fit_trap": trap(4.0 * s, 50.0),
            "heldout_slowramp": tri(5.0 * s, 45.0)}


def segment_limbs(cmd: np.ndarray) -> dict:
    """Boolean masks: pre / rising / peak (≥90 % of max) / falling / post."""
    cmd = np.asarray(cmd, dtype=float)
    peak_val = cmd.max()
    i_peak = int(np.argmax(cmd))
    active = cmd > 0
    first_on = int(np.argmax(active)) if active.any() else len(cmd)
    last_on = len(cmd) - 1 - int(np.argmax(active[::-1])) if active.any() else 0
    idx = np.arange(len(cmd))
    peak = cmd >= 0.9 * peak_val
    rising = active & (idx < i_peak) & ~peak
    falling = active & (idx > i_peak) & ~peak
    pre = idx < first_on
    post = idx > last_on
    return {"pre": pre, "rising": rising, "peak": peak,
            "falling": falling, "post": post}


def _sha16(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def binned_rates_jnp(spikes, dt_ms, bin_ms=50.0, smooth_ms=300.0):
    """(n_bins, n_units) smoothed binned rates in Hz — jax, differentiable.
    Insensitive to spike-time jitter within a bin (kills the chaotic
    decorrelation floor of the pointwise observable)."""
    steps = int(round(bin_ms / dt_ms))
    n_bins = spikes.shape[0] // steps
    counts = spikes[: n_bins * steps].reshape(n_bins, steps, -1).sum(axis=1)
    r = counts / (bin_ms * 1e-3)
    w = max(1, int(round(smooth_ms / bin_ms)))
    kern = jnp.ones((w,)) / w
    return jax.vmap(lambda col: jnp.convolve(col, kern, mode="same"),
                    in_axes=1, out_axes=1)(r)


def bin_mask(mask, dt_ms, bin_ms=50.0):
    """Downsample a per-step boolean mask to per-bin (majority vote)."""
    steps = int(round(bin_ms / dt_ms))
    n_bins = len(mask) // steps
    return (np.asarray(mask[: n_bins * steps], dtype=float)
            .reshape(n_bins, steps).mean(axis=1) > 0.5)


def build_twin_and_events(n_mn, conn_seed, event_seed_offset, commands, dt_ms):
    twin = PoolTwin(TwinSpec(n_mn=n_mn, n_dd=100, n_inh=5, dt_ms=dt_ms,
                             seed=conn_seed))
    events = {}
    for k, (name, cmd) in enumerate(sorted(commands.items())):
        rng = np.random.default_rng(event_seed_offset + k)
        events[name] = jnp.asarray(
            _renewal_events(rng, np.tile(cmd[:, None], (1, 100)), dt_ms, 5))
    return twin, events


def make_observable_fn(twin, events, observable, tau_ms=100.0):
    dt = twin.spec.dt_ms

    def fn(theta_vals, proto, spike_mode="hard"):
        th = twin.default_theta()
        th["gCaL_scale"] = jnp.float32(theta_vals[0])
        th["w_exc"] = jnp.float32(theta_vals[1])
        out = twin.simulate_events(th, events[proto], spike_mode=spike_mode)
        if observable == "pointwise":
            return _smoothed_rates_jax(out["spikes"], dt, tau_ms)
        return binned_rates_jnp(out["spikes"], dt)

    return fn


# legacy alias used by regression tests
def make_rates_fn(twin, events, tau_ms=100.0):
    return make_observable_fn(twin, events, "pointwise", tau_ms)


def numerical_epsilon_spikes(n_mn, conn_seed, events_01, theta_true,
                             fit_protos):
    """Fine-dt (0.05 ms) hard spike matrices at the truth, events re-timed by
    zero-insertion so the same input spikes occur at the same physical
    times."""
    twin_h = PoolTwin(TwinSpec(n_mn=n_mn, n_dd=100, n_inh=5, dt_ms=0.05,
                               seed=conn_seed))
    fine_spikes = []
    for p in fit_protos:
        ev = np.asarray(events_01[p])
        fine = np.zeros((ev.shape[0] * 2, ev.shape[1]), dtype=np.float32)
        fine[::2] = ev
        th = twin_h.default_theta()
        th["gCaL_scale"] = jnp.float32(theta_true[0])
        th["w_exc"] = jnp.float32(theta_true[1])
        out = twin_h.simulate_events(th, jnp.asarray(fine), spike_mode="hard")
        fine_spikes.append(np.asarray(out["spikes"]))
    return fine_spikes


def run_tag(args) -> str:
    """'full' only for scientifically sized runs; everything else is smoke —
    tagged filenames keep smoke runs from overwriting canonical evidence."""
    full = (args.dense_n >= 15 and args.replicates >= 2
            and args.dur_scale >= 1.0 and args.n_mn >= 8)
    return "full" if full else "smoke"


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--true", type=float, nargs=2, default=[1.6, 0.0105])
    ap.add_argument("--dense-n", type=int, default=21)
    ap.add_argument("--grid-n", type=int, default=7)
    ap.add_argument("--refine-n", type=int, default=5)
    ap.add_argument("--budget-iters", type=int, default=15)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--dur-scale", type=float, default=1.0)
    ap.add_argument("--observable", choices=["binned", "pointwise"],
                    default="binned")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    tag = run_tag(args)
    stem = f"deliverable_b_{tag}_{args.observable}"
    t_wall0 = time.time()
    dt_ms = 0.1
    theta_true = np.array(args.true)
    fit_protos = ["fit_ramp", "fit_trap"]
    commands = make_commands(dt_ms, args.dur_scale)

    g_lo, g_hi = 0.25, 4.0
    w_lo, w_hi = 0.0045, 0.018
    g_dense = axis_with_truth(g_lo, g_hi, args.dense_n, theta_true[0])
    w_dense = axis_with_truth(w_lo, w_hi, args.dense_n, theta_true[1])
    gi_true = int(np.argmin(np.abs(g_dense - theta_true[0])))
    wi_true = int(np.argmin(np.abs(w_dense - theta_true[1])))
    assert g_dense[gi_true] == theta_true[0] and w_dense[wi_true] == theta_true[1]

    obs_models = {"full_obs": list(range(args.n_mn)),
                  "partial_obs": list(range(args.n_mn // 2))}

    replicate_reports = []
    rep0 = None
    for r in range(args.replicates):
        conn_seed, ev_off = r, 10 * r
        twin, events = build_twin_and_events(
            args.n_mn, conn_seed, ev_off, commands, dt_ms)
        obs_fn = make_observable_fn(twin, events, args.observable)
        obs = {p: obs_fn(theta_true, p) for p in commands}
        active = {}
        for p in commands:
            th = twin.default_theta()
            th["gCaL_scale"] = jnp.float32(theta_true[0])
            th["w_exc"] = jnp.float32(theta_true[1])
            sp = np.asarray(twin.simulate_events(
                th, events[p], spike_mode="hard")["spikes"])
            active[p] = int((sp.sum(axis=0) > 0).sum())

        cache: dict = {}

        def pu_at(g, w):
            key = (float(g), float(w))
            if key not in cache:
                per = []
                for p in fit_protos:
                    rr = obs_fn(np.array([g, w]), p)
                    per.append(np.asarray(jnp.mean((rr - obs[p]) ** 2, axis=0)))
                cache[key] = np.mean(per, axis=0)
            return cache[key]

        # truth must be exactly 0 (same model, same events)
        truth_pu = pu_at(*theta_true)
        assert float(truth_pu.mean()) == 0.0, "truth loss must be exactly 0"

        t0 = time.time()
        pu = np.zeros((len(g_dense), len(w_dense), args.n_mn))
        for i, g in enumerate(g_dense):
            for j, w in enumerate(w_dense):
                pu[i, j] = pu_at(g, w)
        dense_time = time.time() - t0

        # per-replicate, per-model EPS from dt-halving at the truth
        fine_spikes = numerical_epsilon_spikes(
            args.n_mn, conn_seed, events, theta_true, fit_protos)
        eps_pu_terms = []
        for p, sp_f in zip(fit_protos, fine_spikes):
            if args.observable == "pointwise":
                rf = np.asarray(_smoothed_rates_jax(
                    jnp.asarray(sp_f), 0.05, 100.0))[1::2]
            else:
                rf = np.asarray(binned_rates_jnp(jnp.asarray(sp_f), 0.05))
            rc = np.asarray(obs[p])
            m = min(len(rf), len(rc))
            eps_pu_terms.append(((rf[:m] - rc[:m]) ** 2).mean(axis=0))
        eps_pu = np.mean(eps_pu_terms, axis=0)  # (n_mn,)
        eps = {model: max(float(eps_pu[units].mean()), EPS_FLOOR)
               for model, units in obs_models.items()}

        # profiles with local nuisance refinement, per model
        def refined_profile(units, vary_g: bool):
            axis_p = g_dense if vary_g else w_dense
            axis_n = w_dense if vary_g else g_dense
            prof = np.zeros(len(axis_p))
            for ai, aval in enumerate(axis_p):
                row = [float(pu_at(aval, nv)[units].mean()) if vary_g
                       else float(pu_at(nv, aval)[units].mean())
                       for nv in axis_n]
                jbest = int(np.argmin(row))
                best = row[jbest]
                lo = axis_n[max(jbest - 1, 0)]
                hi = axis_n[min(jbest + 1, len(axis_n) - 1)]
                for nv in np.geomspace(lo, hi, 5)[1:-1]:
                    v = (pu_at(aval, nv) if vary_g
                         else pu_at(nv, aval))[units].mean()
                    best = min(best, float(v))
                prof[ai] = best
            return prof

        rep = {"conn_seed": conn_seed, "event_offset": ev_off,
               "active_units_at_truth": active,
               "truth_fit_loss": float(truth_pu.mean()),
               "eps_by_model": eps,
               "dense_time_s": round(dense_time, 1),
               "event_hashes": {p: _sha16(np.asarray(events[p]))
                                for p in commands}}
        for model, units in obs_models.items():
            pg = refined_profile(units, vary_g=True)
            pw = refined_profile(units, vary_g=False)
            # the profile at the truth MUST be exactly zero in both directions
            assert pg[gi_true] == 0.0 and pw[wi_true] == 0.0, (
                f"profile at truth nonzero ({pg[gi_true]}, {pw[wi_true]}) — "
                "axis/profiling bug")

            def interval(curve, axis):
                within = curve <= eps[model]
                return ([float(axis[within].min()), float(axis[within].max())]
                        if within.any() else None)

            rep[model] = {
                "observed_units": units,
                "eps": eps[model],
                "profile_gCaL": pg.tolist(),
                "profile_w_exc": pw.tolist(),
                "gcal_interval": interval(pg, g_dense),
                "wexc_interval": interval(pw, w_dense),
                "dense_loss": pu[:, :, units].mean(axis=2).tolist(),
            }
        print(f"[rep{r} conn{conn_seed}/ev{ev_off}] dense+profiles "
              f"({len(cache)} evals, {time.time()-t0:.0f}s); truth=0 OK; "
              f"EPS full {eps['full_obs']:.4f} / partial "
              f"{eps['partial_obs']:.4f}; active {active}")
        for model in obs_models:
            print(f"    {model}: gCaL interval {rep[model]['gcal_interval']} "
                  f"| w_exc interval {rep[model]['wexc_interval']}")
        rep["_runtime"] = (twin, events, obs_fn, obs, pu, cache)
        replicate_reports.append(rep)
        if rep0 is None:
            rep0 = rep

    # ------------------------------------- estimator comparison (rep0 only)
    twin, events, obs_fn, obs, pu0, cache0 = rep0["_runtime"]

    def loss_units(theta_vals, units):
        per = []
        for p in fit_protos:
            rr = obs_fn(theta_vals, p)
            per.append(np.asarray(jnp.mean((rr - obs[p]) ** 2, axis=0)))
        return float(np.mean(per, axis=0)[units].mean())

    wall = {}
    estimators = {}
    for model, units in obs_models.items():
        t0 = time.time()
        g_grid = np.geomspace(g_lo, g_hi, args.grid_n)
        w_grid = np.geomspace(w_lo, w_hi, args.grid_n)
        L = np.array([[loss_units(np.array([g, w]), units)
                       for w in w_grid] for g in g_grid])
        bi, bj = np.unravel_index(np.argmin(L), L.shape)

        def nb(grid, k):
            return np.geomspace(grid[max(k - 1, 0)],
                                grid[min(k + 1, len(grid) - 1)], args.refine_n)

        g_ref, w_ref = nb(g_grid, bi), nb(w_grid, bj)
        Lr = np.array([[loss_units(np.array([g, w]), units)
                        for w in w_ref] for g in g_ref])
        ri, rj = np.unravel_index(np.argmin(Lr), Lr.shape)
        wall[f"grid_{model}"] = round(time.time() - t0, 1)
        estimators[(model, "grid_refine")] = {
            "theta_hat": [float(g_ref[ri]), float(w_ref[rj])],
            "fit_loss": float(Lr[ri, rj]),
            "within_eps": bool(Lr[ri, rj] <= rep0[model]["eps"]),
            "boundary_hit": bool(bi in (0, args.grid_n - 1)
                                 or bj in (0, args.grid_n - 1)),
        }

        # unpromoted gradient method; every iterate HARD-evaluated
        t0 = time.time()
        tgt = {p: jax.lax.stop_gradient(obs[p]) for p in fit_protos}
        uidx = jnp.asarray(units)

        def loss_z(z):
            th = twin.default_theta()
            th["gCaL_scale"] = jnp.exp(z[0])
            th["w_exc"] = jnp.exp(z[1])
            tot = 0.0
            for p in fit_protos:
                out = twin._run_events(
                    th, {"exc": events[p],
                         "inh": jnp.zeros((events[p].shape[0], 5), jnp.float32),
                         "exc_conn": twin._exc_mat,
                         "inh_conn": twin._inh_mat},
                    spike_mode="surrogate")
                if args.observable == "pointwise":
                    rr = _smoothed_rates_jax(out["spikes"], dt_ms, 100.0)
                else:
                    rr = binned_rates_jnp(out["spikes"], dt_ms)
                tot = tot + jnp.mean((rr[:, uidx] - tgt[p][:, uidx]) ** 2)
            return tot / len(fit_protos)

        vg = jax.jit(jax.value_and_grad(loss_z))
        runs = []
        for label, start in (("start_default", (1.0, 0.009)),
                             ("start_perturbed", (0.6, 0.014))):
            opt = optax.chain(optax.clip(1.0), optax.adam(0.05))
            z = jnp.log(jnp.asarray(start, dtype=jnp.float32))
            state = opt.init(z)
            best = {"hard_loss": np.inf, "theta": None, "it": -1}
            nonfinite = 0
            hist = []
            for it in range(args.budget_iters):
                th_now = np.exp(np.asarray(z, dtype=np.float64))
                if not np.all(np.isfinite(th_now)):
                    nonfinite += 1
                    hist.append({"it": it, "nonfinite": True})
                    break
                hard = loss_units(th_now, units)
                hist.append({"it": it, "hard_loss": hard,
                             "theta": th_now.tolist()})
                if hard < best["hard_loss"]:
                    best = {"hard_loss": hard, "theta": th_now.tolist(),
                            "it": it}
                _, g = vg(z)
                upd, state = opt.update(g, state)
                z = optax.apply_updates(z, upd)
            runs.append({"label": label, "start": list(start), "best": best,
                         "nonfinite_iterates": nonfinite,
                         "failed": nonfinite > 0, "history": hist})
        wall[f"grad_{model}"] = round(time.time() - t0, 1)
        ok_runs = [r for r in runs if r["best"]["theta"] is not None]
        best_run = min(ok_runs, key=lambda r: r["best"]["hard_loss"])
        estimators[(model, "grad_best_of_starts")] = {
            "theta_hat": best_run["best"]["theta"],
            "fit_loss": best_run["best"]["hard_loss"],
            "within_eps": bool(best_run["best"]["hard_loss"]
                               <= rep0[model]["eps"]),
            "n_failed_starts": sum(r["failed"] for r in runs),
            "n_starts": len(runs),
            "runs": runs,
        }

    # ----------------------------------------------- held-out, limb-specific
    limbs = segment_limbs(commands["heldout_slowramp"])
    if args.observable == "binned":
        limbs = {k: bin_mask(m, dt_ms) for k, m in limbs.items()}

    def heldout_limb_mse(theta_vals):
        rr = np.asarray(obs_fn(np.asarray(theta_vals), "heldout_slowramp"))
        oo = np.asarray(obs["heldout_slowramp"])
        se = (rr - oo) ** 2
        out = {name: float(se[mask].mean()) for name, mask in limbs.items()
               if mask.any()}
        out["overall"] = float(se.mean())
        return out

    recovery_table = []
    for (model, method), est in estimators.items():
        th = np.asarray(est["theta_hat"])
        recovery_table.append({
            "obs_model": model, "method": method,
            "theta_hat": th.tolist(),
            "param_rel_err": (np.abs(th - theta_true) / theta_true).tolist(),
            "fit_loss": est["fit_loss"],
            "fit_within_eps": est["within_eps"],
            "heldout_mse_by_limb": heldout_limb_mse(th),
        })
    heldout_truth = heldout_limb_mse(theta_true)

    # ---------------------------------------------------------------- figure
    clip = EPS_FLOOR / 10  # display floor so the exact-zero truth plots on log axes
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.26)
    for col, model in enumerate(("full_obs", "partial_obs")):
        axL = fig.add_subplot(gs[0, col])
        L = np.array(rep0[model]["dense_loss"])
        pc = axL.pcolormesh(w_dense, g_dense, np.log10(L + clip),
                            cmap="Blues", shading="nearest")
        fig.colorbar(pc, ax=axL, label="log10 fit loss (Hz²)")
        axL.contour(w_dense, g_dense, L, levels=[rep0[model]["eps"]],
                    colors=["#111111"], linewidths=1.4)
        axL.plot(theta_true[1], theta_true[0], "*", ms=16,
                 color=OKABE["truth"], label="truth (loss = 0, on-axis)")
        gb = estimators[(model, "grid_refine")]["theta_hat"]
        axL.plot(gb[1], gb[0], "o", ms=9, mfc="none", mew=2,
                 color=OKABE["grid"], label="grid+refine")
        axL.set_xscale("log")
        axL.set_yscale("log")
        axL.set_xlabel("w_exc (µS)")
        axL.set_ylabel("gCaL_scale")
        axL.set_title(f"{model} ({args.observable}) — rep0; contour: "
                      f"EPS = {rep0[model]['eps']:.3f} Hz²", fontsize=9)
        axL.legend(frameon=False, fontsize=8, loc="lower left")

    for col, (pname, axis, key) in enumerate(
            (("gCaL_scale", g_dense, "profile_gCaL"),
             ("w_exc", w_dense, "profile_w_exc"))):
        ax = fig.add_subplot(gs[1, col])
        for model, c in (("full_obs", OKABE["grid"]),
                         ("partial_obs", OKABE["partial"])):
            for k, rep in enumerate(replicate_reports):
                cur = np.maximum(np.array(rep[model][key]), clip)
                ax.plot(axis, cur, color=c, lw=1.2,
                        alpha=0.45 if k else 1.0,
                        label=model if k == 0 else None)
            ax.axhline(rep0[model]["eps"], color=c, lw=0.9, ls="--",
                       label=f"EPS {model} {rep0[model]['eps']:.3f}")
        ax.axvline(theta_true[0] if pname == "gCaL_scale" else theta_true[1],
                   color=MUTED, lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(pname)
        ax.set_ylabel(f"profile-min fit loss (Hz², display-clipped at {clip:g})")
        ax.set_title(f"Profile over {pname} — truth on-axis, exact 0 at "
                     f"truth; {args.replicates} replicate(s)", fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, color=GRIDC, lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(RESULTS / f"{stem}_landscape.png", dpi=150,
                bbox_inches="tight")

    # ---------------------------------------------------------------- report
    here = Path(__file__)
    report = {
        "manifest": {
            **_manifest(),
            "deliverable_b_sha256_16": hashlib.sha256(
                here.read_bytes()).hexdigest()[:16],
            "args": vars(args),
            "run_tag": tag,
            "protocol_lengths_steps": {k: len(v) for k, v in commands.items()},
            "optimizer": {"chain": "clip(1.0)+adam(0.05)",
                          "starts": [[1.0, 0.009], [0.6, 0.014]],
                          "iters": args.budget_iters},
        },
        "declared": {
            "unknowns": ["gCaL_scale", "w_exc"],
            "experiment_type": "conditional fit with known inputs",
            "observable": args.observable,
            "eps_rule": "per replicate × observation model, dt-halving at "
                        "truth over that model's units; floor %.0e" % EPS_FLOOR,
            "eps_conservative_max": max(
                r["eps_by_model"][m] for r in replicate_reports
                for m in obs_models),
            "theta_true": theta_true.tolist(),
            "axes_contain_truth": True,
            "interval_terminology": "sampled compatibility intervals (deterministic level sets, not statistical confidence intervals)",
        },
        "g_dense": g_dense.tolist(),
        "w_dense": w_dense.tolist(),
        "replicates": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in replicate_reports],
        "estimators": {f"{m}/{meth}": est
                       for (m, meth), est in estimators.items()},
        "recovery_table": recovery_table,
        "heldout_at_truth_by_limb": heldout_truth,
        "wall_times_s": {**wall, "total": round(time.time() - t_wall0, 1)},
    }
    payload = json.dumps(report, indent=1, default=float)
    (RESULTS / f"{stem}.json").write_text(payload)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (EVIDENCE / f"{stem}_{stamp}.json").write_text(payload)

    print(f"\n=== {stem} — sampled compatibility intervals (profile ≤ per-model EPS) ===")
    for k, rep in enumerate(replicate_reports):
        for model in obs_models:
            print(f" rep{k} {model:12s} (EPS {rep[model]['eps']:.4f}): "
                  f"gCaL {rep[model]['gcal_interval']} | "
                  f"w_exc {rep[model]['wexc_interval']}")
    print("\n--- recovery table ---")
    for row in recovery_table:
        th = row["theta_hat"]
        hl = row["heldout_mse_by_limb"]
        print(f"  {row['obs_model']:>11}/{row['method']:<19} "
              f"θ̂=({th[0]:5.3f},{th[1]:8.5f}) fit {row['fit_loss']:8.4f} "
              f"(within EPS: {row['fit_within_eps']}) | held-out overall "
              f"{hl['overall']:.3f}, peak {hl.get('peak', float('nan')):.3f}")
    print(f"held-out at truth: overall {heldout_truth['overall']:.2e}")
    print(f"wall times: {report['wall_times_s']}")
    print(f"saved → {RESULTS / (stem + '.json')} (+ evidence copy, figure "
          f"{stem}_landscape.png)")


if __name__ == "__main__":
    main()

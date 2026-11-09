"""Phase 3 MVP — the inverse problem on synthetic data with known physiology.

Protocol:
1. Simulate "observations" with hidden ground-truth physiology θ* (hard spikes,
   the scientific forward): PIC conductances scaled, altered excitation, altered
   inhibition.
2. Observables = exponentially-smoothed per-MN firing rates (what one gets from
   decomposed EMG in practice).
3. Recover θ = {gCaL_scale, gNaP_scale, drive_gain, w_inh} by gradient descent
   (optax Adam in log-parameter space; surrogate spikes for the backward pass,
   which are *forward-identical* to hard spikes in pathwise mode — so the loss
   at θ = θ* is exactly 0 with the same PRNG key).

``w_exc`` is held fixed at its known value in the MVP (drive_gain and w_exc are
strongly degenerate through the excitatory path; separating them is exactly the
Phase 5 identifiability question).

Exploding-gradient note: BPTT through ~40k timesteps of stiff spiking dynamics
yields gradient magnitudes of 1e16–1e22 (measured; both surrogate and rate
modes). Raw Adam silently freezes — squaring such gradients overflows float32
to inf and the update becomes 0/inf = 0. We therefore clip element-wise
(``optax.clip``) *before* Adam: magnitudes are chaos-dominated but signs remain
informative, so the fit is effectively momentum sign-descent in log-space.
BE CLEAR ABOUT WHAT THIS IS: a heuristic sign method, not gradient descent on
a trustworthy objective — relative sensitivities are discarded, the surrogate
backward pass is not the derivative of the hard-spike objective, and these
gradients must NOT be reused for Fisher-information / identifiability claims.
Results are reported at the best-loss iterate; if the best loss is not clearly
below the initial loss, no recovery has been demonstrated. Taming the
magnitudes properly (truncated BPTT windows, smoothed-dynamics losses,
hard-mode loss-landscape checks) is on the roadmap.

Usage::

    python phase3_inverse_mvp.py [--iters 100] [--lr 0.02]
"""

from __future__ import annotations

import argparse
import pickle
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
import protocols
import observables as obs

RESULTS = Path(__file__).parent / "results"
FIT_KEYS = ("gCaL_scale", "gNaP_scale", "drive_gain", "w_inh")

# Okabe–Ito (colorblind-safe) categorical order for the four parameters.
CAT = {"gCaL_scale": "#0072B2", "gNaP_scale": "#E69F00",
       "drive_gain": "#009E73", "w_inh": "#CC79A7"}
INK, MUTED, GRID = "#1a1a1a", "#666666", "#e3e3e3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--n-dd", type=int, default=100)
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--peak", type=float, default=50.0)
    ap.add_argument("--inh-rate", type=float, default=20.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tau-ms", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    twin = PoolTwin(TwinSpec(n_mn=args.n_mn, n_dd=args.n_dd, dt_ms=args.dt_ms,
                             seed=args.seed))
    _, cmd = protocols.trapezoid(
        args.dt_ms, t_rise_s=1.0, t_hold_s=1.5, t_fall_s=1.0,
        t_pre_s=0.3, t_post_s=0.3, peak=args.peak,
    )
    drives = {
        "dd": jnp.asarray(cmd),
        "inh": jnp.full(len(cmd), args.inh_rate, dtype=jnp.float32),
    }
    key = jax.random.PRNGKey(args.seed)

    # ---------------------------------------------------- ground truth + obs
    theta_true = twin.default_theta()
    theta_true.update({
        "gCaL_scale": jnp.float32(1.6),
        "gNaP_scale": jnp.float32(0.7),
        "drive_gain": jnp.float32(0.85),
        "w_inh": jnp.float32(0.006),
    })
    out_obs = twin.simulate(theta_true, drives, key, spike_mode="hard")
    obs_rates = obs.smoothed_rates(out_obs["spikes"], args.dt_ms, args.tau_ms)
    obs_rates = jax.lax.stop_gradient(obs_rates)
    n_active = int((np.asarray(out_obs["spikes"]).sum(axis=0) >= 3).sum())
    print(f"observations: {n_active}/{args.n_mn} MNs active, "
          f"mean rate {float(obs_rates.mean()):.2f} Hz")

    # ------------------------------------------------------------- fit setup
    theta0 = twin.default_theta()  # deliberately wrong start (all defaults)

    def loss_from_z(z):
        theta = dict(theta0)
        for k in FIT_KEYS:
            theta[k] = jnp.exp(z[k])
        out = twin._run(theta, drives, key, spike_mode="surrogate")
        rates = obs.smoothed_rates(out["spikes"], args.dt_ms, args.tau_ms)
        return jnp.mean((rates - obs_rates) ** 2)

    z = {k: jnp.log(theta0[k]) for k in FIT_KEYS}
    # clip BEFORE adam: raw grads are ~1e20 and overflow adam's g² to inf.
    opt = optax.chain(optax.clip(1.0), optax.adam(args.lr))
    opt_state = opt.init(z)
    vg = jax.jit(jax.value_and_grad(loss_from_z))

    history = {"loss": [], "theta": []}
    best = {"loss": np.inf, "theta": None, "it": -1}
    t0 = time.time()
    for it in range(args.iters):
        loss, grads = vg(z)
        # loss is evaluated at the CURRENT z (before the update), so pair it
        # with the pre-update parameters for best-iterate tracking.
        th_now = {k: float(jnp.exp(z[k])) for k in FIT_KEYS}
        history["loss"].append(float(loss))
        history["theta"].append(th_now)
        if float(loss) < best["loss"]:
            best = {"loss": float(loss), "theta": th_now, "it": it}
        updates, opt_state = opt.update(grads, opt_state)
        z = optax.apply_updates(z, updates)
        if it % 5 == 0 or it == args.iters - 1:
            rel = ", ".join(
                f"{k}={th_now[k] / float(theta_true[k]):.3f}×" for k in FIT_KEYS
            )
            print(f"it {it:4d} | loss {float(loss):.5f} | rel-to-true: {rel} "
                  f"| {time.time()-t0:.0f}s")

    # -------------------------------------------------------------- report
    # Report the BEST iterate (lowest loss), not the last one — with the
    # clipped sign-descent updates the trajectory is non-monotonic and the
    # final iterate can be worse than the start. If best ≈ init loss, the
    # optimization has NOT demonstrated recovery; say so rather than citing
    # per-parameter errors from an unconverged trajectory.
    final = history["theta"][-1]
    print(f"\nbest iterate: {best['it']} (loss {best['loss']:.5f}); "
          f"initial loss {history['loss'][0]:.5f}; "
          f"final loss {history['loss'][-1]:.5f}")
    if history["loss"][-1] > best["loss"]:
        print("WARNING: optimization is non-monotonic — final iterate is worse "
              "than the best; treating the best iterate as the result.")
    print("\nrecovered (best iterate) vs true:")
    print(f"{'param':>12} | {'true':>9} | {'init':>9} | {'best':>9} "
          f"| {'final':>9} | {'best err':>8}")
    for k in FIT_KEYS:
        tv, iv = float(theta_true[k]), float(theta0[k])
        bv, fv = best["theta"][k], final[k]
        print(f"{k:>12} | {tv:9.4f} | {iv:9.4f} | {bv:9.4f} | {fv:9.4f} "
              f"| {abs(bv - tv) / tv * 100:7.1f}%")

    # figure: loss curve + parameter trajectories relative to truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.semilogy(history["loss"], color="#0072B2", linewidth=1.6)
    ax1.plot(best["it"], best["loss"], "o", color=INK, markersize=6, zorder=5)
    ax1.annotate(f"best (it {best['it']})", (best["it"], best["loss"]),
                 textcoords="offset points", xytext=(6, -10), fontsize=8,
                 color=INK)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("rate MSE (Hz²)")
    ax1.set_title("Loss (non-monotonic: sign-descent)", color=INK, fontsize=10)
    its = np.arange(len(history["theta"]))
    for k in FIT_KEYS:
        traj = np.array([h[k] for h in history["theta"]]) / float(theta_true[k])
        ax2.plot(its, traj, color=CAT[k], linewidth=1.6, label=k)
        ax2.annotate(k, (its[-1], traj[-1]), textcoords="offset points",
                     xytext=(4, 0), fontsize=8, color=CAT[k])
    ax2.axhline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("recovered / true")
    ax2.set_title("Parameter recovery (1.0 = truth)", color=INK, fontsize=10)
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "phase3_inverse_mvp.png", dpi=150)
    with open(RESULTS / "phase3_inverse_mvp.pkl", "wb") as f:
        pickle.dump({"args": vars(args), "history": history, "best": best,
                     "theta_true": {k: float(theta_true[k]) for k in FIT_KEYS}}, f)
    print(f"saved → {RESULTS / 'phase3_inverse_mvp.png'}")


if __name__ == "__main__":
    main()

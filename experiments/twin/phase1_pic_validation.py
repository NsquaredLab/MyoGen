"""Phase 1 (gap) — PIC-specific behaviour of the Jaxley NERLab pool.

Triangular-ramp protocol (the ΔF paradigm): descending drive ramps up and back
down; PICs should produce (i) recruitment/derecruitment hysteresis — MNs turn
off at *lower* drive than they turned on, (ii) positive ΔF in paired-MU
analysis, (iii) self-sustained firing after the ramp returns to baseline.

Dose–response over the gCaL grid (default ×{0, 1, 2, 4, 6.5}), optionally
crossed with a NaP multiplier (``--gnap-scale``). The repository's established
strong-PIC regime corresponds to roughly gCaL ×6.5–7.5 (pool ``gamma``
1.3–1.5) with NaP ×5 — a grid restricted to low scales (×0–2) shows no
bistability and supports only the conclusion "none detected in that range",
NOT "the model lacks bistable PICs".

Multiple stochastic-drive seeds (``--n-seeds``) give mean ± sd for ΔF and
hysteresis; single-seed numbers sit inside substantial pair-to-pair
variability. ``--seed`` also sets the connectivity seed.

Caveat for NEURON comparison: with the default synapse options this harness
drives the SOMA with a fixed-driving-force surrogate current, while NEURON
places voltage-dependent Exp2Syn on the DENDRITE (the caL compartment) with
``tau1 = 0.2 ms, tau2 = 0.3 ms, E_exc = +70 mV, E_inh = −75 mV``. For a
matched configuration use::

    --syn-site dendrite --live-df --tau-rise 0.2 --tau-decay 0.3

E_inh convention (must be stated with any paired result): the harness default
−15 mV is the frame-corrected analog of a modern −80 mV (driving force −15 mV
at the NERLab rest of 0 mV); NEURON's literal value is −75 mV. Pass
``--e-inh -75`` to match NEURON's number, or keep −15 to match its *physical*
driving force — these are different experiments.

Variability: ``--n-seeds`` varies the stochastic drive realization;
``--n-conn-seeds`` varies the connectivity matrix. The two sources are
reported separately.

Usage::

    python phase1_pic_validation.py [--n-mn 20] [--peak 60] [--rise 7.5]
    python phase1_pic_validation.py --scales 0 1 2 4 6.5 --gnap-scale 5 \\
        --syn-site dendrite --live-df --tau-rise 0.2 --tau-decay 0.3 \\
        --n-seeds 3 --n-conn-seeds 2
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

from poolbuild import PoolTwin, TwinSpec
import protocols
import observables as obs

RESULTS = Path(__file__).parent / "results"

# Sequential single-hue ramp (light→dark) — gCaL scale is a magnitude.
def make_ramp(n: int) -> list[str]:
    cmap = plt.get_cmap("Blues")
    return [matplotlib.colors.to_hex(cmap(x)) for x in np.linspace(0.35, 0.95, max(n, 2))]


INK, MUTED, GRID = "#1a1a1a", "#666666", "#e3e3e3"


def run_condition(twin, drives, key, gcal_scale, gnap_scale=1.0):
    theta = twin.default_theta()
    theta["gCaL_scale"] = jnp.float32(gcal_scale)
    theta["gNaP_scale"] = jnp.float32(gnap_scale)
    out = twin.simulate(theta, drives, key, spike_mode="hard")
    return np.asarray(out["v"]), np.asarray(out["spikes"])


def analyze(spikes, cmd, dt_ms, t_ramp_end_s):
    t = np.arange(len(cmd)) * dt_ms * 1e-3
    df = obs.delta_f_pairs(spikes, dt_ms)
    hyst = obs.hysteresis_index(cmd, spikes, dt_ms)
    n_active = int((spikes.sum(axis=0) >= 3).sum())
    post = spikes[t >= t_ramp_end_s]
    self_sustained = int((post.sum(axis=0) >= 5).sum())
    return {
        "delta_f_mean": df.mean,
        "delta_f_n_pairs": int(len(df.delta_f)),
        "hysteresis_mean": float(np.nanmean(hyst)) if np.any(~np.isnan(hyst)) else np.nan,
        "n_active": n_active,
        "self_sustained": self_sustained,
        "delta_f": df,
        "hysteresis": hyst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=20)
    ap.add_argument("--n-dd", type=int, default=100)
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--peak", type=float, default=60.0, help="peak DD rate (Hz)")
    ap.add_argument("--rise", type=float, default=7.5, help="ramp rise/fall (s)")
    ap.add_argument("--post", type=float, default=3.0)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 6.5])
    ap.add_argument("--gnap-scale", type=float, default=1.0,
                    help="NaP multiplier applied to every condition "
                         "(strong-PIC regime ≈ 5)")
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="stochastic-drive seeds per condition (mean ± sd)")
    ap.add_argument("--n-conn-seeds", type=int, default=1,
                    help="connectivity seeds (reported separately from drive "
                         "variability)")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed (connectivity and generators derive from it)")
    ap.add_argument("--syn-site", choices=["soma", "dendrite"], default="soma")
    ap.add_argument("--live-df", action="store_true",
                    help="voltage-dependent driving force (NEURON-like)")
    ap.add_argument("--tau-rise", type=float, default=None,
                    help="Exp2Syn rise time constant (ms); omit for single-exp. "
                         "NEURON NERLab uses 0.2")
    ap.add_argument("--tau-decay", type=float, default=None,
                    help="synaptic decay time constant (ms, default 5.0). "
                         "NEURON NERLab uses 0.3")
    ap.add_argument("--w-exc", type=float, default=None, help="excitatory weight (µS)")
    ap.add_argument("--w-inh", type=float, default=None, help="inhibitory weight (µS)")
    ap.add_argument("--e-exc", type=float, default=None,
                    help="excitatory reversal (mV, default +70)")
    ap.add_argument("--e-inh", type=float, default=None,
                    help="inhibitory reversal (mV, default −15 frame-corrected; "
                         "NEURON's literal value is −75 — see docstring)")
    args = ap.parse_args()

    spec_kwargs = dict(
        n_mn=args.n_mn, n_dd=args.n_dd, dt_ms=args.dt_ms,
        syn_site=args.syn_site, live_driving_force=args.live_df,
        syn_tau_rise_ms=args.tau_rise,
    )
    for cli, field in [("tau_decay", "tau_syn_ms"), ("w_exc", "w_exc_uS"),
                       ("w_inh", "w_inh_uS"), ("e_exc", "e_exc_mV"),
                       ("e_inh", "e_inh_mV")]:
        if getattr(args, cli) is not None:
            spec_kwargs[field] = getattr(args, cli)

    twins = [PoolTwin(TwinSpec(seed=args.seed + cs, **spec_kwargs))
             for cs in range(args.n_conn_seeds)]
    twin = twins[0]  # figure + single-run detail come from the first twin
    t, cmd = protocols.triangular_ramp(
        args.dt_ms, t_rise_s=args.rise, t_pre_s=1.0, t_post_s=args.post,
        peak=args.peak,
    )
    t_ramp_end = 1.0 + 2 * args.rise
    drives = {
        "dd": jnp.asarray(cmd),
        "inh": jnp.zeros(len(cmd), dtype=jnp.float32),
    }
    results, spikes_by_scale = {}, {}
    print(f"(gNaP ×{args.gnap_scale:g}, {args.n_seeds} seed(s), "
          f"syn={args.syn_site}, {'live' if args.live_df else 'fixed'} DF)")
    print(f"{'gCaL x':>7} | {'ΔF (Hz)':>14} | {'pairs':>5} "
          f"| {'hyst (Hz drive)':>16} | {'active':>7} | {'self-sust':>9} "
          f"| {'t (s)':>6}")
    for s in args.scales:
        t0 = time.time()
        per_conn = []  # per_conn[cs] = list of per-drive-seed results
        for cs, tw in enumerate(twins):
            runs = []
            for si in range(args.n_seeds):
                key = jax.random.PRNGKey(args.seed + 1000 * cs + si)
                _, sp = run_condition(tw, drives, key, s, args.gnap_scale)
                r = analyze(sp, cmd, args.dt_ms, t_ramp_end)
                runs.append(r)
                if cs == 0 and si == 0:
                    spikes_by_scale[s] = sp
            per_conn.append(runs)
        flat = [r for runs in per_conn for r in runs]
        dfm = np.array([r["delta_f_mean"] for r in flat])
        hym = np.array([r["hysteresis_mean"] for r in flat])
        # separated variability: drive seeds (within first connectivity) vs
        # connectivity seeds (across per-connectivity means)
        drive_sd = float(np.nanstd([r["delta_f_mean"] for r in per_conn[0]]))
        conn_means = [np.nanmean([r["delta_f_mean"] for r in runs])
                      for runs in per_conn]
        conn_sd = float(np.nanstd(conn_means))
        agg = dict(per_conn[0][0])  # first-run detail for the figure
        agg.update({
            "delta_f_mean": float(np.nanmean(dfm)),
            "delta_f_sd": float(np.nanstd(dfm)),
            "delta_f_drive_sd": drive_sd,
            "delta_f_conn_sd": conn_sd,
            "delta_f_per_run": dfm,
            "hysteresis_mean": float(np.nanmean(hym)),
            "hysteresis_sd": float(np.nanstd(hym)),
            "n_active": float(np.mean([r["n_active"] for r in flat])),
            "self_sustained": float(np.mean([r["self_sustained"] for r in flat])),
            "delta_f_n_pairs": int(np.sum([r["delta_f_n_pairs"] for r in flat])),
        })
        results[s] = agg
        multi = len(flat) > 1
        sd = f" ± {agg['delta_f_sd']:4.2f}" if multi else "       "
        hsd = f" ± {agg['hysteresis_sd']:4.2f}" if multi else "       "
        print(f"{s:7.2f} | {agg['delta_f_mean']:7.2f}{sd} "
              f"| {agg['delta_f_n_pairs']:5d} | {agg['hysteresis_mean']:9.2f}{hsd} "
              f"| {agg['n_active']:7.1f} | {agg['self_sustained']:9.1f} "
              f"| {time.time()-t0:6.1f}")
        if args.n_seeds > 1 or args.n_conn_seeds > 1:
            print(f"        ΔF variability — drive seeds: ± {drive_sd:.2f} "
                  f"(n={args.n_seeds}); connectivity seeds: ± {conn_sd:.2f} "
                  f"(n={args.n_conn_seeds})")

    # ---------------------------------------------------------------- figure
    ramp = make_ramp(len(args.scales))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) rasters, lowest vs highest gCaL
    ax = axes[0, 0]
    lo, hi = args.scales[0], args.scales[-1]
    for row, (s, col) in enumerate([(lo, ramp[0]), (hi, ramp[-1])]):
        sp = spikes_by_scale[s]
        for i in range(sp.shape[1]):
            times = np.flatnonzero(sp[:, i] > 0.5) * args.dt_ms * 1e-3
            y = i + row * (args.n_mn + 3)
            ax.plot(times, np.full(len(times), y), "|", color=col, markersize=3)
    ax2 = ax.twinx()  # drive overlay on its own (hidden) scale, gray
    ax2.plot(t, cmd, color=MUTED, linewidth=1.2)
    ax2.set_yticks([])
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax.set_title(f"Raster: gCaL ×{lo:g} (bottom) vs ×{hi:g} (top); drive in gray",
                 color=INK, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MN index")

    # (b) ΔF dose–response
    ax = axes[0, 1]
    means = [results[s]["delta_f_mean"] for s in args.scales]
    for s, m, c in zip(args.scales, means, ramp):
        pts = results[s]["delta_f"].delta_f
        if len(pts):
            ax.plot(np.full(len(pts), s), pts, "o", color=c, alpha=0.4, markersize=4)
    ax.plot(args.scales, means, "-", color=INK, linewidth=1.5)
    if args.n_seeds > 1 or args.n_conn_seeds > 1:
        sds = [results[s]["delta_f_sd"] for s in args.scales]
        ax.errorbar(args.scales, means, yerr=sds, fmt="none", ecolor=INK,
                    elinewidth=1.0, capsize=3)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("ΔF vs gCaL scale (dots: MU pairs; line: mean)", color=INK, fontsize=10)
    ax.set_xlabel("gCaL scale")
    ax.set_ylabel("ΔF (Hz)")

    # (c) drive hysteresis per MN
    ax = axes[1, 0]
    for s, c in zip(args.scales, ramp):
        h = results[s]["hysteresis"]
        ok = ~np.isnan(h)
        ax.plot(np.flatnonzero(ok), h[ok], "o-", color=c, markersize=4,
                linewidth=1.0, label=f"×{s:g}")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, title="gCaL")
    ax.set_title("Derecruitment − recruitment drive (negative ⇒ hysteresis)",
                 color=INK, fontsize=10)
    ax.set_xlabel("MN index")
    ax.set_ylabel("Δ drive (Hz)")

    # (d) population smoothed rate over the ramp
    ax = axes[1, 1]
    for s, c in zip(args.scales, ramp):
        pr = np.asarray(obs.population_rate(jnp.asarray(spikes_by_scale[s]),
                                            args.dt_ms, tau_ms=200.0))
        ax.plot(t, pr, color=c, linewidth=1.4, label=f"×{s:g}")
    ax.legend(frameon=False, fontsize=8, title="gCaL")
    ax.set_title("Population rate (note offset-limb excess with PICs)",
                 color=INK, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean rate (Hz)")

    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "phase1_pic_validation.png", dpi=150)
    with open(RESULTS / "phase1_pic_validation.pkl", "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "results": {
                    s: {k: v for k, v in r.items() if k not in ("delta_f",)}
                    for s, r in results.items()
                },
                "spikes_by_scale": spikes_by_scale,
                "cmd": cmd,
            },
            f,
        )
    print(f"saved → {RESULTS / 'phase1_pic_validation.png'}")


if __name__ == "__main__":
    main()

"""Phase 2 — the computational leap: batched simulation + gradients.

``jax.vmap`` maps the pool twin's forward (and ``value_and_grad``) over K
parameter vectors simultaneously — the thing NEURON cannot do at all. This
script measures throughput vs batch size on whatever device JAX finds (CPU on
the dev Mac; rerun unchanged on a GPU node).

Reported per K: steady-state wall time (compile excluded), simulations/s, and
MN·steps/s, for (a) forward hard-spike runs and (b) full value_and_grad.

Usage::

    python phase2_benchmark.py [--batch-sizes 1 8 32 128] [--t-on 1.0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from poolbuild import PoolTwin, TwinSpec
import protocols
import observables as obs

RESULTS = Path(__file__).parent / "results"


def batched_thetas(twin: PoolTwin, K: int, rng: np.random.Generator) -> dict:
    """Stack K perturbed parameter sets leaf-wise: each leaf becomes (K,)."""
    base = twin.default_theta()
    factors = rng.uniform(0.5, 2.0, size=(K, len(base))).astype(np.float32)
    return {
        k: jnp.asarray(factors[:, j]) * v
        for j, (k, v) in enumerate(sorted(base.items()))
    }


def time_fn(fn, *a, n_rep: int = 3):
    out = fn(*a)  # compile
    jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(n_rep):
        out = fn(*a)
        jax.block_until_ready(out)
    return (time.time() - t0) / n_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mn", type=int, default=10)
    ap.add_argument("--n-dd", type=int, default=100)
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--t-on", type=float, default=1.0)
    ap.add_argument("--level", type=float, default=40.0)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--seed", type=int, default=0,
                    help="connectivity seed and generator seed")
    ap.add_argument("--device", choices=["cpu", "gpu", "tpu"], default=None,
                    help="JAX platform to run on (default: JAX's default)")
    args = ap.parse_args()

    if args.device is not None:
        try:
            jax.config.update("jax_default_device", jax.devices(args.device)[0])
        except RuntimeError:
            print(f"no {args.device} device available; "
                  f"falling back to {jax.devices()[0].platform}")

    twin = PoolTwin(TwinSpec(n_mn=args.n_mn, n_dd=args.n_dd, dt_ms=args.dt_ms,
                             seed=args.seed))
    _, cmd = protocols.step(
        args.dt_ms, t_pre_s=0.2, t_on_s=args.t_on, t_post_s=0.2, level=args.level
    )
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
    key = jax.random.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)
    n_steps = len(cmd)
    dev = jax.devices()[0]
    print(f"device={dev.platform} | n_mn={args.n_mn} | n_steps={n_steps}")

    def loss_fn(th):
        out = twin._run(th, drives, key, spike_mode="surrogate")
        rates = obs.smoothed_rates(out["spikes"], args.dt_ms, tau_ms=150.0)
        return jnp.mean(rates**2)

    fwd_b = jax.jit(
        jax.vmap(lambda th: twin._run(th, drives, key, spike_mode="hard")["spikes"])
    )
    vg_b = jax.jit(jax.vmap(jax.value_and_grad(loss_fn)))

    rows = []
    print(f"{'K':>5} | {'fwd (s)':>8} | {'fwd sims/s':>10} | {'fwd MN·steps/s':>14} "
          f"| {'grad (s)':>8} | {'grad sims/s':>11}")
    for K in args.batch_sizes:
        thetas = batched_thetas(twin, K, rng)
        t_fwd = time_fn(fwd_b, thetas)
        t_vg = time_fn(vg_b, thetas)
        row = {
            "K": K,
            "t_forward_s": t_fwd,
            "forward_sims_per_s": K / t_fwd,
            "forward_mn_steps_per_s": K * args.n_mn * n_steps / t_fwd,
            "t_grad_s": t_vg,
            "grad_sims_per_s": K / t_vg,
        }
        rows.append(row)
        print(f"{K:5d} | {t_fwd:8.2f} | {row['forward_sims_per_s']:10.2f} "
              f"| {row['forward_mn_steps_per_s']:14.3e} | {t_vg:8.2f} "
              f"| {row['grad_sims_per_s']:11.2f}")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "device": dev.platform,
        "n_mn": args.n_mn,
        "n_steps": n_steps,
        "dt_ms": args.dt_ms,
        "rows": rows,
    }
    out_path = RESULTS / f"phase2_benchmark_{dev.platform}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()

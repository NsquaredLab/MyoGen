"""Plain vs checkpointed scan at the benchmark shape (n_mn 10, 14 000 steps):
gradient of the mean soma voltage (well conditioned) and of the benchmark's
surrogate rate loss (exploding), on whatever device JAX uses."""
import json, sys
import jax, jax.numpy as jnp, numpy as np
import protocols, observables as obs
from poolbuild import PoolTwin, TwinSpec

_, cmd = protocols.step(0.1, t_pre_s=0.2, t_on_s=1.0, t_post_s=0.2, level=40.0)
drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
key = jax.random.PRNGKey(0)
res = {"device": jax.devices()[0].platform, "n_steps": len(cmd)}
for name, f in (("mean_voltage", lambda o: jnp.mean(o["v"])),
                ("benchmark_rate_loss",
                 lambda o: jnp.mean(obs.smoothed_rates(o["spikes"], 0.1, tau_ms=150.0) ** 2))):
    out = {}
    for seg in (None, 140):
        twin = PoolTwin(TwinSpec(n_mn=10, n_dd=100, seed=0, remat_segment=seg))
        mode = "hard" if name == "mean_voltage" else "surrogate"
        v, g = jax.jit(jax.value_and_grad(
            lambda th: f(twin._run(th, drives, key, "surrogate"))))(twin.default_theta())
        out[seg] = (float(v), {k: float(x) for k, x in g.items()})
    (v0, g0), (v1, g1) = out[None], out[140]
    rel = {k: abs(g1[k] - g0[k]) / max(abs(g0[k]), 1e-30) for k in g0}
    res[name] = {"value_plain": v0, "value_remat": v1, "grad_plain": g0, "grad_remat": g1,
                 "max_rel_grad_diff": max(rel.values()), "rel": rel}
print(json.dumps(res, indent=1))

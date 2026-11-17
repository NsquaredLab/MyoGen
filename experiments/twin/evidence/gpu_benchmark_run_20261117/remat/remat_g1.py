"""Checkpointing in the verified regime: G1 b_config at dt 0.025 ms, 2 s, x64
(the cell where AD matches converged finite differences). Plain scan vs
remat_segment=400 on Salome's own G1 loss; gradient_repair.py is not edited,
TwinSpec is wrapped to set remat_segment."""
import functools, json, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import poolbuild, gradient_repair as gr

Orig = poolbuild.TwinSpec
res = {"device": jax.devices()[0].platform, "dt_ms": 0.025, "horizon_s": 2.0}
for seg in (None, 400):
    poolbuild.TwinSpec = functools.partial(Orig, remat_segment=seg)
    if hasattr(gr, "TwinSpec"):
        gr.TwinSpec = poolbuild.TwinSpec
    loss, n_spk = gr._make_loss("b_config", 0.025, 2.0)
    s0 = jnp.array([1.0, 1.0])
    t0 = time.time()
    v, g = jax.value_and_grad(loss)(s0)
    res[str(seg)] = {"value": float(v), "grad": np.asarray(g).tolist(), "n_spikes": n_spk,
                     "wall_s": time.time() - t0}
poolbuild.TwinSpec = Orig
a, b = np.array(res["None"]["grad"]), np.array(res["400"]["grad"])
res["max_rel_grad_diff"] = float(np.max(np.abs(b - a) / np.abs(a)))
res["value_rel_diff"] = abs(res["400"]["value"] - res["None"]["value"]) / abs(res["None"]["value"])
print("RESULT " + json.dumps(res))

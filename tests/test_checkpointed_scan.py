"""Gradient checkpointing of the twin's time loop changes memory, not results."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "twin"))
poolbuild = pytest.importorskip("poolbuild")


def test_checkpointed_scan_matches_scan_in_value_and_gradient():
    xs = jnp.linspace(0.0, 1.0, 120)

    def loss(w, segment):
        def body(c, x):
            c = jnp.tanh(w * c + x)
            return c, c ** 2
        c, ys = poolbuild.checkpointed_scan(body, jnp.float32(0.1), xs, segment)
        return c + ys.sum()

    ref = jax.value_and_grad(loss)(0.7, None)
    for seg in (1, 12, 40, 120):
        got = jax.value_and_grad(loss)(0.7, seg)
        np.testing.assert_allclose(got, ref, rtol=1e-6)
    with pytest.raises(ValueError):
        poolbuild.checkpointed_scan(lambda c, x: (c, x), 0.0, xs, 7)


def test_pool_twin_gradient_unchanged_by_checkpointing():
    import protocols

    _, cmd = protocols.step(0.1, t_pre_s=0.05, t_on_s=0.2, t_post_s=0.05, level=40.0)
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
    key = jax.random.PRNGKey(0)

    def grad(segment):
        twin = poolbuild.PoolTwin(poolbuild.TwinSpec(n_mn=2, n_dd=10,
                                                     remat_segment=segment))
        loss = lambda th: jnp.mean(twin._run(th, drives, key, "surrogate")["v"])  # noqa: E731
        return jax.value_and_grad(loss)(twin.default_theta())

    (v0, g0), (v1, g1) = grad(None), grad(100)
    np.testing.assert_allclose(v1, v0, rtol=1e-5)
    for k in g0:
        np.testing.assert_allclose(g1[k], g0[k], rtol=1e-4, atol=1e-8)

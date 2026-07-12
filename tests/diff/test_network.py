"""The Pythonic differentiable-network façade: build imperatively, run purely, differentiate anything.

These tests assert the load-bearing claims of the proposed API:
- you build with plain objects and `connect` by object reference (no strings);
- a multi-population network with a feedback loop compiles to one differentiable scan;
- `jax.grad` flows to *every* connection weight, including the recurrent/feedback one
  (i.e. reflex gains are fittable);
- params and results are indexed by the population objects, not strings;
- zero-delay cycles are rejected at build time;
- the same network wraps into a vmap-batchable RL env.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from myogen.diff.network import LIFPopulation, Network

DRIVE = jnp.ones(20) * 1.3


def _reflex_net(seed: int = 2):
    """mn (motor) → ia (afferent-like) → ⊣ mn : a minimal recurrent reflex loop."""
    net = Network(dt_ms=0.05, seed=seed)
    mn = net.add(LIFPopulation(20, name="mn", r_input=1.0))
    ia = net.add(LIFPopulation(20, name="ia", r_input=300.0))
    net.connect(mn, ia, weight=1.0, p=0.6)
    net.connect(ia, mn, weight=0.5, p=0.6, inhibitory=True, delay_steps=1)  # feedback loop
    net.emg_from(mn, n_channels=4)
    return net, mn, ia


def test_build_and_simulate_with_live_loop():
    net, mn, ia = _reflex_net()
    r = net.simulate(150.0, drive={mn: DRIVE})
    assert r[mn].spikes.shape == (3000, 20)          # result indexed by the object
    assert float(r[mn].spikes.sum()) > 0
    assert float(r[ia].spikes.sum()) > 0             # the loop is actually alive
    assert r.emg.shape[1] == 4


def test_gradient_flows_to_every_connection_including_feedback():
    net, mn, ia = _reflex_net()

    def loss(params):
        return jnp.mean(net.simulate(150.0, drive={mn: DRIVE}, params=params).emg ** 2)

    grads = jax.grad(loss)(net.params)
    assert np.all([np.isfinite(float(w)) for w in grads.weights])
    assert float(grads[mn >> ia]) != 0.0
    assert float(grads[ia >> mn]) != 0.0             # gradient through the recurrent reflex gain


def test_params_and_results_indexed_by_object():
    net, mn, ia = _reflex_net()
    assert float(net.params[mn >> ia]) == 1.0        # net.params[src >> tgt], not "mn->ia"
    assert float(net.params[ia >> mn]) == 0.5
    r = net.simulate(50.0, drive={mn: DRIVE})
    assert hasattr(r[mn], "spikes")


def test_zero_delay_edge_is_rejected():
    net, mn, ia = _reflex_net()
    with pytest.raises(ValueError, match="delay_steps"):
        net.connect(mn, mn, weight=0.1, delay_steps=0)


def test_same_network_wraps_into_a_batchable_env():
    net, mn, ia = _reflex_net()
    env = net.as_env(
        action=mn, observe=mn, reward=lambda r: -jnp.mean(r.emg ** 2),
        duration_ms=100.0, base_drive=1.3,
    )
    key, obs = env.reset(jax.random.PRNGKey(0))
    key, obs, reward, done = env.step(key, obs, jnp.zeros(20))
    assert jnp.ndim(reward) == 0 and bool(done)
    rewards = jax.vmap(lambda a: env.step(key, obs, a)[2])(
        jnp.stack([jnp.zeros(20), jnp.ones(20) * 0.3])
    )
    assert rewards.shape == (2,)

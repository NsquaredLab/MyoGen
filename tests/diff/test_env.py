"""RL mode: the same forward model wraps into a deterministic, vmap-batchable environment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from myogen.diff import autodiff as A
from myogen.diff import env as ENV


def _env(n_units=3, n_steps=1500):
    cfg = A.make_config(n_units=n_units, n_steps=n_steps)
    target_action = jnp.linspace(0.8, 1.6, n_units)
    target_env = A.emg_envelope(A.rollout({"drive_gain": target_action}, cfg)["emg"])
    return ENV.PoolEnv(cfg, target_env), target_action


def test_reset_and_step_shapes():
    env, target_action = _env()
    state, obs0 = env.reset(jax.random.PRNGKey(0))
    _, obs, reward, done = env.step(state, target_action)
    assert obs0.shape == obs.shape
    assert jnp.ndim(reward) == 0
    assert bool(done)


def test_reward_is_maximal_at_target_action():
    env, target_action = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    r_target = float(env.step(state, target_action)[2])
    r_off = float(env.step(state, target_action + 0.4)[2])
    assert r_target == pytest.approx(0.0, abs=1e-9)
    assert r_target > r_off


def test_env_is_deterministic():
    env, _ = _env()
    state, _ = env.reset(jax.random.PRNGKey(0))
    a = jnp.array([1.0, 1.1, 1.2])
    assert float(env.step(state, a)[2]) == float(env.step(state, a)[2])


def test_env_vmap_batches_actions():
    env, target_action = _env()
    batch = jnp.stack([target_action, target_action + 0.3, target_action - 0.3])
    rewards = ENV.batched_rewards(env, batch)
    assert rewards.shape == (3,)
    assert int(np.argmax(np.asarray(rewards))) == 0  # target action is best

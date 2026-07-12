"""RL mode: the *same* forward model wrapped as a vectorizable environment.

This is the other half of "do both": :mod:`myogen.diff.autodiff` wraps
:func:`myogen.diff.model.forward` in ``jax.grad``; here the identical function is wrapped as a
gymnax-style env (``reset``/``step``, ``jax.vmap``-batchable, deterministic given a key). RL
never backprops through the dynamics, so it is untroubled by the ill-conditioned,
non-differentiable parts that make the autodiff inverse hard — the two modes are complementary
over one core.

The env here is single-decision (bandit-style): an action is a per-unit drive vector, a step
runs one rollout and returns an EMG-envelope observation plus a reward = −‖envelope − target‖².
Temporal, closed-loop control is the natural extension — jaxley's ``return_states``/
``all_states`` gives chunked stepping that carries simulation state across steps (mechanism
validated), and a stochastic ``jax.random`` drive (see :func:`myogen.diff.spikes.drive_spike_trains`)
makes the key load-bearing — but the contract this proves is the one that matters: one pure
core serves both grad and env.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from myogen.diff.autodiff import rollout
from myogen.diff.emg import emg_envelope


@dataclass(frozen=True)
class PoolEnv:
    """A minimal RL environment over the shared MU-pool forward model.

    Parameters
    ----------
    cfg : dict
        The shared static config from :func:`myogen.diff.autodiff.make_config`.
    target_envelope : Array
        The EMG envelope the agent is rewarded for matching, shape ``(n_steps, n_channels)``.
    """

    cfg: dict
    target_envelope: jnp.ndarray

    @property
    def action_dim(self) -> int:
        return self.cfg["n_units"]

    def reset(self, key):
        """Return ``(state, obs)``. State is the RNG key; obs is a zero placeholder."""
        n_channels = self.target_envelope.shape[1]
        return key, jnp.zeros((n_channels,))

    def step(self, state, action, key=None):
        """Apply a per-unit drive ``action`` → ``(state, obs, reward, done)``.

        ``obs`` is the per-channel mean EMG envelope; ``reward`` is the negative MSE to the
        target envelope; the episode is single-step (``done=True``). Pure and deterministic
        given ``action`` (and ``key``, once a stochastic drive is wired in).
        """
        out = rollout({"drive_gain": action}, self.cfg, key)
        envelope = emg_envelope(out["emg"])  # (n_steps, n_channels)
        reward = -jnp.mean((envelope - self.target_envelope) ** 2)
        obs = jnp.mean(envelope, axis=0)  # per-channel mean envelope
        done = jnp.array(True)
        return state, obs, reward, done


def batched_rewards(env: PoolEnv, actions: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a batch of actions ``(batch, n_units)`` in parallel via ``jax.vmap``.

    This is the RL-relevant capability: the same forward core, vectorized across many
    environment instances/actions at once (the substrate a batched policy rollout needs).
    """
    state, _ = env.reset(jax.random.PRNGKey(0))
    return jax.vmap(lambda a: env.step(state, a)[2])(actions)

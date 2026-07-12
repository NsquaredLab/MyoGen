"""Autodiff mode: fit MU-pool parameters to a target EMG by gradient descent.

This is the first inverse solver for "EMG → MU-pool parameters" and the proof that the whole
pipeline (jaxley dynamics → surrogate spikes → linear EMG) is differentiable end to end. The
demo is self-supervised: plant per-unit drive gains, generate a target EMG with the forward
model, then recover the gains from that EMG alone using ``jax.grad`` + optax.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from myogen.diff.emg import default_emg_config, emg_envelope
from myogen.diff.model import build_pool, default_base_drive, forward


def make_config(
    n_units: int = 4,
    dt_s: float = 0.025,
    n_steps: int = 4000,
    n_channels: int = 4,
    i_amp: float = 1.0,
    seed: int = 0,
) -> dict:
    """Static config shared by autodiff and the RL env: the jaxley net, EMG geometry, drive."""
    return {
        "net": build_pool(n_units),
        "n_units": n_units,
        "dt_s": dt_s,
        "n_steps": n_steps,
        "emg_cfg": default_emg_config(n_units, n_channels, seed=seed),
        "base": default_base_drive(n_steps, i_amp),
    }


def rollout(params: dict, cfg: dict, key=None) -> dict:
    """Run the shared forward model for a given parameter pytree."""
    return forward(
        params,
        cfg["base"],
        key,
        net=cfg["net"],
        n_units=cfg["n_units"],
        dt_s=cfg["dt_s"],
        emg_cfg=cfg["emg_cfg"],
    )


def emg_loss(params: dict, target_envelope: jnp.ndarray, cfg: dict) -> jnp.ndarray:
    """MSE between the model EMG envelope and a target EMG envelope (smooth, drive-encoding)."""
    return jnp.mean((emg_envelope(rollout(params, cfg)["emg"]) - target_envelope) ** 2)


def recover_drive_gains(true_gains, init_gains, cfg: dict, *, steps: int = 200, lr: float = 0.05):
    """Plant ``true_gains`` → target EMG, then recover them from the EMG envelope by gradient descent.

    Returns ``(recovered_gains, loss_history)``. Gains are kept inside HH's repetitive-firing
    window by clipping after each step, so a stray step can't push a unit into depolarization
    block (a flat, gradient-less region).
    """
    true_gains = jnp.asarray(true_gains, dtype=float)
    init_gains = jnp.asarray(init_gains, dtype=float)

    target_envelope = emg_envelope(rollout({"drive_gain": true_gains}, cfg)["emg"])

    params = {"drive_gain": init_gains}
    opt = optax.adam(lr)
    opt_state = opt.init(params)
    loss_and_grad = jax.jit(jax.value_and_grad(lambda p: emg_loss(p, target_envelope, cfg)))

    history = []
    for _ in range(steps):
        loss, grads = loss_and_grad(params)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params = {"drive_gain": jnp.clip(params["drive_gain"], 0.4, 2.0)}
        history.append(float(loss))

    return params["drive_gain"], jnp.asarray(history)

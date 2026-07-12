"""Differentiable spikes and a JAX re-home of the descending-drive point process.

Two independent concerns live here:

1. **Surrogate-gradient spikes** (:func:`spike_train`): a spike is a threshold crossing — a
   step function whose gradient is zero/undefined. We keep the hard threshold in the forward
   pass but substitute a smooth ("fast sigmoid") derivative in the backward pass, so gradients
   can flow from an EMG/loss back through spike timing. This is the standard differentiable-SNN
   trick and it is the single hardest unknown the PoC has to clear.

2. **A functional Poisson/Gamma drive generator** (:func:`poisson_spike_train`,
   :func:`gamma_spike_train`): a ``jax.random`` re-home of the Cython
   ``_poisson_process_generator``. It uses the *same* corrected integrate-and-threshold scheme
   as the fixed Cython path — a single ``Exp(1)`` inter-spike threshold drawn as ``-log(1 - U)``
   — so at a constant rate the ISIs are exponential with CV = 1 (a true Poisson process). It is
   stochastic and non-differentiable by design; it feeds the RL env, not the autodiff loss.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------------------------
# Surrogate-gradient spikes (differentiable)
# --------------------------------------------------------------------------------------------


@jax.custom_jvp
def _surrogate_heaviside(x):
    """Heaviside step (1 where x > 0) in the forward pass; smooth derivative in backward."""
    return (x > 0).astype(x.dtype)


@_surrogate_heaviside.defjvp
def _surrogate_heaviside_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    beta = 10.0  # fast-sigmoid surrogate: derivative 1 / (1 + beta|x|)^2
    surrogate = 1.0 / (1.0 + beta * jnp.abs(x)) ** 2
    return _surrogate_heaviside(x), surrogate * dx


def spike_train(voltages, threshold_mV: float = 0.0):
    """Differentiable spike train from voltage traces via upward threshold crossings.

    Parameters
    ----------
    voltages : Array, shape ``(..., n_steps)``
        Membrane potential over time (last axis is time).
    threshold_mV : float
        Spike detection threshold.

    Returns
    -------
    Array, shape ``(..., n_steps)``
        Approximately-impulse spikes (1 at an upward crossing). Differentiable w.r.t.
        ``voltages`` through the surrogate gradient.
    """
    above = _surrogate_heaviside(voltages - threshold_mV)
    prev = jnp.pad(above[..., :-1], [(0, 0)] * (above.ndim - 1) + [(1, 0)])
    return above * (1.0 - prev)  # 1 only on the step it first goes above threshold


# --------------------------------------------------------------------------------------------
# Functional Poisson / Gamma drive (jax.random re-home of the Cython generator)
# --------------------------------------------------------------------------------------------


def poisson_spike_train(key, rate_hz: float, dt_s: float, n_steps: int):
    """A true (inhomogeneous-ready) Poisson spike train by integrate-and-threshold.

    Mirrors the corrected Cython generator: accumulate ``rate * dt`` and emit a spike when it
    crosses an ``Exp(1)`` threshold, then redraw the threshold as ``-log(1 - U)`` (the exact,
    ``log(0)``-safe inverse-transform). At constant rate the ISIs are exponential (CV = 1).
    """
    key, sub = jax.random.split(key)
    thres0 = -jnp.log1p(-jax.random.uniform(sub))

    def step(carry, _):
        yi, thres, k = carry
        yi = yi + rate_hz * dt_s
        spike = yi >= thres
        k, sub = jax.random.split(k)
        new_thres = jnp.where(spike, -jnp.log1p(-jax.random.uniform(sub)), thres)
        new_yi = jnp.where(spike, 0.0, yi)
        return (new_yi, new_thres, k), spike.astype(jnp.float32)

    _, spikes = jax.lax.scan(step, (0.0, thres0, key), None, length=n_steps)
    return spikes


def gamma_spike_train(key, rate_hz: float, shape: int, dt_s: float, n_steps: int):
    """Gamma renewal spike train (CV = 1/sqrt(shape)); ``shape=1`` reduces to Poisson.

    The inter-spike threshold is the mean of ``shape`` ``Exp(1)`` draws — i.e. ``Gamma(shape,
    shape)`` with unit mean — so the mean rate is preserved and only regularity changes, exactly
    like the Cython Gamma generator.
    """

    def draw_threshold(k):
        us = jax.random.uniform(k, (shape,))
        return jnp.mean(-jnp.log1p(-us))

    key, sub = jax.random.split(key)
    thres0 = draw_threshold(sub)

    def step(carry, _):
        yi, thres, k = carry
        yi = yi + rate_hz * dt_s
        spike = yi >= thres
        k, sub = jax.random.split(k)
        new_thres = jnp.where(spike, draw_threshold(sub), thres)
        new_yi = jnp.where(spike, 0.0, yi)
        return (new_yi, new_thres, k), spike.astype(jnp.float32)

    _, spikes = jax.lax.scan(step, (0.0, thres0, key), None, length=n_steps)
    return spikes


def drive_spike_trains(key, rates_hz, dt_s: float, n_steps: int, shape: int = 1):
    """Vectorized per-unit drive: one independent (Poisson or Gamma) train per rate.

    ``rates_hz`` is a 1-D array of per-unit rates. Each unit gets its own split key via
    ``fold_in`` (the JAX analog of the Cython ``derive_subseed`` per-unit seeding).
    """
    rates_hz = jnp.atleast_1d(jnp.asarray(rates_hz))
    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(rates_hz.shape[0]))
    if shape == 1:
        gen = lambda k, r: poisson_spike_train(k, r, dt_s, n_steps)
    else:
        gen = lambda k, r: gamma_spike_train(k, r, shape, dt_s, n_steps)
    return jax.vmap(gen)(keys, rates_hz)  # (n_units, n_steps)

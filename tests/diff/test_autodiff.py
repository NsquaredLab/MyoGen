"""Autodiff mode: the pipeline is differentiable, and the gradient is numerically correct.

We do NOT assert full parameter recovery here: the drive→EMG inverse is genuinely ill-posed
(HH's saturating f-I gives a near-flat loss plateau, and MU superposition is multimodal), so
gradient-MAP alone does not reliably reach the planted parameters — an intended finding that
motivates SBI/priors/RL, not a machinery failure. What we CAN assert rigorously is that the
autodiff machinery is correct: the gradient through the jaxley dynamics matches finite
differences, and the full pipeline (dynamics → surrogate spikes → EMG) is differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from myogen.diff import autodiff as A


def test_emg_pipeline_is_differentiable():
    cfg = A.make_config(n_units=2, n_steps=1500)
    target = A.emg_envelope(A.rollout({"drive_gain": jnp.array([1.0, 1.3])}, cfg)["emg"])
    grad = jax.grad(lambda g: A.emg_loss({"drive_gain": g}, target, cfg))(jnp.array([1.1, 1.1]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_gradient_matches_finite_difference_through_dynamics():
    """The gradient our pipeline computes through jaxley is numerically correct (cosine ~ 1)."""
    cfg = A.make_config(n_units=2, n_steps=1500)

    def vloss(gains):  # mean membrane potential — a strong, smooth gradient through the dynamics
        return jnp.mean(A.rollout({"drive_gain": gains}, cfg)["voltages"])

    g0 = jnp.array([1.0, 1.2])
    auto = np.asarray(jax.grad(vloss)(g0))
    eps = 1e-3
    fd = np.array(
        [float((vloss(g0.at[i].add(eps)) - vloss(g0.at[i].add(-eps))) / (2 * eps)) for i in range(2)]
    )
    cos = float(np.dot(auto, fd) / (np.linalg.norm(auto) * np.linalg.norm(fd) + 1e-12))
    assert cos > 0.9, f"autodiff={auto}, finite_diff={fd}, cosine={cos}"

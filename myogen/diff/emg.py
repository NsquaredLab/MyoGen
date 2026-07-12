"""A trivial, fully-differentiable linear surface-EMG readout.

This is the deliberately minimal stand-in for the real EMG pipeline
(``myogen/simulator/core/emg/surface/surface_emg.py``): each motor unit has a MUAP waveform
(a temporal template) and a spatial weight per electrode channel; the surface EMG is the sum
over units of (spike train ⊛ MUAP) projected onto the electrode grid. It is one convolution +
one matmul, so gradients flow cleanly from the EMG all the way back to spikes and drive.

The real volume-conductor / MUAP physics (FEM-JAX later) slots in behind the same
``spikes_to_emg`` signature; nothing downstream needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class EMGConfig:
    """Fixed (non-trained) linear-EMG geometry: per-unit MUAP templates + spatial weights."""

    muap_templates: jnp.ndarray  # (n_units, kernel_len)
    spatial_weights: jnp.ndarray  # (n_units, n_channels)


def default_emg_config(
    n_units: int,
    n_channels: int | None = None,
    kernel_len: int = 41,
    *,
    spatial_overlap: float = 0.4,
    seed: int = 0,
):
    """Build biphasic MUAP templates and per-unit spatial weights (fixed, not trained).

    ``spatial_overlap`` is the Gaussian width (in channels) of each unit's projection onto the
    electrode grid. Small values (≈0.4, the default, with ``n_channels == n_units``) give a
    near-diagonal, *identifiable* mixing — each unit dominates one channel — so the drive→EMG
    inverse is well-posed. Larger values reproduce the realistic, ill-posed MU-superposition
    regime where many drive sets explain the same EMG.
    """
    if n_channels is None:
        n_channels = n_units
    key = jax.random.PRNGKey(seed)
    t = jnp.linspace(-3.0, 3.0, kernel_len)
    base = -t * jnp.exp(-(t**2))  # a simple biphasic MUAP shape
    base = base / jnp.max(jnp.abs(base))
    amps = 0.5 + jax.random.uniform(key, (n_units,))  # per-unit MUAP amplitude
    templates = amps[:, None] * base[None, :]
    # unit i is centered on channel i (spread over n_channels), width = spatial_overlap
    centers = jnp.linspace(0, n_channels - 1, n_units)
    chan = jnp.arange(n_channels)
    weights = jnp.exp(-0.5 * ((chan[None, :] - centers[:, None]) / spatial_overlap) ** 2)
    return EMGConfig(muap_templates=templates, spatial_weights=weights)


def spikes_to_emg(spikes: jnp.ndarray, cfg: EMGConfig) -> jnp.ndarray:
    """Spikes ``(n_units, n_steps)`` → surface EMG ``(n_steps, n_channels)`` (differentiable)."""
    muaps = jax.vmap(lambda s, k: jnp.convolve(s, k, mode="same"))(
        spikes, cfg.muap_templates
    )  # (n_units, n_steps)
    return muaps.T @ cfg.spatial_weights  # (n_steps, n_channels)


def emg_envelope(emg: jnp.ndarray, win: int = 301) -> jnp.ndarray:
    """Smooth per-channel power envelope of the EMG ``(n_steps, n_channels)``.

    Raw spike-level EMG is hypersensitive to exact spike timing, which makes an MSE loss a
    rugged staircase that gradient descent stalls on. The envelope (a low-pass of ``emg**2``,
    i.e. EMG amplitude/RMS — what actually encodes drive) is a smooth, monotone-in-drive
    observable, giving a well-behaved loss.
    """
    kernel = jnp.ones(win) / win
    return jax.vmap(lambda c: jnp.convolve(c, kernel, mode="same"), in_axes=1, out_axes=1)(
        emg**2
    )

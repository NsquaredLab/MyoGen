"""
Differentiable EMG synthesis (JAX) for the MyoGen Jaxley backend.

The surface / intramuscular EMG of a motor-unit pool is a linear superposition:

    EMG[channel, t] = Σ_MU (spike_train_MU ⋆ MUAP_{MU, channel})[t]

where ``⋆`` is cross-correlation (``np.correlate(..., mode="same")``) and the MUAP
templates are **static** — they depend only on muscle geometry, tissue
conductivities and conduction velocity, not on the spike trains. Because the
templates are frozen, the whole map from spike trains to EMG is a convolution and
is trivially differentiable: gradients flow ``spikes → EMG`` with no Bessel /
volume-conductor machinery involved.

These pure-JAX functions mirror the summation in
``core/emg/surface/surface_emg.py`` and ``core/emg/intramuscular/intramuscular_emg.py``
but operate on JAX arrays so they compose with :func:`myogen.simulator.jaxley.closed_loop.run_jax`
and ``jax.grad``. The existing Neo-returning methods remain the scientific
default; these are the differentiable runtime form.

Notes
-----
* Inactive motor units simply contribute zero (their spike trains are all-zero),
  so no explicit active-MU masking is needed — pass the full pool.
* MUAP templates must already be resampled to the spike-train timestep. Use
  :func:`resample_muaps` for that (differentiable ``jnp.interp``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "surface_emg_jax",
    "intramuscular_emg_jax",
    "resample_muaps",
]


def _correlate_same(signal, template):
    """Cross-correlation with ``mode="same"`` — matches ``np.correlate``.

    ``signal`` shape ``(T,)``, ``template`` shape ``(Tm,)`` → output ``(T,)``.
    """
    return jnp.correlate(signal, template, mode="same")


def surface_emg_jax(spike_trains, muap_shapes):
    """Differentiable surface EMG for a 2-D electrode grid.

    Parameters
    ----------
    spike_trains : array, shape ``(n_pools, n_mu, T)``
        Per-MU spike (or continuous rate/surrogate) trains at the EMG timestep.
    muap_shapes : array, shape ``(n_mu, n_rows, n_cols, Tm)``
        Static MUAP templates per grid electrode, resampled to the spike-train
        timestep (see :func:`resample_muaps`).

    Returns
    -------
    array, shape ``(n_pools, n_rows, n_cols, T)``
        ``emg[p, r, c] = Σ_mu correlate(spike[p, mu], muap[mu, r, c], "same")``.
    """
    # (n_mu, n_rows, n_cols, Tm) → (n_rows, n_cols, n_mu, Tm)
    muap_grid = jnp.transpose(muap_shapes, (1, 2, 0, 3))

    def channel_emg(spike_pool, muap_channel):
        # spike_pool: (n_mu, T); muap_channel: (n_mu, Tm) → (T,)
        corrs = jax.vmap(_correlate_same)(spike_pool, muap_channel)
        return jnp.sum(corrs, axis=0)

    def pool_emg(spike_pool):
        # vmap channel_emg over (n_rows, n_cols)
        return jax.vmap(jax.vmap(lambda mc: channel_emg(spike_pool, mc)))(muap_grid)

    return jax.vmap(pool_emg)(spike_trains)


def intramuscular_emg_jax(spike_trains, muap_shapes):
    """Differentiable intramuscular EMG for a 1-D electrode set.

    Parameters
    ----------
    spike_trains : array, shape ``(n_pools, n_mu, T)``
    muap_shapes : array, shape ``(n_mu, n_elec, Tm)``
        Static per-electrode MUAP templates at the spike-train timestep.

    Returns
    -------
    array, shape ``(n_pools, n_elec, T)``
        ``emg[p, e] = Σ_mu correlate(spike[p, mu], muap[mu, e], "same")``.
    """
    muap_grid = jnp.transpose(muap_shapes, (1, 0, 2))  # (n_elec, n_mu, Tm)

    def channel_emg(spike_pool, muap_channel):
        corrs = jax.vmap(_correlate_same)(spike_pool, muap_channel)
        return jnp.sum(corrs, axis=0)

    def pool_emg(spike_pool):
        return jax.vmap(lambda mc: channel_emg(spike_pool, mc))(muap_grid)

    return jax.vmap(pool_emg)(spike_trains)


def resample_muaps(muap_shapes, muap_dt_s: float, target_dt_s: float):
    """Resample MUAP templates along their time axis with differentiable interp.

    Mirrors the ``np.interp`` resampling in the Neo EMG path but with
    ``jnp.interp`` so the operation stays in the JAX graph (useful if the
    templates themselves become differentiable via the volume-conductor port).

    Parameters
    ----------
    muap_shapes : array, shape ``(..., Tm)``  Templates on the MUAP grid.
    muap_dt_s : float   Template sample period [s].
    target_dt_s : float   Desired sample period [s] (the spike-train timestep).

    Returns
    -------
    array, shape ``(..., T)``  Resampled templates.
    """
    Tm = muap_shapes.shape[-1]
    duration = Tm * muap_dt_s
    xp = jnp.arange(Tm, dtype=jnp.float32) * jnp.float32(muap_dt_s)
    x = jnp.arange(0.0, duration, target_dt_s, dtype=jnp.float32)

    flat = muap_shapes.reshape(-1, Tm)
    resampled = jax.vmap(lambda fp: jnp.interp(x, xp, fp))(flat)
    return resampled.reshape(*muap_shapes.shape[:-1], x.shape[0])

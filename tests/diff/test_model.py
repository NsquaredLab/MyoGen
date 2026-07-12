"""The shared jaxley forward model: shapes, firing, and end-to-end differentiability."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from myogen.diff import emg as E
from myogen.diff import model as M


def _build(n_units=3, n_steps=1500, n_channels=4):
    net = M.build_pool(n_units)
    emg_cfg = E.default_emg_config(n_units, n_channels)
    base = M.default_base_drive(n_steps, i_amp=1.0)
    return net, emg_cfg, base, n_units, n_steps, n_channels


def test_forward_produces_spikes_and_emg():
    net, emg_cfg, base, n_units, n_steps, n_channels = _build()
    out = M.forward(
        M.make_params(n_units, 1.2), base, None,
        net=net, n_units=n_units, dt_s=0.025, emg_cfg=emg_cfg,
    )
    assert out["voltages"].shape[0] == n_units
    assert out["spikes"].shape == out["voltages"].shape
    assert out["emg"].shape[1] == n_channels
    # the HH cells actually spike, and the EMG is non-trivial
    assert float(out["voltages"].max()) > 0.0  # crosses 0 mV
    assert float(out["spikes"].sum()) > 0.0
    assert float(jnp.abs(out["emg"]).max()) > 0.0


def test_forward_is_end_to_end_differentiable():
    """One jax.grad through jaxley dynamics + surrogate spikes + EMG returns finite gradients."""
    net, emg_cfg, base, n_units, n_steps, n_channels = _build()

    def scalar(gains):
        out = M.forward(
            {"drive_gain": gains}, base, None,
            net=net, n_units=n_units, dt_s=0.025, emg_cfg=emg_cfg,
        )
        return jnp.mean(out["emg"] ** 2)

    grad = jax.grad(scalar)(M.make_params(n_units, 1.2)["drive_gain"])
    assert np.all(np.isfinite(np.asarray(grad)))

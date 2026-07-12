"""The shared forward model: a small jaxley motor-unit pool → spikes → EMG.

``forward(params, base_drive, key)`` is the single pure function the whole PoC is built on.
The autodiff mode (:mod:`myogen.diff.autodiff`) wraps it in ``jax.grad``; the RL mode
(:mod:`myogen.diff.env`) wraps the identical function as an environment. Keeping it pure,
pytree-stated and ``jax.random``-keyed is the only requirement for one core to serve both.

The pool is intentionally minimal — ``n_units`` single-compartment Hodgkin–Huxley cells, each
driven by its own current. The differentiable "MU-pool parameter" is the per-unit **drive
gain** (a smooth scaling of the input current), injected via jaxley ``data_stimulate`` so
``jax.grad`` flows drive-gain → dynamics → surrogate spikes → EMG. Real jaxley biophysics /
FEM-JAG muscle slot in behind this same signature later.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxley as jx
from jaxley.channels import HH

from myogen.diff.emg import EMGConfig, spikes_to_emg
from myogen.diff.spikes import spike_train


# HH is the only jaxley spiking channel that is smooth/differentiable through the spike
# (Izhikevich/Fire warn "gradient will be zero after every spike"). A single default HH
# compartment is tiny, so any injectable current is a huge current *density* and the cell
# goes into depolarization block (fires once). Enlarging the compartment puts a ~0.5-2 nA
# drive in HH's repetitive-firing window, where spike rate tracks the drive — exactly what a
# rate-recruited motor unit needs, and what makes the drive→EMG inverse well-posed.
_HH_RADIUS_UM = 10.0
_HH_LENGTH_UM = 100.0


def build_pool(n_units: int) -> jx.Network:
    """A network of ``n_units`` single-compartment HH cells (sized for repetitive firing)."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    net = jx.Network([jx.Cell(branch, parents=[-1]) for _ in range(n_units)])
    net.insert(HH())
    net.set("radius", _HH_RADIUS_UM)
    net.set("length", _HH_LENGTH_UM)
    for i in range(n_units):
        net.cell(i).branch(0).comp(0).record("v")
    return net


def default_base_drive(n_steps: int, i_amp: float = 1.0) -> jnp.ndarray:
    """A short ramp-on then sustained baseline current (nA) in HH's repetitive-firing window."""
    ramp = jnp.clip(jnp.arange(n_steps) / max(1, n_steps // 20), 0.0, 1.0)
    return i_amp * ramp


def make_params(n_units: int, drive_gain: float = 1.0) -> dict:
    """The trainable parameter pytree: one differentiable drive gain per motor unit."""
    return {"drive_gain": jnp.full((n_units,), float(drive_gain))}


def forward(
    params: dict,
    base_drive: jnp.ndarray,
    key=None,
    *,
    net: jx.Network,
    n_units: int,
    dt_s: float,
    emg_cfg: EMGConfig,
    checkpoint_lengths=None,
) -> dict:
    """Pure forward: params + drive → voltages, spikes, EMG.

    Parameters
    ----------
    params : dict
        ``{"drive_gain": (n_units,)}`` — the differentiable per-unit drive gains.
    base_drive : Array
        Baseline current template, shape ``(n_steps,)`` (shared) or ``(n_units, n_steps)``.
    key : jax.random.PRNGKey, optional
        Unused in the deterministic forward; present so the signature matches the RL/stochastic
        path (which injects a jax.random drive). Threading it keeps the function pure.
    net, n_units, dt_s, emg_cfg, checkpoint_lengths
        Static configuration (the jaxley module, timestep, EMG geometry, BPTT checkpointing).

    Returns
    -------
    dict
        ``{"voltages": (n_units, n_t), "spikes": (n_units, n_t), "emg": (n_t, n_channels)}``.
    """
    gains = params["drive_gain"]
    n_steps = base_drive.shape[-1]
    currents = gains[:, None] * (base_drive if base_drive.ndim == 2 else base_drive[None, :])

    data_stimuli = None
    for i in range(n_units):
        data_stimuli = net.cell(i).branch(0).comp(0).data_stimulate(
            currents[i], data_stimuli
        )

    voltages = jx.integrate(
        net,
        delta_t=dt_s,
        t_max=n_steps * dt_s,
        data_stimuli=data_stimuli,
        checkpoint_lengths=checkpoint_lengths,
    )  # (n_units, n_t)

    spikes = spike_train(voltages)
    emg = spikes_to_emg(spikes, emg_cfg)
    return {"voltages": voltages, "spikes": spikes, "emg": emg}

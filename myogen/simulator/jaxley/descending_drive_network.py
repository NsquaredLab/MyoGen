"""Calibrated descending-drive network for the Jaxley NERLab pool.

This is the Jaxley counterpart of
``myogen.simulator.neuron.descending_drive_network``.  It deliberately uses
the same profile-card parameters and model semantics:

* gamma-renewal descending drive implemented by hazard accumulation;
* equal fan-in and dendritic-transfer weight scaling;
* peak-normalized dendritic Exp2Syn kinetics (0.2/0.3 ms by default);
* live dendritic driving force and the production synaptic delay;
* live-voltage Gfluctdv conductances on the dendritic compartment.

The random streams are backend-specific, so ordinary profile simulations are
expected to agree statistically rather than spike-for-spike.  Exact paired
validation must continue to use the frozen event/connectivity artifact in
``experiments/twin/validation_protocol.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
from jaxley.integrate import build_init_and_step_fn

from myogen.simulator.jaxley.jax_models import gamma_init, gamma_step
from myogen.simulator.jaxley.populations import AlphaMN__Pool


@dataclass(frozen=True)
class DescendingDriveResult:
    """Arrays returned by :func:`run_descending_drive`."""

    voltage_mV: np.ndarray
    dendrite_voltage_mV: np.ndarray
    spikes: np.ndarray
    dd_spikes: np.ndarray
    connectivity: np.ndarray
    weights_uS: np.ndarray
    transfer: np.ndarray


def excitation_to_drive_hz(segments, threshold_hz, max_hz, timestep_ms=0.1, noise_seed=42):
    """Map profile excitation segments to the calibrated noisy DD rate.

    Kept byte-for-byte equivalent in semantics to the NEURON helper while
    avoiding a NEURON import in a JAX process.
    """
    parts = [
        np.linspace(a, b, round(duration * 1000 / timestep_ms), endpoint=False)
        for duration, a, b in segments
    ]
    excitation = np.concatenate(parts)
    hz = np.where(
        excitation > 0,
        threshold_hz + excitation * (max_hz - threshold_hz),
        0.8 * threshold_hz,
    )
    hz = hz + np.abs(
        np.random.default_rng(noise_seed).normal(0, hz.max() * 0.01, size=hz.size)
    )
    return excitation, hz


def equal_fan_in_connectivity(n_mn: int, n_dd: int, n_inputs: int, seed: int) -> np.ndarray:
    """Return the profile network's target-by-source equal-fan-in matrix."""
    if not 1 <= n_inputs <= n_dd:
        raise ValueError(f"n_inputs must be 1..{n_dd}, got {n_inputs}")
    rng = np.random.default_rng(seed)
    out = np.zeros((n_mn, n_dd), dtype=np.float32)
    for i in range(n_mn):
        out[i, rng.choice(n_dd, size=n_inputs, replace=False)] = 1.0
    return out


def _exp2_constants(dt_ms: float, tau1_ms: float, tau2_ms: float) -> tuple[float, float, float]:
    if not 0 < tau1_ms < tau2_ms:
        raise ValueError("Exp2Syn requires 0 < tau1_ms < tau2_ms")
    t_peak = tau1_ms * tau2_ms / (tau2_ms - tau1_ms) * np.log(tau2_ms / tau1_ms)
    norm = 1.0 / (np.exp(-t_peak / tau2_ms) - np.exp(-t_peak / tau1_ms))
    return np.exp(-dt_ms / tau1_ms), np.exp(-dt_ms / tau2_ms), norm


def _network(cells, n_steps: int, initial_voltage_mV: float):
    net = jx.Network([cw.cell for cw in cells])
    placeholder = jnp.zeros(n_steps, dtype=jnp.float32)
    for i in range(len(cells)):
        net.cell(i).branch(0).loc(0.5).record("v", verbose=False)
    for i in range(len(cells)):
        net.cell(i).branch(1).loc(0.5).record("v", verbose=False)
    for i in range(len(cells)):
        net.cell(i).branch(1).loc(0.5).stimulate(placeholder, verbose=False)
    net.set("v", float(initial_voltage_mV))
    net.init_states()
    return net


def _dendritic_transfer(cells, dt_ms: float, initial_voltage_mV: float,
                        probe_nA: float = 0.01, settle_ms: float = 200.0) -> np.ndarray:
    """Measure DC dendrite-to-soma transfer relative to soma injection."""
    n_steps = round(settle_ms / dt_ms)

    def steady(branch: int, amplitude: float):
        for cw in cells:
            cw.cell.delete_recordings()
            cw.cell.delete_stimuli()
        net = jx.Network([cw.cell for cw in cells])
        waveform = jnp.full(n_steps, amplitude, dtype=jnp.float32)
        for i in range(len(cells)):
            net.cell(i).branch(branch).loc(0.5).stimulate(waveform, verbose=False)
            net.cell(i).branch(0).loc(0.5).record("v", verbose=False)
        net.set("v", float(initial_voltage_mV))
        net.init_states()
        v = np.asarray(jx.integrate(net, delta_t=dt_ms, t_max=settle_ms - dt_ms))
        return v[:, -1]

    baseline = steady(0, 0.0)
    dend = steady(1, probe_nA) - baseline
    soma = steady(0, probe_nA) - baseline
    if np.any(~np.isfinite(dend)) or np.any(~np.isfinite(soma)) or np.any(soma == 0):
        raise RuntimeError("non-finite dendritic-transfer measurement")
    return dend / soma


def run_descending_drive(
    thresholds,
    drive_hz,
    dd_params,
    *,
    timestep_ms: float = 0.1,
    seed: int = 42,
    initial_voltage_mV: float = -67.0,
    tau1_ms: float = 0.2,
    tau2_ms: float = 0.3,
    e_exc_mV: float = 70.0,
    e_noise_mV: float = 0.0,
    i_noise_mV: float = -75.0,
    external_event_delay_ms: float = 1.0,
    synaptic_delay_ms: float = 1.0,
) -> DescendingDriveResult:
    """Run the calibrated profile-card descending drive on Jaxley.

    ``initial_voltage_mV=-67`` reproduces the current NEURON profile runner.
    Matched intrinsic validation may instead pass 0 mV explicitly.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    drive_hz = np.asarray(drive_hz, dtype=np.float32)
    if thresholds.ndim != 1 or drive_hz.ndim != 1:
        raise ValueError("thresholds and drive_hz must be one-dimensional")
    if not np.all(np.isfinite(thresholds)) or not np.all(np.isfinite(drive_hz)):
        raise ValueError("thresholds and drive_hz must be finite")

    n_mn = thresholds.size
    n_dd = int(dd_params["dd_neurons"])
    n_steps = drive_hz.size
    n_inputs = round(float(dd_params["conn_probability"]) * n_dd)
    shape = float(dd_params["gamma_shape"])
    weight_exponent = float(dd_params["network"]["weight_exponent"])
    base_weight = float(dd_params["synaptic_weight"])
    gf_scale = float(dd_params.get("gfluctdv_scale", 0.0))

    pool = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        model="NERLab",
        mode="active",
        initial_voltage__mV=float(initial_voltage_mV),
        spike_threshold__mV=50.0,
        normalize_nerlab_thresholds=False,
    )
    cells = list(pool)
    transfer = _dendritic_transfer(cells, timestep_ms, initial_voltage_mV)
    weights = base_weight * (transfer.max() / transfer) ** weight_exponent
    connectivity = equal_fan_in_connectivity(n_mn, n_dd, n_inputs, seed)
    weighted_conn_np = connectivity * weights[:, None]

    for cw in cells:
        cw.cell.delete_recordings()
        cw.cell.delete_stimuli()
    net = _network(cells, n_steps, initial_voltage_mV)
    net.to_jax()
    init_fn, step_fn = build_init_and_step_fn(net)
    states, params = init_fn(net.get_parameters())
    sim_dtype = states["v"].dtype
    weighted_conn = jnp.asarray(weighted_conn_np, dtype=sim_dtype)
    external_inds = net.external_inds.copy()
    rec = jnp.asarray(net.recordings.rec_index.to_numpy(), dtype=jnp.int32)
    rec_soma, rec_dend = rec[:n_mn], rec[n_mn:]

    a1, a2, exp2_norm = _exp2_constants(timestep_ms, tau1_ms, tau2_ms)
    delay_steps = round((external_event_delay_ms + synaptic_delay_ms) / timestep_ms)
    delay_len = max(delay_steps, 1)

    # Gfluctdv density values set by AlphaMN.insert_Gfluctdv, converted to uS.
    area_cm2 = np.asarray([
        2.0 * np.pi * (cw.cell.branch(1).nodes["radius"].to_numpy()[0] * 1e-4)
        * (cw.cell.branch(1).nodes["length"].to_numpy()[0] * 1e-4)
        for cw in cells
    ], dtype=np.float32)
    ge0 = jnp.asarray(1e-5 * gf_scale * area_cm2 * 1e6, dtype=sim_dtype)
    gi0 = jnp.asarray(1e-5 * gf_scale * area_cm2 * 1e6, dtype=sim_dtype)
    ge_sd = jnp.asarray(1.2e-5 * gf_scale * area_cm2 * 1e6, dtype=sim_dtype)
    gi_sd = jnp.asarray(1.2e-5 * gf_scale * area_cm2 * 1e6, dtype=sim_dtype)
    ou_decay = jnp.asarray(np.exp(-timestep_ms / 20.0), dtype=sim_dtype)
    ou_scale = jnp.asarray(np.sqrt(1.0 - np.exp(-2.0 * timestep_ms / 20.0)), dtype=sim_dtype)

    dd_key, noise_key = jax.random.split(jax.random.PRNGKey(seed))
    carry0 = {
        "states": states,
        "prev_soma": jnp.full(n_mn, initial_voltage_mV, dtype=sim_dtype),
        "prev_dend": jnp.full(n_mn, initial_voltage_mV, dtype=sim_dtype),
        "dd": gamma_init(n_dd, shape, key=dd_key),
        "delay": jnp.zeros((delay_len, n_dd), dtype=jnp.float32),
        "delay_idx": jnp.int32(0),
        "A": jnp.zeros(n_mn, dtype=sim_dtype),
        "B": jnp.zeros(n_mn, dtype=sim_dtype),
        "ge": ge0,
        "gi": gi0,
        "noise_key": noise_key,
    }

    def body(carry, rate_t):
        dd_state, emitted_bool = gamma_step(
            carry["dd"], rate_t, timestep_ms, shape, mode="hard")
        emitted = emitted_bool.astype(jnp.float32)
        if delay_steps:
            arriving = carry["delay"][carry["delay_idx"]]
            delay = carry["delay"].at[carry["delay_idx"]].set(emitted)
            delay_idx = (carry["delay_idx"] + 1) % delay_len
        else:
            arriving, delay, delay_idx = emitted, carry["delay"], carry["delay_idx"]

        increment = weighted_conn @ arriving.astype(sim_dtype)
        A = jnp.asarray(a1, dtype=sim_dtype) * carry["A"] + jnp.asarray(exp2_norm, dtype=sim_dtype) * increment
        B = jnp.asarray(a2, dtype=sim_dtype) * carry["B"] + jnp.asarray(exp2_norm, dtype=sim_dtype) * increment
        conductance = B - A

        key, ke, ki = jax.random.split(carry["noise_key"], 3)
        ge = ge0 + (carry["ge"] - ge0) * ou_decay + ge_sd * ou_scale * jax.random.normal(ke, (n_mn,), dtype=sim_dtype)
        gi = gi0 + (carry["gi"] - gi0) * ou_decay + gi_sd * ou_scale * jax.random.normal(ki, (n_mn,), dtype=sim_dtype)
        ge, gi = jnp.maximum(ge, 0.0), jnp.maximum(gi, 0.0)

        vd = carry["prev_dend"]
        i_syn = conductance * (jnp.asarray(e_exc_mV, dtype=sim_dtype) - vd)
        i_noise = ge * (jnp.asarray(e_noise_mV, dtype=sim_dtype) - vd) + gi * (jnp.asarray(i_noise_mV, dtype=sim_dtype) - vd)
        new_states = step_fn(
            carry["states"], params, {"i": i_syn + i_noise},
            external_inds, delta_t=timestep_ms,
        )
        soma = new_states["v"][rec_soma]
        dend = new_states["v"][rec_dend]
        spikes = (carry["prev_soma"] < 50.0) & (soma >= 50.0)
        new_carry = {
            "states": new_states, "prev_soma": soma, "prev_dend": dend,
            "dd": dd_state, "delay": delay, "delay_idx": delay_idx,
            "A": A, "B": B, "ge": ge, "gi": gi, "noise_key": key,
        }
        return new_carry, (soma, dend, spikes.astype(jnp.float32), emitted)

    _, (voltage, dend_voltage, spikes, dd_spikes) = jax.jit(
        lambda c, x: jax.lax.scan(body, c, x)
    )(carry0, jnp.asarray(drive_hz))
    voltage, dend_voltage, spikes, dd_spikes = jax.device_get(
        (voltage, dend_voltage, spikes, dd_spikes)
    )
    return DescendingDriveResult(
        voltage_mV=np.asarray(voltage),
        dendrite_voltage_mV=np.asarray(dend_voltage),
        spikes=np.asarray(spikes),
        dd_spikes=np.asarray(dd_spikes),
        connectivity=connectivity,
        weights_uS=np.asarray(weights),
        transfer=np.asarray(transfer),
    )


__all__ = [
    "DescendingDriveResult",
    "equal_fan_in_connectivity",
    "excitation_to_drive_hz",
    "run_descending_drive",
]

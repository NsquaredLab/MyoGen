"""Spike Train Generation with Descending Drive — Jaxley Backend
=================================================================

This is the Jaxley counterpart of
``03_simulate_spike_trains_descending_drive_neuron.py``.  Both examples use
the current ``fdi_semg`` profile card, the same 25 % MVC protocol, the same
NERLab pool parameters, and the same calibrated descending-drive network.

The backend-specific random streams are deliberately independent, so normal
profile runs are compared at the population/statistical level.  Exact
spike-for-spike backend validation uses the frozen-event protocol in
``experiments/twin/validation_protocol.py``.

Unlike the former Jaxley example, this path does not use a fitted 5 ms IIR
current surrogate.  It uses the NEURON model's dendritic, peak-normalized
0.2/0.3 ms Exp2Syn kinetics, live driving force, equal fan-in, dendritic
transfer scaling, two-stage event delay, and live-voltage Gfluctdv noise.
"""

from pathlib import Path

import joblib
import numpy as np
import quantities as pq
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain

from myogen import load_profile
from myogen.simulator.jaxley.descending_drive_network import (
    excitation_to_drive_hz,
    run_descending_drive,
)

plt.style.use("fivethirtyeight")


def mean_firing_rate(spiketrain):
    return (len(spiketrain) / (spiketrain.t_stop - spiketrain.t_start)).rescale(pq.Hz)


def population_psth(spiketrains, bin_size):
    """Population spike count and bin centres."""
    t_start = min(st.t_start for st in spiketrains).rescale(pq.s).magnitude
    t_stop = max(st.t_stop for st in spiketrains).rescale(pq.s).magnitude
    bs = bin_size.rescale(pq.s).magnitude
    edges = np.arange(t_start, t_stop + 0.5 * bs, bs)
    spikes = np.concatenate([st.rescale(pq.s).magnitude for st in spiketrains])
    spikes = spikes[(spikes >= edges[0]) & (spikes < edges[-1])]
    counts, _ = np.histogram(spikes, bins=edges)
    return counts, edges[:-1] + 0.5 * bs


##############################################################################
# Build the same profile-card protocol as the NEURON example
# -----------------------------------------------------------

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

card = load_profile()  # fdi_semg
protocol = [(1, 0, 0), (4, 0, 25), (4, 25, 25), (4, 25, 0), (1, 0, 0)]
segments = [
    (float(duration), card.excitation_for_pct_mvc(start), card.excitation_for_pct_mvc(stop))
    for duration, start, stop in protocol
]
dd = card.dd_params()
dt_ms = float(card.value("drive.timestep_ms"))
timestep = dt_ms * pq.ms
excitation, drive_hz = excitation_to_drive_hz(
    segments,
    float(card.value("drive.threshold_hz")),
    float(dd["dd_drive__Hz"]),
    timestep_ms=dt_ms,
    noise_seed=card.seeds["drive"],
)
simulation_time = len(drive_hz) * dt_ms * pq.ms
ramp_up_end = 5.0 * pq.s
plateau_end = 9.0 * pq.s

print(f"Running calibrated Jaxley drive: {simulation_time.rescale(pq.s)}")
print(f"  MNs={len(card.thresholds())}, DD sources={dd['dd_neurons']}, dt={dt_ms} ms")
print(f"  DD rate={drive_hz.min():.1f}–{drive_hz.max():.1f} Hz")

##############################################################################
# Run Jaxley with the matched conductance pathway
# ------------------------------------------------

run = run_descending_drive(
    card.thresholds(),
    drive_hz,
    dd,
    timestep_ms=dt_ms,
    seed=int(card.seeds["drive"]),
    # The current NEURON card inherits -67 mV from alpha_mn_default.yaml.
    # Use the same initial state here. The 1 s pre-ramp lets the pool settle.
    initial_voltage_mV=-67.0,
)

n_steps, n_motor_neurons = run.spikes.shape
n_dd = run.dd_spikes.shape[1]
mn_spike_times = [
    (np.flatnonzero(run.spikes[:, i]) + 1) * dt_ms for i in range(n_motor_neurons)
]
dd_spike_times = [
    np.flatnonzero(run.dd_spikes[:, i]) * dt_ms for i in range(n_dd)
]

print(f"  dendritic transfer={run.transfer.min():.3f}–{run.transfer.max():.3f}")
print(f"  synaptic weights={run.weights_uS.min():.5f}–{run.weights_uS.max():.5f} µS")

##############################################################################
# Convert to Neo and generate the same force observable
# ------------------------------------------------------

spike_train_block = Block(name="Calibrated FDI descending drive — Jaxley")
mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        (times * pq.ms).rescale(pq.s),
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / timestep).rescale(pq.Hz),
        sampling_period=timestep.rescale(pq.s),
        name=str(i),
    )
    for i, times in enumerate(mn_spike_times)
]
spike_train_block.segments.append(mn_segment)

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        (times * pq.ms).rescale(pq.s),
        t_stop=simulation_time.rescale(pq.s),
        sampling_rate=(1 / timestep).rescale(pq.Hz),
        sampling_period=timestep.rescale(pq.s),
        name=f"DD_{i}",
    )
    for i, times in enumerate(dd_spike_times)
]

force_model = card.force_model()
force = np.asarray(force_model.generate_force(spike_train_block, verbose=False).magnitude)[:, 0]
force_pct_mvc = 100.0 * force / float(card.value("drive.mvc_model_units"))
fs_force_hz = float(card.value("force.recording_frequency__Hz"))

excitation_signal = AnalogSignal(
    excitation[:, None], units=pq.dimensionless, sampling_period=timestep.rescale(pq.s)
)
drive_signal = AnalogSignal(
    drive_hz[:, None], units=pq.Hz, sampling_period=timestep.rescale(pq.s)
)
joblib.dump(spike_train_block, save_path / "trapezoid_dd_spike_trains_jaxley.pkl")
joblib.dump(excitation_signal, save_path / "trapezoid_drive_pattern_jaxley.pkl")
joblib.dump(drive_signal, save_path / "trapezoid_drive_hz_jaxley.pkl")

diag_ids = sorted(set(np.linspace(0, n_motor_neurons - 1, 6, dtype=int)))
joblib.dump(
    {
        **{f"MN_{i}": run.voltage_mV[:, i].astype(np.float32) for i in diag_ids},
        "__dt_ms": np.float32(dt_ms),
        "__sim_ms": np.float32(simulation_time.rescale(pq.ms).magnitude),
    },
    save_path / "trapezoid_dd_voltage_traces_jaxley.pkl",
)

##############################################################################
# Summary metrics
# ---------------

all_counts = np.asarray([len(st) for st in mn_segment.spiketrains])
plateau_counts = np.asarray([
    len(st.time_slice(ramp_up_end, plateau_end)) for st in mn_segment.spiketrains
])
active = int(np.sum(all_counts > 0))
sustained = int(np.sum(plateau_counts >= 5))
plateau_total = int(plateau_counts.sum())
plateau_rate = plateau_total / max(sustained, 1) / 4.0

print("Motor neurons:")
print(f"  active: {active}/{n_motor_neurons}")
print(f"  total spikes: {all_counts.sum()}")
print(f"  sustained during hold: {sustained}/{n_motor_neurons}")
print(f"  hold spikes: {plateau_total}")
print(f"  mean sustained hold rate: {plateau_rate:.1f} Hz")
print(f"  peak force: {force_pct_mvc.max():.1f} % MVC")

##############################################################################
# Matched visualization
# ---------------------

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
t_force = np.arange(force_pct_mvc.size) / fs_force_hz
t_target = np.cumsum([0] + [duration for duration, _, _ in protocol])
target = [protocol[0][1]] + [stop for _, _, stop in protocol]
axes[0].plot(t_target, target, "k--", linewidth=2, label="Target")
axes[0].plot(t_force, force_pct_mvc, "b-", linewidth=2, label="Jaxley force")
axes[0].set_ylabel("Force (% MVC)")
axes[0].set_title("Trapezoidal contraction — calibrated FDI drive")
axes[0].legend()

colors = plt.get_cmap("Reds")(np.linspace(0.3, 0.9, n_motor_neurons))
for i, (st, color) in enumerate(zip(mn_segment.spiketrains, colors)):
    if len(st):
        axes[1].scatter(st.rescale(pq.s).magnitude, np.full(len(st), i), c=[color], s=1)
axes[1].set_ylabel("Motor neuron ID")
axes[1].set_title(f"Motor-neuron activity ({active}/{n_motor_neurons} active)")
axes[1].set_ylim(-1, n_motor_neurons)

bin_ms = 100
counts, centres = population_psth(mn_segment.spiketrains, bin_ms * pq.ms)
population_rate = counts / (bin_ms / 1000.0) / n_motor_neurons
axes[2].plot(centres, population_rate, "r-", linewidth=2)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Population rate (Hz)")
axes[2].set_title("Population firing rate")

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, simulation_time.rescale(pq.s).magnitude)
plt.tight_layout()
plt.savefig(save_path / "trapezoid_dd_jaxley_matched.png", dpi=180)
plt.show()

##############################################################################
# Individual smoothed discharge rates
# ------------------------------------

window_samples = round(0.4 / timestep.rescale(pq.s).magnitude)
window = np.hanning(window_samples)
window /= window.sum() * timestep.rescale(pq.s).magnitude
rates, active_ids = [], []
for i, st in enumerate(mn_segment.spiketrains):
    if len(st) <= 2:
        continue
    binary = np.zeros(n_steps)
    idx = np.rint(st.rescale(pq.ms).magnitude / dt_ms).astype(int)
    idx = idx[(idx >= 0) & (idx < n_steps)]
    binary[idx] = 1.0
    rates.append(np.convolve(binary, window, mode="same"))
    active_ids.append(i)

if rates:
    rates = np.asarray(rates)
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    image = axes2[0].imshow(
        rates, aspect="auto", cmap="hot", interpolation="bilinear", origin="lower",
        extent=[0, simulation_time.rescale(pq.s).magnitude, 0, len(active_ids)],
        vmin=0, vmax=np.percentile(rates, 95),
    )
    axes2[0].set_ylabel("Active motor units")
    axes2[0].set_title("Individual discharge rates — 400 ms Hanning smoothing")
    plt.colorbar(image, ax=axes2[0], label="Firing rate (Hz)")
    t = np.arange(n_steps) * dt_ms / 1000.0
    for rate in rates:
        axes2[1].plot(t, rate, linewidth=0.7, alpha=0.7)
    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Firing rate (Hz)")
    axes2[1].set_xlim(0, simulation_time.rescale(pq.s).magnitude)
    axes2[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "trapezoid_dd_discharge_rates_jaxley.png", dpi=180)
    plt.show()

print("[DONE] Matched profile-card Jaxley descending-drive simulation complete.")

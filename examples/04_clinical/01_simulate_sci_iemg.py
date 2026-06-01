"""
SCI-like Pathological iEMG: Modulation, Loss of Derecruitment, and Spasticity
=============================================================================

This example builds **three intramuscular EMG (iEMG) signals** that illustrate a
progression of motor-control pathology seen after **spinal cord injury (SCI)**.
All three come from the *same* muscle, electrode, and NEURON pipeline
(``DescendingDrive__Pool`` -> ``AlphaMN__Pool`` -> spike trains ->
``IntramuscularEMG``). They differ **only in how the descending drive is shaped**:

1. **Voluntary modulation** -- a sinusoidal cortical drive that returns to zero
   each cycle, so motor units recruit and **derecruit** every cycle.
2. **Loss of derecruitment** -- the same sinusoid riding on a tonic *floor*, so
   once recruited the units **never go silent** (a functional stand-in for
   persistent-inward-current self-sustained firing).
3. **Modulation then spasticity** -- voluntary modulation that switches to
   involuntary **~6 Hz clonic bursts**.

.. note::
    This example is *self-contained*: it generates its own recruitment
    thresholds and muscle model, so it does not depend on the ``01_basic``
    gallery outputs.
"""
# sphinx_gallery_thumbnail_number = -1

# %%

##############################################################################
# Import Libraries
# ----------------

import itertools
from pathlib import Path

import joblib
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import pyplot as plt
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator, set_random_seed, simulator
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.types import pps

plt.style.use("fivethirtyeight")

# %%

##############################################################################
# Setup
# -----
#
# Fix the seed, load NEURON mechanisms, and create a results directory next to
# this script (the gallery runs each subsection in its own directory, so we use
# a ``__file__``-relative path rather than a bare ``./results``).

set_random_seed(42)
load_nmodl_mechanisms()

try:
    _script_dir = Path(__file__).parent
except NameError:  # __file__ is undefined when run as a raw cell
    _script_dir = Path.cwd()
save_path = _script_dir / "results"
save_path.mkdir(exist_ok=True, parents=True)

# Simulation constants shared by all three conditions
N_MU = 60  # motor units (kept lean so the gallery build stays fast)
timestep = 0.1 * pq.ms
simulation_time = 9000 * pq.ms
time_points = int(simulation_time / timestep)
base_freq__Hz = 0.5  # voluntary modulation frequency
clonus_freq__Hz = 6.0  # spastic clonus burst frequency

# %%

##############################################################################
# Recruitment Thresholds and Muscle Model
# ---------------------------------------
#
# Generate recruitment thresholds (Combined model) and build the muscle model.
# Both are cached so re-runs during authoring are cheap.

recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=N_MU,
    recruitment_range__ratio=100,
    deluca__slope=5,
    konstantin__max_threshold__ratio=1.0,
    mode="combined",
)

muscle_cache = save_path / "sci_muscle_model.pkl"
if muscle_cache.exists():
    muscle = joblib.load(muscle_cache)
else:
    muscle = simulator.Muscle(
        recruitment_thresholds=recruitment_thresholds,
        radius_bone__mm=1.0 * pq.mm,
        fiber_density__fibers_per_mm2=400 * pq.mm**-2,
        fat_thickness__mm=10 * pq.mm,
        autorun=True,
    )
    joblib.dump(muscle, muscle_cache)

print(f"Muscle built: {N_MU} MUs, "
      f"{sum(muscle.resulting_number_of_innervated_fibers)} fibers total")

# %%

##############################################################################
# Intramuscular EMG Setup
# -----------------------
#
# Create a differential needle electrode and the iEMG simulator, then compute
# the motor unit action potentials (MUAPs) and cache them. Computing MUAPs is the
# most expensive step, so the simulator (with its MUAPs) is pickled and reloaded
# on subsequent runs; the three conditions below then only re-run the cheaper
# spike-train -> EMG convolution.

iemg_cache = save_path / "sci_iemg_simulator.pkl"
if iemg_cache.exists():
    iemg_sim = joblib.load(iemg_cache)
else:
    electrode = simulator.IntramuscularElectrodeArray(
        num_electrodes=4,
        inter_electrode_distance__mm=2.0 * pq.mm,
        differentiation_mode="consecutive",
        position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm),
    )
    iemg_sim = simulator.IntramuscularEMG(
        muscle_model=muscle,
        electrode_array=electrode,
        MUs_to_simulate=list(range(N_MU)),
    )
    print("Computing MUAPs (once)...")
    iemg_sim.simulate_muaps(n_jobs=2)
    joblib.dump(iemg_sim, iemg_cache)

# %%

##############################################################################
# Descending-Drive Builders
# --------------------------
#
# Each builder returns a 1-D ``neo.AnalogSignal`` (in ``pps``) of length
# ``time_points``. The *only* thing that differs between the three SCI signals is
# which builder produces the drive fed into the descending-drive pool.

_t__s = np.linspace(
    0.0, float(simulation_time.rescale(pq.s)), time_points, endpoint=False
)  # time vector in seconds


def _as_drive(signal__pps: np.ndarray) -> AnalogSignal:
    """Wrap a pps array as an AnalogSignal, adding small non-negative noise."""
    noise = np.clip(get_random_generator().normal(0, 1.0, size=time_points), 0, None)
    return AnalogSignal(
        signal=(signal__pps + noise) * pps,
        sampling_period=timestep.rescale(pq.s),
    )


def build_voluntary_drive(peak__pps: float = 55.0) -> AnalogSignal:
    """Sinusoid sweeping 0 -> peak -> 0 each cycle (units derecruit at troughs)."""
    drive = (peak__pps / 2.0) * (1.0 - np.cos(2.0 * np.pi * base_freq__Hz * _t__s))
    return _as_drive(drive)


def build_no_derecruit_drive(
    peak__pps: float = 55.0, floor__pps: float = 42.0, ramp__s: float = 1.0
) -> AnalogSignal:
    """Sinusoid riding on a tonic floor after an initial recruiting ramp.

    The trough never returns to zero, so recruited units keep firing (rate
    modulated, never silent).
    """
    osc = floor__pps + ((peak__pps - floor__pps) / 2.0) * (
        1.0 - np.cos(2.0 * np.pi * base_freq__Hz * (_t__s - ramp__s))
    )
    ramp = np.clip(_t__s / ramp__s, 0.0, 1.0) * floor__pps  # 0 -> floor over ramp__s
    drive = np.where(_t__s < ramp__s, ramp, osc)
    return _as_drive(drive)


def build_clonus_drive(
    peak__pps: float = 55.0, tone__pps: float = 8.0, burst__pps: float = 60.0
) -> AnalogSignal:
    """Voluntary sinusoid for the first half, ~6 Hz clonic bursts for the second."""
    half = float(simulation_time.rescale(pq.s)) / 2.0
    voluntary = (peak__pps / 2.0) * (1.0 - np.cos(2.0 * np.pi * base_freq__Hz * _t__s))
    clonus = tone__pps + burst__pps * np.clip(
        np.sin(2.0 * np.pi * clonus_freq__Hz * _t__s), 0.0, None
    )
    drive = np.where(_t__s < half, voluntary, clonus)
    return _as_drive(drive)


drive_builders = {
    "Voluntary modulation": build_voluntary_drive,
    "Loss of derecruitment": build_no_derecruit_drive,
    "Modulation -> clonus": build_clonus_drive,
}

# %%

##############################################################################
# Motor Neuron Pool, Descending Drive Pool, and Network
# -----------------------------------------------------
#
# Built once and reused for all three conditions.

h.secondorder = 2  # Crank-Nicolson
motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file="alpha_mn_default.yaml",
)
descending_drive_pool = DescendingDrive__Pool(
    n=100, poisson_batch_size=5, timestep__ms=timestep
)

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})
network.connect(source="DD", target="aMN", probability=0.5, weight__uS=0.15 * pq.uS)
network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
dd_netcons = network.get_netcons("cortical_input", "DD")

h.load_file("stdrun.hoc")
h.dt = timestep
h.tstop = simulation_time


def run_drive(drive_signal: AnalogSignal, label: str) -> Block:
    """Run one NEURON simulation for ``drive_signal`` and return MN spike trains.

    Re-initializes NEURON state, drives the DD pool step-by-step with
    ``drive_signal``, records motor-neuron spikes, and packages them into a Neo
    ``Block`` with one ``SpikeTrain`` per motor unit.
    """
    # Fresh motor-neuron spike recorders for this run
    mn_spike_recorders = []
    for cell in motor_neuron_pool:
        rec = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(rec)
        mn_spike_recorders.append((nc, rec))  # keep nc alive for the run

    # Initialize voltages for both pools, then reset NEURON time/state
    for section, voltage in itertools.chain.from_iterable(
        zip(*pool.get_initialization_data())
        for pool in [motor_neuron_pool, descending_drive_pool]
    ):
        section.v = voltage
    h.finitialize()

    step_counter = 0
    print(f"  running '{label}'...")
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                spike_time = h.t + 1
                if spike_time < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(spike_time)
        h.fadvance()
        step_counter += 1

    block = Block(name=label)
    seg = Segment(name="Motor Neurons")
    seg.spiketrains = [
        SpikeTrain(
            (rec.as_numpy() * pq.ms).rescale(pq.s),
            t_stop=simulation_time.rescale(pq.s),
            sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
            sampling_period=h.dt * pq.ms,
            name=f"MN_{i}",
        )
        for i, (_, rec) in enumerate(mn_spike_recorders)
    ]
    block.segments.append(seg)
    return block


# %%

##############################################################################
# Simulate the Three Conditions
# -----------------------------
#
# For each condition: build the drive, run the network, synthesize iEMG from the
# spike trains (reusing the precomputed MUAPs), and add realistic noise.

conditions = {}  # label -> dict(drive, spikes, iemg)
for label, builder in drive_builders.items():
    print(f"Condition: {label}")
    drive = builder()
    spikes__Block = run_drive(drive, label)
    iemg_sim.simulate_intramuscular_emg(spike_train__Block=spikes__Block)
    noisy__Block = iemg_sim.add_noise(snr__dB=20)
    conditions[label] = {
        "drive": drive,
        "spikes": spikes__Block,
        "iemg": noisy__Block.segments[0].analogsignals[0],
    }

    # Per-condition statistics
    sts = spikes__Block.segments[0].spiketrains
    active = sum(1 for st in sts if len(st) > 0)
    rates = [
        float((len(st) / simulation_time.rescale(pq.s)).magnitude)
        for st in sts
        if len(st) > 0
    ]
    rms = float(np.sqrt(np.mean(conditions[label]["iemg"].magnitude ** 2)))
    mean_rate = float(np.mean(rates)) if rates else 0.0
    print(f"  active MUs: {active}/{N_MU} | mean rate: {mean_rate:.1f} pps "
          f"| iEMG RMS: {rms:.3f} mV")

# %%

##############################################################################
# Figure 1 — Descending Drive
# ---------------------------
#
# The three command signals. Same hardware below; only this input changes.

labels = list(conditions.keys())
fig, axes = plt.subplots(len(labels), 1, figsize=(12, 8), sharex=True)
for ax, label in zip(axes, labels):
    drive = conditions[label]["drive"]
    ax.plot(drive.times.rescale(pq.s).magnitude, drive.magnitude, linewidth=1.2)
    ax.set_ylabel("Drive (pps)")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)
plt.tight_layout()
plt.show()

# %%

##############################################################################
# Figure 2 — Motor Neuron Rasters with Mean Discharge Rate
# --------------------------------------------------------
#
# Derecruitment at troughs (top) vs. persistent tonic firing (middle) vs. clonic
# bursting (bottom), at the spike level. The black line overlays the mean
# population discharge rate (averaged over units that fire at least once), which
# makes the rate modulation explicit -- note how it never returns to zero for the
# loss-of-derecruitment condition.


def population_discharge_rate(spiketrains, bin__s=0.1, smooth_bins=5):
    """Mean per-unit discharge rate (pps) over time across units that fire.

    Returns ``(bin_centers__s, rate__pps)``. Each active unit contributes its
    binned rate; the population mean is smoothed with a short moving average.
    """
    active = [st.rescale(pq.s).magnitude for st in spiketrains if len(st) > 0]
    t_stop = float(simulation_time.rescale(pq.s))
    edges = np.arange(0.0, t_stop + bin__s, bin__s)
    centers = edges[:-1] + bin__s / 2.0
    if not active:
        return centers, np.zeros_like(centers)
    per_unit = np.stack([np.histogram(sp, bins=edges)[0] / bin__s for sp in active])
    mean_rate = per_unit.mean(axis=0)
    if smooth_bins > 1:
        mean_rate = np.convolve(mean_rate, np.ones(smooth_bins) / smooth_bins, mode="same")
    return centers, mean_rate


fig, axes = plt.subplots(len(labels), 1, figsize=(12, 8), sharex=True)
for ax, label in zip(axes, labels):
    sts = conditions[label]["spikes"].segments[0].spiketrains
    for i, st in enumerate(sts):
        if len(st) > 0:
            ax.scatter(st.rescale(pq.s).magnitude, [i] * len(st), s=0.6, alpha=0.7)
    ax.set_ylabel("MU #")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)

    rate_t__s, rate__pps = population_discharge_rate(sts)
    ax_rate = ax.twinx()
    ax_rate.plot(rate_t__s, rate__pps, color="black", linewidth=1.5, alpha=0.85)
    ax_rate.set_ylabel("Discharge rate (pps)")
    ax_rate.set_ylim(bottom=0)
axes[-1].set_xlabel("Time (s)")
sns.despine(fig=fig, top=True)
plt.tight_layout()
plt.show()

# %%

##############################################################################
# Figure 3 — Intramuscular EMG
# ----------------------------
#
# The headline result: three pathological iEMG recordings. (1) bursts with silent
# gaps, (2) modulated but never silent, (3) voluntary modulation turning into
# ~6 Hz clonus. First electrode channel shown.

fig, axes = plt.subplots(len(labels), 1, figsize=(12, 8), sharex=True)
for ax, label in zip(axes, labels):
    iemg = conditions[label]["iemg"][:, 0]
    ax.plot(iemg.times.rescale(pq.s).magnitude, iemg.magnitude, linewidth=0.6)
    ax.set_ylabel("iEMG (mV)")
    ax.set_title(label)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Time (s)")
sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)
plt.tight_layout()
plt.show()

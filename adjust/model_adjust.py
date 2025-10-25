"""
Motor Neuron Model Adjustment and Validation
=============================================

This script performs model parameter adjustment and validation by comparing simulated motor neuron
behavior with experimental data. It uses descending drive populations to generate realistic spike
trains and analyzes the resulting firing rate and ISI statistics against empirical measurements.

.. note::
    This script uses the following components for model adjustment:

    - **DescendingDrive__Pool**: Poisson process neurons modeling cortical input
    - **AlphaMN__Pool**: Biophysically detailed motor neurons (Powers2017 model)
    - **Network**: Synaptic connections between descending drive and motor neuron populations
    - **ForceModel**: Conversion of spike trains to force output
    - **ISI Analysis**: Comparison with experimental data from ISI_statistics.csv

.. important::
    **Purpose**: This script validates that the motor neuron model parameters (recruitment thresholds,
    synaptic weights, neural dynamics) produce physiologically realistic firing patterns that match
    experimental observations from multiple muscles (VL, VM, TA, FDI).
"""

##############################################################################
# Import Libraries
# ----------------
#
# .. important::
#    In **MyoGen** all **random number generation** is handled by the ``RANDOM_GENERATOR`` object.
#
#    This object is a wrapper around the ``numpy.random`` module and is used to generate random numbers.
#
#    It is intended to be used with the following API:
#
#    .. code-block:: python
#
#       from myogen import simulator, RANDOM_GENERATOR
#
#    To change the default seed, use ``set_random_seed``:
#
#    .. code-block:: python
#
#       from myogen import set_random_seed
#       set_random_seed(42)

# %%

import itertools
from pathlib import Path

import elephant
import joblib
import numpy as np
import pandas as pd
import quantities as pq
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h
from tqdm import tqdm

from myogen import simulator, RANDOM_GENERATOR
from myogen.simulator.core.force.force_model import ForceModel
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.nmodl import load_nmodl_mechanisms

##############################################################################
# Load NEURON Mechanisms
# ----------------------
#
# Compile and load the NMODL mechanisms required for biophysically detailed neuron simulations.
# This must be done before creating any neuron populations.
#
load_nmodl_mechanisms()

save_path = Path("./results")
save_path.mkdir(exist_ok=True)


##############################################################################
# Generate Recruitment Thresholds
# --------------------------------
#
# Generate recruitment thresholds using the combined model from example 0.
# This creates physiologically realistic recruitment patterns.

n_motor_units = 100
recruitment_range = 100  # Recruitment range (max_threshold / min_threshold)
combined_max_threshold = 1.0  # Maximum threshold for combined model

# Generate thresholds using combined model with slope=5 (same as example 0)
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=n_motor_units,
    recruitment_range__ratio=recruitment_range,
    deluca__slope=5,
    konstantin__max_threshold__ratio=combined_max_threshold,
    mode="combined",
)

# Save the generated thresholds
joblib.dump(recruitment_thresholds, save_path / "thresholds.pkl")

##############################################################################
# Create Neuron Populations
# --------------------------
#
# Create the motor neuron pool and descending drive population with the following
# simulation parameters:
#
# - ``timestep``: 0.1 ms (high temporal resolution for accurate spike timing)
# - ``n_dd_neurons``: 400 descending drive neurons (Poisson processes)
# - ``motor_neuron_pool``: AlphaMN pool using VLVM configuration
#

timestep = 0.1  # ms

motor_neuron_pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    config_file="alpha_mn_VLVM.yaml",
)

descending_drive_pool = DescendingDrive__Pool(
    n=400, poisson_batch_size=16, timestep__ms=timestep
)

##############################################################################
# Generate Drive Pattern
# -----------------------
#
# Create a time-varying input pattern to drive the descending drive population's Poisson processes.
# This pattern represents cortical motor commands with the following characteristics:
#
# - **Baseline activity**: Continuous low-level drive at 20 Hz
# - **Sinusoidal modulation**: 0.25 Hz oscillation (disabled in current configuration)
# - **Gaussian noise**: Small random fluctuations (std=1.0 Hz) for realism
#
# The pattern is generated at the simulation timestep resolution and provided as an AnalogSignal.
#

simulation_time = 3000  # ms
time_points = int(simulation_time / timestep)

# Drive pattern parameters
dd_frequency__Hz = 0.25  # Sinusoidal modulation frequency (currently not applied)
dd_amplitude__Hz = 60.0  # Maximum drive amplitude
dd_baseline__Hz = 20.0   # Baseline drive level

sinusoidal_drive = AnalogSignal(
    signal=(
        np.maximum(
            dd_baseline__Hz
            + (
                0
                * (dd_amplitude__Hz - dd_baseline__Hz)
                * np.sin(
                    2
                    * np.pi
                    * dd_frequency__Hz
                    * np.linspace(0, simulation_time, time_points)
                    / 1000.0
                )
            ),
            dd_baseline__Hz,
        )
        + np.clip(RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None)
    ),
    units=pq.Hz,
    sampling_period=(timestep * pq.ms).rescale(pq.s),
)

joblib.dump(sinusoidal_drive, save_path / "sinusoidal_drive_pattern.pkl")

##############################################################################
# Create Network and Connections
# -------------------------------
#
# In MyoGen, populations can be connected using the **Network** class from the
# `myogen.simulator.neuron` module.
#
# The **Network** class provides a high-level interface for creating and managing
# connections between neuron populations.

# Use the **Network** class to create synaptic connections between the descending drive
# population and the motor neuron pool. This creates realistic synaptic transmission
# with appropriate delays and weights.
#

network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})

# Connect DD neurons to motor neurons with realistic synaptic parameters
network.connect(source="DD", target="aMN", probability=0.5, weight__μS=0.1)

# Set up external input to DD population
network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)

# Get NetCons for manual DD stimulation
dd_netcons = network.get_netcons("cortical_input", "DD")

##############################################################################
# Setup Spike Recording
# ----------------------
#
# Configure spike recording for both populations:
# - Motor neurons: Use NetCon objects with voltage threshold detection (50 mV)
# - Descending drive neurons: Track spike times manually during simulation loop
#

# Manual spike tracking for DD neurons (they use Poisson processes)
dd_spike_times = [[] for _ in range(len(descending_drive_pool))]

# Record spikes from motor neurons
mn_spike_recorders = []
for cell in motor_neuron_pool:
    spike_recorder = h.Vector()
    nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
    nc.threshold = 50  # Standard threshold for motor neurons
    nc.record(spike_recorder)
    mn_spike_recorders.append(spike_recorder)

##############################################################################
# Run Simulation
# --------------
#
# Execute the NEURON simulation with real-time injection of the drive pattern into descending
# drive neurons. At each timestep, the DD Poisson processes integrate the current drive level
# and may generate spikes, which then propagate through the synaptic network to motor neurons.

h.load_file("stdrun.hoc")  # Load standard run library for NEURON
h.dt = timestep
h.tstop = simulation_time

# Initialize voltages for all pools
for section, voltage in itertools.chain.from_iterable(
    zip(*pool.get_initialization_data())
    for pool in [motor_neuron_pool, descending_drive_pool]
):
    section.v = voltage


h.finitialize()

# Calculate total simulation steps for progress bar
total_steps = int(simulation_time / timestep)

step_counter = 0
with tqdm(
    total=simulation_time,
    desc="Running simulation",
    unit="ms",
    bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} ms [{elapsed}<{remaining}, {rate_fmt}]",
) as pbar:
    while h.t < h.tstop:
        current_drive = sinusoidal_drive[min(step_counter, len(sinusoidal_drive) - 1)]

        # Drive DD neurons with current input level
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                # Record spike time for DD neuron
                dd_spike_times[dd_cell.pool__ID].append(h.t)
                # Generate spike in DD neuron
                spike_time = h.t + np.clip(RANDOM_GENERATOR.normal(0, 10), 0, None)
                if spike_time < h.tstop:  # Avoid scheduling beyond simulation end
                    dd_netcons[dd_cell.pool__ID].event(spike_time)

        # Progress simulation
        h.fadvance()
        step_counter += 1
        pbar.update(timestep)


##############################################################################
# Convert Spike Data to Neo Format
# ---------------------------------
#
# Convert recorded spike times from both populations into Neo SpikeTrain objects.
# While both DD and MN spike trains are converted, only the motor neuron segment
# is saved to disk for subsequent force generation and analysis.
#

spike_train_block = Block(name="Motor Neuron Spike Trains")

dd_segment = Segment(name="Descending Drive")
dd_segment.spiketrains = [
    SpikeTrain(
        spike_times * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"DD_{i}",
    )
    for i, spike_times in enumerate(dd_spike_times)
]

mn_segment = Segment(name="Motor Neurons")
mn_segment.spiketrains = [
    SpikeTrain(
        recorder.as_numpy() * pq.ms,
        t_stop=simulation_time * pq.ms,
        sampling_rate=(1 / h.dt * (1 / pq.ms)),
        sampling_period=h.dt * pq.ms,
        name=f"MN_{i}",
    )
    for i, recorder in enumerate(mn_spike_recorders)
]

# Add only the motor neuron segment to the block (DD spikes not needed for force generation)
spike_train_block.segments.append(mn_segment)

joblib.dump(spike_train_block, save_path / "sinusoidal_dd_spike_trains.pkl")

##############################################################################
# Generate Force Output
# ---------------------
#
# Convert spike trains to force using the ForceModel.
# This simulates individual motor unit twitches and their temporal summation.

# Force model parameters
recording_frequency__Hz = 2048  # 2048 Hz sampling rate
longest_duration_rise_time__ms = 90.0  # Maximum twitch rise time
contraction_time_range = 3  # Contraction time range factor

# Create force model
force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds,
    recording_frequency__Hz=recording_frequency__Hz,
    longest_duration_rise_time__ms=longest_duration_rise_time__ms,
    contraction_time_range__unitless=contraction_time_range,
)

print("\nForce model statistics:")
print(f"  Number of motor units: {force_model._number_of_neurons}")
print(f"  Recruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    f"  Peak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - {force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    f"  Contraction time range: {force_model.contraction_times__samples[0]:.1f} - {force_model.contraction_times__samples[-1]:.1f} samples"
)

# Generate force from spike trains
force_output = force_model.generate_force(spike_train__Block=spike_train_block)

# Add realistic noise to the force signal
noise_level = 0.015  # 1.5% of mean force
noisy_force = force_output.magnitude[:, 0] + RANDOM_GENERATOR.normal(
    0,
    noise_level * np.mean(force_output.magnitude[:, 0]),
    size=len(force_output.magnitude[:, 0]),
)

# Save force output
joblib.dump(force_output, save_path / "force_output.pkl")

print("\nForce output statistics:")
print(f"  Peak force: {np.max(force_output.magnitude[:, 0]):.3f} a.u.")
print(f"  Mean force: {np.mean(force_output.magnitude[:, 0]):.3f} a.u.")
print(f"  Force duration: {force_output.times[-1]:.2f}")

##############################################################################
# Calculate Firing Rate Statistics
# ---------------------------------
#

print("\nFiring rate analysis:")

# Calculate DD firing rates
dd_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in dd_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

# Calculate MN firing rates
mn_firing_rates = np.array(
    [
        elephant.statistics.mean_firing_rate(st__s.time_slice(st__s.min(), st__s.max()))
        for st__ms in mn_segment.spiketrains
        if len(st__s := st__ms.rescale(pq.s)) > 0
    ]
)

print("Descending Drive neurons:")
print(f"\tActive neurons: {len(dd_firing_rates)}/{descending_drive_pool.n}")
if len(dd_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(dd_firing_rates):.1f} ± {np.std(dd_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(dd_firing_rates):.1f} - {np.max(dd_firing_rates):.1f} Hz"
    )

print("Motor neurons (VLVM model):")
print(f"\tActive neurons: {len(mn_firing_rates)}/{motor_neuron_pool.n}")
if len(mn_firing_rates) > 0:
    print(
        f"\tMean firing rate: {np.mean(mn_firing_rates):.1f} ± {np.std(mn_firing_rates):.1f} Hz"
    )
    print(
        f"\tRate range: {np.min(mn_firing_rates):.1f} - {np.max(mn_firing_rates):.1f} Hz"
    )

##############################################################################
# Network Activity Visualization
# -------------------------------
#
# Streamlined visualization for model adjustment showing:
# 1. Drive input pattern (baseline + noise)
# 2. Motor neuron raster plot (recruitment order)
# 3. Population firing rates over time (binned)

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# 1. Drive pattern
time_s = sinusoidal_drive.times.rescale(pq.s).magnitude
axes[0].plot(time_s, sinusoidal_drive, "b-", linewidth=2, label="DD Input")
axes[0].axhline(dd_baseline__Hz, color="r", linestyle="--", alpha=0.7, label="Baseline")
axes[0].set_ylabel("Drive (Hz)")
axes[0].set_title("Descending Drive Pattern (Baseline + Noise)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Motor neuron raster plot
mn_colors = plt.cm.get_cmap("Reds")(np.linspace(0.3, 0.9, len(mn_segment.spiketrains)))
active_mn_count = sum(1 for st in mn_segment.spiketrains if len(st) > 0)
for i, (spiketrain, color) in enumerate(zip(mn_segment.spiketrains, mn_colors)):
    if len(spiketrain) > 0:
        spike_times = spiketrain.rescale(pq.s).magnitude
        axes[1].scatter(spike_times, [i] * len(spike_times), c=[color], s=1.0, alpha=0.8)

axes[1].set_ylabel("Motor Neuron ID\n(Recruitment Order)")
axes[1].set_title(f"Motor Neuron Activity (n={active_mn_count}/{motor_neuron_pool.n} active)")
axes[1].set_ylim(-1, motor_neuron_pool.n)
axes[1].grid(True, alpha=0.3)

# 3. Population firing rates over time
bin_size_ms = 100
dd_psth = elephant.statistics.time_histogram(dd_segment.spiketrains, bin_size_ms * pq.ms)
mn_psth = elephant.statistics.time_histogram(mn_segment.spiketrains, bin_size_ms * pq.ms)

dd_rates_binned = (dd_psth / (bin_size_ms * pq.ms) / descending_drive_pool.n).rescale(pq.Hz).magnitude
mn_rates_binned = (mn_psth / (bin_size_ms * pq.ms) / motor_neuron_pool.n).rescale(pq.Hz).magnitude
bin_centers_s = dd_psth.times.rescale(pq.s).magnitude

axes[2].plot(bin_centers_s, dd_rates_binned, "b-", linewidth=2, label="DD Population", alpha=0.8)
axes[2].plot(bin_centers_s, mn_rates_binned, "r-", linewidth=2, label="MN Population", alpha=0.8)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Population Rate (Hz)")
axes[2].set_title("Population Firing Rates Over Time")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Format all axes
for ax in axes:
    ax.set_xlim(0, simulation_time / 1000.0)

plt.tight_layout()
plt.show()

##############################################################################
# Helper Function: Calculate ISI Statistics
# ------------------------------------------
#


def calculate_isi_statistics(spiketrains):
    """
    Calculate inter-spike interval (ISI) statistics from spike trains.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of spike train objects to analyze.

    Returns
    -------
    tuple
        Arrays of (firing_rates, cv_values, neuron_indices) for neurons with valid ISIs.
    """
    simulated_fr = []
    simulated_cv = []
    simulated_neuron_idx = []

    for idx, spiketrain in enumerate(spiketrains):
        if len(spiketrain) > 1:  # Need at least 2 spikes to calculate ISI
            spike_times_s = spiketrain.rescale(pq.s).magnitude
            isis = np.diff(spike_times_s)

            if len(isis) > 0:
                mean_isi = np.mean(isis)
                fr = 1.0 / mean_isi if mean_isi > 0 else 0
                cv = np.std(isis) / mean_isi if mean_isi > 0 else 0

                # Only include neurons with meaningful firing rates
                if fr >= 0.01:
                    simulated_fr.append(fr)
                    simulated_cv.append(cv)
                    simulated_neuron_idx.append(idx)

    return np.array(simulated_fr), np.array(simulated_cv), np.array(simulated_neuron_idx)


##############################################################################
# Plot ISI Statistics with Recruitment Thresholds
# ------------------------------------------------
#
# This function creates a comprehensive visualization combining:
# 1. Generated recruitment thresholds from the simulation
# 2. Experimental ISI statistics from ISI_statistics.csv
#
# The plot shows how the model's recruitment thresholds compare with
# real experimental data from different muscles and force levels.


def plot_isi_statistics_with_thresholds():
    """
    Plot ISI statistics from CSV file alongside generated recruitment thresholds.

    This function loads experimental ISI data and plots it together with the
    recruitment thresholds generated using the combined model.
    """
    # Load ISI statistics from CSV
    csv_path = Path(__file__).parent / "ISI_statistics.csv"
    isi_data = pd.read_csv(csv_path)

    # Calculate ISI statistics from simulated spike trains
    simulated_fr, simulated_cv, simulated_neuron_idx = calculate_isi_statistics(
        mn_segment.spiketrains
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Recruitment Thresholds
    ax1 = axes[0, 0]
    ax1.plot(
        range(1, len(recruitment_thresholds) + 1),
        recruitment_thresholds,
        "o-",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Generated Thresholds",
    )
    ax1.set_xlabel("Motor Unit Index", fontsize=12)
    ax1.set_ylabel("Recruitment Threshold", fontsize=12)
    ax1.set_title(
        "Generated Recruitment Thresholds (Combined Model)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Subplot 2: Mean Firing Rate by Muscle and Force Level
    ax2 = axes[0, 1]
    muscles = isi_data["Muscle"].unique()
    colors = {"VM": "#E63946", "VL": "#F77F00", "TA": "#06A77D", "FDI": "#4361EE"}

    for muscle in muscles:
        muscle_data = isi_data[isi_data["Muscle"] == muscle]
        force_levels = muscle_data["Force Level"].unique()

        for force in sorted(force_levels):
            force_data = muscle_data[muscle_data["Force Level"] == force]
            mean_fr = force_data["FR mean"].mean()
            std_fr = force_data["FR mean"].std()

            ax2.errorbar(
                force,
                mean_fr,
                yerr=std_fr,
                marker="o",
                markersize=8,
                color=colors.get(muscle, "#000000"),
                label=f"{muscle} - {force}%"
                if force == sorted(force_levels)[0]
                else "",
                alpha=0.7,
            )

    ax2.set_xlabel("Force Level (%)", fontsize=12)
    ax2.set_ylabel("Mean Firing Rate (Hz)", fontsize=12)
    ax2.set_title(
        "Experimental Firing Rates by Muscle and Force", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    # Subplot 3: ISI Coefficient of Variation (CV) Distribution
    ax3 = axes[1, 0]

    for muscle in muscles:
        muscle_data = isi_data[isi_data["Muscle"] == muscle]
        ax3.hist(
            muscle_data["ISI CV"],
            bins=20,
            alpha=0.5,
            color=colors.get(muscle, "#000000"),
            label=muscle,
            edgecolor="black",
        )

    ax3.set_xlabel("ISI Coefficient of Variation", fontsize=12)
    ax3.set_ylabel("Frequency", fontsize=12)
    ax3.set_title(
        "ISI Variability Distribution by Muscle", fontsize=14, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=10)

    # Subplot 4: CV vs Firing Rate (Simulated + Experimental) - Styled like fr_cv function
    ax4 = axes[1, 1]

    # Plot simulated data with colormap based on neuron index (recruitment order)
    if len(simulated_fr) > 0:
        scatter_sim = ax4.scatter(
            simulated_cv,
            simulated_fr,
            c=simulated_neuron_idx,
            cmap="Blues",
            s=80,
            alpha=0.7,
            vmin=0,
            vmax=len(mn_segment.spiketrains),
            edgecolors="black",
            linewidth=0.5,
            label="Simulated (VLVM)",
        )

        # Add colorbar for simulated data
        cax = inset_axes(
            ax4,
            width="3%",
            height="30%",
            loc="upper right",
            bbox_to_anchor=(0.15, 0, 1, 1),
            bbox_transform=ax4.transAxes,
        )
        cbar = plt.colorbar(scatter_sim, cax=cax, label="MN Index")
        cbar.ax.tick_params(labelsize=8)

    # Plot experimental data from CSV
    for muscle in muscles:
        muscle_data = isi_data[isi_data["Muscle"] == muscle]
        ax4.scatter(
            muscle_data["ISI CV"],
            muscle_data["FR mean"],
            s=60,
            alpha=0.5,
            color=colors.get(muscle, "#000000"),
            label=f"{muscle} (Exp)",
            edgecolors="black",
            linewidth=0.3,
            marker="x",
        )

    ax4.set_xlabel("CV (Coefficient of Variation)", fontsize=12)
    ax4.set_ylabel("Firing Rate (Hz)", fontsize=12)
    ax4.set_title(
        "CV vs Firing Rate (Simulated + Experimental)", fontsize=14, fontweight="bold"
    )
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend(fontsize=8, loc="upper left")

    # Add overall title
    fig.suptitle(
        "Recruitment Thresholds and ISI Statistics Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save the figure
    output_path = save_path / "isi_statistics_with_thresholds.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.show()


# Call the plotting function
plot_isi_statistics_with_thresholds()

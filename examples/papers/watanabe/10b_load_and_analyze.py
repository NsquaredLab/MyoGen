"""
Load and Analyze Watanabe Simulation Results
=============================================

This script demonstrates loading the chunked simulation data as a NEO Block,
which is compatible with all existing analysis code that expects SimulationRunner output.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from myogen.utils.continuous_saver import convert_chunks_to_neo

plt.style.use("fivethirtyeight")

##############################################################################
# Load Simulation Parameters
# --------------------------
#
# Load the parameters from the simulation to ensure analysis matches the actual run

results_path = Path("./results")
params_file = results_path / "watanabe__simulation_params.pkl"

if not params_file.exists():
    raise FileNotFoundError(
        f"Simulation parameters file not found: {params_file}\n"
        "Please run 10a_paper_watanabe.py first to generate the simulation data."
    )

sim_params = joblib.load(params_file)
segment_duration__s = sim_params["segment_duration__s"]
tstop__ms = sim_params["tstop__ms"]
dt__ms = sim_params["dt__ms"]

print("=" * 70)
print("SIMULATION PARAMETERS")
print("=" * 70)
print(f"Segment duration: {segment_duration__s} s")
print(f"Total duration: {tstop__ms / 1000} s ({tstop__ms} ms)")
print(f"Timestep: {dt__ms} ms")
print(f"Motor neurons: {sim_params['naMN']}")

# Calculate phase boundaries
phase1_end = segment_duration__s
phase2_end = 2 * segment_duration__s
phase3_end = 3 * segment_duration__s

##############################################################################
# Load Data as NEO Block
# ----------------------
#
# This is the RECOMMENDED approach - converts chunks to a NEO Block that's
# identical in structure to what SimulationRunner.run() returns

chunks_path = results_path / "watanabe_chunks"

print("=" * 70)
print("LOADING SIMULATION DATA AS NEO BLOCK")
print("=" * 70)
print(f"Chunks directory: {chunks_path}\n")

# Convert chunks to NEO Block format (compatible with SimulationRunner output)
# All data (spikes + membrane potentials) is self-contained in the chunks
results = convert_chunks_to_neo(chunks_path)

print("\nNEO Block loaded successfully!")
print(f"\tType: {type(results)}")
print(f"\tSegments: {len(results.segments)}")

for seg in results.segments:
    print(
        f"\t{seg.name}: {len(seg.spiketrains)} spike trains, "
        f"\t{len(seg.analogsignals)} analog signals"
    )

##############################################################################
# Analyze Membrane Potentials
# ---------------------------
#
# Access membrane potentials from NEO AnalogSignals

print("\n" + "=" * 70)
print("ANALYZING MEMBRANE POTENTIALS")
print("=" * 70)

# Find the aMN segment
aMN_segment = None
for seg in results.segments:
    if seg.name == "aMN":
        aMN_segment = seg
        break

if aMN_segment and len(aMN_segment.analogsignals) > 0:
    # Extract analog signals
    analog_signals = aMN_segment.analogsignals
    n_neurons = len(analog_signals)

    # Get time vector from first signal
    first_signal = analog_signals[0]
    times_ms = first_signal.times.rescale("ms").magnitude
    sampling_period = float(first_signal.sampling_period.rescale("ms"))

    print("\nMembrane potential recording:")
    print("\tPopulation: aMN (alpha motor neurons)")
    print(f"\tRecorded neurons: {n_neurons}")
    print(f"\tTime range: {times_ms[0]:.1f} - {times_ms[-1]:.1f} ms")
    print(f"\tDuration: {(times_ms[-1] - times_ms[0]) / 1000:.1f} seconds")
    print(f"\tSampling period: {sampling_period} ms")
    print(f"\tSamples per neuron: {len(times_ms)}")

    # Plot membrane potentials for early, middle, and late recruited neurons
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Select neurons to plot
    indices_to_plot = [0, n_neurons // 2, n_neurons - 1]

    for ax, idx in zip(axes, indices_to_plot):
        signal = analog_signals[idx]
        neuron_id = signal.name  # Actual neuron ID
        voltage = signal.rescale("mV").magnitude.flatten()

        ax.plot(times_ms / 1000.0, voltage, linewidth=0.5)
        ax.set_ylabel(f"aMN[{neuron_id}]\nVoltage (mV)")
        ax.set_title(f"Motor Neuron {neuron_id} Membrane Potential")
        ax.grid(True, alpha=0.3)

        # Highlight the three experimental phases
        ax.axvspan(0, phase1_end, alpha=0.1, color="gray", label="Phase 1: Constant")
        ax.axvspan(phase1_end, phase2_end, alpha=0.1, color="blue", label="Phase 2: Sinusoid DC=65")
        ax.axvspan(phase2_end, phase3_end, alpha=0.1, color="red", label="Phase 3: Sinusoid DC=58")

        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_membrane_potentials.png", dpi=150)
    print(f"\n✓ Saved plot: {chunks_path.parent / 'watanabe_membrane_potentials.png'}")
    plt.show()

##############################################################################
# Analyze Spike Data
# ------------------
#
# Access spike trains from NEO SpikeTrain objects

print("\n" + "=" * 70)
print("ANALYZING SPIKE DATA")
print("=" * 70)

# Print summary for all populations
for seg in results.segments:
    if len(seg.spiketrains) > 0:
        n_units = len(seg.spiketrains)
        n_spikes = sum(len(st) for st in seg.spiketrains)
        duration_s = float(seg.spiketrains[0].t_stop.rescale("s"))

        print(f"\n\t{seg.name}:")
        print(f"\t\tTotal units: {n_units}")
        print(f"\t\tTotal spikes: {n_spikes}")
        if duration_s > 0 and n_units > 0:
            print(f"\t\tMean firing rate: {n_spikes / n_units / duration_s:.2f} Hz")

# Plot raster plot for aMN population
if aMN_segment and len(aMN_segment.spiketrains) > 0:
    print("\nGenerating spike raster plot...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract spike times and IDs from NEO spike trains
    for st in aMN_segment.spiketrains:
        neuron_id = int(st.name)
        spike_times_s = st.times.rescale("s").magnitude
        if len(spike_times_s) > 0:
            ax.scatter(
                spike_times_s,
                [neuron_id] * len(spike_times_s),
                s=0.5,
                c="black",
                alpha=0.5,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motor Neuron ID")
    ax.set_title("Alpha Motor Neuron Spike Raster")
    ax.grid(True, alpha=0.3)

    # Highlight the three experimental phases
    ax.axvspan(0, phase1_end, alpha=0.1, color="gray", label="Phase 1: Constant")
    ax.axvspan(phase1_end, phase2_end, alpha=0.1, color="blue", label="Phase 2: Sinusoid DC=65")
    ax.axvspan(phase2_end, phase3_end, alpha=0.1, color="red", label="Phase 3: Sinusoid DC=58")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_spike_raster.png", dpi=150)
    print(f"✓ Saved plot: {chunks_path.parent / 'watanabe_spike_raster.png'}")
    plt.show()

    # Plot population firing rate over time
    print("\nGenerating population firing rate plot...")

    fig, ax = plt.subplots(figsize=(14, 4))

    # Collect all spike times
    all_spike_times = []
    for st in aMN_segment.spiketrains:
        all_spike_times.extend(st.times.rescale("ms").magnitude)
    all_spike_times = np.array(all_spike_times)

    # Calculate population firing rate in 100 ms bins
    bin_size_ms = 100
    duration_ms = float(aMN_segment.spiketrains[0].t_stop.rescale("ms"))
    bins = np.arange(0, duration_ms + bin_size_ms, bin_size_ms)
    counts, _ = np.histogram(all_spike_times, bins=bins)

    # Convert to firing rate (spikes/sec/neuron)
    n_neurons = len(aMN_segment.spiketrains)
    firing_rate = counts / (bin_size_ms / 1000.0) / n_neurons
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.plot(bin_centers / 1000.0, firing_rate, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population Firing Rate (Hz)")
    ax.set_title(f"Alpha Motor Neuron Population Firing Rate ({bin_size_ms} ms bins)")
    ax.grid(True, alpha=0.3)

    # Highlight the three experimental phases
    ax.axvspan(0, phase1_end, alpha=0.1, color="gray", label="Phase 1")
    ax.axvspan(phase1_end, phase2_end, alpha=0.1, color="blue", label="Phase 2")
    ax.axvspan(phase2_end, phase3_end, alpha=0.1, color="red", label="Phase 3")
    ax.legend()

    plt.tight_layout()
    plt.savefig(chunks_path.parent / "watanabe_firing_rate.png", dpi=150)
    print(f"✓ Saved plot: {chunks_path.parent / 'watanabe_firing_rate.png'}")
    plt.show()

##############################################################################
# Save NEO Block for Future Use (OPTIONAL)
# ----------------------------------------

# NOTE: Saving the full NEO Block is slow (~23 GB of data) and unnecessary
# since we can quickly regenerate it from chunks using convert_chunks_to_neo()
#
# If you really want to save it, uncomment below and use compress=0 for speed:
#
print("\nSaving NEO Block for future use...")


neo_output_path = chunks_path.parent / "watanabe_results_neo.pkl"
joblib.dump(results, neo_output_path, compress=0)  # No compression = faster
print(f"✓ NEO Block saved to: {neo_output_path}")

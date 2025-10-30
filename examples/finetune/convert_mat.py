# %%# Load the data

from scipy.io import loadmat
import numpy as np
import quantities as pq
from neo.core import SpikeTrain
import matplotlib.pyplot as plt
from helper import calculate_firing_rate_statistics
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
save_folder = "experimental_results"
muscle_type = "VL"
subject_id = "P2"
extra = "ramp1"
mvc_percent = 30

# Calculate per-neuron statistics
plateau_start_ms = 4.5e4
plateau_end_ms = 6.0e4

# ============================================================================
# Load the data
# ============================================================================
data = loadmat(
    r"/home/oj98yqyk/code/simulators/MyoGen/data/02_Decomposed_iEMG/Decomposed_iEMG/VL_30.mat"
)

# Extract spike times in seconds and flatten to ensure 1D
mu_pulses_raw = [
    np.array(x[0]).flatten() / data["fsamp"][0, 0] for x in data["MUPulses"][0]
]

print(f"Loaded {len(mu_pulses_raw)} motor units from experimental data")

# Determine total recording duration
# Find the maximum spike time across all motor units
max_spike_time = max(
    [pulses.max() if len(pulses) > 0 else 0.0 for pulses in mu_pulses_raw]
)
# Convert to Neo SpikeTrain objects
spiketrains = [
    SpikeTrain(
        times=pulses * pq.s, t_start=0 * pq.s, t_stop=max_spike_time * pq.s
    ).rescale(pq.ms)
    for pulses in mu_pulses_raw
]

# Print start and end time for each spiketrain
print("\nSpiketrain temporal information:")
for i, st in enumerate(spiketrains):
    if len(st) > 0:
        first_spike = st[0]
        last_spike = st[-1]
        print(
            f"MU {i:2d}: n_spikes={len(st):3d}, first_spike={first_spike:8.2f}, last_spike={last_spike:8.2f}, t_start={st.t_start:8.2f}, t_stop={st.t_stop:8.2f}"
        )
    else:
        print(f"MU {i:2d}: n_spikes=  0, (no spikes)")
print()

per_neuron_stats = calculate_firing_rate_statistics(
    spiketrains,
    return_per_neuron=True,
    min_spikes_for_cv=3,
    plateau_start_ms=plateau_start_ms,
    plateau_end_ms=plateau_end_ms,
)

# Calculate ISI statistics per neuron
from elephant.statistics import isi

isi_means = []
isi_sds = []
isi_cvs = []

for st in spiketrains:
    # Filter to plateau phase
    plateau_st = st.time_slice(plateau_start_ms * pq.ms, plateau_end_ms * pq.ms)

    if len(plateau_st) > 1:
        isis = isi(plateau_st.rescale(pq.ms))
        if len(isis) > 0:
            isis_array = isis.magnitude
            isi_means.append(np.mean(isis_array))
            isi_sds.append(np.std(isis_array, ddof=1))
            isi_cvs.append(np.std(isis_array, ddof=1) / np.mean(isis_array))
        else:
            isi_means.append(np.nan)
            isi_sds.append(np.nan)
            isi_cvs.append(np.nan)
    else:
        isi_means.append(np.nan)
        isi_sds.append(np.nan)
        isi_cvs.append(np.nan)

# Add ISI statistics to dataframe
per_neuron_stats["ISI mean"] = [isi_means[i] for i in per_neuron_stats["MU_ID"]]
per_neuron_stats["ISI SD"] = [isi_sds[i] for i in per_neuron_stats["MU_ID"]]
per_neuron_stats["ISI CV"] = [isi_cvs[i] for i in per_neuron_stats["MU_ID"]]

# Add metadata and rename columns to match desired format
per_neuron_stats["Muscle"] = muscle_type
per_neuron_stats["Force Level"] = mvc_percent
per_neuron_stats["FR mean"] = per_neuron_stats["mean_firing_rate_Hz"]
per_neuron_stats["subject_id"] = subject_id
per_neuron_stats["extra"] = extra

print("Per-Neuron Statistics:")
print(per_neuron_stats)

# Reorder columns to put metadata first
columns_order = [
    "Muscle",
    "Force Level",
    "ISI mean",
    "ISI SD",
    "ISI CV",
    "FR mean",
    "subject_id",
    "extra",
    "MU_ID",
]
per_neuron_stats = per_neuron_stats[columns_order]

# Save to CSV
Path(save_folder).mkdir(parents=True, exist_ok=True)
csv_filename = (
    f"{save_folder}/{muscle_type}_{subject_id}_{extra}_MVC{mvc_percent}_stats.csv"
)
per_neuron_stats.to_csv(csv_filename, index=False)
print(f"\nStatistics saved to: {csv_filename}")

# ============================================================================
# Plot the spiketrains
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

# Create raster plot
for i, st in enumerate(spiketrains):
    if len(st) > 0:
        spike_times = st.magnitude  # Get spike times in ms
        ax.scatter(
            spike_times,
            [i] * len(spike_times),
            marker="|",
            s=50,
            c="black",
            linewidths=0.5,
        )

# Add plateau region shading
ax.axvspan(
    plateau_start_ms, plateau_end_ms, alpha=0.2, color="green", label="Plateau region"
)

# Formatting
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_ylabel("Motor Unit ID", fontsize=12)
title = f"Experimental Motor Unit Spike Trains - {muscle_type} ({subject_id}, MVC {mvc_percent}%)"
ax.set_title(title, fontsize=14, fontweight="bold")
ax.set_ylim(-0.5, len(spiketrains) - 0.5)
ax.set_xlim(0, max_spike_time * 1000)  # Convert to ms
ax.grid(True, alpha=0.3, axis="x")
ax.legend()

plt.tight_layout()
plot_filename = (
    f"{save_folder}/{muscle_type}_{subject_id}_{extra}_MVC{mvc_percent}_raster.png"
)
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
print(f"\nRaster plot saved to: {plot_filename}")
plt.show()

# %%

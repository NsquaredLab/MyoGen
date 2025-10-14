"""
Watanabe Paper - Force Computation
===================================

This script loads pre-generated spike trains and computes muscle force output
using the Fuglevand force model. The force output is saved as a Neo Block for
later visualization.

**Pipeline:**
1. Load spike trains from 10a (spinal_network_results.pkl)
2. Setup force model parameters
3. Compute force from motor neuron spike trains
4. Save force output as Neo Block (watanabe_force_results.pkl)

**Outputs:**
- watanabe_force_results.pkl: Neo Block containing force AnalogSignal and metadata
"""

# %%

from pathlib import Path
import joblib
import neo
import numpy as np
import quantities as pq

from myogen import simulator
from myogen.simulator.core.force.force_model import ForceModel

##############################################################################
# Setup Paths and Parameters
# --------------------------

# Paths
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

spinal_results_path = save_path / Path("watanabe__spinal_network_results.pkl")

# Simulation parameters (must match spike train generation)

# Motor neuron pool parameters
naMN = 400  # Number of alpha motor neurons

# Memory optimization parameters
motor_unit_batch_size = 50  # Process motor units in batches instead of all at once

##############################################################################
# Load Spike Train Results
# ------------------------

with open(spinal_results_path, "rb") as f:
    results: neo.Block = joblib.load(f)
tstop = results.annotations['time__ms']
dt = results.annotations['timestep__ms']
active_MNs = results.annotations['active_MNs']
# Extract alpha motor neuron spike trains
aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

##############################################################################
# Setup Force Model Parameters
# ----------------------------

# Generate recruitment thresholds for motor neuron pool
# These determine motor unit properties (twitch amplitude, contraction time)
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=naMN,
    recruitment_range__ratio=100,
    mode="combined",
    deluca__slope=10,
)

force_params = {
    "recording_frequency__Hz": 2048,
    "longest_duration_rise_time__ms": 90.0,  # Slowest motor unit twitch rise time
    "contraction_time_range__unitless": 2,  # Spread of contraction times
}

# Create global force model to compute correct motor unit properties
global_force_model = ForceModel(recruitment_thresholds=recruitment_thresholds, **force_params)

# Pre-compute global motor unit properties for all motor units
global_peak_forces = global_force_model.peak_twitch_forces__unitless
global_contraction_times = global_force_model.contraction_times__samples
global_twitch_list = global_force_model.twitch_list

print(f"Global motor unit properties computed:")
print(f"  Peak force range: {global_peak_forces[0]:.3f} to {global_peak_forces[-1]:.3f}")
print(f"  Contraction time range: {global_contraction_times[0]/force_params['recording_frequency__Hz']*1000:.1f} to {global_contraction_times[-1]/force_params['recording_frequency__Hz']*1000:.1f} ms")

##############################################################################
# Generate Force Output in Motor Unit Batches (Memory Efficient)
# --------------------------------------------------------------

print("\nGenerating force output by processing motor units in batches...")


# Calculate number of batches
n_batches = int(np.ceil(naMN / motor_unit_batch_size))

# Initialize force accumulator
force_duration_samples = int(tstop / 1000 * force_params["recording_frequency__Hz"])
force_total = np.zeros(force_duration_samples)

for batch_idx in range(n_batches):
    # Define motor unit batch indices
    mu_start = batch_idx * motor_unit_batch_size
    mu_end = min((batch_idx + 1) * motor_unit_batch_size, naMN)

    print(
        f"\tProcessing motor units {mu_start + 1}-{mu_end} (batch {batch_idx + 1}/{n_batches})..."
    )

    # Create batch-specific Block with subset of motor units
    batch_block = neo.Block(name=f"MU_Batch_{batch_idx}")
    batch_segment = neo.Segment(name=f"aMN_batch_{batch_idx}")

    # Add spike trains for this batch of motor units
    # We need to ensure we have exactly (mu_end - mu_start) spike trains
    # Create a mapping of existing spike trains by motor neuron ID
    spike_train_map = {}
    for spike_train in aMN_spikes:
        mn_id = int(spike_train.name)
        spike_train_map[mn_id] = spike_train

    # Create spike trains for all motor units in this batch (including empty ones)
    batch_spike_trains = []
    for mn_id in range(mu_start, mu_end):
        if mn_id in spike_train_map:
            # Use existing spike train
            spike_train = spike_train_map[mn_id]
            batch_spike_train = neo.SpikeTrain(
                times=spike_train.times,
                t_start=spike_train.t_start,
                t_stop=spike_train.t_stop,
                sampling_rate=spike_train.sampling_rate,
                name=str(mn_id - mu_start)  # Renumber relative to batch start
            )
        else:
            # Create empty spike train for inactive motor neuron
            batch_spike_train = neo.SpikeTrain(
                times=np.array([]) * pq.ms,
                t_start=0.0 * pq.ms,
                t_stop=tstop * pq.ms,
                sampling_rate=(1000.0 / dt) * pq.Hz,
                name=str(mn_id - mu_start)  # Renumber relative to batch start
            )

        batch_segment.spiketrains.append(batch_spike_train)
        batch_spike_trains.append(mn_id)

    batch_block.segments.append(batch_segment)

    active_in_batch = [mn_id for mn_id in range(mu_start, mu_end) if mn_id in spike_train_map]
    

    # Create force model for this batch using CORRECT global properties
    batch_recruitment_thresholds = recruitment_thresholds[mu_start:mu_end]
    batch_force_model = ForceModel(
        recruitment_thresholds=batch_recruitment_thresholds, **force_params
    )

    # CRITICAL FIX: Override the batch model's properties with the correct global ones
    # This ensures each motor unit gets its correct physiological properties
    batch_force_model._peak_twitch_forces__unitless = global_peak_forces[mu_start:mu_end]
    batch_force_model._contraction_times__samples = global_contraction_times[mu_start:mu_end]
    batch_force_model._twitch_list = global_twitch_list[mu_start:mu_end]

    # Also update the twitch matrix if it exists
    if hasattr(batch_force_model, '_twitch_mat') and batch_force_model._twitch_mat is not None:
        # Slice the global twitch matrix for this batch
        global_twitch_mat = global_force_model.twitch_mat
        batch_force_model._twitch_mat = global_twitch_mat[:, mu_start:mu_end]

    # Generate force for this batch (now with correct properties)
    force_batch = batch_force_model.generate_force(spike_train__Block=batch_block)
    force_batch_array = force_batch.magnitude.squeeze()

    
    
    # Accumulate force (sum contributions from all motor units)
    force_total += force_batch_array

    # Clean up to free memory
    del batch_block, batch_segment, batch_force_model, force_batch, force_batch_array

# Create final AnalogSignal
force_output = neo.AnalogSignal(
    force_total * pq.dimensionless,
    t_start=0 * pq.ms,
    sampling_rate=force_params["recording_frequency__Hz"] * pq.Hz,
)

print(f"Force generation complete! Final signal shape: {force_output.shape}")

##############################################################################
# Save Force Results as Neo Block
# -------------------------------

# Create Neo Block to store force results with metadata
force_block = neo.Block(name="Watanabe Force Results")
force_segment = neo.Segment(name="Force")

# Add force output (already a Neo AnalogSignal)
force_segment.analogsignals.append(force_output)

# Store metadata as annotations
force_block.annotations["simulation_params"] = {
    "dt_ms": dt,
    "tstop_ms": tstop,
    "n_motor_neurons": naMN,
}

force_block.annotations["recruitment_thresholds"] = recruitment_thresholds
force_block.annotations["force_model_params"] = force_params

force_block.annotations["force_model_stats"] = {
    "n_motor_units": global_force_model._number_of_neurons,
    "recruitment_ratio": float(global_force_model._recruitment_ratio),
    "peak_force_range": [
        float(global_force_model.peak_twitch_forces__unitless[0]),
        float(global_force_model.peak_twitch_forces__unitless[-1]),
    ],
    "contraction_time_range_samples": [
        float(global_force_model.contraction_times__samples[0]),
        float(global_force_model.contraction_times__samples[-1]),
    ],
}

# Add segment to block
force_block.segments.append(force_segment)

# Save using joblib (compatible with Neo)
output_file = save_path / "watanabe__force_results.pkl"

with open(output_file, "wb") as f:
    joblib.dump(force_block, f)

print("Force computation complete!")

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

from pathlib import Path
import numpy as np
import joblib
import neo
import quantities as pq

from myogen import simulator
from myogen.simulator.core.force.force_model import ForceModel

##############################################################################
# Setup Paths and Parameters
# --------------------------

# Paths
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

spinal_results_path = Path(
    r"C:\Users\raulc\Research\papers_server\in_progress\simulator\spinal_network_results.pkl"
)

# Simulation parameters (must match spike train generation)
dt = 0.1  # ms - Integration timestep
tstop = 180 * 1e3  # ms - Total simulation duration (180 seconds)

# Motor neuron pool parameters
naMN = 100  # Number of alpha motor neurons

print("=" * 80)
print("Watanabe Paper - Force Computation")
print("=" * 80)
print(f"\nSimulation parameters:")
print(f"  - Duration: {tstop/1000:.1f} s")
print(f"  - Timestep: {dt} ms")
print(f"  - Motor neurons: {naMN}")

##############################################################################
# Load Spike Train Results
# ------------------------

print(f"\nLoading spike trains from:")
print(f"  {spinal_results_path}")

with open(spinal_results_path, "rb") as f:
    results: neo.Block = joblib.load(f)
    print("[OK] Spike trains loaded successfully")

# Extract alpha motor neuron spike trains
aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

print(f"  - Alpha motor neurons: {len(aMN_spikes)}")
print(f"  - Duration: {aMN_spikes[0].t_stop.rescale('s')}")

##############################################################################
# Setup Force Model Parameters
# ----------------------------

# Generate recruitment thresholds for motor neuron pool
# These determine motor unit properties (twitch amplitude, contraction time)
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=naMN,
    recruitment_range__ratio=50,  # Physiological recruitment range (50:1)
    mode="combined",
    deluca__slope=5,
)

# Force model parameters
force_params = {
    "recording_frequency__Hz": 100,  # 100 Hz sampling (sufficient for force)
    "longest_duration_rise_time__ms": 90.0,  # Slowest motor unit twitch rise time
    "contraction_time_range__unitless": 3,  # Spread of contraction times
}

print(f"\nForce model parameters:")
for key, value in force_params.items():
    print(f"  - {key}: {value}")

# Create force model
force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds, **force_params
)

print(f"\nForce model statistics:")
print(f"  - Motor units: {force_model._number_of_neurons}")
print(f"  - Recruitment ratio: {force_model._recruitment_ratio:.1f}")
print(
    f"  - Peak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - "
    f"{force_model.peak_twitch_forces__unitless[-1]:.3f}"
)
print(
    f"  - Contraction time range: {force_model.contraction_times__samples[0]:.1f} - "
    f"{force_model.contraction_times__samples[-1]:.1f} samples"
)

##############################################################################
# Prepare Spike Trains for Force Generation
# -----------------------------------------

# Create a new Block containing only aMN spikes, trimmed to exact duration
# This ensures consistent timing and avoids edge effects

print(f"\nPreparing spike trains for force generation...")

aMN_block = neo.Block(name="Alpha Motor Neurons")
aMN_segment_trimmed = neo.Segment(name="aMN")

# Trim each spike train to exact duration
for st in aMN_spikes:
    st_trimmed = st.time_slice(0 * pq.ms, tstop * pq.ms)
    # Manually set t_stop to exact value to avoid rounding issues
    st_trimmed.t_stop = tstop * pq.ms
    aMN_segment_trimmed.spiketrains.append(st_trimmed)

aMN_block.segments.append(aMN_segment_trimmed)

print(f"  - Trimmed spike trains: {len(aMN_segment_trimmed.spiketrains)}")
print(f"  - Time range: {aMN_segment_trimmed.spiketrains[0].t_start} - {aMN_segment_trimmed.spiketrains[0].t_stop}")

##############################################################################
# Generate Force Output
# ---------------------

print(f"\nGenerating force output...")
print("  This may take several minutes for 180s simulation...")

force_output = force_model.generate_force(spike_train__Block=aMN_block)

print(f"[OK] Force generation complete!")
print(f"  - Force samples: {force_output.shape[0]}")
print(f"  - Sampling rate: {force_output.sampling_rate}")
print(f"  - Duration: {force_output.t_stop.rescale('s')}")
print(f"  - Force range: {np.min(force_output.magnitude):.3f} - {np.max(force_output.magnitude):.3f} (unitless)")

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
    "n_motor_units": force_model._number_of_neurons,
    "recruitment_ratio": float(force_model._recruitment_ratio),
    "peak_force_range": [
        float(force_model.peak_twitch_forces__unitless[0]),
        float(force_model.peak_twitch_forces__unitless[-1]),
    ],
    "contraction_time_range_samples": [
        float(force_model.contraction_times__samples[0]),
        float(force_model.contraction_times__samples[-1]),
    ],
}

# Add segment to block
force_block.segments.append(force_segment)

# Save using joblib (compatible with Neo)
output_file = save_path / "watanabe_force_results.pkl"
print(f"\nSaving force results to:")
print(f"  {output_file}")

with open(output_file, "wb") as f:
    joblib.dump(force_block, f)

print(f"[OK] Force results saved successfully!")

##############################################################################
# Summary
# -------

print("\n" + "=" * 80)
print("Force Computation Complete!")
print("=" * 80)
print(f"\nOutput file: {output_file}")
print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print(f"\nNext step: Run 10c_watanabe_visualize.py to generate plots")
print("=" * 80)

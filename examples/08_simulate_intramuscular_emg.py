"""
Intramuscular EMG Signals
========================

This example demonstrates how to simulate **intramuscular EMG signals** using
needle electrodes. It shows the complete pipeline from muscle model creation
to EMG signal generation with realistic noise and motor unit detectability.

.. note::
    **Intramuscular EMG** (iEMG) is recorded using needle electrodes inserted
    directly into the muscle tissue. This provides high spatial resolution
    and allows for the detection of individual motor unit action potentials
    (MUAPs). Unlike surface EMG, intramuscular recordings can capture the
    activity of deeper motor units and provide better selectivity.

Key Features:
    - **High spatial resolution**: Needle electrodes can detect individual MUAPs
    - **Deep muscle access**: Can record from muscles not accessible by surface electrodes
    - **Motor unit discrimination**: Individual motor units can be identified and tracked
    - **Realistic noise modeling**: Includes physiological noise and recording artifacts
"""
# sphinx_gallery_thumbnail_number = -1

##############################################################################
# Import Libraries
# ----------------

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from myogen import simulator
from myogen.utils.currents import create_sinusoidal_current
from myogen.utils.plotting.spikes import plot_spike_trains

##############################################################################
# Define Parameters
# -----------------
#

# Simulation parameters
N_motor_units = 25  # Smaller number for faster computation
recruitment_range = 50.0

# Electrode parameters
inter_electrode_distance = 0.5  # mm
electrode_position = (0.0, 0.0, 0.0)  # mm (start of muscle)

##############################################################################
# Generate Motor Unit Recruitment Thresholds
# -------------------------------------------
#
# First, we generate the **recruitment thresholds** for the motor unit pool.

thresholds, _ = simulator.generate_mu_recruitment_thresholds(
    N=N_motor_units,
    recruitment_range=recruitment_range,
    deluca__slope=5,
    mode="combined"
)

plt.figure()
plt.plot(thresholds, "o")
plt.tight_layout()
plt.show()

##############################################################################
# Load Muscle Model
# -------------------
#
# Load the **muscle model** with the generated recruitment thresholds.

muscle = joblib.load("results/muscle_model.pkl")
muscle.length__mm = 30

##############################################################################
# Create Intramuscular Electrode Array
# ------------------------------------
#
# Set up a **differential needle electrode** for intramuscular recordings.

electrode = simulator.IntramuscularElectrodeArray(
    num_electrodes=4,
    inter_electrode_distance__mm=inter_electrode_distance,
    differentiation_mode="consecutive",
    position__mm=electrode_position,
    orientation__rad=(-np.pi / 2, 0, -np.pi / 2),  # perpendicular to muscle
    trajectory_distance__mm=0.125,  # mm
    trajectory_steps=1,  # number of steps
)

##############################################################################
# Initialize Intramuscular EMG Simulator
# --------------------------------------
#
# Create the **intramuscular EMG simulator** with the muscle model and electrode.

print("Initializing iEMG simulator...")
iemg_sim = simulator.IntramuscularEMG(
    muscle_model=muscle,
    electrode_array=electrode,
    MUs_to_simulate=list(range(0, N_motor_units, 3)),
)

##############################################################################
# Calculate Motor Unit Action Potentials
# --------------------------------------
#
# Compute the **MUAPs** for each motor unit at the electrode positions.

print("Computing motor unit action potentials...")
iemg_sim.simulate_muaps()
print(f"  - Generated MUAPs for {N_motor_units} motor units")
print(f"  - Electrode array with {electrode.num_electrodes} electrodes")


##############################################################################
# Create Motor Neuron Pool
# ------------------------
#
# Set up the **motor neuron pool** for spike train generation.

mn_pool = simulator.MotorNeuronPool(recruitment_thresholds=thresholds)
mvc_current = mn_pool.mvc_current_threshold


##############################################################################
# Generate Input Currents
# -----------------------
#
# Create a **sinusoidal current profile** for the contraction simulation.

n_pools = 2  # Number of distinct motor neuron pools

timestep = 0.01  # Simulation timestep in ms (high resolution)
simulation_time = 500  # Total simulation duration in ms

# Calculate number of time points
t_points = int(simulation_time / timestep)

# Create the input current matrix
sin_amplitudes = [
    mvc_current * 1,
    mvc_current * 1,
]  # Same amplitude for both pools
sin_frequencies = [1.0, 1.0]  # Same frequency for both pools (1 Hz)
sin_offsets = [
    0.0,
    0.0,
]
sin_phases = [0.0, np.pi]  # 0 and 90 degrees (in radians)

sinusoidal_currents = create_sinusoidal_current(
    n_pools=n_pools,
    t_points=t_points,
    timestep__ms=timestep,
    amplitudes__muV=sin_amplitudes,
    frequencies__Hz=sin_frequencies,
    offsets__muV=sin_offsets,
    phases__rad=sin_phases,
)

##############################################################################
# Generate Spike Trains
# ---------------------
#
# Simulate the **neural spike trains** using the motor neuron pool.

print("Generating spike trains...")
spike_trains, active_indices, _ = mn_pool.generate_spike_trains(
    input_current__matrix=sinusoidal_currents, timestep__ms=timestep
)
print(f"  - Generated spike trains for {len(active_indices)} active motor units")
print(f"  - Total simulation time: {simulation_time} ms")

##############################################################################
# Visualize Spike Trains
# ----------------------
#
# Display the **spike trains** generated by the motor neuron pool in response 
# to the sinusoidal input current. This shows the neural firing patterns
# that will be convolved with the MUAPs to generate the final EMG signal.

spike_fig, spike_ax = plt.subplots(figsize=(10, 6))
plot_spike_trains(
    spike_trains__matrix=spike_trains,
    timestep__ms=timestep,
    axs=[spike_ax],
    pool_current__matrix=sinusoidal_currents,
    pool_to_plot=[0]  # Show only the first pool
)
plt.tight_layout()
plt.show()

##############################################################################
# Simulate Intramuscular EMG
# -------------------------
#
# Generate the final **intramuscular EMG signals** by convolving spike trains with MUAPs.

print("Simulating intramuscular EMG signals...")
emg_signals = iemg_sim.simulate_intramuscular_emg(
    motor_neuron_pool=mn_pool,
)

print("Intramuscular EMG simulation completed!")
print(f"  - EMG signal shape: {emg_signals.shape}")
print(f"  - Signal RMS (before noise): {np.sqrt(np.mean(emg_signals**2)):.3f} nA")

print("Adding realistic noise (SNR = 20 dB)...")
noisy_emg_signals = iemg_sim.add_noise(snr_db=20)
print(f"  - Signal RMS (after noise): {np.sqrt(np.mean(noisy_emg_signals**2)):.3f} nA")

##############################################################################
# Visualize Intramuscular EMG Results
# ----------------------------------
#
# Create an **xkcd-style plot** comparing the **intramuscular EMG** signal 
# with the input current, similar to the surface EMG example.
#
# .. note::
#   The intramuscular EMG provides **high spatial resolution** and can detect
#   individual motor unit action potentials (MUAPs) with excellent signal quality.

# Clear matplotlib cache and set up xkcd style
matplotlib.get_cachedir()
with plt.xkcd():
    plt.rcParams.update({"font.size": 24})
    
    # Create single plot with normalized signals
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get the signals - use first electrode's differential recording
    iemg_signal = noisy_emg_signals[0, 0, :]  # First electrode differential pair
    current_signal = sinusoidal_currents[0]  # First current pool
    
    # Create time axes
    emg_time = np.arange(len(iemg_signal)) / iemg_sim.sampling_frequency__Hz  # Convert to seconds
    current_time = mn_pool.times / 1000
    
    # Normalize iEMG signal by dividing by maximum absolute value
    iemg_normalized = iemg_signal / np.max(np.abs(iemg_signal))
    
    # Normalize current between 0 and 1
    current_normalized = (current_signal - np.min(current_signal)) / (
        np.max(current_signal) - np.min(current_signal)
    )
    
# Plot both normalized signals
ax.plot(
    emg_time,
    iemg_normalized,
    linewidth=2,
    label="Intramuscular EMG",
)

with plt.xkcd():
    ax.plot(
        current_time,
        current_normalized,
        linewidth=2,
        label="Input Current",
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    sns.despine(trim=True, left=False, bottom=False, right=True, top=True, offset=5)
    
    plt.title("Normalized Intramuscular EMG and Input Current")

plt.tight_layout()
plt.show()
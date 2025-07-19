"""
Intramuscular EMG Signals
========================

This example demonstrates how to simulate **intramuscular EMG signals** using
needle electrodes. It shows the complete pipeline from muscle model creation
to EMG signal generation with realistic noise and motor unit detectability.

.. note::
    **Intramuscular EMG** (iEMG) is recorded using needle electrodes inserted
    directly into the muscle tissue. This provides high spatial resolution
    and allows for the detection of individual motor unit action potentials.

Based on the MATLAB iemg_simulator functionality, now available in Python.
"""

##############################################################################
# Import Libraries
# ----------------

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from myogen import simulator
from myogen.utils.currents import create_trapezoid_current

##############################################################################
# Define Parameters
# -----------------
#
# The **intramuscular EMG** simulation requires several key parameters:
#
# - ``N_motor_units``: Number of motor units to simulate
# - ``recruitment_range``: Range of recruitment thresholds
# - ``fiber_density``: Number of muscle fibers per mm²
# - ``sampling_frequency``: EMG sampling rate in Hz
# - ``snr_db``: Signal-to-noise ratio in decibels
#
# For the electrode configuration:
#
# - ``inter_electrode_distance``: Distance between differential contacts
# - ``electrode_position``: 3D position in muscle coordinates
# - ``trajectory_distance``: Scanning distance for multiple positions

# Simulation parameters
N_motor_units = 5  # Smaller number for faster computation
recruitment_range = 30.0
fiber_density = 300  # fibers per mm² (reduced for faster computation)
sampling_frequency = 10000.0  # Hz
snr_db = 20  # Signal-to-noise ratio

# Electrode parameters
inter_electrode_distance = 0.5  # mm
electrode_position = (0.0, 0.0, 15.0)  # mm (center of muscle)
trajectory_distance = 2.0  # mm
trajectory_steps = 5

# Contraction parameters
duration_s = 3.0
current_amplitude = 100  # nA
ramp_start_ratio = 0.2
ramp_end_ratio = 0.8

##############################################################################
# Generate Motor Unit Recruitment Thresholds
# -------------------------------------------
#
# First, we generate the **recruitment thresholds** for the motor unit pool.

print("Generating motor unit recruitment thresholds...")
thresholds, _ = simulator.generate_mu_recruitment_thresholds(
    N=N_motor_units,
    recruitment_range=recruitment_range,
)
print(f"Created {N_motor_units} motor units with recruitment range {recruitment_range}")

##############################################################################
# Create Muscle Model
# -------------------
#
# Create the **muscle model** with the generated recruitment thresholds.

print("Creating muscle model...")
muscle = simulator.Muscle(
    recruitment_thresholds=thresholds,
    fiber_density__fibers_per_mm2=fiber_density,
    autorun=True,  # Automatically run full initialization
)
print(
    f"Muscle: {muscle.muscle_area__mm2:.1f} mm² area, {muscle.number_of_muscle_fibers} fibers"
)
print(f"Motor units: {muscle._number_of_neurons} total")

##############################################################################
# Create Intramuscular Electrode Array
# ------------------------------------
#
# Set up a **differential needle electrode** for intramuscular recordings.

print("Setting up intramuscular electrode...")

# Create a single differential electrode (2-contact needle)
electrode = simulator.IntramuscularElectrodeArray.create_single_differential(
    inter_electrode_distance__mm=inter_electrode_distance,
    position__mm=electrode_position,
    trajectory_distance__mm=trajectory_distance,
    trajectory_steps=trajectory_steps,
)

print(f"Electrode: {electrode.type}")
print(f"Channels: {electrode.num_channels}")
print(
    f"Trajectory: {electrode.trajectory_steps} steps over {electrode.trajectory_distance__mm} mm"
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
    sampling_frequency__Hz=sampling_frequency,
    snr__dB=snr_db,
)

print(f"Sampling frequency: {iemg_sim.sampling_frequency__Hz} Hz")
print(f"Simulating {len(iemg_sim.MUs_to_simulate)} motor units")

##############################################################################
# Calculate Motor Unit Action Potentials
# --------------------------------------
#
# Compute the **MUAPs** for each motor unit at the electrode positions.

print("Computing motor unit action potentials...")
iemg_sim.initialize_motor_units()
iemg_sim.simulate_neuromuscular_junctions()
iemg_sim.calculate_muaps()

print(f"Calculated MUAPs for {len(iemg_sim.motor_units)} motor units")
print(f"MUAP duration: {iemg_sim.max_muap_length * iemg_sim.dt * 1000:.1f} ms")

##############################################################################
# Generate Noise Reference
# ------------------------
#
# Create a **noise reference** from simulated maximum voluntary contraction EMG.

print("Generating noise reference...")
mvc_emg = iemg_sim.generate_mvc_emg(duration__s=2.0)

mvc_std = np.mean(iemg_sim.mvc_emg_std) if iemg_sim.mvc_emg_std is not None else 0.0
noise_std = (
    np.mean(iemg_sim.emg_noise_std) if iemg_sim.emg_noise_std is not None else 0.0
)

print(f"MVC EMG RMS: {mvc_std:.3f}")
print(f"Noise level: {noise_std:.3f}")

##############################################################################
# Analyze Motor Unit Detectability
# --------------------------------
#
# Determine which **motor units** will be detectable given the electrode position and noise level.

print("Analyzing motor unit detectability...")
detectable, detectable_indices = iemg_sim.analyze_detectable_motor_units()
print(f"Detectable motor units: {len(detectable_indices)}/{len(iemg_sim.motor_units)}")

##############################################################################
# Create Motor Neuron Pool
# ------------------------
#
# Set up the **motor neuron pool** for spike train generation.

print("Setting up neural simulation...")
mn_pool = simulator.MotorNeuronPool(
    recruitment_thresholds=thresholds[: len(iemg_sim.MUs_to_simulate)]
)

##############################################################################
# Generate Input Currents
# -----------------------
#
# Create a **trapezoidal current profile** for the contraction simulation.

print("Generating input currents...")
n_time_points = int(duration_s * iemg_sim.sampling_frequency__Hz)


# Create trapezoidal current
def generate_trapezoidal_current(
    n_time_points: int,
    amplitude_start: float,
    amplitude_end: float,
    ramp_start_ratio: float,
    ramp_end_ratio: float,
) -> np.ndarray:
    """Generate a trapezoidal current profile."""
    current = np.zeros(n_time_points)

    ramp_start_idx = int(ramp_start_ratio * n_time_points)
    ramp_end_idx = int(ramp_end_ratio * n_time_points)

    # Initial ramp up
    current[:ramp_start_idx] = np.linspace(
        amplitude_start, amplitude_end, ramp_start_idx
    )

    # Plateau
    current[ramp_start_idx:ramp_end_idx] = amplitude_end

    # Ramp down
    current[ramp_end_idx:] = np.linspace(
        amplitude_end, amplitude_start, n_time_points - ramp_end_idx
    )

    return current


input_currents = generate_trapezoidal_current(
    n_time_points=n_time_points,
    amplitude_start=0.0,
    amplitude_end=current_amplitude,
    ramp_start_ratio=ramp_start_ratio,
    ramp_end_ratio=ramp_end_ratio,
).reshape(1, -1)  # Shape: (1, n_time_points)

print(f"Duration: {duration_s} s")
print(f"Peak current: {current_amplitude * 0.2:.0f} nA")

##############################################################################
# Generate Spike Trains
# ---------------------
#
# Simulate the **neural spike trains** using the motor neuron pool.

print("Generating neural spike trains...")
spike_trains, active_indices, _ = mn_pool.generate_spike_trains(
    input_current__matrix=input_currents, timestep__ms=iemg_sim.dt * 1000
)

n_active = len(active_indices[0])
print(f"Active motor units: {n_active}/{len(iemg_sim.MUs_to_simulate)}")

##############################################################################
# Simulate Intramuscular EMG
# -------------------------
#
# Generate the final **intramuscular EMG signals** by convolving spike trains with MUAPs.

print("Generating intramuscular EMG signals...")
emg_signals = iemg_sim.simulate_emg(
    spike_trains=spike_trains, use_jitter=True, add_noise=True
)

print(f"EMG shape: {emg_signals.shape}")
print(f"Signal RMS: {np.sqrt(np.mean(emg_signals**2)):.3f}")

##############################################################################
# Save Results
# -----------
#
# Save the simulation results for later analysis.

# Create results directory
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

# Save EMG signals
np.save(save_path / "intramuscular_emg.npy", emg_signals)

# Save input currents
np.save(save_path / "input_currents.npy", input_currents)

# Save electrode array
joblib.dump(electrode, save_path / "electrode_arrays_v2.pkl")

# Save input current matrix for compatibility
joblib.dump(input_currents, save_path / "input_current_matrix.pkl")

# Save simulation summary
summary = iemg_sim.get_simulation_summary()
time_s = np.arange(emg_signals.shape[2]) / iemg_sim.sampling_frequency__Hz
summary.update(
    {
        "simulation_duration_s": float(time_s[-1]),
        "emg_signal_shape": emg_signals.shape,
        "signal_rms": float(np.sqrt(np.mean(emg_signals**2))),
    }
)

# Save as text file
with open(save_path / "simulation_summary.txt", "w") as f:
    f.write("Intramuscular EMG Simulation Summary\n")
    f.write("=" * 40 + "\n\n")

    for key, value in summary.items():
        f.write(f"{key}: {value}\n")

print(f"Results saved to: {save_path}")

##############################################################################
# Plot Results
# -----------
#
# Visualize the **intramuscular EMG simulation** results.

# Time axis
time_s = np.arange(emg_signals.shape[2]) / iemg_sim.sampling_frequency__Hz

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Intramuscular EMG Simulation Results", fontsize=16)

# Plot 1: Input current and EMG signal
ax1 = axes[0, 0]
ax1.plot(time_s, input_currents[0], "b-", linewidth=2, label="Input Current")
ax1.set_ylabel("Current (nA)", color="b")
ax1.tick_params(axis="y", labelcolor="b")

ax1_twin = ax1.twinx()
ax1_twin.plot(time_s, emg_signals[0, 0, :], "r-", alpha=0.7, label="iEMG Signal")
ax1_twin.set_ylabel("iEMG Amplitude", color="r")
ax1_twin.tick_params(axis="y", labelcolor="r")
ax1.set_xlabel("Time (s)")
ax1.set_title("Input Current and iEMG Signal")
ax1.grid(True, alpha=0.3)

# Plot 2: Sample MUAPs from detectable motor units
ax2 = axes[0, 1]
if len(detectable_indices) > 0 and iemg_sim.muaps is not None:
    muap_time = np.arange(iemg_sim.muaps.shape[1]) * iemg_sim.dt * 1000
    n_muaps_to_show = min(5, len(detectable_indices))

    for i, mu_idx in enumerate(detectable_indices[:n_muaps_to_show]):
        muap = iemg_sim.muaps[mu_idx, :, 0]  # First electrode
        ax2.plot(muap_time, muap + i * 0.5, label=f"MU {mu_idx + 1}", linewidth=1.5)

    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("MUAP Amplitude (offset)")
    ax2.set_title("Detectable Motor Unit Action Potentials")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(
        0.5,
        0.5,
        "No detectable MUAPs",
        ha="center",
        va="center",
        transform=ax2.transAxes,
    )
    ax2.set_title("Motor Unit Action Potentials")

# Plot 3: Signal spectrum
ax3 = axes[1, 0]
freqs, psd = welch(
    emg_signals[0, 0, :], fs=iemg_sim.sampling_frequency__Hz, nperseg=1024, noverlap=512
)

ax3.semilogy(freqs, psd, "g-", linewidth=2)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Power Spectral Density")
ax3.set_title("iEMG Power Spectrum")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 1000)

# Plot 4: Motor unit territory visualization
ax4 = axes[1, 1]

# Plot muscle fibers and motor unit territories
fiber_positions = iemg_sim.muscle_model.mf_centers
if fiber_positions is not None:
    ax4.scatter(
        fiber_positions[:, 0],
        fiber_positions[:, 1],
        c="lightgray",
        s=1,
        alpha=0.5,
        label="Muscle fibers",
    )

# Plot innervation centers
centers = iemg_sim.muscle_model.innervation_center_positions
if centers is not None:
    ax4.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        s=50,
        marker="x",
        linewidth=2,
        label="MU centers",
    )

    # Highlight detectable motor units
    if len(detectable_indices) > 0:
        det_centers = centers[detectable_indices]
        ax4.scatter(
            det_centers[:, 0],
            det_centers[:, 1],
            c="blue",
            s=100,
            marker="o",
            alpha=0.7,
            label="Detectable MUs",
        )

# Plot electrode position
electrode_pos = iemg_sim.electrode_array.position__mm
if electrode_pos is not None:
    ax4.scatter(
        electrode_pos[0],
        electrode_pos[1],
        c="black",
        s=200,
        marker="^",
        label="Electrode",
    )

ax4.set_xlabel("X (mm)")
ax4.set_ylabel("Y (mm)")
ax4.set_title("Motor Unit Territories and Electrode")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis("equal")

plt.tight_layout()
plt.show()

##############################################################################
# Muscle Cross-Section Visualization
# ----------------------------------
#
# Show the **muscle cross-section** with motor unit territories and electrode position.

# Plot muscle cross-section with territories
fig2, ax = plt.subplots(1, 1, figsize=(10, 8))

# Draw muscle boundary
theta = np.linspace(0, 2 * np.pi, 100)
muscle_x = iemg_sim.muscle_model.radius__mm * np.cos(theta)
muscle_y = iemg_sim.muscle_model.radius__mm * np.sin(theta)
ax.plot(muscle_x, muscle_y, "k-", linewidth=2, label="Muscle boundary")

# Plot fiber positions colored by motor unit
if fiber_positions is not None:
    cmap = plt.cm.get_cmap("tab10")
    for mu_idx in range(min(10, len(iemg_sim.motor_units))):
        fiber_mask = iemg_sim.muscle_model.assignment == mu_idx
        mu_fibers = fiber_positions[fiber_mask]

        if len(mu_fibers) > 0:
            color = cmap(mu_idx % 10)
            ax.scatter(
                mu_fibers[:, 0],
                mu_fibers[:, 1],
                c=[color],
                s=5,
                alpha=0.6,
                label=f"MU {mu_idx + 1}" if mu_idx < 5 else "",
            )

# Plot electrode
ax.scatter(
    electrode_pos[0],
    electrode_pos[1],
    c="red",
    s=200,
    marker="^",
    label="Electrode",
    zorder=10,
)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Muscle Cross-Section with Motor Unit Territories")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)
ax.axis("equal")

plt.tight_layout()
plt.show()

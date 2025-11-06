#!/usr/bin/env python
"""
Intramuscular EMG Simulation with Command-Line Interface
==========================================================

This script provides a command-line interface for simulating intramuscular EMG
with configurable motor unit selection, SNR, and trapezoid drive patterns.

Usage:
    # Basic usage with default trapezoid
    python run_iemg.py --mus "0,5,10,15,20" --snr 20
    python run_iemg.py --mus all --snr 15
    python run_iemg.py --mus "0-20" --snr 10
    python run_iemg.py --mus "0-100-10" --snr 25

    # Negative indices (reference active MUs from spike trains)
    python run_iemg.py --mus "-1" --snr 20          # Last active MU (largest)
    python run_iemg.py --mus "-5,-1" --snr 20       # 5th from last and last
    python run_iemg.py --mus "-10--1" --snr 20      # Last 10 active MUs

    # Custom trapezoid pattern
    python run_iemg.py --mus "0,5,10" --snr 20 --peak-hz 80 --plateau-time 8000
    python run_iemg.py --mus all --snr 15 --rise-time 1000 --fall-time 1000

Parameters:
    --mus            Motor units to simulate (see examples above)
    --snr            Signal-to-noise ratio in dB (default: 20)
    --no-plot        Skip plotting for faster execution

    Trapezoid Drive Pattern (optional - uses existing pkl if not specified):
    --sim-time       Simulation time in ms (default: 13000)
    --timestep       Time step in ms (default: 0.1)
    --rise-time      Ramp-up duration in ms (default: 500)
    --plateau-time   Plateau duration in ms (default: 10000)
    --fall-time      Ramp-down duration in ms (default: 500)
    --rest-before    Initial rest in ms (default: 1000)
    --rest-after     Final rest in ms (default: 1000)
    --baseline-hz    Baseline drive in Hz (default: 0.0)
    --peak-hz        Peak drive in Hz (default: 65.0)
    --noise-std      Noise std deviation in Hz (default: 1.0)
    --no-noise       Disable noise in trapezoid

Output:
    Creates a subdirectory in results/synthetic_gen/ containing:
    - signals.pkl: EMG signals
    - decomp.pkl: Decomposition package (templates + spike trains)
    - emg_plot.png: EMG visualization (if --no-plot not set)
    - muap_templates.png: MUAP templates (if --no-plot not set)
    - trapezoid_drive.pkl: Custom trapezoid (if parameters provided)
"""

import argparse
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import seaborn as sns
from neo.core import AnalogSignal

from myogen import RANDOM_GENERATOR, simulator
from myogen.utils.cortical_inputs import create_trapezoid_cortical_input
from myogen.utils.types import CURRENT__AnalogSignal, SPIKE_TRAIN__Block

# Configure matplotlib
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def get_active_motor_units(spike_train_path):
    """
    Get list of motor units that actually fired in the spike train simulation.

    Parameters
    ----------
    spike_train_path : Path
        Path to the spike train pickle file (sinusoidal_dd_spike_trains.pkl)

    Returns
    -------
    list[int]
        Sorted list of MU indices that have at least one spike
    """
    spike_train_block = joblib.load(spike_train_path)
    
    # Get motor neuron segment (should be first/only segment)
    mn_segment = spike_train_block.segments[0]
    
    # Find MUs with spikes
    active_mus = []
    for i, spiketrain in enumerate(mn_segment.spiketrains):
        if len(spiketrain) > 0:
            active_mus.append(i)
    
    return active_mus


def parse_mus(mus_str, active_mus=None):
    """
    Parse motor unit specification string with support for negative indices.

    Negative indices reference the list of active (firing) motor units,
    where -1 is the last MU that fired, -2 is second-to-last, etc.

    Parameters
    ----------
    mus_str : str
        MU specification:
        - "all": All motor units
        - "0,5,10": Specific indices (comma-separated)
        - "-1": Last active MU (largest that fired)
        - "-5,-1": 5th from last and last active MUs
        - "0-20": Range (0 to 20 inclusive)
        - "-10--1": Last 10 active MUs
        - "0-100-10": Every 10th from 0 to 100
    active_mus : list[int], optional
        List of active MU indices (those that fired). Required if using
        negative indices. If None, negative indices will raise an error.

    Returns
    -------
    list[int] or None
        List of MU indices or None for all MUs

    Raises
    ------
    ValueError
        If negative indices used without active_mus, or invalid specification
    """
    if mus_str.lower() == "all":
        return None

    # Helper function to convert negative index to positive
    def convert_index(idx_str):
        idx = int(idx_str.strip())
        if idx < 0:
            if active_mus is None:
                raise ValueError(
                    f"Negative indices (like {idx}) require spike train data. "
                    "Make sure spike trains have been generated first."
                )
            if abs(idx) > len(active_mus):
                raise ValueError(
                    f"Negative index {idx} out of range. "
                    f"Only {len(active_mus)} motor units are active."
                )
            return active_mus[idx]
        return idx

    # Comma-separated list
    if "," in mus_str:
        return [convert_index(x) for x in mus_str.split(",")]

    # Range specification
    if "-" in mus_str:
        # Handle negative numbers in ranges carefully
        # Split and track if we started with negative
        parts = []
        current = ""
        for i, char in enumerate(mus_str):
            if char == "-":
                if current == "" or mus_str[i-1] == "-":
                    # This is a negative sign, not a separator
                    current += char
                else:
                    # This is a separator
                    parts.append(current)
                    current = ""
            else:
                current += char
        if current:
            parts.append(current)
        
        if len(parts) == 2:
            # Range: start-end
            start = convert_index(parts[0])
            end = convert_index(parts[1])
            return list(range(start, end + 1))
        elif len(parts) == 3:
            # Range with step: start-end-step
            start = convert_index(parts[0])
            end = convert_index(parts[1])
            step = int(parts[2])
            return list(range(start, end, step))

    # Single value (possibly negative)
    try:
        return [convert_index(mus_str)]
    except ValueError:
        raise ValueError(f"Invalid MU specification: {mus_str}")


def generate_filename_suffix(mus, snr):
    """
    Generate filename suffix from parameters.

    Parameters
    ----------
    mus : list[int] or None
        Motor unit indices
    snr : float
        SNR in dB

    Returns
    -------
    str
        Filename suffix like "mu_0_5_10_snr20"
    """
    if mus is None:
        mu_str = "mu_all"
    else:
        # Use first few indices in filename
        if len(mus) <= 5:
            mu_str = "mu_" + "_".join(map(str, mus))
        else:
            mu_str = "mu_" + "_".join(map(str, mus[:3])) + f"_plus{len(mus)-3}"

    snr_str = f"snr{int(snr)}"
    return f"{mu_str}_{snr_str}"


def generate_trapezoid_drive(
    simulation_time__ms=13000.0,
    timestep__ms=0.1,
    rise_time__ms=500.0,
    plateau_time__ms=10000.0,
    fall_time__ms=500.0,
    rest_before__ms=1000.0,
    rest_after__ms=1000.0,
    baseline__Hz=0.0,
    peak__Hz=65.0,
    noise_std__Hz=1.0,
    add_noise=True,
):
    """
    Generate trapezoid cortical drive pattern with optional noise.

    Parameters
    ----------
    simulation_time__ms : float
        Total simulation duration in milliseconds
    timestep__ms : float
        Time step in milliseconds
    rise_time__ms : float
        Ramp-up duration in milliseconds
    plateau_time__ms : float
        Plateau duration in milliseconds
    fall_time__ms : float
        Ramp-down duration in milliseconds
    rest_before__ms : float
        Initial rest period before trapezoid starts
    rest_after__ms : float
        Final rest period after trapezoid ends
    baseline__Hz : float
        Baseline drive level in Hz
    peak__Hz : float
        Peak drive level in Hz
    noise_std__Hz : float
        Standard deviation of Gaussian noise in Hz
    add_noise : bool
        Whether to add Gaussian noise

    Returns
    -------
    CURRENT__AnalogSignal
        Trapezoid drive pattern as Neo AnalogSignal with Hz units
    """
    # Calculate total time points
    n_points = int(simulation_time__ms / timestep__ms)

    # Use MyoGen's built-in trapezoid generation
    trapezoid = create_trapezoid_cortical_input(
        n_pools=1,
        t_points=n_points,
        timestep__ms=timestep__ms,
        amplitudes__pps=peak__Hz,
        rise_times__ms=rise_time__ms,
        plateau_times__ms=plateau_time__ms,
        fall_times__ms=fall_time__ms,
        offsets__pps=baseline__Hz,
        delays__ms=rest_before__ms,
    )

    # Extract the single pool's data
    trapezoid_array = trapezoid[:, 0]

    # Add noise if requested (matching current implementation)
    if add_noise and noise_std__Hz > 0:
        noise = RANDOM_GENERATOR.normal(0, noise_std__Hz, size=trapezoid_array.shape)
        # Clip to ensure no negative values
        trapezoid_array = trapezoid_array + np.clip(noise, 0, None)

    # Wrap in Neo AnalogSignal
    analog_signal = AnalogSignal(
        signal=trapezoid_array,
        units=pq.Hz,
        sampling_period=(timestep__ms * pq.ms).rescale(pq.s),
    )

    return analog_signal


def main():
    parser = argparse.ArgumentParser(
        description="Simulate intramuscular EMG with custom parameters"
    )
    parser.add_argument(
        "--mus",
        type=str,
        default="0,5,10,15,20,25,30",
        help='Motor units to simulate: "all", "0,5,10", "0-20", or "0-100-10"',
    )
    parser.add_argument(
        "--snr", type=float, default=20.0, help="Signal-to-noise ratio in dB"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting (faster execution)"
    )

    # Trapezoid drive pattern parameters (optional)
    parser.add_argument(
        "--sim-time", type=float, default=None,
        help="Simulation time in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--timestep", type=float, default=None,
        help="Time step in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--rise-time", type=float, default=None,
        help="Trapezoid rise time in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--plateau-time", type=float, default=None,
        help="Trapezoid plateau time in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--fall-time", type=float, default=None,
        help="Trapezoid fall time in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--rest-before", type=float, default=None,
        help="Rest period before trapezoid in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--rest-after", type=float, default=None,
        help="Rest period after trapezoid in ms (default: load from pkl)"
    )
    parser.add_argument(
        "--baseline-hz", type=float, default=None,
        help="Baseline drive in Hz (default: load from pkl)"
    )
    parser.add_argument(
        "--peak-hz", type=float, default=None,
        help="Peak drive in Hz (default: load from pkl)"
    )
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="Noise standard deviation in Hz (default: load from pkl)"
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="Disable noise in trapezoid generation"
    )

    args = parser.parse_args()

    # Set up paths first to access spike train data
    base_path = Path("./results/synthetic_gen")
    spike_train_file = base_path / "sinusoidal_dd_spike_trains.pkl"

    # Get active motor units from spike train data (for negative index support)
    if spike_train_file.exists():
        active_mus = get_active_motor_units(spike_train_file)
        print(f"Found {len(active_mus)} active motor units (indices {min(active_mus)}-{max(active_mus)})")
    else:
        print("Warning: Spike train file not found. Negative indices not supported.")
        active_mus = None

    # Parse MU specification (supports negative indices if active_mus available)
    MUs_to_simulate = parse_mus(args.mus, active_mus)
    snr_db = args.snr

    # Generate filename suffix
    filename_suffix = generate_filename_suffix(MUs_to_simulate, snr_db)

    print("=" * 70)
    print("INTRAMUSCULAR EMG SIMULATION")
    print("=" * 70)
    print(f"Motor Units: {MUs_to_simulate if MUs_to_simulate else 'All'}")
    print(f"SNR: {snr_db} dB")
    print(f"Output suffix: {filename_suffix}")
    print("=" * 70 + "\n")
    save_path = base_path / f"iemg_{filename_suffix}"
    save_path.mkdir(parents=True, exist_ok=True)

    ##########################################################################
    # Load Prerequisites
    ##########################################################################

    print("Loading muscle model and spike trains...")
    muscle: simulator.Muscle = joblib.load(base_path / "muscle_model.pkl")
    spike_train__Block: SPIKE_TRAIN__Block = joblib.load(
        base_path / "sinusoidal_dd_spike_trains.pkl"
    )

    # Check if any trapezoid parameters were provided
    trapezoid_params_provided = any([
        args.sim_time is not None,
        args.timestep is not None,
        args.rise_time is not None,
        args.plateau_time is not None,
        args.fall_time is not None,
        args.rest_before is not None,
        args.rest_after is not None,
        args.baseline_hz is not None,
        args.peak_hz is not None,
        args.noise_std is not None,
        args.no_noise,
    ])

    if trapezoid_params_provided:
        # Generate custom trapezoid
        print("\nGenerating custom trapezoid drive pattern...")

        # Use defaults from current implementation if not specified
        trap_kwargs = {
            "simulation_time__ms": args.sim_time if args.sim_time is not None else 13000.0,
            "timestep__ms": args.timestep if args.timestep is not None else 0.1,
            "rise_time__ms": args.rise_time if args.rise_time is not None else 500.0,
            "plateau_time__ms": args.plateau_time if args.plateau_time is not None else 10000.0,
            "fall_time__ms": args.fall_time if args.fall_time is not None else 500.0,
            "rest_before__ms": args.rest_before if args.rest_before is not None else 1000.0,
            "rest_after__ms": args.rest_after if args.rest_after is not None else 1000.0,
            "baseline__Hz": args.baseline_hz if args.baseline_hz is not None else 0.0,
            "peak__Hz": args.peak_hz if args.peak_hz is not None else 65.0,
            "noise_std__Hz": args.noise_std if args.noise_std is not None else 1.0,
            "add_noise": not args.no_noise,
        }

        input_current__AnalogSignal = generate_trapezoid_drive(**trap_kwargs)

        # Save generated trapezoid to subdirectory
        trapezoid_file = save_path / "trapezoid_drive.pkl"
        with open(trapezoid_file, 'wb') as f:
            pickle.dump(input_current__AnalogSignal, f)
        print(f"  → Generated trapezoid parameters:")
        print(f"     Peak: {trap_kwargs['peak__Hz']:.1f} Hz, Plateau: {trap_kwargs['plateau_time__ms']:.0f} ms")
        print(f"     Rise: {trap_kwargs['rise_time__ms']:.0f} ms, Fall: {trap_kwargs['fall_time__ms']:.0f} ms")
        print(f"  → Saved to: {trapezoid_file}")
    else:
        # Load existing trapezoid
        print("Loading existing trapezoid drive pattern...")
        input_current__AnalogSignal: CURRENT__AnalogSignal = joblib.load(
            base_path / "trapezoid_drive_pattern.pkl"
        )

    ##########################################################################
    # Create Electrode Array
    ##########################################################################

    electrode = simulator.IntramuscularElectrodeArray(
        num_electrodes=4,
        inter_electrode_distance__mm=0.5,
        differentiation_mode="consecutive",
        position__mm=(0.0, 0.0, 0.0),
        orientation__rad=(-np.pi / 2, 0, -np.pi / 2),
        trajectory_distance__mm=0.125,
        trajectory_steps=1,
    )

    ##########################################################################
    # Initialize Simulator
    ##########################################################################

    print("\nInitializing iEMG simulator...")
    if MUs_to_simulate is not None:
        print(f"  → Simulating {len(MUs_to_simulate)} selected MUs: {MUs_to_simulate}")
    else:
        n_mus = len(muscle.resulting_number_of_innervated_fibers)
        print(f"  → Simulating all {n_mus} MUs")

    iemg_sim = simulator.IntramuscularEMG(
        muscle_model=muscle,
        electrode_array=electrode,
        MUs_to_simulate=MUs_to_simulate,
    )

    ##########################################################################
    # Simulate MUAPs
    ##########################################################################

    print("\nComputing motor unit action potentials...")
    iemg_sim.simulate_muaps()

    ##########################################################################
    # Simulate EMG
    ##########################################################################

    print("\nSimulating intramuscular EMG signals...")
    emg_signals = iemg_sim.simulate_intramuscular_emg(
        spike_train__Block=spike_train__Block
    )

    first_emg_signal = emg_signals.segments[0].analogsignals[0]
    print(f"Generated EMG shape: {first_emg_signal.shape}")
    print(f"  - {first_emg_signal.shape[0]} time samples")
    print(f"  - {first_emg_signal.shape[1]} electrode channels")

    ##########################################################################
    # Add Noise
    ##########################################################################

    print(f"\nAdding realistic noise (SNR = {snr_db} dB)...")
    noisy_emg_signals__Block = iemg_sim.add_noise(snr__dB=snr_db)

    ##########################################################################
    # Save Results
    ##########################################################################

    print("\nSaving results...")

    # Save EMG signals
    emg_file = save_path / "signals.pkl"
    with open(emg_file, 'wb') as f:
        pickle.dump(noisy_emg_signals__Block, f)
    print(f"✓ EMG signals: {emg_file}")

    ##########################################################################
    # Package for Decomposition
    ##########################################################################

    print("\nPackaging data for decomposition...")

    # Get MU indices
    if MUs_to_simulate is not None:
        mu_indices = MUs_to_simulate
    else:
        mu_indices = list(range(len(muscle.resulting_number_of_innervated_fibers)))

    # Extract data
    emg_signal = noisy_emg_signals__Block.segments[0].analogsignals[0]
    emg_data = emg_signal.magnitude

    muap_templates = []
    muap_durations_ms = []
    for segment in iemg_sim.muaps__Block.segments:
        muap = segment.analogsignals[0].magnitude
        muap_templates.append(muap)
        duration_ms = muap.shape[0] / iemg_sim.sampling_frequency__Hz * 1000
        muap_durations_ms.append(duration_ms)

    spike_trains_list = []
    all_spike_trains = spike_train__Block.segments[0].spiketrains
    for mu_idx in mu_indices:
        spike_times_ms = all_spike_trains[mu_idx].magnitude
        spike_trains_list.append(spike_times_ms)

    decomposition_package = {
        "emg_signal": emg_data,
        "muap_templates": muap_templates,
        "spike_trains": spike_trains_list,
        "mu_indices": mu_indices,
        "n_motor_units": len(mu_indices),
        "recruitment_thresholds": muscle.recruitment_thresholds[mu_indices],
        "muap_durations_ms": np.array(muap_durations_ms),
        "sampling_rate_hz": float(iemg_sim.sampling_frequency__Hz),
        "time_duration_s": float(emg_signal.t_stop.rescale("s").magnitude),
        "time_start_s": float(emg_signal.t_start.rescale("s").magnitude),
        "n_samples": emg_data.shape[0],
        "electrode_positions_mm": electrode.pts,
        "n_electrodes": electrode.num_electrodes,
        "inter_electrode_distance_mm": electrode.inter_electrode_distance__mm,
        "snr_db": snr_db,
        "muscle_radius_mm": muscle.radius__mm,
        "muscle_length_mm": muscle.length__mm,
        "endplate_center_percent": iemg_sim.endplate_center__percent,
        "endplate_center_mm": iemg_sim.endplate_center__mm,
    }

    decomp_file = save_path / "decomp.pkl"
    with open(decomp_file, 'wb') as f:
        pickle.dump(decomposition_package, f)
    print(f"✓ Decomposition package: {decomp_file}")

    print(f"\n✅ Simulation complete!")
    print(f"   Output directory: {save_path}")

    ##########################################################################
    # Optional Plotting
    ##########################################################################

    if not args.no_plot:
        print("\nGenerating plot...")
        
        # Extract EMG signals and time axis
        emg_signal_obj = noisy_emg_signals__Block.segments[0].analogsignals[0]
        emg_data = emg_signal_obj.magnitude  # Shape: (time, n_channels)
        emg_times = emg_signal_obj.times.rescale("s").magnitude
        
        current_signal = input_current__AnalogSignal[:, 0].magnitude
        current_times = input_current__AnalogSignal.times.rescale("s").magnitude
        
        # Normalize current for overlay
        current_normalized = (current_signal - np.min(current_signal)) / (
            np.max(current_signal) - np.min(current_signal)
        )
        
        # Determine how many channels to plot (max 4)
        n_channels = min(4, emg_data.shape[1])
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 2), sharex=True)
        if n_channels == 1:
            axes = [axes]
        
        # Plot each channel
        for i, ax in enumerate(axes):
            # Plot EMG signal
            ax.plot(emg_times, emg_data[:, i], linewidth=0.5, color="#2874A6", alpha=0.8, label="iEMG")
            
            # Overlay input current on first subplot
            if i == 0:
                ax2 = ax.twinx()
                ax2.plot(current_times, current_normalized, linewidth=1.5, color="#E67E22", 
                        alpha=0.7, label="Input Current")
                ax2.set_ylabel("Normalized Current", fontsize=9, color="#E67E22")
                ax2.tick_params(axis='y', labelcolor="#E67E22")
                ax2.set_ylim([0, 1])
                
                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
            
            # Formatting
            ax.set_ylabel(f"Ch {i + 1}\n[µV]", fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax, trim=True, offset=5)
            
            # Only show x-label on bottom subplot
            if i == len(axes) - 1:
                ax.set_xlabel("Time [s]", fontsize=10)
        
        # Overall title
        fig.suptitle(
            f"Intramuscular EMG: {len(mu_indices)} MUs, SNR={snr_db}dB",
            fontsize=14,
            fontweight="bold",
        )
        
        plt.tight_layout()
        
        plot_file = save_path / "emg_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ EMG plot saved: {plot_file}")

        ##########################################################################
        # Plot MUAPs
        ##########################################################################

        print("\nGenerating MUAP template plots...")

        # Determine how many MUs to plot (max 12)
        n_mus_to_plot = min(12, len(mu_indices))

        # Create grid layout for MUAPs
        n_cols = 3
        n_rows = int(np.ceil(n_mus_to_plot / n_cols))

        fig_muap, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.5), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        # Plot each MUAP
        for i in range(n_mus_to_plot):
            ax = axes[i]
            mu_idx = mu_indices[i]

            # Get MUAP template
            muap = muap_templates[i]  # Shape: (time, n_channels)
            muap_times = np.arange(muap.shape[0]) / iemg_sim.sampling_frequency__Hz * 1000  # ms

            # Plot all channels for this MUAP
            for ch in range(muap.shape[1]):
                ax.plot(muap_times, muap[:, ch], linewidth=1, alpha=0.7, label=f"Ch {ch+1}")

            # Formatting
            ax.set_title(f"MU {mu_idx}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Amplitude [µV]", fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax, trim=True, offset=5)

            # Add legend only to first subplot
            if i == 0:
                ax.legend(loc="upper right", fontsize=7)

            # X-labels on bottom row
            if i >= (n_rows - 1) * n_cols:
                ax.set_xlabel("Time [ms]", fontsize=9)

        # Hide unused subplots
        for i in range(n_mus_to_plot, len(axes)):
            axes[i].axis('off')

        # Overall title
        fig_muap.suptitle(
            f"Intramuscular MUAP Templates ({n_mus_to_plot} MUs)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        muap_plot_file = save_path / "muap_templates.png"
        plt.savefig(muap_plot_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ MUAP plot saved: {muap_plot_file}")

        plt.show()


if __name__ == "__main__":
    main()

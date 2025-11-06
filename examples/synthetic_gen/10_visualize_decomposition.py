r"""
Decomposition Visualization Script
===================================

This script visualizes all components from a decomposition pickle file:
- Individual EMG signal plots (one per channel/electrode)
- Individual MUAP mini plots (one per motor unit)
- Input current waveform

All plots are saved as SVG files with publication-quality styling in a 'plots/'
subfolder within the same directory as the input decomp.pkl file.

Visual style follows:
- EMG signals: plot_isi_cv_multi_muscle_comparison.py style (large, clean)
- MUAPs: plot_isi_cv_individual_mini.py style (small, minimal)

Usage:
------
python examples/synthetic_gen/10_visualize_decomposition.py \
    --decomp-file results/synthetic_gen/iemg_mu_72_snr5/decomp.pkl \
    --dpi 300
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots  # noqa
import colorsys
from matplotlib.colors import LinearSegmentedColormap

##############################################################################
# Configure Matplotlib Style
##############################################################################

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# Keep text editable in SVG/PDF exports
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Set font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Roboto", "DejaVu Sans"]

# Remove top and right spines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# Make ticks and axis lines thicker
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0

# Remove minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False

# Adjust subplot spacing
plt.rcParams["figure.subplot.left"] = 0.15
plt.rcParams["figure.subplot.bottom"] = 0.12

##############################################################################
# Configuration
##############################################################################

# Base color for motor unit gradient
MU_BASE_COLOR = "#d62728"  # MyoGen red


##############################################################################
# Utility Functions
##############################################################################


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def get_recruitment_colors(n_mus):
    """
    Generate colors using rainbow colormap for motor units.

    Parameters
    ----------
    n_mus : int
        Number of motor units.

    Returns
    -------
    np.ndarray
        Array of RGBA colors with shape (n_mus, 4).
    """
    # Use rainbow colormap
    cmap = plt.get_cmap("rainbow")

    # Normalize recruitment order to [0, 1]
    recruitment_order = np.arange(n_mus)
    norm_recruitment = recruitment_order / max(1, n_mus - 1)

    # Sample from colormap
    colors = cmap(norm_recruitment)

    return colors


##############################################################################
# Data Loading
##############################################################################


def load_decomposition(decomp_path):
    """
    Load decomposition data from pickle file.

    Parameters
    ----------
    decomp_path : Path
        Path to decomp.pkl file.

    Returns
    -------
    dict
        Decomposition data dictionary.
    """
    if not decomp_path.exists():
        raise FileNotFoundError(f"Decomposition file not found: {decomp_path}")

    try:
        decomp = joblib.load(decomp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load decomposition file: {e}")

    # Validate required keys
    required_keys = ["emg_signal", "muap_templates", "sampling_rate_hz", "mu_indices"]
    missing_keys = [key for key in required_keys if key not in decomp]
    if missing_keys:
        raise ValueError(f"Decomposition file missing required keys: {missing_keys}")

    return decomp


def load_input_current(decomp_folder):
    """
    Load input current from trapezoid_drive.pkl.

    Tries local folder first, then fallback to shared location.

    Parameters
    ----------
    decomp_folder : Path
        Folder containing decomp.pkl.

    Returns
    -------
    dict or None
        Input current data dictionary, or None if not found.
    """
    # Try local folder first
    local_path = decomp_folder / "trapezoid_drive.pkl"
    if local_path.exists():
        try:
            return joblib.load(local_path)
        except Exception as e:
            print(f"⚠️  Failed to load {local_path}: {e}")

    # Try fallback location
    fallback_path = Path("results/synthetic_gen/trapezoid_drive_pattern.pkl")
    if fallback_path.exists():
        try:
            return joblib.load(fallback_path)
        except Exception as e:
            print(f"⚠️  Failed to load {fallback_path}: {e}")

    return None


def load_spike_times_from_xml(xml_path):
    """
    Load spike timing data from emglab XML file.

    Parameters
    ----------
    xml_path : Path
        Path to XML file containing spike timing data.

    Returns
    -------
    dict
        Dictionary mapping motor unit IDs (int) to lists of spike times (float, in seconds).
        Example: {0: [0.5, 1.2, 1.8], 1: [0.7, 1.5, 2.1], ...}
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    # Parse XML by simple text processing
    # Format: <emglab_spike_events> contains lines like "time unit chan"
    spike_dict = {}
    
    try:
        with open(xml_path, 'r') as f:
            content = f.read()
        
        # Find the emglab_spike_events section
        start_tag = '<emglab_spike_events>'
        end_tag = '</emglab_spike_events>'
        
        start_idx = content.find(start_tag)
        end_idx = content.find(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not find <emglab_spike_events> tags in XML file")
        
        # Extract the data section
        data_section = content[start_idx + len(start_tag):end_idx].strip()
        
        # Parse each line: time unit chan
        for line in data_section.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            time_s = float(parts[0])
            unit_id = int(parts[1])
            # Ignore chan (parts[2]) as requested
            
            if unit_id not in spike_dict:
                spike_dict[unit_id] = []
            spike_dict[unit_id].append(time_s)
    
    except Exception as e:
        raise RuntimeError(f"Failed to parse XML file: {e}")
    
    # Sort spike times for each unit
    for unit_id in spike_dict:
        spike_dict[unit_id] = sorted(spike_dict[unit_id])
    
    return spike_dict


##############################################################################
# PNR Calculation Functions
##############################################################################


def calculate_pnr(decomp, rest_duration_s=0.5, use_signal_extraction=True):
    """
    Calculate Pulse-to-Noise Ratio (PNR) for each motor unit using DEMUSE methodology.

    Implements the DEMUSE method (Holobar et al., 2014):
    - Reconstructs each MU's pulse train by convolving spike times with MUAP template
    - Computes residual = Total EMG - MU pulse train
    - PNR (dB) = 10 * log10(power_of_pulse_train / power_of_residual)

    Also computes amplitude-based PNR for comparison using background noise.

    Parameters
    ----------
    decomp : dict
        Decomposition data dictionary containing:
        - emg_signal: noisy EMG signal
        - spike_trains: spike times for each MU
        - muap_templates: MUAP waveforms
        - sampling_rate_hz: sampling frequency
    rest_duration_s : float, optional
        Duration in seconds for background noise estimation, by default 0.5.
    use_signal_extraction : bool, optional
        If True, extract MUAPs from actual signal and compute DEMUSE PNR.
        If False, use templates (fallback, not recommended), by default True.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mu_indices': list of motor unit indices
        - 'pnr_rest_dB': DEMUSE power-based PNR in dB
        - 'pnr_snr_dB': Amplitude-based PNR in dB (for reference)
        - 'muap_amplitudes_uV': Peak-to-peak MUAP amplitudes
        - 'noise_rms_rest_uV': RMS of residual noise per MU
        - 'noise_rms_snr_uV': RMS of background noise

    Notes
    -----
    PNR ≥ 30 dB is generally considered high-quality decomposition.
    """
    emg_signal = decomp['emg_signal']
    muap_templates = decomp['muap_templates']
    mu_indices = decomp['mu_indices']
    sampling_rate_hz = decomp['sampling_rate_hz']
    snr_db = decomp['snr_db']

    # Determine EMG type and reshape if needed
    if emg_signal.ndim == 2:
        # iEMG: (n_samples, n_electrodes)
        emg_2d = emg_signal
    elif emg_signal.ndim == 3:
        # sEMG: (n_samples, rows, cols) -> flatten to (n_samples, rows*cols)
        n_samples, n_rows, n_cols = emg_signal.shape
        emg_2d = emg_signal.reshape(n_samples, n_rows * n_cols)
    else:
        raise ValueError(f"Unexpected EMG signal shape: {emg_signal.shape}")

    n_samples_total, n_channels = emg_2d.shape

    # ========================================================================
    # Method 1: Estimate noise from rest periods
    # ========================================================================
    rest_samples = int(rest_duration_s * sampling_rate_hz)

    # Extract rest periods from beginning and end
    rest_start = emg_2d[:rest_samples, :]
    rest_end = emg_2d[-rest_samples:, :]
    rest_combined = np.concatenate([rest_start, rest_end], axis=0)

    # Calculate RMS noise per channel, then average
    noise_rms_per_channel_rest = np.sqrt(np.mean(rest_combined**2, axis=0))
    noise_rms_rest = np.mean(noise_rms_per_channel_rest)

    # ========================================================================
    # Method 2: Reconstruct noise from SNR
    # ========================================================================
    # Calculate signal power per channel
    signal_power_per_channel = np.mean(emg_2d**2, axis=0)

    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power and RMS per channel
    noise_power_per_channel = signal_power_per_channel / snr_linear
    noise_rms_per_channel_snr = np.sqrt(noise_power_per_channel)
    noise_rms_snr = np.mean(noise_rms_per_channel_snr)


    # ========================================================================
    # Calculate MUAP amplitudes (peak-to-peak) per motor unit
    # ========================================================================
    muap_amplitudes = []
    pnr_rest_values = []
    pnr_snr_values = []

    spike_trains = decomp['spike_trains']

    # Extract MUAP amplitudes and calculate DEMUSE PNR
    # Use BEST ELECTRODE method for PNR calculation
    if use_signal_extraction:
        # Define window around spike for MUAP template extraction (±5ms to minimize contamination)
        window_ms = 5.0
        window_samples = int((window_ms / 1000.0) * sampling_rate_hz)

        # Store reconstructed signals for each MU
        reconstructed_signals_all_mus = []
        best_electrodes = []
        muap_templates_per_mu = []

        # ================================================================
        # Step 1: Extract MUAP templates and find best electrode for each MU
        # ================================================================
        for idx, mu_idx in enumerate(mu_indices):
            spike_times_ms = spike_trains[idx]

            if len(spike_times_ms) == 0:
                muap_amplitudes.append(0.0)
                reconstructed_signals_all_mus.append(None)
                best_electrodes.append(0)
                muap_templates_per_mu.append(None)
                continue

            # Convert spike times to sample indices
            spike_indices = (np.array(spike_times_ms) / 1000.0 * sampling_rate_hz).astype(int)

            # Extract spike-triggered average
            segments = []
            for spike_idx in spike_indices:
                start_idx = spike_idx - window_samples
                end_idx = spike_idx + window_samples
                if start_idx < 0 or end_idx >= n_samples_total:
                    continue
                segments.append(emg_2d[start_idx:end_idx, :])

            if len(segments) == 0:
                muap_amplitudes.append(0.0)
                reconstructed_signals_all_mus.append(None)
                best_electrodes.append(0)
                muap_templates_per_mu.append(None)
                continue

            # Average segments to get MUAP template
            muap_template_sta = np.mean(segments, axis=0)  # (window_samples*2, n_channels)

            # Calculate peak-to-peak per channel
            p2p_per_channel = np.max(muap_template_sta, axis=0) - np.min(muap_template_sta, axis=0)

            # Find best electrode (maximum amplitude)
            best_electrode_idx = np.argmax(p2p_per_channel)
            muap_amplitude = p2p_per_channel[best_electrode_idx]
            muap_amplitudes.append(muap_amplitude)
            best_electrodes.append(best_electrode_idx)
            muap_templates_per_mu.append(muap_template_sta)

            # Debug first MU
            if idx == 0:
                print(f"\n  DEBUG - First MU MUAP extraction:")
                print(f"    Number of spikes: {len(spike_indices)}")
                print(f"    Number of valid segments: {len(segments)}")
                print(f"    MUAP template shape: {muap_template_sta.shape}")
                print(f"    MUAP amplitude (best channel): {muap_amplitude:.6e}")
                print(f"    Best electrode index: {best_electrode_idx}")
                print(f"    MUAP template range: [{np.min(muap_template_sta):.6e}, {np.max(muap_template_sta):.6e}]")

            # ================================================================
            # Step 2: Reconstruct this MU's signal contribution
            # ================================================================
            # Directly place MUAP templates at spike times (no convolution to avoid phase issues)
            muap_waveform = muap_template_sta[:, best_electrode_idx]
            muap_length = len(muap_waveform)
            half_muap = muap_length // 2

            reconstructed_signal = np.zeros(n_samples_total)
            for spike_idx in spike_indices:
                # Place MUAP centered at spike
                start_idx = spike_idx - half_muap
                end_idx = spike_idx + half_muap

                # Handle boundary conditions
                muap_start = 0
                muap_end = muap_length
                if start_idx < 0:
                    muap_start = -start_idx
                    start_idx = 0
                if end_idx > n_samples_total:
                    muap_end = muap_length - (end_idx - n_samples_total)
                    end_idx = n_samples_total

                # Add MUAP to signal
                if start_idx >= 0 and end_idx <= n_samples_total:
                    reconstructed_signal[start_idx:end_idx] += muap_waveform[muap_start:muap_end]

            reconstructed_signals_all_mus.append(reconstructed_signal)

            # Debug first MU
            if idx == 0:
                recon_power = np.mean(reconstructed_signal**2)
                print(f"    Reconstructed signal power: {recon_power:.6e}")
                print(f"    Number of spikes placed: {len(spike_indices)}")

        # ================================================================
        # Step 3: Calculate GLOBAL residual (sum of ALL MUs removed)
        # ================================================================
        # For each electrode, sum all MU contributions and compute residual
        # Store per-electrode residuals for later use
        residual_per_electrode = {}

        for ch_idx in range(n_channels):
            # Sum all MU contributions on this channel
            total_mu_signal = np.zeros(n_samples_total)
            mu_powers_for_debug = []

            for idx, mu_idx in enumerate(mu_indices):
                if muap_templates_per_mu[idx] is None:
                    continue

                # Get spike train for this MU
                spike_times_ms = spike_trains[idx]
                spike_indices = (np.array(spike_times_ms) / 1000.0 * sampling_rate_hz).astype(int)

                # Directly place MUAPs at spike times for THIS channel
                muap_waveform_ch = muap_templates_per_mu[idx][:, ch_idx]
                muap_length = len(muap_waveform_ch)
                half_muap = muap_length // 2

                mu_signal_ch = np.zeros(n_samples_total)
                for spike_idx in spike_indices:
                    start_idx = spike_idx - half_muap
                    end_idx = spike_idx + half_muap

                    muap_start = 0
                    muap_end = muap_length
                    if start_idx < 0:
                        muap_start = -start_idx
                        start_idx = 0
                    if end_idx > n_samples_total:
                        muap_end = muap_length - (end_idx - n_samples_total)
                        end_idx = n_samples_total

                    if start_idx >= 0 and end_idx <= n_samples_total:
                        mu_signal_ch[start_idx:end_idx] += muap_waveform_ch[muap_start:muap_end]

                total_mu_signal += mu_signal_ch

                # Track power for debug
                if ch_idx == 0:
                    mu_power = np.mean(mu_signal_ch**2)
                    mu_powers_for_debug.append((mu_idx, mu_power))

            # Residual = EMG - sum of all MUs
            residual_per_electrode[ch_idx] = emg_2d[:, ch_idx] - total_mu_signal

            # Debug first electrode
            if ch_idx == 0:
                orig_power = np.mean(emg_2d[:, ch_idx]**2)
                mu_sum_power = np.mean(total_mu_signal**2)
                residual_power = np.mean(residual_per_electrode[ch_idx]**2)
                print(f"\n  DEBUG - Electrode 0 residual calculation:")
                print(f"    Original EMG power: {orig_power:.6e}")
                print(f"    Sum of all MU signals power: {mu_sum_power:.6e}")
                print(f"    Residual power: {residual_power:.6e}")
                print(f"    First 5 MU powers:")
                for mu_idx, mu_power in mu_powers_for_debug[:5]:
                    print(f"      MU {mu_idx}: {mu_power:.6e}")
                print(f"    Sum of individual MU powers: {sum(p for _, p in mu_powers_for_debug):.6e}")

        # ================================================================
        # Step 4: Calculate PNR for each MU using windowed approach
        # ================================================================
        for idx, mu_idx in enumerate(mu_indices):
            if reconstructed_signals_all_mus[idx] is None:
                pnr_rest_values.append(0.0)
                continue

            # Get spike times and signals
            spike_times_ms = spike_trains[idx]
            spike_indices = (np.array(spike_times_ms) / 1000.0 * sampling_rate_hz).astype(int)

            # This MU's reconstructed signal (on its best electrode)
            mu_pulse_train = reconstructed_signals_all_mus[idx]
            best_electrode_idx = best_electrodes[idx]

            # Global residual on this electrode (same for all MUs)
            residual_signal = residual_per_electrode[best_electrode_idx]

            # Calculate PNR using windows around each spike
            pulse_powers = []
            noise_powers = []

            # Window size = ±window_samples
            for spike_idx in spike_indices:
                # Extract window around spike
                start_idx = spike_idx - window_samples
                end_idx = spike_idx + window_samples

                # Check bounds
                if start_idx < 0 or end_idx >= n_samples_total:
                    continue

                # Extract THIS MU's reconstructed signal in this window
                pulse_window = mu_pulse_train[start_idx:end_idx]

                # Extract residual (global, with ALL MUs removed) in this window
                noise_window = residual_signal[start_idx:end_idx]

                # Compute power (mean of squared amplitudes)
                pulse_power_window = np.mean(pulse_window**2)
                noise_power_window = np.mean(noise_window**2)

                pulse_powers.append(pulse_power_window)
                noise_powers.append(noise_power_window)

            # Average powers across all spikes
            if len(pulse_powers) > 0:
                avg_pulse_power = np.mean(pulse_powers)
                avg_noise_power = np.mean(noise_powers)

                # Compute PNR in dB: 10 * log10(avg_pulse_power / avg_noise_power)
                if avg_noise_power > 0 and avg_pulse_power > 0:
                    pnr_db = 10 * np.log10(avg_pulse_power / avg_noise_power)
                else:
                    pnr_db = 0.0

                # Debug first MU
                if idx == 0:
                    print(f"\n  DEBUG - DEMUSE PNR for first MU:")
                    print(f"    Number of spikes: {len(spike_indices)}")
                    print(f"    Number of valid windows: {len(pulse_powers)}")
                    print(f"    Avg pulse power (MU reconstructed signal): {avg_pulse_power:.6e}")
                    print(f"    Avg noise power (global residual): {avg_noise_power:.6e}")
                    print(f"    PNR: {pnr_db:.1f} dB")
            else:
                pnr_db = 0.0

            pnr_rest_values.append(pnr_db)

        # Store global residual noise RMS (average across electrodes)
        global_residual_rms = np.mean([np.sqrt(np.mean(res**2)) for res in residual_per_electrode.values()])
        noise_rms_per_mu = [global_residual_rms] * len(mu_indices)

        # Also calculate SNR-based PNR for comparison (using background noise)
        for idx, muap_amplitude in enumerate(muap_amplitudes):
            # SNR-based uses amplitude ratio with background noise
            if noise_rms_snr > 0:
                pnr_snr_linear = muap_amplitude / noise_rms_snr
                pnr_snr_dB = 20 * np.log10(pnr_snr_linear)
            else:
                pnr_snr_dB = 0.0
            pnr_snr_values.append(pnr_snr_dB)

        # Convert to array for easier processing
        noise_rms_per_mu = np.array(noise_rms_per_mu)
    else:
        # Fallback: use templates (not recommended)
        noise_rms_per_mu = []
        for idx, mu_idx in enumerate(mu_indices):
            muap = muap_templates[mu_idx]

            # Flatten electrodes for consistent processing
            if muap.ndim == 2:
                muap_2d = muap
            elif muap.ndim == 3:
                n_s, n_r, n_c = muap.shape
                muap_2d = muap.reshape(n_s, n_r * n_c)

            # Calculate peak-to-peak amplitude per electrode
            p2p_per_electrode = np.max(muap_2d, axis=0) - np.min(muap_2d, axis=0)
            mean_muap_amplitude = np.mean(p2p_per_electrode)
            muap_amplitudes.append(mean_muap_amplitude)

            # Fallback PNR calculation
            pnr_rest_dB = 20 * np.log10(mean_muap_amplitude / noise_rms_rest) if noise_rms_rest > 0 else 0.0
            pnr_snr_dB = 20 * np.log10(mean_muap_amplitude / noise_rms_snr) if noise_rms_snr > 0 else 0.0

            pnr_rest_values.append(pnr_rest_dB)
            pnr_snr_values.append(pnr_snr_dB)
            noise_rms_per_mu.append(noise_rms_rest)

        noise_rms_per_mu = np.array(noise_rms_per_mu)

    return {
        'mu_indices': mu_indices,
        'pnr_rest_dB': np.array(pnr_rest_values),
        'pnr_snr_dB': np.array(pnr_snr_values),
        'muap_amplitudes_uV': np.array(muap_amplitudes),
        'noise_rms_rest_uV': noise_rms_rest,
        'noise_rms_snr_uV': noise_rms_snr,
    }


def save_pnr_to_csv(pnr_data, output_path):
    """
    Save PNR data to CSV file.

    Parameters
    ----------
    pnr_data : dict
        PNR data dictionary from calculate_pnr().
    output_path : Path
        Output CSV file path.
    """
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'motor_unit_index',
            'pnr_demuse_dB',
            'pnr_amplitude_based_dB',
            'muap_peak_to_peak_amplitude',
            'residual_noise_rms',
            'background_noise_rms'
        ])

        # Write data rows
        for i, mu_idx in enumerate(pnr_data['mu_indices']):
            writer.writerow([
                mu_idx,
                f"{pnr_data['pnr_rest_dB'][i]:.2f}",
                f"{pnr_data['pnr_snr_dB'][i]:.2f}",
                f"{pnr_data['muap_amplitudes_uV'][i]:.6f}",
                f"{pnr_data['noise_rms_rest_uV']:.6f}",
                f"{pnr_data['noise_rms_snr_uV']:.6f}"
            ])


##############################################################################
# Plotting Functions - EMG Signals
##############################################################################


def plot_emg_signal_channel(signal_data, time_vector, channel_idx, output_path):
    """
    Plot a single EMG channel signal.

    Parameters
    ----------
    signal_data : np.ndarray
        EMG signal for this channel (1D array).
    time_vector : np.ndarray
        Time vector in seconds.
    channel_idx : int
        Channel index for labeling.
    output_path : Path
        Output SVG file path.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot signal
    ax.plot(time_vector, signal_data, color="#1f77b4", linewidth=0.5, alpha=0.8)

    # Format axes
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (μV)", fontsize=12)
    ax.set_title(f"EMG Signal - Channel {channel_idx}", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    sns.despine(ax=ax, offset=10, trim=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_all_emg_signals(decomp, output_folder):
    """
    Plot all individual EMG signal channels.

    Parameters
    ----------
    decomp : dict
        Decomposition data dictionary.
    output_folder : Path
        Output folder for signal plots.

    Returns
    -------
    int
        Number of plots created.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    emg_signal = decomp["emg_signal"]
    sampling_rate_hz = decomp["sampling_rate_hz"]

    # Create time vector
    n_samples = emg_signal.shape[0]
    time_vector = np.arange(n_samples) / sampling_rate_hz

    # Detect EMG type (iEMG vs sEMG)
    if emg_signal.ndim == 2:
        # iEMG: (n_samples, n_electrodes)
        n_channels = emg_signal.shape[1]
        signals_2d = emg_signal
    elif emg_signal.ndim == 3:
        # sEMG: (n_samples, rows, cols) - flatten to (n_samples, rows*cols)
        n_rows, n_cols = emg_signal.shape[1], emg_signal.shape[2]
        n_channels = n_rows * n_cols
        signals_2d = emg_signal.reshape(n_samples, n_channels)
    else:
        raise ValueError(f"Unexpected EMG signal shape: {emg_signal.shape}")

    # Plot each channel
    print(f"\n📊 Creating {n_channels} EMG signal plots...")
    for ch_idx in range(n_channels):
        output_path = output_folder / f"channel_{ch_idx:02d}.svg"
        plot_emg_signal_channel(
            signals_2d[:, ch_idx], time_vector, ch_idx, output_path
        )

    return n_channels


##############################################################################
# Plotting Functions - MUAPs
##############################################################################


def plot_muap_mini_iemg(muap_template, mu_idx, sampling_rate_hz, color, output_path, mu_indices, global_ylim, pnr_value=None):
    """
    Plot a single MUAP for iEMG in mini style.

    Parameters
    ----------
    muap_template : np.ndarray
        MUAP template with shape (n_samples, n_electrodes).
    mu_idx : int
        Motor unit index for labeling.
    sampling_rate_hz : float
        Sampling rate in Hz.
    color : tuple
        RGBA color tuple.
    output_path : Path
        Output SVG file path.
    mu_indices : list
        List of all motor unit indices (for debug check).
    global_ylim : tuple
        Global (y_min, y_max) limits for consistent scaling across all MUAPs.
    pnr_value : float, optional
        Pulse-to-Noise Ratio in dB to display in title.
    """
    fig, ax = plt.subplots(figsize=(4, 3))  # Larger for better visibility

    n_samples, n_electrodes = muap_template.shape
    time_ms = np.arange(n_samples) / sampling_rate_hz * 1000

    # Plot all electrodes overlaid
    # Convert color to hex for matplotlib
    if isinstance(color, (list, tuple, np.ndarray)):
        color_rgb = tuple(color[:3])
    else:
        color_rgb = color

    # Debug: print color for first MU
    if mu_idx == mu_indices[0]:
        print(f"  Debug - First MU color (RGB): {color_rgb}")

    for e_idx in range(n_electrodes):
        ax.plot(
            time_ms,
            muap_template[:, e_idx],
            color=color_rgb,
            linewidth=2.0,
            alpha=1.0
        )

    # Format axes with explicit limits
    ax.set_xlim(time_ms[0], time_ms[-1])

    # Use GLOBAL y-axis limits for consistent scaling
    ax.set_ylim(global_ylim[0], global_ylim[1])

    # Add labels for clarity
    ax.set_xlabel("Time (ms)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    sns.despine(ax=ax, offset=5, trim=True)

    # Add title with PNR if available
    if pnr_value is not None:
        ax.set_title(f"MU {mu_idx} | PNR: {pnr_value:.1f} dB", fontsize=12)
    else:
        ax.set_title(f"MU {mu_idx}", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_muap_mini_semg(muap_template, mu_idx, sampling_rate_hz, color, output_path, pnr_value=None):
    """
    Plot a single MUAP for sEMG in mini style (grid of waveforms).

    Parameters
    ----------
    muap_template : np.ndarray
        MUAP template with shape (n_samples, rows, cols).
    mu_idx : int
        Motor unit index for labeling.
    sampling_rate_hz : float
        Sampling rate in Hz.
    color : tuple
        RGBA color tuple.
    output_path : Path
        Output SVG file path.
    pnr_value : float, optional
        Pulse-to-Noise Ratio in dB to display in title.
    """
    n_samples, n_rows, n_cols = muap_template.shape

    # Convert color to RGB
    if isinstance(color, (list, tuple, np.ndarray)):
        color_rgb = tuple(color[:3])
    else:
        color_rgb = color

    # Create figure with grid of subplots matching electrode layout
    # Scale figure size based on grid dimensions
    fig_width = max(8, n_cols * 1.5)
    fig_height = max(6, n_rows * 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height),
                             sharex=True, sharey=True)

    # Handle single electrode case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Create time vector in milliseconds
    time_ms = np.arange(n_samples) / sampling_rate_hz * 1000

    # Find global amplitude limits for consistent scaling across all subplots
    global_min = muap_template.min()
    global_max = muap_template.max()
    y_margin = (global_max - global_min) * 0.1

    # Crop to fixed 40ms window centered on MUAP
    # Find center of active MUAP region (>5% of max amplitude)
    threshold = 0.05 * np.max(np.abs(muap_template))
    active_mask = np.any(np.abs(muap_template) > threshold, axis=(1, 2))

    if np.any(active_mask):
        # Find center of active region
        active_indices = np.where(active_mask)[0]
        center_idx = (active_indices[0] + active_indices[-1]) // 2

        # Calculate 40ms window (convert to samples)
        window_duration_ms = 40.0
        window_samples = int((window_duration_ms / 1000.0) * sampling_rate_hz)
        half_window = window_samples // 2

        # Center window on MUAP, with bounds checking
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window

        # Shift window if it extends beyond template boundaries
        if start_idx < 0:
            end_idx -= start_idx
            start_idx = 0
        if end_idx >= n_samples:
            start_idx -= (end_idx - n_samples + 1)
            end_idx = n_samples - 1

        # Final bounds check
        start_idx = max(0, start_idx)
        end_idx = min(n_samples - 1, end_idx)

        # Crop arrays to 40ms window
        muap_template = muap_template[start_idx:end_idx+1, :, :]
        time_ms = time_ms[start_idx:end_idx+1]
        n_samples = len(time_ms)  # Update sample count

    # Plot waveform for each electrode in grid layout
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            waveform = muap_template[:, row, col]

            # Plot waveform with MU color
            ax.plot(time_ms, waveform, color=color_rgb, linewidth=1.5, alpha=1.0)

            # Set consistent limits
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
            ax.set_xlim(time_ms[0], time_ms[-1])

            # Minimal styling - remove ticks for clean grid appearance
            ax.set_xticks([])
            ax.set_yticks([])

            # Keep spines but make them thin
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color('gray')
                spine.set_alpha(0.3)

    # Add shared axis labels for the entire grid
    fig.text(0.5, 0.02, 'Time (ms)', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=12)

    # Add title with MU index and PNR if available
    if pnr_value is not None:
        fig.suptitle(f'MU {mu_idx} - Surface EMG Grid | PNR: {pnr_value:.1f} dB', fontsize=14, y=0.98)
    else:
        fig.suptitle(f'MU {mu_idx} - Surface EMG Grid', fontsize=14, y=0.98)

    # Adjust subplot spacing for compact grid
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94,
                       hspace=0.05, wspace=0.05)

    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_all_muaps(decomp, output_folder):
    """
    Plot all individual MUAP templates in mini style.

    Parameters
    ----------
    decomp : dict
        Decomposition data dictionary.
    output_folder : Path
        Output folder for MUAP plots.

    Returns
    -------
    int
        Number of plots created.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    muap_templates = decomp["muap_templates"]
    mu_indices = decomp["mu_indices"]
    sampling_rate_hz = decomp["sampling_rate_hz"]
    n_mus = len(muap_templates)
    n_selected = len(mu_indices)

    # Get recruitment gradient colors for selected MUs
    colors = get_recruitment_colors(n_selected)

    # Detect MUAP type (iEMG vs sEMG)
    first_muap = muap_templates[0]
    is_iemg = first_muap.ndim == 2  # (n_samples, n_electrodes)
    is_semg = first_muap.ndim == 3  # (n_samples, rows, cols)

    if not is_iemg and not is_semg:
        raise ValueError(f"Unexpected MUAP template shape: {first_muap.shape}")

    print(f"\n📊 Creating {n_selected} MUAP mini plots (from {n_mus} total templates)...")

    # Debug first MUAP BEFORE normalization
    if n_selected > 0:
        first_muap = muap_templates[0]
        print(f"  Debug - First MUAP shape: {first_muap.shape}")
        print(f"  Debug - First MUAP range (raw): [{first_muap.min():.6f}, {first_muap.max():.6f}]")

    # NORMALIZE each MUAP individually to [-1, 1] to see shapes clearly
    muap_templates_normalized = []
    for muap in muap_templates:
        max_abs = np.max(np.abs(muap))
        if max_abs > 0:
            muap_templates_normalized.append(muap / max_abs)
        else:
            muap_templates_normalized.append(muap)

    print(f"  Normalized each MUAP individually to [-1, 1]")

    # Debug first MUAP AFTER normalization
    if n_mus > 0:
        first_muap_norm = muap_templates_normalized[0]
        print(f"  Debug - First MUAP range (normalized): [{first_muap_norm.min():.6f}, {first_muap_norm.max():.6f}]")

    # Use FIXED y-axis limits for all plots: [-1, 1] with padding
    global_ylim = (-1.15, 1.15)
    print(f"  Fixed y-axis limits for all plots: [{global_ylim[0]:.2f}, {global_ylim[1]:.2f}]")

    # Calculate PNR for all motor units (using DEMUSE power-based method)
    print(f"  Calculating PNR values...")
    pnr_data = calculate_pnr(decomp)

    # Create lookup dictionary for PNR values (use DEMUSE method for plots)
    pnr_lookup = {mu_idx: pnr_val for mu_idx, pnr_val in zip(pnr_data['mu_indices'], pnr_data['pnr_rest_dB'])}

    # Use NORMALIZED templates for plotting (only selected MUs)
    for idx, mu_idx in enumerate(mu_indices):
        muap = muap_templates_normalized[mu_idx]
        output_path = output_folder / f"mu_{mu_idx:03d}.svg"
        color = colors[idx]
        pnr_value = pnr_lookup.get(mu_idx, None)

        if is_iemg:
            plot_muap_mini_iemg(muap, mu_idx, sampling_rate_hz, color, output_path, mu_indices, global_ylim, pnr_value)
        elif is_semg:
            plot_muap_mini_semg(muap, mu_idx, sampling_rate_hz, color, output_path, pnr_value)

    return n_selected, pnr_data


##############################################################################
# Plotting Functions - Input Current
##############################################################################


def plot_input_current(current_data, output_path):
    """
    Plot input current waveform.

    Parameters
    ----------
    current_data : neo.AnalogSignal or dict
        Input current data (Neo AnalogSignal or dictionary with current/time).
    output_path : Path
        Output SVG file path.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle Neo AnalogSignal objects
    if hasattr(current_data, 'magnitude'):
        # It's a Neo AnalogSignal
        current = current_data.magnitude.flatten()
        time_s = current_data.times.rescale('s').magnitude
    else:
        # It's a dictionary
        # Extract current and time data
        if "current__matrix" in current_data:
            current_matrix = current_data["current__matrix"]
            # Sum across motor units if multiple
            if current_matrix.ndim == 2:
                current = current_matrix.sum(axis=1)
            else:
                current = current_matrix
        elif "current" in current_data:
            current = current_data["current"]
        else:
            raise ValueError("Input current data missing 'current__matrix' or 'current' key")

        if "time__s" in current_data:
            time_s = current_data["time__s"]
        elif "time" in current_data:
            time_s = current_data["time"]
        else:
            # Create time vector from length
            time_s = np.arange(len(current)) / 1000.0  # Assume 1kHz if not specified

    # Plot
    ax.plot(time_s, current, color="#2ca02c", linewidth=2.0)

    # Format axes
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Current (nA)", fontsize=12)
    ax.set_title("Input Current - Trapezoid Drive", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    sns.despine(ax=ax, offset=10, trim=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)


##############################################################################
# Plotting Functions - Spike Raster
##############################################################################


def plot_spike_train_single(spike_times, mu_idx, color, output_path, duration_s):
    """
    Plot a single motor unit's spike train.

    Parameters
    ----------
    spike_times : list
        List of spike times in seconds.
    mu_idx : int
        Motor unit index for labeling.
    color : tuple
        RGBA color tuple.
    output_path : Path
        Output SVG file path.
    duration_s : float
        Total duration of recording in seconds.
    """
    fig, ax = plt.subplots(figsize=(10, 2))

    # Convert color to RGB
    if isinstance(color, (list, tuple, np.ndarray)):
        color_rgb = tuple(color[:3])
    else:
        color_rgb = color

    # Plot spikes as dots at y=0 with colored edges only
    y_positions = np.zeros(len(spike_times))
    ax.scatter(spike_times, y_positions, facecolors='none', edgecolors=color_rgb,
               s=30, linewidths=1.5, marker='o')

    # Format axes
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("", fontsize=10)
    ax.set_title(f"MU {mu_idx} Spike Train", fontsize=12)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.5, 0.5)
    
    # Remove y-axis ticks since there's only one row
    ax.set_yticks([])
    
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis='x')
    sns.despine(ax=ax, left=True, offset=5, trim=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_all_spike_trains(spike_dict, mu_indices, output_folder, duration_s):
    """
    Plot individual spike trains for each motor unit.

    Parameters
    ----------
    spike_dict : dict
        Dictionary mapping motor unit IDs to lists of spike times (in seconds).
    mu_indices : list
        List of motor unit indices to plot (from original motor pool).
    output_folder : Path
        Output folder for spike train plots.
    duration_s : float
        Total duration of recording in seconds.

    Returns
    -------
    int
        Number of plots created.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get rainbow colors for motor units
    n_mus = len(mu_indices)
    colors = get_recruitment_colors(n_mus)

    # The XML file uses sequential unit IDs (1, 2, 3, ...) for the selected MUs
    xml_unit_ids = sorted(spike_dict.keys())
    
    if len(xml_unit_ids) != n_mus:
        print(f"⚠️  Warning: Number of units in XML ({len(xml_unit_ids)}) != selected MUs ({n_mus})")
        print(f"    XML units: {xml_unit_ids}")
        print(f"    Expected MU indices: {mu_indices}")

    print(f"  Creating {n_mus} decomposed spike train plots...")
    
    # Map XML unit IDs to mu_indices positions and plot each one
    plots_created = 0
    for idx in range(min(len(xml_unit_ids), n_mus)):
        xml_unit_id = xml_unit_ids[idx]
        mu_idx = mu_indices[idx]
        
        if xml_unit_id not in spike_dict:
            continue
        
        spike_times = spike_dict[xml_unit_id]
        if len(spike_times) == 0:
            continue
        
        # Get color for this MU
        color = colors[idx]

        # Create output file with "decomposed" prefix and XML unit number
        output_path = output_folder / f"spike_train_decomposed_unit_{xml_unit_id:03d}.svg"

        # Plot this spike train (using xml_unit_id for the label)
        plot_spike_train_single(spike_times, xml_unit_id, color, output_path, duration_s)
        plots_created += 1

    return plots_created


def plot_spike_trains_from_decomp(decomp, output_folder):
    """
    Plot spike trains from decomposition file (ground truth).

    Parameters
    ----------
    decomp : dict
        Decomposition data dictionary containing spike_trains.
    output_folder : Path
        Output folder for spike train plots.

    Returns
    -------
    int
        Number of plots created.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    spike_trains = decomp['spike_trains']
    mu_indices = decomp['mu_indices']
    duration_s = decomp['time_duration_s']
    
    # Get rainbow colors for motor units
    n_mus = len(mu_indices)
    colors = get_recruitment_colors(n_mus)

    print(f"  Creating {n_mus} ground truth spike train plots...")
    
    plots_created = 0
    for idx, mu_idx in enumerate(mu_indices):
        spike_times_ms = spike_trains[idx]
        
        if len(spike_times_ms) == 0:
            continue
        
        # Convert from milliseconds to seconds
        spike_times_s = spike_times_ms / 1000.0
        
        # Get color for this MU
        color = colors[idx]
        
        # Create output file with "simulated" prefix to indicate ground truth
        output_path = output_folder / f"spike_train_simulated_mu_{mu_idx:03d}.svg"
        
        # Plot this spike train
        plot_spike_train_single(spike_times_s, mu_idx, color, output_path, duration_s)
        plots_created += 1

    return plots_created


##############################################################################
# Main Function
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize decomposition pickle file with publication-quality plots"
    )
    parser.add_argument(
        "--decomp-file",
        type=Path,
        required=True,
        help="Path to decomp.pkl file",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output figures (default: 300)",
    )

    args = parser.parse_args()

    # Update DPI if specified
    if args.dpi != 300:
        plt.rcParams["figure.dpi"] = args.dpi
        plt.rcParams["savefig.dpi"] = args.dpi

    print("=" * 80)
    print("Decomposition Visualization")
    print("=" * 80)
    print(f"Input: {args.decomp_file}")

    # Load decomposition data
    print("\n📂 Loading decomposition data...")
    decomp = load_decomposition(args.decomp_file)

    # Print summary
    n_mus = len(decomp["muap_templates"])
    n_samples = decomp["emg_signal"].shape[0]
    sampling_rate_hz = decomp["sampling_rate_hz"]
    duration_s = n_samples / sampling_rate_hz

    if decomp["emg_signal"].ndim == 2:
        n_electrodes = decomp["emg_signal"].shape[1]
        emg_type = "iEMG"
    else:
        n_rows, n_cols = decomp["emg_signal"].shape[1], decomp["emg_signal"].shape[2]
        n_electrodes = n_rows * n_cols
        emg_type = "sEMG"

    print(f"  Type: {emg_type}")
    print(f"  Motor units: {n_mus}")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Duration: {duration_s:.2f} s")
    print(f"  Sampling rate: {sampling_rate_hz:.0f} Hz")

    # Create output folders
    decomp_folder = args.decomp_file.parent
    plots_folder = decomp_folder / "plots"
    signals_folder = plots_folder / "signals"
    muaps_folder = plots_folder / "muaps"

    # Plot EMG signals
    n_signal_plots = plot_all_emg_signals(decomp, signals_folder)
    print(f"✅ Created {n_signal_plots} signal plots in {signals_folder.relative_to(decomp_folder)}/")

    # Plot MUAPs and calculate PNR
    n_muap_plots, pnr_data = plot_all_muaps(decomp, muaps_folder)
    print(f"✅ Created {n_muap_plots} MUAP plots in {muaps_folder.relative_to(decomp_folder)}/")

    # Save PNR data to CSV
    pnr_csv_path = plots_folder / "pnr_values.csv"
    save_pnr_to_csv(pnr_data, pnr_csv_path)
    print(f"✅ Saved PNR values to {pnr_csv_path.relative_to(decomp_folder)}")

    # Print PNR summary
    print(f"\n📊 PNR Summary:")
    print(f"   DEMUSE method (power-based): {pnr_data['pnr_rest_dB'].mean():.1f} ± {pnr_data['pnr_rest_dB'].std():.1f} dB (range: {pnr_data['pnr_rest_dB'].min():.1f} - {pnr_data['pnr_rest_dB'].max():.1f} dB)")
    print(f"   Amplitude-based (for reference): {pnr_data['pnr_snr_dB'].mean():.1f} ± {pnr_data['pnr_snr_dB'].std():.1f} dB (range: {pnr_data['pnr_snr_dB'].min():.1f} - {pnr_data['pnr_snr_dB'].max():.1f} dB)")

    # Count high-quality MUs (PNR >= 30 dB)
    high_quality_count = np.sum(pnr_data['pnr_rest_dB'] >= 30.0)
    print(f"   High-quality MUs (PNR ≥ 30 dB): {high_quality_count} / {len(pnr_data['mu_indices'])}")

    # Plot input current
    print("\n📂 Loading input current...")
    current_data = load_input_current(decomp_folder)
    if current_data is not None:
        output_path = plots_folder / "input_current.svg"
        try:
            plot_input_current(current_data, output_path)
            print(f"✅ Created input current plot: {output_path.relative_to(decomp_folder)}")
        except Exception as e:
            print(f"⚠️  Failed to plot input current: {e}")
    else:
        print("⚠️  Input current file not found (skipped)")

    # Plot ground truth spike trains from decomp file
    print("\n📊 Plotting ground truth spike trains...")
    spikes_folder = plots_folder / "spikes"
    try:
        n_gt_plots = plot_spike_trains_from_decomp(decomp, spikes_folder)
        print(f"✅ Created {n_gt_plots} ground truth spike train plots in {spikes_folder.relative_to(decomp_folder)}/")
    except Exception as e:
        print(f"⚠️  Failed to plot ground truth spike trains: {e}")

    # Plot decomposed spike trains from XML
    print("\n📂 Loading decomposed spike timing data...")
    # Look for XML file with same base name as decomp folder
    xml_path = decomp_folder / f"{decomp_folder.name}.xml"

    if xml_path.exists():
        try:
            spike_dict = load_spike_times_from_xml(xml_path)
            mu_indices = decomp["mu_indices"]

            # Create spikes output folder
            spikes_folder = plots_folder / "spikes"

            n_spike_plots = plot_all_spike_trains(spike_dict, mu_indices, spikes_folder, duration_s)
            print(f"✅ Created {n_spike_plots} decomposed spike train plots in {spikes_folder.relative_to(decomp_folder)}/")
        except Exception as e:
            print(f"⚠️  Failed to plot decomposed spike trains: {e}")
    else:
        print(f"⚠️  Spike timing XML file not found: {xml_path.name} (skipped)")

    # Final summary
    print("\n" + "=" * 80)
    print("✅ All visualizations saved!")
    print(f"   Output folder: {plots_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()

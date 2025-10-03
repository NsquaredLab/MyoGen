"""
Paper Watanabe
=====================================================
"""
##############################################################################
# Import Libraries
# ----------------
#

from pathlib import Path

import numpy as np
import joblib
import matplotlib.pyplot as plt
import neo
import elephant
from scipy import signal as scipy_signal
from tqdm import tqdm

import quantities as pq

from myogen import load_nmodl_mechanisms, simulator
from myogen.utils.plotting import plot_membrane_potentials, plot_raster_spikes
from myogen.simulator.core.force.force_model import ForceModel

##############################################################################
# Load NEURON Mechanisms and Dependencies
# ---------------------------------------
#
# Load the compiled NMODL mechanisms required for biophysical neuron modeling
# and load results from previous examples that serve as inputs to this simulation.

# Load NEURON mechanisms
load_nmodl_mechanisms()

# Setup results directory
save_path = Path("./results")
save_path.mkdir(exist_ok=True)

dt = 0.1  # ms - Integration timestep
tstop = 180 * 1e3  # ms - Total simulation duration
n_steps = int(tstop / dt)
# Add margin for NEURON's potential overstep
time = np.linspace(0, tstop, n_steps + 100)

print("Simulation parameters:")
print(f"  - Duration: {tstop} ms")
print(f"  - Timestep: {dt} ms")
print(f"  - Time samples: {len(time)}")

##############################################################################
# Define Neural Population Sizes
# -----------------------------
#
# Population sizes are based on physiological estimates from cat and human studies.
# These numbers represent typical motor pool compositions for a single muscle.

# Motor neurons (output to muscles)
naMN = 100

# Descending drive (cortical input)
nDD = 400  # Total descending drive neurons
DDorder = 1  # Poisson process order for realistic spike patterns

time_s = time / 1000.0  # Convert to seconds for easier calculation

# Initialize with constant value
DDdrive = np.full_like(time, 65.0)

# Phase 2: 20 Hz sinusoid with DC=65, amplitude=20 from 60-120s
phase2_mask = (time_s >= 60) & (time_s < 120)
DDdrive[phase2_mask] = 65 + 20 * np.sin(2 * np.pi * 20 * time_s[phase2_mask])

# Phase 3: Same sinusoid from 120-180s but DC=58
phase3_mask = (time_s >= 120) & (time_s <= 180)
DDdrive[phase3_mask] = 58 + 20 * np.sin(2 * np.pi * 20 * time_s[phase3_mask])

# The time array now includes margin, so DDdrive is automatically padded
# Set extra points beyond 180s to maintain the last phase value (DC=58)
beyond_180_mask = time_s > 180

##############################################################################
# DDdrive Power Spectrum Analysis
# ---------------------------------
#
# Compute and display power spectra of DDdrive for the three time windows

# Define time windows for analysis
time_windows = [(3, 60), (63, 120), (123, 180)]
# Colors matching the paper figures
window_colors = ["#0001f9", "#966562", "#2efe37"]

# Create figure with 3 side-by-side subplots
fig_dd, axes_dd = plt.subplots(1, 3, figsize=(18, 5))

for idx, (t_start, t_stop) in enumerate(time_windows):
    # Extract DDdrive for the time window (convert seconds to ms)
    start_idx = int(t_start * 1000 / dt)
    stop_idx = int(t_stop * 1000 / dt)
    DDdrive_windowed = DDdrive[start_idx:stop_idx]

    # Remove linear trend
    DDdrive_detrended = scipy_signal.detrend(DDdrive_windowed, type="linear")

    # Compute power spectrum using Welch's method with full signal as segment
    sampling_rate_dd = 1000.0 / dt  # Hz (dt is in ms)
    signal_length = len(DDdrive_detrended)

    # Use the full signal length as nperseg for maximum resolution
    freqs_dd, psd_dd = scipy_signal.welch(
        DDdrive_detrended,
        fs=sampling_rate_dd,
        window="hamming",
        nperseg=signal_length,
        noverlap=0,
        nfft=signal_length,
    )

    # Normalize power spectrum between 0 and 1
    psd_dd_normalized = (psd_dd - np.min(psd_dd)) / (np.max(psd_dd) - np.min(psd_dd))

    # Plot normalized power spectrum with filled area
    axes_dd[idx].fill_between(
        freqs_dd, psd_dd_normalized, color=window_colors[idx], alpha=0.7
    )
    axes_dd[idx].plot(
        freqs_dd, psd_dd_normalized, color=window_colors[idx], linewidth=1.5
    )
    axes_dd[idx].set_xlabel("Frequency (Hz)", fontsize=12)
    axes_dd[idx].set_ylabel("Conductance Spectrum (A.U.)", fontsize=12)
    axes_dd[idx].set_xlim(0, 25)
    axes_dd[idx].set_ylim(0, 1.05)
    axes_dd[idx].spines["top"].set_visible(False)
    axes_dd[idx].spines["right"].set_visible(False)
    # Add panel label (A, B, C)
    axes_dd[idx].text(
        -0.15,
        1.05,
        chr(65 + idx),
        transform=axes_dd[idx].transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )

plt.tight_layout()
plt.savefig(
    save_path / "DDdrive_power_spectra_all_windows.png", dpi=150, bbox_inches="tight"
)
plt.show()
print("DDdrive power spectra computed and plotted for all three time windows")


with open(
    r"C:\Users\raulc\Research\papers_server\in_progress\simulator\spinal_network_results.pkl",
    "rb",
) as f:
    results: neo.Block = joblib.load(f)
    print("Previous results loaded successfully")

aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

# Define time windows for analysis
time_windows = [(3, 60), (63, 120), (123, 180)]

##############################################################################
# Compute Force Signal for Full Duration (Panel G)
# -------------------------------------------------
#
# Calculate force using the Fuglevand ForceModel

print("\nComputing force signal from motor neuron activity...")

# Generate recruitment thresholds for motor neuron pool
recruitment_thresholds, _ = simulator.RecruitmentThresholds(
    N=naMN,
    recruitment_range__ratio=50,  # Physiological recruitment range
    mode="combined",
    deluca__slope=5,
)

# Create force model with low sampling rate for efficiency
# Using vectorized version for better performance
force_model = ForceModel(
    recruitment_thresholds=recruitment_thresholds,
    recording_frequency__Hz=100,  # 100 Hz sampling rate (sufficient for force, reduces memory)
    longest_duration_rise_time__ms=90.0,  # Maximum twitch rise time
    contraction_time_range__unitless=3,  # Contraction time range factor
)

print("Force model created:")
print(f"  - Number of motor units: {force_model._number_of_neurons}")
print(f"  - Peak force range: {force_model.peak_twitch_forces__unitless[0]:.3f} - {force_model.peak_twitch_forces__unitless[-1]:.3f}")

# Generate force from spike trains (only aMN segment)
# Create a new Block containing only aMN spikes
# Trim spike trains to exact duration to avoid timing issues
aMN_block = neo.Block()
aMN_segment_trimmed = neo.Segment(name="aMN")

# Trim each spike train to exact duration
for st in aMN_spikes:
    st_trimmed = st.time_slice(0 * pq.ms, tstop * pq.ms)
    # Manually set t_stop to exact value to avoid rounding issues
    st_trimmed.t_stop = tstop * pq.ms
    aMN_segment_trimmed.spiketrains.append(st_trimmed)

aMN_block.segments.append(aMN_segment_trimmed)

force_output = force_model.generate_force(spike_train__Block=aMN_block)

# Extract force signal and time vector
force_signal = force_output.magnitude[:, 0]
time_spikes = force_output.times.rescale("s").magnitude
sampling_rate_spikes = force_output.sampling_rate.rescale("Hz").magnitude

# Scale force signal to reasonable range (e.g., 0-500 N)
force_signal = (force_signal / np.max(force_signal)) * 500 if np.max(force_signal) > 0 else force_signal

print(f"Force signal computed (length: {len(force_signal)} samples)")

# Create Panel G: Force over time with color-coded intervals
fig_force, ax_force = plt.subplots(1, 1, figsize=(15, 4))

# Plot force signal in segments with different colors
for window_idx, (t_start, t_stop) in enumerate(time_windows):
    start_idx = int(t_start * sampling_rate_spikes)
    stop_idx = int(t_stop * sampling_rate_spikes)
    time_segment = time_spikes[start_idx:stop_idx]
    force_segment = force_signal[start_idx:stop_idx]

    ax_force.plot(
        time_segment, force_segment, color=window_colors[window_idx], linewidth=1.5
    )

ax_force.set_xlabel("Time (s)", fontsize=12)
ax_force.set_ylabel("Force (N)", fontsize=12)
ax_force.set_xlim(0, 180)
ax_force.spines["top"].set_visible(False)
ax_force.spines["right"].set_visible(False)
# Add panel label G
ax_force.text(
    -0.05,
    1.05,
    "G",
    transform=ax_force.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

plt.tight_layout()
plt.savefig(save_path / "watanabe_force_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("Force timeseries plot created (Panel G)")

##############################################################################
# Create Full Duration Raster Plot (Panel H)
# -------------------------------------------
#
# Display motor neuron spikes across the entire simulation with color-coded backgrounds

print("\nCreating full duration raster plot (Panel H)...")

fig_raster, ax_raster = plt.subplots(1, 1, figsize=(15, 6))

# Add color-coded background regions
for window_idx, (t_start, t_stop) in enumerate(time_windows):
    ax_raster.axvspan(t_start, t_stop, facecolor=window_colors[window_idx], alpha=0.3)

# Plot all motor neuron spikes
for neuron_idx, spike_train in enumerate(aMN_spikes):
    spike_times = spike_train.times.rescale("s").magnitude
    ax_raster.plot(
        spike_times,
        np.ones_like(spike_times) * neuron_idx,
        "k.",
        markersize=1,
        alpha=0.5,
    )

ax_raster.set_xlabel("Time (s)", fontsize=12)
ax_raster.set_ylabel("MN #", fontsize=12)
ax_raster.set_xlim(0, 180)
ax_raster.set_ylim(-1, len(aMN_spikes))
ax_raster.spines["top"].set_visible(False)
ax_raster.spines["right"].set_visible(False)
# Add panel label H
ax_raster.text(
    -0.05,
    1.05,
    "H",
    transform=ax_raster.transAxes,
    fontsize=16,
    fontweight="bold",
    va="top",
)

# Add zoomed insets for each time window (like in paper)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

for window_idx, (t_start, t_stop) in enumerate(time_windows):
    # Create a small time window to zoom into (0.15s window)
    zoom_center = (t_start + t_stop) / 2
    zoom_start = zoom_center - 0.075
    zoom_stop = zoom_center + 0.075

    # Create inset
    axins = inset_axes(
        ax_raster,
        width="15%",
        height="20%",
        bbox_to_anchor=(0.18 + window_idx * 0.31, 0.75, 0.3, 0.25),
        bbox_transform=ax_raster.transAxes,
    )

    # Plot spikes in zoomed window
    for neuron_idx, spike_train in enumerate(aMN_spikes):
        spike_times = spike_train.times.rescale("s").magnitude
        in_window = (spike_times >= zoom_start) & (spike_times <= zoom_stop)
        if np.any(in_window):
            axins.plot(
                spike_times[in_window],
                np.ones_like(spike_times[in_window]) * neuron_idx,
                "k.",
                markersize=2,
            )

    axins.set_xlim(zoom_start, zoom_stop)
    axins.set_ylim(50, 450)  # Focus on middle neurons
    axins.set_facecolor(window_colors[window_idx])
    axins.set_alpha(0.3)
    axins.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(
    save_path / "watanabe_raster_full_duration.png", dpi=150, bbox_inches="tight"
)
plt.show()
print("Full duration raster plot created (Panel H)")

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.

print("\nGenerating comprehensive visualizations...")

# Loop through each time window
for window_idx, (t_start, t_stop) in enumerate(time_windows):
    print(f"\nProcessing time window: {t_start}-{t_stop}s")
    window_color = window_colors[window_idx]

    # Filter spike trains for the current time window
    aMN_spikes_windowed = [
        st.time_slice(t_start * pq.s, t_stop * pq.s) for st in aMN_spikes
    ]

    # Convert to binned spike trains
    spike_trains_not_zeros = elephant.conversion.BinnedSpikeTrain(
        aMN_spikes_windowed,
        n_bins=int(
            (
                aMN_spikes_windowed[0].sampling_rate.rescale("1/s")
                * aMN_spikes_windowed[0].duration.rescale("s")
            ).magnitude
        ),
    ).to_sparse_bool_array()

    # Generate random pairs
    np.random.seed(42)
    random_indices = np.random.choice(
        spike_trains_not_zeros.shape[0], size=(100, 5), replace=True
    )

    random_pairs = np.array(
        [
            spike_trains_not_zeros[indices].max(axis=0).todense()
            for indices in random_indices
        ]
    )[:, 0]

    # Convolve each random pair with a square signal
    # Square pulse: amplitude 20k, duration 0.05 ms
    sampling_rate = aMN_spikes_windowed[0].sampling_rate.rescale("Hz").magnitude
    dt = 1.0 / sampling_rate  # time step in seconds
    pulse_duration = 0.05e-3  # 0.05 ms in seconds
    pulse_samples = int(pulse_duration / dt)
    square_pulse = np.ones(pulse_samples) * 20000  # amplitude 20k

    # Convolve each random pair with the square pulse
    convolved_signals = []
    for pair in random_pairs:
        pair_array = np.asarray(pair).flatten()
        convolved = np.convolve(pair_array, square_pulse, mode="same")
        convolved_signals.append(convolved)

    convolved_signals = np.array(convolved_signals)

    # Compute power spectrum for each convolved signal
    # Using 60,000 FFT points, Hamming window, no overlap
    # Remove linear trends before computing power spectrum
    nfft = 60000
    power_spectra = []
    frequencies = None

    for conv_sig in tqdm(convolved_signals, desc="Computing power spectra"):
        # Detrend to remove linear trends
        conv_sig_detrended = scipy_signal.detrend(conv_sig, type="linear")

        freqs, psd = scipy_signal.welch(
            conv_sig_detrended,
            fs=sampling_rate,
            window="hamming",
            nperseg=nfft,
            noverlap=0,
            nfft=nfft,
        )
        power_spectra.append(psd)
        if frequencies is None:
            frequencies = freqs

    power_spectra = np.array(power_spectra)

    print(f"Convolved {len(random_pairs)} random pairs with square pulse")
    print(f"Computed power spectra (shape: {power_spectra.shape})")

    # 1. POWER SPECTRA: Individual and mean power spectra
    fig0, ax0 = plt.subplots(1, 1, figsize=(12, 6))
    # Plot individual power spectra with low alpha
    for psd in power_spectra:
        ax0.plot(frequencies, psd, alpha=0.1, color=window_color)
    # Plot mean power spectrum with alpha=1.0
    mean_psd = np.mean(power_spectra, axis=0)
    ax0.plot(
        frequencies, mean_psd, alpha=1.0, color=window_color, linewidth=2, label="Mean"
    )
    ax0.set_xlabel("Frequency (Hz)")
    ax0.set_ylabel("Power Spectral Density")
    ax0.set_title(f"Power Spectra of Convolved Random Pairs ({t_start}-{t_stop}s)")
    ax0.set_xlim(0, 25)
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        save_path / f"watanabe_power_spectra_{t_start}_{t_stop}s.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # 1b. CORTICOMUSCULAR COHERENCE: Between MN membrane potential and CST
    print("Computing corticomuscular coherence...")

    # Extract membrane potentials from aMN segment
    aMN_membrane_potentials = []
    for analog_sig in aMN_results.analogsignals:
        # Check if analog signal covers the requested time window
        sig_t_start = analog_sig.t_start.rescale("s").magnitude
        sig_t_stop = analog_sig.t_stop.rescale("s").magnitude

        if sig_t_start <= t_start and sig_t_stop >= t_stop:
            # Time slice the analog signal
            analog_windowed = analog_sig.time_slice(t_start * pq.s, t_stop * pq.s)
            aMN_membrane_potentials.append(analog_windowed.magnitude.flatten())

    if len(aMN_membrane_potentials) == 0:
        print(
            f"⚠ No membrane potentials available for time window {t_start}-{t_stop}s, skipping coherence computation"
        )
    else:
        # Average membrane potentials across all MNs
        avg_membrane_potential = np.mean(aMN_membrane_potentials, axis=0)

        # Detrend membrane potential
        avg_membrane_potential_detrended = scipy_signal.detrend(
            avg_membrane_potential, type="linear"
        )

        # Resample membrane potential to match CST sampling rate if needed
        # Membrane potential sampling rate
        mp_sampling_rate = (
            aMN_results.analogsignals[0].sampling_rate.rescale("Hz").magnitude
        )

        # Resample membrane potential to CST sampling rate
        from scipy import interpolate

        mp_time = np.arange(len(avg_membrane_potential_detrended)) / mp_sampling_rate
        cst_time = np.arange(len(convolved_signals[0])) / sampling_rate

        # Create interpolation function
        interp_func = interpolate.interp1d(
            mp_time,
            avg_membrane_potential_detrended,
            kind="linear",
            fill_value="extrapolate",
        )

        # Resample to CST time base
        avg_membrane_potential_resampled = interp_func(cst_time)

        # Compute coherence for each CST
        coherences = []
        coherence_freqs = None

        for conv_sig in tqdm(convolved_signals, desc="Computing coherences"):
            # Detrend CST
            conv_sig_detrended = scipy_signal.detrend(conv_sig, type="linear")

            # Compute coherence with higher resolution
            # Use larger nperseg for better frequency resolution
            nperseg_coherence = min(
                120000, len(conv_sig_detrended)
            )  # Increased for higher resolution
            freqs_coh, coh = scipy_signal.coherence(
                avg_membrane_potential_resampled,
                conv_sig_detrended,
                fs=sampling_rate,
                window="hamming",
                nperseg=nperseg_coherence,
                noverlap=nperseg_coherence // 2,  # 50% overlap for smoother estimate
            )
            coherences.append(coh)
            if coherence_freqs is None:
                coherence_freqs = freqs_coh

        coherences = np.array(coherences)

        # Calculate 95% confidence level
        from scipy.stats import f as f_dist

        K = 1  # Number of segments (1 when using full signal)
        alpha = 0.05
        F_value = f_dist.ppf(1 - alpha, 2, 2 * K - 1)
        confidence_level = (F_value * (K - 1)) / ((K - 1) + F_value)

        print(f"Computed coherences (shape: {coherences.shape})")
        print(f"95% Confidence Level: {confidence_level:.4f}")

        # Plot coherence
        fig0b, ax0b = plt.subplots(1, 1, figsize=(8, 5))
        # Plot individual coherences with low alpha
        for coh in coherences:
            ax0b.plot(
                coherence_freqs, coh, alpha=0.15, color=window_color, linewidth=0.5
            )
        # Plot mean coherence with alpha=1.0
        mean_coh = np.mean(coherences, axis=0)
        ax0b.plot(coherence_freqs, mean_coh, alpha=1.0, color=window_color, linewidth=2)
        # Plot 95% confidence level
        ax0b.axhline(
            y=confidence_level, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        ax0b.set_xlabel("Frequency (Hz)", fontsize=12)
        ax0b.set_ylabel("Coherence", fontsize=12)
        ax0b.set_xlim(0, 25)
        ax0b.set_ylim(0, 1)
        ax0b.spines["top"].set_visible(False)
        ax0b.spines["right"].set_visible(False)
        # Add panel label (D, E, F)
        panel_letter = chr(68 + window_idx)  # D=68, E=69, F=70
        ax0b.text(
            -0.15,
            1.05,
            panel_letter,
            transform=ax0b.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )
        plt.tight_layout()
        plt.savefig(
            save_path / f"watanabe_coherence_{t_start}_{t_stop}s.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    # 2. NEURAL ACTIVITY: Raster plot showing motor neuron spike patterns
    populations_list = ["aMN", "DD"]
    fig1, axes1 = plt.subplots(len(populations_list), 1, figsize=(15, 8), sharex=True)

    # Filter results for the current time window
    results_windowed = neo.Block()
    for segment in results.segments:
        seg_windowed = neo.Segment(name=segment.name)
        # Filter spike trains
        for st in segment.spiketrains:
            st_windowed = st.time_slice(t_start * pq.s, t_stop * pq.s)
            seg_windowed.spiketrains.append(st_windowed)
        results_windowed.segments.append(seg_windowed)

    plot_raster_spikes(
        results_windowed,
        axes1,
        populations=populations_list,
        title=f"Motor Neuron Pool Activity - Watanabe Paper ({t_start}-{t_stop}s)",
    )
    plt.tight_layout()
    plt.savefig(
        save_path / f"watanabe_raster_plot_{t_start}_{t_stop}s.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # 3. MOTOR NEURON DYNAMICS: Membrane potentials showing integration
    fig2, axes2 = plt.subplots(1, 1, figsize=(15, 8))
    plot_membrane_potentials(
        results_windowed,
        [axes2],
        populations=["aMN"],
        cell_indices=[0, 10, 20, 30, 40, 50, 60, 70],
        title=f"Motor Neuron Membrane Potentials - Watanabe Paper ({t_start}-{t_stop}s)",
    )
    plt.tight_layout()
    plt.savefig(
        save_path / f"watanabe_membrane_potentials_{t_start}_{t_stop}s.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

print("\n✅ All visualizations completed for all time windows!")

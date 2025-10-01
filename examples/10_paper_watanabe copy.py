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

from myogen import load_nmodl_mechanisms
from myogen.utils.plotting import plot_membrane_potentials, plot_raster_spikes

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
naMN = 11

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

    # Plot normalized power spectrum
    axes_dd[idx].plot(freqs_dd, psd_dd_normalized, color="blue", linewidth=2)
    axes_dd[idx].set_xlabel("Frequency (Hz)")
    axes_dd[idx].set_ylabel("Power Spectral Density")
    axes_dd[idx].set_title(f"DDdrive Power Spectrum ({t_start}-{t_stop}s)")
    axes_dd[idx].set_xlim(0, 25)
    axes_dd[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    save_path / "DDdrive_power_spectra_all_windows.png", dpi=150, bbox_inches="tight"
)
plt.show()
print("✓ DDdrive power spectra computed and plotted for all three time windows")


with open(r"C:\Users\raulc\Downloads\spinal_network_results.pkl", "rb") as f:
    results: neo.Block = joblib.load(f)
    print("✓ Previous results loaded successfully")

aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

# Define time windows for analysis
time_windows = [(3, 60), (63, 120), (123, 180)]

##############################################################################
# Comprehensive Results Visualization
# ---------------------------------
#
# Create a series of plots that tell the complete story of spinal network
# function, from neural activity to mechanical output.

print("\n📊 Generating comprehensive visualizations...")

import quantities as pq

# Loop through each time window
for t_start, t_stop in time_windows:
    print(f"\n⏱️  Processing time window: {t_start}-{t_stop}s")

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

    print(f"✓ Convolved {len(random_pairs)} random pairs with square pulse")
    print(f"✓ Computed power spectra (shape: {power_spectra.shape})")

    # 1. POWER SPECTRA: Individual and mean power spectra
    fig0, ax0 = plt.subplots(1, 1, figsize=(12, 6))
    # Plot individual power spectra with low alpha
    for psd in power_spectra:
        ax0.plot(frequencies, psd, alpha=0.1, color="blue")
    # Plot mean power spectrum with alpha=1.0
    mean_psd = np.mean(power_spectra, axis=0)
    ax0.plot(frequencies, mean_psd, alpha=1.0, color="red", linewidth=2, label="Mean")
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
        print(f"⚠ No membrane potentials available for time window {t_start}-{t_stop}s, skipping coherence computation")
    else:
        # Average membrane potentials across all MNs
        avg_membrane_potential = np.mean(aMN_membrane_potentials, axis=0)

        # Detrend membrane potential
        avg_membrane_potential_detrended = scipy_signal.detrend(avg_membrane_potential, type="linear")

        # Resample membrane potential to match CST sampling rate if needed
        # Membrane potential sampling rate
        mp_sampling_rate = aMN_results.analogsignals[0].sampling_rate.rescale("Hz").magnitude

        # Resample membrane potential to CST sampling rate
        from scipy import interpolate
        mp_time = np.arange(len(avg_membrane_potential_detrended)) / mp_sampling_rate
        cst_time = np.arange(len(convolved_signals[0])) / sampling_rate

        # Create interpolation function
        interp_func = interpolate.interp1d(
            mp_time, avg_membrane_potential_detrended,
            kind='linear', fill_value='extrapolate'
        )

        # Resample to CST time base
        avg_membrane_potential_resampled = interp_func(cst_time)

        # Compute coherence for each CST
        coherences = []
        coherence_freqs = None

        for conv_sig in tqdm(convolved_signals, desc="Computing coherences"):
            # Detrend CST
            conv_sig_detrended = scipy_signal.detrend(conv_sig, type="linear")

            # Compute coherence
            freqs_coh, coh = scipy_signal.coherence(
                avg_membrane_potential_resampled,
                conv_sig_detrended,
                fs=sampling_rate,
                window="hamming",
                nperseg=min(nfft, len(conv_sig_detrended)),
                noverlap=0,
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

        print(f"✓ Computed coherences (shape: {coherences.shape})")
        print(f"✓ 95% Confidence Level: {confidence_level:.4f}")

        # Plot coherence
        fig0b, ax0b = plt.subplots(1, 1, figsize=(12, 6))
        # Plot individual coherences with low alpha
        for coh in coherences:
            ax0b.plot(coherence_freqs, coh, alpha=0.1, color="blue")
        # Plot mean coherence with alpha=1.0
        mean_coh = np.mean(coherences, axis=0)
        ax0b.plot(coherence_freqs, mean_coh, alpha=1.0, color="red", linewidth=2, label="Mean")
        # Plot 95% confidence level
        ax0b.axhline(y=confidence_level, color="black", linestyle="--", linewidth=1.5, label="95% CL")
        ax0b.set_xlabel("Frequency (Hz)")
        ax0b.set_ylabel("Coherence")
        ax0b.set_title(f"Corticomuscular Coherence ({t_start}-{t_stop}s)")
        ax0b.set_xlim(0, 25)
        ax0b.set_ylim(0, 1)
        ax0b.legend()
        ax0b.grid(True, alpha=0.3)
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

"""
Watanabe Paper - Results Visualization
======================================

This script loads pre-computed spike trains and force output to generate
all figures for the Watanabe paper reproduction.

**Pipeline:**
1. Load spike trains from 10a (spinal_network_results.pkl)
2. Load force output from 10b (watanabe_force_results.pkl)
3. Generate all visualizations (panels A-H and supplementary figures)

**Outputs:**
- Multiple PNG figures in ./results/ directory
"""

from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
import neo
import elephant
from scipy import signal as scipy_signal
from tqdm import tqdm
import quantities as pq

from myogen.utils.plotting import plot_membrane_potentials, plot_raster_spikes

##############################################################################
# Setup Paths and Load Data
# -------------------------

save_path = Path("./results")
save_path.mkdir(exist_ok=True)

spinal_results_path = Path(
    r"C:\Users\raulc\Research\papers_server\in_progress\simulator\spinal_network_results.pkl"
)
force_results_path = save_path / "watanabe_force_results.pkl"

print("=" * 80)
print("Watanabe Paper - Results Visualization")
print("=" * 80)

# Load spike train results
print(f"\nLoading data...")
print(f"  Spike trains: {spinal_results_path}")
with open(spinal_results_path, "rb") as f:
    results: neo.Block = joblib.load(f)
    print("  [OK] Spike trains loaded")

# Load force results
print(f"  Force output: {force_results_path}")
with open(force_results_path, "rb") as f:
    force_block: neo.Block = joblib.load(f)
    print("  [OK] Force output loaded")

# Extract data
aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

force_segment = force_block.segments[0]
force_output = force_segment.analogsignals[0]

print(f"\nData summary:")
print(f"  - Motor neurons: {len(aMN_spikes)}")
print(f"  - Force samples: {force_output.shape[0]}")
print(f"  - Duration: {force_output.t_stop.rescale('s')}")

##############################################################################
# Define Simulation Parameters
# ----------------------------

dt = 0.1  # ms - Integration timestep
tstop = 180 * 1e3  # ms - Total simulation duration
n_steps = int(tstop / dt)
time = np.linspace(0, tstop, n_steps + 100)  # Add margin for NEURON overstep

# Define time windows for analysis (matching paper)
time_windows = [(3, 60), (63, 120), (123, 180)]
window_colors = ["#0001f9", "#966562", "#2efe37"]  # Blue, brown, green

##############################################################################
# Generate DDdrive Signal (for power spectrum analysis)
# ---------------------------------------------------

time_s = time / 1000.0  # Convert to seconds

# Initialize with constant value
DDdrive = np.full_like(time, 65.0)

# Phase 2: 20 Hz sinusoid with DC=65, amplitude=20 from 60-120s
phase2_mask = (time_s >= 60) & (time_s < 120)
DDdrive[phase2_mask] = 65 + 20 * np.sin(2 * np.pi * 20 * time_s[phase2_mask])

# Phase 3: Same sinusoid from 120-180s but DC=58
phase3_mask = (time_s >= 120) & (time_s <= 180)
DDdrive[phase3_mask] = 58 + 20 * np.sin(2 * np.pi * 20 * time_s[phase3_mask])

##############################################################################
# PANELS A-C: DDdrive Power Spectra
# ----------------------------------

print(f"\n{'='*80}")
print("Generating Panels A-C: DDdrive Power Spectra")
print(f"{'='*80}")

fig_dd, axes_dd = plt.subplots(1, 3, figsize=(18, 5))

for idx, (t_start, t_stop) in enumerate(time_windows):
    # Extract DDdrive for the time window
    start_idx = int(t_start * 1000 / dt)
    stop_idx = int(t_stop * 1000 / dt)
    DDdrive_windowed = DDdrive[start_idx:stop_idx]

    # Remove linear trend
    DDdrive_detrended = scipy_signal.detrend(DDdrive_windowed, type="linear")

    # Compute power spectrum using Welch's method
    sampling_rate_dd = 1000.0 / dt  # Hz
    signal_length = len(DDdrive_detrended)

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

    # Plot
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
    # Panel label
    axes_dd[idx].text(
        -0.15, 1.05, chr(65 + idx), transform=axes_dd[idx].transAxes,
        fontsize=16, fontweight="bold", va="top"
    )

plt.tight_layout()
plt.savefig(save_path / "DDdrive_power_spectra_all_windows.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Panels A-C saved")

##############################################################################
# PANEL G: Force Timeseries
# -------------------------

print(f"\nGenerating Panel G: Force Timeseries")

# Extract force signal and scale to reasonable range (0-500 N)
force_signal = force_output.magnitude[:, 0]
time_force = force_output.times.rescale("s").magnitude
sampling_rate_force = force_output.sampling_rate.rescale("Hz").magnitude

force_signal_scaled = (force_signal / np.max(force_signal)) * 500 if np.max(force_signal) > 0 else force_signal

# Create figure
fig_force, ax_force = plt.subplots(1, 1, figsize=(15, 4))

# Plot force in segments with different colors
for window_idx, (t_start, t_stop) in enumerate(time_windows):
    start_idx = int(t_start * sampling_rate_force)
    stop_idx = int(t_stop * sampling_rate_force)
    time_segment = time_force[start_idx:stop_idx]
    force_segment = force_signal_scaled[start_idx:stop_idx]

    ax_force.plot(time_segment, force_segment, color=window_colors[window_idx], linewidth=1.5)

ax_force.set_xlabel("Time (s)", fontsize=12)
ax_force.set_ylabel("Force (N)", fontsize=12)
ax_force.set_xlim(0, 180)
ax_force.spines["top"].set_visible(False)
ax_force.spines["right"].set_visible(False)
ax_force.text(-0.05, 1.05, "G", transform=ax_force.transAxes,
              fontsize=16, fontweight="bold", va="top")

plt.tight_layout()
plt.savefig(save_path / "watanabe_force_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Panel G saved")

##############################################################################
# PANEL H: Full Duration Raster Plot
# ----------------------------------

print(f"\nGenerating Panel H: Full Duration Raster Plot")

fig_raster, ax_raster = plt.subplots(1, 1, figsize=(15, 6))

# Add color-coded background regions
for window_idx, (t_start, t_stop) in enumerate(time_windows):
    ax_raster.axvspan(t_start, t_stop, facecolor=window_colors[window_idx], alpha=0.3)

# Plot all motor neuron spikes
for neuron_idx, spike_train in enumerate(aMN_spikes):
    spike_times = spike_train.times.rescale("s").magnitude
    ax_raster.plot(spike_times, np.ones_like(spike_times) * neuron_idx,
                   "k.", markersize=1, alpha=0.5)

ax_raster.set_xlabel("Time (s)", fontsize=12)
ax_raster.set_ylabel("MN #", fontsize=12)
ax_raster.set_xlim(0, 180)
ax_raster.set_ylim(-1, len(aMN_spikes))
ax_raster.spines["top"].set_visible(False)
ax_raster.spines["right"].set_visible(False)
ax_raster.text(-0.05, 1.05, "H", transform=ax_raster.transAxes,
               fontsize=16, fontweight="bold", va="top")

# Add zoomed insets
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

for window_idx, (t_start, t_stop) in enumerate(time_windows):
    zoom_center = (t_start + t_stop) / 2
    zoom_start = zoom_center - 0.075
    zoom_stop = zoom_center + 0.075

    axins = inset_axes(
        ax_raster, width="15%", height="20%",
        bbox_to_anchor=(0.18 + window_idx * 0.31, 0.75, 0.3, 0.25),
        bbox_transform=ax_raster.transAxes
    )

    for neuron_idx, spike_train in enumerate(aMN_spikes):
        spike_times = spike_train.times.rescale("s").magnitude
        in_window = (spike_times >= zoom_start) & (spike_times <= zoom_stop)
        if np.any(in_window):
            axins.plot(spike_times[in_window],
                      np.ones_like(spike_times[in_window]) * neuron_idx,
                      "k.", markersize=2)

    axins.set_xlim(zoom_start, zoom_stop)
    axins.set_ylim(50, 450)
    axins.set_facecolor(window_colors[window_idx])
    axins.set_alpha(0.3)
    axins.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(save_path / "watanabe_raster_full_duration.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Panel H saved")

##############################################################################
# PANELS D-F: Per-Window Analysis (Coherence, Rasters, Membrane Potentials)
# --------------------------------------------------------------------------

print(f"\n{'='*80}")
print("Generating Panels D-F: Per-Window Coherence and Activity")
print(f"{'='*80}")

for window_idx, (t_start, t_stop) in enumerate(time_windows):
    print(f"\nProcessing window {window_idx+1}: {t_start}-{t_stop}s")
    window_color = window_colors[window_idx]

    # Filter spike trains for current window
    aMN_spikes_windowed = [
        st.time_slice(t_start * pq.s, t_stop * pq.s) for st in aMN_spikes
    ]

    # Convert to binned spike trains
    spike_trains_binned = elephant.conversion.BinnedSpikeTrain(
        aMN_spikes_windowed,
        n_bins=int(
            (
                aMN_spikes_windowed[0].sampling_rate.rescale("1/s")
                * aMN_spikes_windowed[0].duration.rescale("s")
            ).magnitude
        ),
    ).to_sparse_bool_array()

    # Generate random pairs (composite spike trains)
    np.random.seed(42)
    random_indices = np.random.choice(
        spike_trains_binned.shape[0], size=(100, 5), replace=True
    )

    random_pairs = np.array(
        [spike_trains_binned[indices].max(axis=0).todense() for indices in random_indices]
    )[:, 0]

    # Convolve with square pulse (simulating CST)
    sampling_rate = aMN_spikes_windowed[0].sampling_rate.rescale("Hz").magnitude
    dt_s = 1.0 / sampling_rate
    pulse_duration = 0.05e-3  # 0.05 ms
    pulse_samples = int(pulse_duration / dt_s)
    square_pulse = np.ones(pulse_samples) * 20000

    convolved_signals = []
    for pair in random_pairs:
        pair_array = np.asarray(pair).flatten()
        convolved = np.convolve(pair_array, square_pulse, mode="same")
        convolved_signals.append(convolved)

    convolved_signals = np.array(convolved_signals)

    # Compute power spectra
    nfft = 60000
    power_spectra = []
    frequencies = None

    for conv_sig in tqdm(convolved_signals, desc="  Power spectra"):
        conv_sig_detrended = scipy_signal.detrend(conv_sig, type="linear")
        freqs, psd = scipy_signal.welch(
            conv_sig_detrended, fs=sampling_rate, window="hamming",
            nperseg=nfft, noverlap=0, nfft=nfft
        )
        power_spectra.append(psd)
        if frequencies is None:
            frequencies = freqs

    power_spectra = np.array(power_spectra)

    # Plot power spectra
    fig_psd, ax_psd = plt.subplots(1, 1, figsize=(12, 6))
    for psd in power_spectra:
        ax_psd.plot(frequencies, psd, alpha=0.1, color=window_color)
    mean_psd = np.mean(power_spectra, axis=0)
    ax_psd.plot(frequencies, mean_psd, alpha=1.0, color=window_color,
               linewidth=2, label="Mean")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power Spectral Density")
    ax_psd.set_title(f"Power Spectra of CST ({t_start}-{t_stop}s)")
    ax_psd.set_xlim(0, 25)
    ax_psd.legend()
    ax_psd.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / f"watanabe_power_spectra_{t_start}_{t_stop}s.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Compute corticomuscular coherence (Panels D-F)
    print("  Computing coherence...")

    # Extract membrane potentials
    aMN_membrane_potentials = []
    for analog_sig in aMN_results.analogsignals:
        sig_t_start = analog_sig.t_start.rescale("s").magnitude
        sig_t_stop = analog_sig.t_stop.rescale("s").magnitude

        if sig_t_start <= t_start and sig_t_stop >= t_stop:
            analog_windowed = analog_sig.time_slice(t_start * pq.s, t_stop * pq.s)
            aMN_membrane_potentials.append(analog_windowed.magnitude.flatten())

    if len(aMN_membrane_potentials) > 0:
        # Average and detrend membrane potential
        avg_membrane_potential = np.mean(aMN_membrane_potentials, axis=0)
        avg_membrane_potential_detrended = scipy_signal.detrend(
            avg_membrane_potential, type="linear"
        )

        # Resample to CST sampling rate
        from scipy import interpolate
        mp_sampling_rate = aMN_results.analogsignals[0].sampling_rate.rescale("Hz").magnitude
        mp_time = np.arange(len(avg_membrane_potential_detrended)) / mp_sampling_rate
        cst_time = np.arange(len(convolved_signals[0])) / sampling_rate

        interp_func = interpolate.interp1d(
            mp_time, avg_membrane_potential_detrended,
            kind="linear", fill_value="extrapolate"
        )
        avg_membrane_potential_resampled = interp_func(cst_time)

        # Compute coherence for each CST
        coherences = []
        coherence_freqs = None

        for conv_sig in tqdm(convolved_signals, desc="  Coherence"):
            conv_sig_detrended = scipy_signal.detrend(conv_sig, type="linear")
            nperseg_coherence = min(120000, len(conv_sig_detrended))
            freqs_coh, coh = scipy_signal.coherence(
                avg_membrane_potential_resampled, conv_sig_detrended,
                fs=sampling_rate, window="hamming",
                nperseg=nperseg_coherence, noverlap=nperseg_coherence // 2
            )
            coherences.append(coh)
            if coherence_freqs is None:
                coherence_freqs = freqs_coh

        coherences = np.array(coherences)

        # 95% confidence level
        from scipy.stats import f as f_dist
        K = 1
        alpha = 0.05
        F_value = f_dist.ppf(1 - alpha, 2, 2 * K - 1)
        confidence_level = (F_value * (K - 1)) / ((K - 1) + F_value)

        # Plot coherence (Panel D, E, or F)
        fig_coh, ax_coh = plt.subplots(1, 1, figsize=(8, 5))
        for coh in coherences:
            ax_coh.plot(coherence_freqs, coh, alpha=0.15,
                       color=window_color, linewidth=0.5)
        mean_coh = np.mean(coherences, axis=0)
        ax_coh.plot(coherence_freqs, mean_coh, alpha=1.0,
                   color=window_color, linewidth=2)
        ax_coh.axhline(y=confidence_level, color="black",
                      linestyle="--", linewidth=1, alpha=0.7)
        ax_coh.set_xlabel("Frequency (Hz)", fontsize=12)
        ax_coh.set_ylabel("Coherence", fontsize=12)
        ax_coh.set_xlim(0, 25)
        ax_coh.set_ylim(0, 1)
        ax_coh.spines["top"].set_visible(False)
        ax_coh.spines["right"].set_visible(False)
        # Panel label (D, E, F)
        panel_letter = chr(68 + window_idx)
        ax_coh.text(-0.15, 1.05, panel_letter, transform=ax_coh.transAxes,
                   fontsize=16, fontweight="bold", va="top")
        plt.tight_layout()
        plt.savefig(save_path / f"watanabe_coherence_{t_start}_{t_stop}s.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [OK] Panel {panel_letter} saved (coherence)")

    # Raster plot for this window
    populations_list = ["aMN", "DD"]
    fig_raster_win, axes_raster_win = plt.subplots(
        len(populations_list), 1, figsize=(15, 8), sharex=True
    )

    results_windowed = neo.Block()
    for segment in results.segments:
        seg_windowed = neo.Segment(name=segment.name)
        for st in segment.spiketrains:
            st_windowed = st.time_slice(t_start * pq.s, t_stop * pq.s)
            seg_windowed.spiketrains.append(st_windowed)
        results_windowed.segments.append(seg_windowed)

    plot_raster_spikes(
        results_windowed, axes_raster_win, populations=populations_list,
        title=f"Motor Neuron Activity ({t_start}-{t_stop}s)"
    )
    plt.tight_layout()
    plt.savefig(save_path / f"watanabe_raster_plot_{t_start}_{t_stop}s.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Raster plot saved")

    # Membrane potential plot
    fig_vm, ax_vm = plt.subplots(1, 1, figsize=(15, 8))
    plot_membrane_potentials(
        results_windowed, [ax_vm], populations=["aMN"],
        cell_indices=[0, 10, 20, 30, 40, 50, 60, 70],
        title=f"Motor Neuron Membrane Potentials ({t_start}-{t_stop}s)"
    )
    plt.tight_layout()
    plt.savefig(save_path / f"watanabe_membrane_potentials_{t_start}_{t_stop}s.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [OK] Membrane potential plot saved")

##############################################################################
# Summary
# -------

print(f"\n{'='*80}")
print("All Visualizations Complete!")
print(f"{'='*80}")
print(f"\nGenerated figures:")
print(f"  - DDdrive power spectra (Panels A-C)")
print(f"  - Corticomuscular coherence (Panels D-F)")
print(f"  - Force timeseries (Panel G)")
print(f"  - Full raster plot (Panel H)")
print(f"  - Per-window power spectra (3 figures)")
print(f"  - Per-window raster plots (3 figures)")
print(f"  - Per-window membrane potentials (3 figures)")
print(f"\nAll figures saved to: {save_path}")
print(f"{'='*80}")

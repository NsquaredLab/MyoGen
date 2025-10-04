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

# %%

from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import elephant
from scipy import signal as scipy_signal
import quantities as pq


# Set seaborn style with larger fonts
sns.set_context("talk", font_scale=2.0)
sns.set_style("ticks", {"xtick.direction": "in", "ytick.direction": "in"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 5
plt.rcParams["xtick.major.width"] = 3
plt.rcParams["ytick.major.width"] = 3
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["xtick.major.pad"] = 10
plt.rcParams["ytick.major.pad"] = 10

##############################################################################
# Setup Paths and Load Data
# -------------------------

save_path = Path(
    r"/home/oj98yqyk/code/simulators/MyoGen/examples/papers/watanabe/results"
)
save_path.mkdir(exist_ok=True)

spinal_results_path = save_path / Path("watanabe__spinal_network_results.pkl")
force_results_path = save_path / "watanabe__force_results.pkl"

# Load spike train results
with open(spinal_results_path, "rb") as f:
    results: neo.Block = joblib.load(f)

# Load force results
with open(force_results_path, "rb") as f:
    force_block: neo.Block = joblib.load(f)

# Extract data
aMN_results: neo.Segment = results.filter(name="aMN", container=True)[0]
aMN_spikes = aMN_results.spiketrains

force_segment = force_block.segments[0]
force_output = force_segment.analogsignals[0]

##############################################################################
# Define Simulation Parameters
# ----------------------------

dt = 0.025  # ms - Integration timestep
tstop = 180 * 1e3  # ms - Total simulation duration
n_steps = int(tstop / dt)
time = np.linspace(0, tstop, n_steps + 100)  # Add margin for NEURON overstep

# Define time windows for analysis (matching paper)
time_windows = [(3, 60), (63, 120), (123, 180)]
window_colors = ["#0001f9", "#966562", "#2efe37"]  # Blue, brown, green

# Force plot time windows (continuous segments)
force_windows = [(0, 60), (60, 120), (120, 180)]

##############################################################################
# PANELS A-C: Net Membrane Potential Power Spectra
# -------------------------------------------------

fig_mp, axes_mp = plt.subplots(1, 3, figsize=(18, 5))

for idx, (t_start, t_stop) in enumerate(time_windows):
    # Extract membrane potentials for this time window
    aMN_membrane_potentials = []
    for analog_sig in aMN_results.analogsignals:
        sig_t_start = analog_sig.t_start.rescale("s").magnitude
        sig_t_stop = analog_sig.t_stop.rescale("s").magnitude

        if sig_t_start <= t_start and sig_t_stop >= t_stop:
            analog_windowed = analog_sig.time_slice(t_start * pq.s, t_stop * pq.s)
            aMN_membrane_potentials.append(analog_windowed.magnitude.flatten())

    if len(aMN_membrane_potentials) > 0:
        # Average across all motor neurons
        avg_membrane_potential = np.mean(aMN_membrane_potentials, axis=0)

        # Get sampling rate
        mp_sampling_rate = (
            aMN_results.analogsignals[0].sampling_rate.rescale("Hz").magnitude
        )

        # Compute power spectrum using Welch's method with built-in detrending
        signal_length = len(avg_membrane_potential)

        freqs_mp, psd_mp = scipy_signal.welch(
            avg_membrane_potential,
            fs=mp_sampling_rate,
            window="hamming",
            nperseg=60000,
            noverlap=0,
            nfft=signal_length,
            detrend="linear",
        )

        # Normalize power spectrum between 0 and 1
        psd_mp_normalized = (psd_mp - np.min(psd_mp)) / (
            np.max(psd_mp) - np.min(psd_mp)
        )

        # Plot
        axes_mp[idx].fill_between(
            freqs_mp, psd_mp_normalized, color=window_colors[idx], alpha=0.7
        )
        axes_mp[idx].plot(
            freqs_mp, psd_mp_normalized, color=window_colors[idx], linewidth=1.5
        )
        axes_mp[idx].set_xlabel("Frequency (Hz)")
        if idx == 0:
            axes_mp[idx].set_ylabel("Membrane Potential\nSpectrum (A.U.)")
        axes_mp[idx].set_xlim(0, 25)
        axes_mp[idx].set_xticks([0, 5, 10, 15, 20, 25])
        axes_mp[idx].set_ylim(0, 1.05)
        axes_mp[idx].set_yticks([0, 0.5, 1])

        sns.despine(ax=axes_mp[idx], trim=True)

plt.tight_layout()
plt.savefig(
    save_path / "membrane_potential_power_spectra_all_windows.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

##############################################################################
# PANELS D-F: Corticomuscular Coherence Spectra
# ----------------------------------------------

# Create figure with 3 panels
fig_coh, axes_coh = plt.subplots(1, 3, figsize=(18, 5))

# Pre-compute coherences for all windows
all_coherences = []
coherence_freqs = None

for window_idx, (t_start, t_stop) in enumerate(time_windows):
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
        [
            spike_trains_binned[indices].max(axis=0).todense()
            for indices in random_indices
        ]
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

        mp_sampling_rate = (
            aMN_results.analogsignals[0].sampling_rate.rescale("Hz").magnitude
        )
        mp_time = np.arange(len(avg_membrane_potential_detrended)) / mp_sampling_rate
        cst_time = np.arange(len(convolved_signals[0])) / sampling_rate

        interp_func = interpolate.interp1d(
            mp_time,
            avg_membrane_potential_detrended,
            kind="linear",
            fill_value="extrapolate",
        )
        avg_membrane_potential_resampled = interp_func(cst_time)

        # Compute coherence for each CST using parallel processing
        # Optimized parameters for faster computation
        nperseg_coherence = min(60000, len(convolved_signals[0]))  # Reduced from 300000
        noverlap_coherence = nperseg_coherence // 2  # 50% overlap instead of 75%

        def compute_single_coherence(conv_sig):
            freqs_coh, coh = scipy_signal.coherence(
                avg_membrane_potential_resampled,
                conv_sig,
                fs=sampling_rate,
                window="hamming",
                nperseg=nperseg_coherence,
                noverlap=noverlap_coherence,
                detrend="linear",
            )
            return freqs_coh, coh

        # Parallel computation across all pairs
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_single_coherence)(conv_sig)
            for conv_sig in convolved_signals
        )

        coherence_freqs = results[0][0]
        coherences = np.array([coh for _, coh in results])

        coherences = np.array(coherences)
        all_coherences.append((coherences, window_color))

        # Plot in corresponding panel
        # Plot all 100 individual pairs with alpha
        for coh in coherences:
            axes_coh[window_idx].plot(
                coherence_freqs, coh, color=window_color, alpha=0.1, linewidth=0.5
            )

        # Plot mean coherence spectrum with solid colored line
        mean_coh = np.mean(coherences, axis=0)
        axes_coh[window_idx].plot(
            coherence_freqs, mean_coh, color=window_color, linewidth=2, alpha=1.0
        )

        # Plot horizontal line showing overall mean coherence value
        overall_mean = np.mean(mean_coh)
        axes_coh[window_idx].axhline(
            y=overall_mean, color="black", linestyle="--", linewidth=1.5, alpha=1.0
        )

        axes_coh[window_idx].set_xlabel("Frequency (Hz)")
        if window_idx == 0:
            axes_coh[window_idx].set_ylabel("Coherence")
        axes_coh[window_idx].set_xlim(0, 25)
        axes_coh[window_idx].set_xticks([0, 5, 10, 15, 20, 25])
        axes_coh[window_idx].set_ylim(0, 1.05)
        axes_coh[window_idx].set_yticks([0, 0.5, 1])

        sns.despine(ax=axes_coh[window_idx], trim=True)

plt.tight_layout()
plt.savefig(
    save_path / "coherence_spectra_all_windows.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()

##############################################################################
# PANEL G: Force Timeseries
# -------------------------

# Extract force signal in original unitless scale
force_signal = force_output.magnitude[:, 0]
time_force = force_output.times.rescale("s").magnitude
sampling_rate_force = force_output.sampling_rate.rescale("Hz").magnitude

# Create figure
fig_force, ax_force = plt.subplots(1, 1, figsize=(15, 4))

# Plot force in continuous segments with different colors (0-60s, 60-120s, 120-180s)
for window_idx, (t_start, t_stop) in enumerate(force_windows):
    start_idx = int(t_start * sampling_rate_force)
    stop_idx = int(t_stop * sampling_rate_force)
    time_segment = time_force[start_idx:stop_idx]
    force_segment = force_signal[start_idx:stop_idx]

    ax_force.plot(
        time_segment, force_segment, color=window_colors[window_idx], linewidth=3
    )

ax_force.set_xlabel("Time (s)")
ax_force.set_ylabel("Force (a.u.)")
ax_force.set_xlim(0, 180)

# Dynamically set y-axis limits based on data range (skip initial baseline)
# Skip first 5 seconds to avoid baseline period
skip_samples = int(5 * sampling_rate_force)
force_signal_active = force_signal[skip_samples:]
y_min = np.min(force_signal_active)
y_max = np.max(force_signal_active)
y_range = y_max - y_min
padding = 0.05 * y_range  # 5% padding
ax_force.set_ylim(y_min - padding, y_max + padding)

sns.despine(ax=ax_force, trim=True)

plt.tight_layout()
plt.savefig(save_path / "watanabe_force_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

##############################################################################
# PANEL H: Full Duration Raster Plot
# ----------------------------------

fig_raster, ax_raster = plt.subplots(1, 1, figsize=(15, 6))

# Plot all motor neuron spikes with colors based on time window (using force windows)
for neuron_idx, spike_train in enumerate(aMN_spikes):
    spike_times = spike_train.times.rescale("s").magnitude

    # Color spikes based on which time window they fall in
    for window_idx, (t_start, t_stop) in enumerate(force_windows):
        in_window = (spike_times >= t_start) & (spike_times < t_stop)
        if np.any(in_window):
            ax_raster.plot(
                spike_times[in_window],
                np.ones_like(spike_times[in_window]) * neuron_idx,
                ".",
                color=window_colors[window_idx],
                markersize=1,
                alpha=1.0,
            )

ax_raster.set_xlabel("Time (s)")
ax_raster.set_ylabel("MN #")
ax_raster.set_xlim(0, 180)
ax_raster.set_ylim(-1, len(aMN_spikes))
sns.despine(ax=ax_raster, trim=True)

plt.tight_layout()
plt.savefig(
    save_path / "watanabe_raster_full_duration.png", dpi=150, bbox_inches="tight"
)
plt.show()

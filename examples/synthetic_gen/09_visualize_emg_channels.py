"""
EMG Channel Visualization
=================================

This script loads saved **EMG signals** from a pickle file and generates
**publication-quality PNG plots** of the EMG channels.

.. note::
    This script can handle EMG data saved in different formats:

    - **NEO Block objects**: Direct EMG signal data
    - **Simulator objects**: IntramuscularEMG or SurfaceEMG simulator instances

    For **intramuscular EMG**, the data structure is:
        - Block.segments[0].analogsignals[0]: 2D array (time × channels)

    For **surface EMG**, the data structure is:
        - Block.groups[0].segments[0].analogsignals[0]: 3D array (time × rows × columns)

Features:
    - Automatic detection of data format and EMG type
    - Multi-channel visualization with proper time axes
    - Configurable subplot layouts for optimal visualization
    - High-resolution PNG export for publication quality

.. important::
    If you want to visualize EMG signals, you need to save them from your
    simulation script. Add this line after generating the signals:

    .. code-block:: python

        joblib.dump(noisy_emg_signals__Block, save_path / "emg_signals.pkl")
"""

# %%

##############################################################################
# Import Libraries
# ----------------
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from neo.core import Block

from myogen import simulator

# Configure matplotlib for high-quality output
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

##############################################################################
# Configuration
# -------------
# Configure the input file path and output settings

# Input file - this should be the saved EMG signals Block object
# To generate this file, add the following to your simulation script after
# calling add_noise() or simulate_intramuscular_emg():
#   joblib.dump(noisy_emg_signals__Block, save_path / "emg_signals.pkl")
INPUT_FILE = "./results/synthetic_gen/noisy_surface_emg_signals.pkl"

# Output settings
OUTPUT_DIR = Path("./results/synthetic_gen/emg_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting parameters
TIME_WINDOW__s = None  # None = plot all, or (start_time, end_time) in seconds
MAX_CHANNELS_PER_FIGURE = 8  # Maximum number of channels per figure
FIGURE_HEIGHT_PER_CHANNEL = 1.5  # Height in inches per channel subplot

##############################################################################
# Load EMG Data
# -------------
#
# Load the EMG signals from the pickle file. This script handles both
# NEO Block objects (signal data) and simulator objects.

print(f"Loading data from: {INPUT_FILE}")
loaded_data = joblib.load(INPUT_FILE)

# Detect data type and extract EMG signals
if isinstance(loaded_data, Block):
    print("Successfully loaded NEO Block (direct EMG signal data)")
    emg_data = loaded_data
elif isinstance(loaded_data, simulator.IntramuscularEMG):
    print("Loaded IntramuscularEMG simulator object")
    if (
        hasattr(loaded_data, "_noisy_emg_signals__Block")
        and loaded_data._noisy_emg_signals__Block is not None
    ):
        print("  → Extracting noisy EMG signals from simulator")
        emg_data = loaded_data._noisy_emg_signals__Block
    elif (
        hasattr(loaded_data, "_emg_signals__Block") and loaded_data._emg_signals__Block is not None
    ):
        print("  → Extracting clean EMG signals from simulator")
        emg_data = loaded_data._emg_signals__Block
    else:
        raise ValueError(
            "IntramuscularEMG simulator has no generated signals. "
            "Please run simulate_intramuscular_emg() first, then save the returned Block object."
        )
elif isinstance(loaded_data, simulator.SurfaceEMG):
    print("Loaded SurfaceEMG simulator object")
    if (
        hasattr(loaded_data, "_noisy_emg_signals__Block")
        and loaded_data._noisy_emg_signals__Block is not None
    ):
        print("  → Extracting noisy EMG signals from simulator")
        emg_data = loaded_data._noisy_emg_signals__Block
    elif (
        hasattr(loaded_data, "_emg_signals__Block") and loaded_data._emg_signals__Block is not None
    ):
        print("  → Extracting clean EMG signals from simulator")
        emg_data = loaded_data._emg_signals__Block
    else:
        raise ValueError(
            "SurfaceEMG simulator has no generated signals. "
            "Please run simulate_surface_emg() first, then save the returned Block object."
        )
else:
    raise TypeError(
        f"Unsupported data type: {type(loaded_data).__name__}. "
        "Expected NEO Block, IntramuscularEMG, or SurfaceEMG object. "
        "Please save the EMG signals Block object directly using:\n"
        "  joblib.dump(noisy_emg_signals__Block, save_path / 'emg_signals.pkl')"
    )

##############################################################################
# Detect EMG Type and Extract Signals
# ------------------------------------
#
# Determine whether this is surface EMG or intramuscular EMG data.


def extract_emg_signals(block: Block):
    """
    Extract EMG signals and metadata from a NEO Block object.

    Parameters
    ----------
    block : neo.core.Block
        The NEO Block containing EMG data.

    Returns
    -------
    signals : np.ndarray
        The EMG signal data.
    times : np.ndarray
        Time axis in seconds.
    emg_type : str
        Either 'surface' or 'intramuscular'.
    shape_info : dict
        Dictionary containing shape information.
    """
    # Try to access as surface EMG first (has groups)
    if hasattr(block, "groups") and len(block.groups) > 0:
        # Surface EMG structure: groups[0].segments[0].analogsignals[0]
        analog_signal = block.groups[0].segments[0].analogsignals[0]
        emg_type = "surface"
        print("Detected Surface EMG data")
    elif hasattr(block, "segments") and len(block.segments) > 0:
        # Intramuscular EMG structure: segments[0].analogsignals[0]
        analog_signal = block.segments[0].analogsignals[0]
        emg_type = "intramuscular"
        print("Detected Intramuscular EMG data")
    else:
        raise ValueError("Could not extract signals from Block object")

    # Extract signal data and time axis
    signals = analog_signal.magnitude
    times = analog_signal.times.rescale("s").magnitude

    # Get shape information
    shape_info = {
        "n_samples": signals.shape[0],
        "signal_shape": signals.shape,
    }

    if emg_type == "surface":
        shape_info["n_rows"] = signals.shape[1]
        shape_info["n_cols"] = signals.shape[2]
        shape_info["n_channels"] = signals.shape[1] * signals.shape[2]
        print(
            f"  - Shape: {signals.shape[0]} samples × "
            f"{signals.shape[1]} rows × {signals.shape[2]} columns"
        )
        print(f"  - Total channels: {shape_info['n_channels']}")
    else:  # intramuscular
        shape_info["n_channels"] = signals.shape[1]
        print(f"  - Shape: {signals.shape[0]} samples × {signals.shape[1]} channels")

    print(f"  - Duration: {times[-1]:.2f} seconds")
    print(f"  - Sampling rate: {len(times) / times[-1]:.1f} Hz")

    return signals, times, emg_type, shape_info


signals, times, emg_type, shape_info = extract_emg_signals(emg_data)

##############################################################################
# Apply Time Window (Optional)
# ----------------------------
#
# Optionally select a specific time window for visualization.

if TIME_WINDOW__s is not None:
    start_time, end_time = TIME_WINDOW__s
    time_mask = (times >= start_time) & (times <= end_time)
    signals = signals[time_mask]
    times = times[time_mask]
    print(f"Applied time window: {start_time:.2f}s to {end_time:.2f}s")

##############################################################################
# Reshape Surface EMG Data
# ------------------------
#
# For surface EMG, reshape from 3D (time × rows × cols) to 2D (time × channels).

if emg_type == "surface":
    n_rows, n_cols = shape_info["n_rows"], shape_info["n_cols"]
    # Reshape to (time × channels)
    signals = signals.reshape(signals.shape[0], -1)
    print(f"Reshaped surface EMG to: {signals.shape[0]} samples × {signals.shape[1]} channels")

##############################################################################
# Create Channel Plots
# --------------------
#
# Generate plots showing all EMG channels, organized into multiple figures
# if necessary.


def plot_channels(signals, times, n_channels, max_channels_per_fig=8):
    """
    Create plots of EMG channels.

    Parameters
    ----------
    signals : np.ndarray
        Signal data (time × channels).
    times : np.ndarray
        Time axis in seconds.
    n_channels : int
        Number of channels to plot.
    max_channels_per_fig : int
        Maximum number of channels per figure.

    Returns
    -------
    figures : list
        List of matplotlib figure objects.
    """
    figures = []
    n_figures = int(np.ceil(n_channels / max_channels_per_fig))

    for fig_idx in range(n_figures):
        # Determine which channels to plot in this figure
        start_ch = fig_idx * max_channels_per_fig
        end_ch = min(start_ch + max_channels_per_fig, n_channels)
        n_subplots = end_ch - start_ch

        # Create figure with subplots
        fig_height = n_subplots * FIGURE_HEIGHT_PER_CHANNEL
        fig, axes = plt.subplots(n_subplots, 1, figsize=(12, fig_height), sharex=True)

        # Handle single subplot case
        if n_subplots == 1:
            axes = [axes]

        # Plot each channel
        for i, ax in enumerate(axes):
            ch_idx = start_ch + i
            channel_data = signals[:, ch_idx]

            # Plot the signal
            ax.plot(times, channel_data, linewidth=0.5, color="#2874A6", alpha=0.8)

            # Formatting
            if emg_type == "surface":
                # Convert linear channel index to (row, col)
                row = ch_idx // shape_info["n_cols"]
                col = ch_idx % shape_info["n_cols"]
                ax.set_ylabel(f"Ch ({row},{col})\n[uV]", fontsize=9)
                ax.set_title(
                    f"Electrode ({row},{col}) - Channel {ch_idx + 1}/{n_channels}",
                    fontsize=10,
                    pad=5,
                )
            else:
                ax.set_ylabel(f"Ch {ch_idx + 1}\n[uV]", fontsize=9)
                ax.set_title(f"Channel {ch_idx + 1}/{n_channels}", fontsize=10, pad=5)

            # Add grid and remove spines
            ax.grid(True, alpha=0.3, linewidth=0.5)
            sns.despine(ax=ax, trim=True, offset=5)

            # Only show x-label on bottom subplot
            if i == len(axes) - 1:
                ax.set_xlabel("Time [s]", fontsize=10)

        # Overall title
        fig.suptitle(
            f"{emg_type.capitalize()} EMG Channels {start_ch + 1}-{end_ch}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()
        figures.append(fig)

        print(f"Created figure {fig_idx + 1}/{n_figures} with channels {start_ch + 1}-{end_ch}")

    return figures


print("\nGenerating plots...")
figures = plot_channels(signals, times, shape_info["n_channels"], MAX_CHANNELS_PER_FIGURE)

##############################################################################
# Save Plots as PNG
# -----------------
#
# Save all generated figures as high-resolution PNG files.

print(f"\nSaving plots to: {OUTPUT_DIR}")

for fig_idx, fig in enumerate(figures):
    output_file = OUTPUT_DIR / f"emg_channels_fig{fig_idx + 1:02d}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {output_file.name}")

print(
    f"\nCompleted! Generated {len(figures)} figure(s) with {shape_info['n_channels']} total channels."
)

##############################################################################
# Create Overview Plot
# --------------------
#
# Generate a single overview plot showing all channels in a compact format.

print("\nGenerating overview plot...")

fig_overview, ax = plt.subplots(figsize=(14, 8))

# Plot all channels with offset for visibility
n_channels = shape_info["n_channels"]
channel_offset = np.max(np.abs(signals)) * 2.5  # Spacing between channels

for ch_idx in range(n_channels):
    offset = ch_idx * channel_offset
    ax.plot(
        times,
        signals[:, ch_idx] + offset,
        linewidth=0.3,
        alpha=0.7,
        label=f"Ch {ch_idx + 1}" if n_channels <= 16 else None,
    )

# Formatting
ax.set_xlabel("Time [s]", fontsize=12)
ax.set_ylabel("Channel + Offset [uV]", fontsize=12)
ax.set_title(
    f"{emg_type.capitalize()} EMG - All Channels Overview ({n_channels} channels)",
    fontsize=14,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)
sns.despine(trim=True, offset=5)

# Add legend only if not too many channels
if n_channels <= 16:
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=True)

plt.tight_layout()

# Save overview
overview_file = OUTPUT_DIR / "emg_channels_overview.png"
fig_overview.savefig(overview_file, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  ✓ Saved: {overview_file.name}")

print("\n" + "=" * 60)
print("All plots saved successfully!")
print(f"Output directory: {OUTPUT_DIR.absolute()}")
print("=" * 60 + "\n")

plt.show()

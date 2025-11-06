r"""
Surface EMG Grid Visualization
================================

Visualizes 5×5 surface EMG electrode grid from decomposition files.

Creates a subplot grid matching the physical electrode layout, showing
the full time-series EMG signal for each electrode position.

Usage:
------
python examples/synthetic_gen/13_plot_emg_grid.py \
    --decomp-file results/synthetic_gen/semg_mu_41_42_43_plus18_snr10/decomp.pkl \
    --dpi 300
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scienceplots  # noqa

##############################################################################
# Configure Matplotlib Style
##############################################################################

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=1.5)

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
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.width"] = 1.5

# Remove minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False


##############################################################################
# Data Loading Functions
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
    required_keys = ["emg_signal", "sampling_rate_hz"]
    missing_keys = [key for key in required_keys if key not in decomp]
    if missing_keys:
        raise ValueError(f"Decomposition file missing required keys: {missing_keys}")

    return decomp


##############################################################################
# Plotting Functions
##############################################################################


def plot_emg_grid(decomp, output_path, time_window=None):
    """
    Plot 5×5 surface EMG electrode grid.

    Parameters
    ----------
    decomp : dict
        Decomposition data dictionary containing emg_signal and metadata.
    output_path : Path
        Output file path for saving the plot.
    time_window : tuple, optional
        Time window (start_s, end_s) to plot. If None, plots full recording.
    """
    emg_signal = decomp["emg_signal"]
    sampling_rate_hz = decomp["sampling_rate_hz"]

    # Validate shape
    if emg_signal.ndim != 3:
        raise ValueError(f"Expected 3D EMG signal (n_samples, rows, cols), got shape {emg_signal.shape}")

    n_samples, n_rows, n_cols = emg_signal.shape

    if n_rows != 5 or n_cols != 5:
        print(f"⚠️  Warning: Expected 5×5 grid, got {n_rows}×{n_cols}. Adjusting layout...")

    # Create time vector
    time_s = np.arange(n_samples) / sampling_rate_hz
    duration_s = n_samples / sampling_rate_hz

    # Apply time window if specified
    if time_window is not None:
        start_s, end_s = time_window
        start_idx = int(start_s * sampling_rate_hz)
        end_idx = int(end_s * sampling_rate_hz)
        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)

        time_s = time_s[start_idx:end_idx]
        emg_signal = emg_signal[start_idx:end_idx, :, :]
        print(f"  Plotting time window: {start_s:.2f} - {end_s:.2f} s")
    else:
        print(f"  Plotting full recording: 0 - {duration_s:.2f} s")

    # Find global amplitude range for consistent scaling
    global_min = np.min(emg_signal)
    global_max = np.max(emg_signal)
    amplitude_range = global_max - global_min
    y_margin = amplitude_range * 0.05  # 5% margin

    print(f"  Amplitude range: [{global_min:.2e}, {global_max:.2e}] μV")

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12),
                             sharex=True, sharey=True)

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each electrode
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            signal = emg_signal[:, row, col]

            # Plot signal
            ax.plot(time_s, signal, color='#1f77b4', linewidth=0.5, alpha=0.8)

            # Set consistent y-limits
            ax.set_ylim(global_min - y_margin, global_max + y_margin)

            # Add electrode label
            ax.text(0.02, 0.98, f'[{row},{col}]',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))

            # Remove all axis elements for clean look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    # Add shared axis labels
    fig.text(0.5, 0.02, 'Time (s)', ha='center', fontsize=14, weight='bold')
    fig.text(0.02, 0.5, 'Amplitude (μV)', va='center', rotation='vertical',
             fontsize=14, weight='bold')

    # Add title with metadata
    snr_db = decomp.get('snr_db', 'N/A')
    title = f'Surface EMG - {n_rows}×{n_cols} Electrode Grid\n'
    title += f'Duration: {duration_s:.1f}s | Sampling Rate: {sampling_rate_hz:.0f} Hz | SNR: {snr_db} dB'
    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)

    # Adjust spacing
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.94,
                       hspace=0.08, wspace=0.08)

    # Save figure
    plt.savefig(output_path, dpi=plt.rcParams["savefig.dpi"], bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved EMG grid plot: {output_path}")


##############################################################################
# Main Function
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize 5×5 surface EMG electrode grid"
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
        help="DPI for output figure (default: 300)",
    )
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="Time window to plot in seconds (e.g., --time-window 5.0 7.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output filename (default: emg_grid_5x5.svg in decomp folder)",
    )

    args = parser.parse_args()

    # Update DPI if specified
    if args.dpi != 300:
        plt.rcParams["figure.dpi"] = args.dpi
        plt.rcParams["savefig.dpi"] = args.dpi

    print("=" * 80)
    print("Surface EMG Grid Visualization")
    print("=" * 80)
    print(f"Input: {args.decomp_file}")

    # Load decomposition data
    print("\n📂 Loading decomposition data...")
    decomp = load_decomposition(args.decomp_file)

    # Print summary
    emg_signal = decomp["emg_signal"]
    n_samples, n_rows, n_cols = emg_signal.shape
    sampling_rate_hz = decomp["sampling_rate_hz"]
    duration_s = n_samples / sampling_rate_hz

    print(f"  Grid size: {n_rows}×{n_cols}")
    print(f"  Samples: {n_samples}")
    print(f"  Duration: {duration_s:.2f} s")
    print(f"  Sampling rate: {sampling_rate_hz:.0f} Hz")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        decomp_folder = args.decomp_file.parent
        output_path = decomp_folder / "emg_grid_5x5.svg"

    # Create time window tuple if specified
    time_window = tuple(args.time_window) if args.time_window else None

    # Plot EMG grid
    print("\n📊 Creating EMG grid plot...")
    plot_emg_grid(decomp, output_path, time_window=time_window)

    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

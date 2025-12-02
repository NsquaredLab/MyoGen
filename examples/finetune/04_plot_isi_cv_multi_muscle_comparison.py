"""
Multi-Muscle ISI and CV Statistics Comparison Plot
===================================================

This script loads and visualizes ISI/CV data from multiple muscle types across
multiple force levels (auto-detected) and compares them against experimental data.

Each muscle type is assigned a distinct color-to-white gradient colormap, with
points fading from the base color (early recruited units) to white (late recruited
units) based on recruitment order. All glyphs have black edges for visibility.

Force levels are distinguished by marker shapes (auto-detected from available data).

Usage:
------
python plot_isi_cv_multi_muscle_comparison.py \
    --muscles THIRTY TWENTYFIVE TWENTY FIFTEEN TEN FIVE \
    --output-format jpg
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

plt.style.use("fivethirtyeight")

##############################################################################
# Configure Matplotlib Style
# ---------------------------

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (10, 8)

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
# CONFIGURATION
# -------------

# Default path to results directory
RESULTS_PATH = Path("./results")

# Muscle-specific colormaps (base color to white gradients)
MUSCLE_COLORMAPS = {
    "TEST": "Reds",
    "THIRTY": "Reds",
    "TWENTYFIVE": "Blues",
    "TWENTY": "Greens",
    "FIFTEEN": "Purples",
    "TEN": "Oranges",
    "FIVE": "YlOrBr",
}

# Base colors for legend (extracted from colormaps)
MUSCLE_LEGEND_COLORS = {
    "TEST": "#d62728",  # Red
    "THIRTY": "#d62728",  # Red
    "TWENTYFIVE": "#1f77b4",  # Blue
    "TWENTY": "#2ca02c",  # Green
    "FIFTEEN": "#9467bd",  # Purple
    "TEN": "#ff7f0e",  # Orange
    "FIVE": "#8c564b",  # Brown
}

# Experimental data colors (all gray)
EXP_COLORS = {"VM": "#808080", "VL": "#808080", "TA": "#808080", "FDI": "#808080"}

# Available marker styles (will cycle through these for force levels)
AVAILABLE_MARKERS = ["o", "s", "^", "D", "v", "p", "*", "h", "<", ">", "d", "P", "X"]


##############################################################################
# Data Loading Functions
# -----------------------


def auto_detect_force_levels(results_path, muscle_type, study_prefix, short_muscle):
    """
    Auto-detect available force levels for a given muscle type.

    Parameters
    ----------
    results_path : Path
        Path to results directory containing CSV files.
    muscle_type : str
        Full muscle type identifier (e.g., "THIRTY_gamma2.0-3.0").
    study_prefix : str
        Study prefix for file naming (e.g., "THIRTY_gamma2.0-3.0_").
    short_muscle : str
        Short muscle name used in filename (e.g., "THIRTY").

    Returns
    -------
    list of int
        Sorted list of detected force levels.
    """
    pattern = f"{study_prefix}isi_cv_data_{short_muscle}_*.csv"
    files = list(results_path.glob(pattern))

    force_levels = []
    for file in files:
        # Extract force level from filename
        # Pattern: {prefix}isi_cv_data_{short_muscle}_{force}.csv
        parts = file.stem.split("_")
        try:
            force_level = int(parts[-1])
            force_levels.append(force_level)
        except (ValueError, IndexError):
            continue

    return sorted(force_levels)


def load_multi_muscle_data(results_path, muscle_types):
    """
    Load ISI/CV data for multiple muscle types with auto-detected force levels.

    Parameters
    ----------
    results_path : Path
        Path to results directory containing CSV files.
    muscle_types : list of str
        List of muscle type identifiers (e.g., ["THIRTY_gamma2.0-3.0", "TWENTYFIVE_gamma2.0-3.0"]).

    Returns
    -------
    dict
        Nested dictionary: {muscle: {force_level: DataFrame}}
        DataFrame columns:
        - MU_ID: Motor unit identifier
        - mean_firing_rate_Hz: Mean firing rate in Hz
        - CV_ISI: Coefficient of variation of inter-spike intervals
    """
    all_data = {}

    for muscle in muscle_types:
        study_prefix = f"{muscle}_"

        # Extract short muscle name (first part before underscore)
        short_muscle = muscle.split("_")[0]

        # Auto-detect force levels
        force_levels = auto_detect_force_levels(results_path, muscle, study_prefix, short_muscle)

        if not force_levels:
            print(f"No data found for {muscle}")
            continue

        print(f"Detected force levels for {muscle}: {force_levels}")

        muscle_data = {}
        for force in force_levels:
            file_path = results_path / f"{study_prefix}isi_cv_data_{short_muscle}_{force}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if len(df) == 0:
                        print(f"Empty data for {force}% in {file_path.name} - skipping")
                        continue
                    muscle_data[force] = df
                    print(f"Loaded {len(df)} motor units for {force}% from {file_path.name}")
                except pd.errors.EmptyDataError:
                    print(f"Malformed/empty file for {force}%: {file_path.name} - skipping")
                except Exception as e:
                    print(f"Error reading {file_path.name}: {e}")
            else:
                print(f"File not found: {file_path}")

        if muscle_data:
            all_data[muscle] = muscle_data

    return all_data


def load_experimental_data(csv_path):
    """
    Load experimental ISI statistics from CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to ISI_statistics.csv file.

    Returns
    -------
    pd.DataFrame
        Experimental data with columns including:
        - Muscle: Muscle identifier
        - Force Level: Force level (%)
        - FR mean: Mean firing rate (pps)
        - ISI CV: Coefficient of variation
    """
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded experimental data: {len(df)} records")
        return df
    else:
        print(f"Experimental data file not found: {csv_path}")
        return None


##############################################################################
# Color Mapping Functions
# ------------------------


def get_muscle_colors(recruitment_order, colormap_name="Reds"):
    """
    Generate colors from base color to white based on recruitment order.

    Early recruited units (low index) receive the full base color,
    while late recruited units (high index) fade to white.

    Parameters
    ----------
    recruitment_order : np.ndarray
        Array of recruitment indices (typically MU_ID).
    colormap_name : str
        Matplotlib colormap name (e.g., 'Reds', 'Blues', 'Greens').

    Returns
    -------
    np.ndarray
        Array of RGBA colors with shape (N, 4).
    """
    # Get colormap
    cmap = plt.get_cmap(colormap_name)

    # Normalize recruitment order to [0, 1]
    norm_recruitment = (recruitment_order - recruitment_order.min()) / (
        recruitment_order.max() - recruitment_order.min() + 1e-10
    )

    # Reverse the mapping: early recruited (0) → full color (1.0)
    #                      late recruited (1) → white (0.0)
    color_values = 1.0 - norm_recruitment

    # Sample from colormap
    colors = cmap(color_values)

    return colors


def generate_force_markers(force_levels):
    """
    Generate marker mappings for force levels.

    Parameters
    ----------
    force_levels : set or list
        Set of force levels (e.g., {5, 15, 30, 50}).

    Returns
    -------
    dict
        Dictionary mapping force_level to marker shape.
    """
    force_markers = {}

    for i, force in enumerate(sorted(force_levels)):
        # Assign marker (cycle through available markers)
        force_markers[force] = AVAILABLE_MARKERS[i % len(AVAILABLE_MARKERS)]

    return force_markers


##############################################################################
# Plotting Function
# -----------------


def plot_cv_vs_fr_multi_muscle(all_muscle_data, exp_data):
    """
    Create CV vs FR plot comparing multiple muscles across force levels.

    Parameters
    ----------
    all_muscle_data : dict
        Nested dictionary: {muscle: {force_level: DataFrame}}.
    exp_data : pd.DataFrame
        Experimental ISI statistics.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect all force levels across all muscles
    all_force_levels = set()
    for muscle_data in all_muscle_data.values():
        all_force_levels.update(muscle_data.keys())

    force_markers = generate_force_markers(all_force_levels)

    # 1. Plot experimental data (convex hulls + scatter) - INCLUDING FDI
    if exp_data is not None:
        muscles = exp_data["Muscle"].unique()
        for muscle in muscles:
            # No longer skip FDI - include all experimental muscles
            muscle_data = exp_data[exp_data["Muscle"] == muscle]
            cv_data = muscle_data["ISI CV"].values
            fr_data = muscle_data["FR mean"].values

            # Draw convex hull for experimental data
            if len(cv_data) > 2:
                points = np.column_stack([cv_data, fr_data])
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    polygon = Polygon(
                        hull_points,
                        facecolor=EXP_COLORS.get(muscle, "#808080"),
                        alpha=0.25,
                        edgecolor=EXP_COLORS.get(muscle, "#808080"),
                        linewidth=1.5,
                        linestyle="-",
                        zorder=0,
                    )
                    ax.add_patch(polygon)
                except Exception:
                    # Skip if convex hull cannot be computed
                    pass

            # Scatter points for experimental data
            ax.scatter(
                cv_data,
                fr_data,
                s=20,
                alpha=1.0,
                color=EXP_COLORS.get(muscle, "#808080"),
                edgecolors="white",
                linewidth=0.5,
                marker="x",
                zorder=1,
            )

    # 2. Plot simulated data for each muscle and force level
    for muscle_type in sorted(all_muscle_data.keys()):
        muscle_data = all_muscle_data[muscle_type]

        # Extract short muscle name (first part before underscore)
        # e.g., "THIRTY_gamma0.5-0.75" → "THIRTY"
        short_muscle = muscle_type.split("_")[0]
        colormap_name = MUSCLE_COLORMAPS.get(short_muscle, "Greys")

        for force_level in sorted(muscle_data.keys()):
            df = muscle_data[force_level]

            # Extract recruitment order from MU_ID
            if "MU_ID" in df.columns:
                recruitment_order = df["MU_ID"].values
            else:
                recruitment_order = np.arange(len(df))

            # Get color-to-white gradient for this muscle
            colors = get_muscle_colors(recruitment_order, colormap_name)

            # Get marker for this force level
            marker = force_markers.get(force_level, "o")

            # Plot with muscle-specific colors and force-level-specific marker
            ax.scatter(
                df["CV_ISI"],
                df["mean_firing_rate_Hz"],
                s=50,
                alpha=0.8,
                c=colors,
                edgecolors="black",
                linewidth=0.6,
                marker=marker,
                zorder=2,
            )

    # 3. Create combined legend
    legend_elements = []

    # Section 1: Muscle types (colored patches)
    legend_elements.append(Line2D([0], [0], color="none", label="Muscle Types:", marker=""))
    for muscle in sorted(all_muscle_data.keys()):
        # Extract short muscle name for cleaner legend
        short_muscle = muscle.split("_")[0]
        color = MUSCLE_LEGEND_COLORS.get(short_muscle, "#000000")
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=f"  {short_muscle}"))

    # Add spacing
    legend_elements.append(Line2D([0], [0], color="none", label=" ", marker=""))

    # Section 2: Force levels (marker shapes)
    legend_elements.append(Line2D([0], [0], color="none", label="Force Levels:", marker=""))
    for force in sorted(all_force_levels):
        marker = force_markers[force]
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                label=f"{force}%",
                linewidth=0,
            )
        )

    # Add legend to plot
    ax.legend(
        handles=legend_elements,
        frameon=True,
        fontsize=9,
        framealpha=1.0,
        edgecolor="none",
        loc="upper right",
        ncol=1,
    )

    # 4. Format plot
    ax.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
    ax.set_ylabel("Mean Firing Rate (pps)", fontsize=12)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(4, 25)
    ax.set_title("ISI Statistics Comparison - Multi-Muscle", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    # sns.despine(ax=ax, offset=10, trim=True)

    return fig, ax


##############################################################################
# Main Execution
# --------------


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot ISI/CV comparison for multiple muscles with auto-detected force levels"
    )
    parser.add_argument(
        "--muscles",
        type=str,
        nargs="+",
        default=["TEST"],
        help="Muscle types to compare (e.g., THIRTY TWENTYFIVE TWENTY FIFTEEN TEN FIVE)",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
        help=f"Path to results directory (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "svg", "pdf"],
        help="Output format for figures (default: jpg)",
    )

    args = parser.parse_args()

    # Use muscle names as provided (no conversion)
    muscles = args.muscles

    print("=" * 80)
    print("Multi-Muscle ISI/CV Comparison Plot")
    print("=" * 80)
    print(f"\tMuscles: {', '.join(muscles)}")
    print(f"\tOutput Format: {args.output_format.upper()}")

    # Load simulation data with auto-detection
    print("\nLoading simulation data (auto-detecting force levels)...")
    all_muscle_data = load_multi_muscle_data(args.results_path, muscles)

    if not all_muscle_data:
        print("\nNo simulation data found. Please run extract_isi_and_cv_per_ramps.py first.")
        exit(1)

    # Load experimental data
    print("\nLoading experimental data...")
    exp_csv_path = Path(__file__).parent / "ISI_statistics.csv"
    exp_data = load_experimental_data(exp_csv_path)

    # Create comparison plot
    print("\nCreating multi-muscle comparison plot...")
    fig, ax = plot_cv_vs_fr_multi_muscle(all_muscle_data, exp_data)

    # Generate descriptive output filename
    muscle_str = "_".join(muscles)
    output_file = args.results_path / f"isi_cv_comparison_{muscle_str}.{args.output_format}"
    plt.tight_layout()

    # Set quality based on format
    if args.output_format in ["jpg", "jpeg"]:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)

    print(f"\nPlot saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_motor_units = 0
    for muscle in sorted(all_muscle_data.keys()):
        print(f"\n{muscle}:")
        muscle_data = all_muscle_data[muscle]

        for force_level in sorted(muscle_data.keys()):
            df = muscle_data[force_level]
            total_motor_units += len(df)

            print(f"\n{force_level}% Force:")
            print(f"\tMotor units (N): {len(df)}")
            print(
                f"\tMean firing rate: {df['mean_firing_rate_Hz'].mean():.2f} ± "
                f"{df['mean_firing_rate_Hz'].std():.2f} Hz"
            )
            print(f"\tMean CV: {df['CV_ISI'].mean():.3f} ± {df['CV_ISI'].std():.3f}")
            print(
                f"\tFR range: {df['mean_firing_rate_Hz'].min():.2f} - "
                f"{df['mean_firing_rate_Hz'].max():.2f} Hz"
            )
            print(f"\tCV range: {df['CV_ISI'].min():.3f} - {df['CV_ISI'].max():.3f}")

    print(f"\n{'=' * 80}")
    print(f"Total motor units plotted: {total_motor_units}")
    print(f"Total muscle types: {len(all_muscle_data)}")
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

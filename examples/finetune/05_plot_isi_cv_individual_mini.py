r"""
Individual Mini Plots for ISI and CV Statistics
===============================================

This script generates small individual plots for each muscle-gamma-force
combination, showing CV vs Firing Rate without experimental data overlay.

Visual encoding:
- Color: Muscle-gamma specific color with recruitment gradient
- Each plot: One muscle-gamma-force combination only
- No experimental data, no legend, minimal clutter

Usage:
------
python plot_isi_cv_individual_mini.py \
    --muscles THIRTY TWENTYFIVE TWENTY FIFTEEN TEN FIVE \
    --output-format jpg \
    --exclude-gammas 50 30 \
    --min-firing-rate 4.0
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots  # noqa
import colorsys

##############################################################################
# Configure Matplotlib Style
##############################################################################

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (2, 2)  # Very small plots

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

##############################################################################
# CONFIGURATION
##############################################################################

# Default path to results directory
RESULTS_PATH = Path("./results")

# Base colors for each muscle type
MUSCLE_BASE_COLORS = {
    "THIRTY": "#d62728",  # Red
    "TWENTYFIVE": "#1f77b4",  # Blue
    "TWENTY": "#2ca02c",  # Green
    "FIFTEEN": "#9467bd",  # Purple
    "TEN": "#ff7f0e",  # Orange
    "FIVE": "#8c564b",  # Brown
}

##############################################################################
# Color Utility Functions
##############################################################################


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex color."""
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def generate_color_shades(base_hex, n_shades):
    """
    Generate n_shades of a base color from light to dark.

    Parameters
    ----------
    base_hex : str
        Base color in hex format (e.g., "#d62728").
    n_shades : int
        Number of shades to generate.

    Returns
    -------
    list of str
        List of hex colors from light to dark.
    """
    if n_shades == 1:
        return [base_hex]

    # Convert to RGB
    rgb = hex_to_rgb(base_hex)

    # Convert to HSL for easier manipulation
    h, l, s = colorsys.rgb_to_hls(*rgb)  # noqa: E741

    # Generate shades from light to dark
    shades = []
    for i in range(n_shades):
        # Interpolate lightness from 0.85 (light) to base lightness (dark)
        t = i / max(1, n_shades - 1)
        new_l = 0.85 * (1 - t) + l * t

        # Convert back to RGB then hex
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        shades.append(rgb_to_hex(new_rgb))

    return shades


##############################################################################
# Sorting Helper Functions
##############################################################################


def muscle_name_to_number(muscle_name):
    """Convert muscle name to its numerical value for sorting."""
    return {
        "FIVE": 5,
        "TEN": 10,
        "FIFTEEN": 15,
        "TWENTY": 20,
        "TWENTYFIVE": 25,
        "THIRTY": 30,
    }.get(muscle_name, 0)


def gamma_str_to_number(gamma_str):
    """Convert gamma string to numerical value for sorting."""
    gamma_value = gamma_str.replace("gamma", "")
    if "-" in gamma_value:
        gamma_value = gamma_value.split("-")[0]
    try:
        return float(gamma_value)
    except ValueError:
        return 0.0


def parse_filename(filename):
    """
    Parse filename to extract muscle, gamma, and force information.

    Expected pattern: {MUSCLE}_gamma{X}-{Y}_isi_cv_data_{MUSCLE}_{FORCE}.csv

    Parameters
    ----------
    filename : str
        Filename to parse.

    Returns
    -------
    tuple or None
        (muscle, gamma_str, force) if successful, None otherwise.
    """
    stem = Path(filename).stem

    # Split by underscore
    parts = stem.split("_")

    try:
        # Find "gamma" part
        gamma_idx = None
        for i, part in enumerate(parts):
            if part.startswith("gamma"):
                gamma_idx = i
                break

        if gamma_idx is None:
            return None

        # Extract muscle (everything before gamma)
        muscle = parts[0]

        # Extract gamma (the gamma part itself)
        gamma_str = parts[gamma_idx]

        # Extract force (last part after splitting)
        force = int(parts[-1])

        return (muscle, gamma_str, force)
    except (ValueError, IndexError):
        return None


def auto_detect_all_data(results_path, muscle_types):
    """
    Auto-detect all muscle/gamma/force combinations in results directory.

    Parameters
    ----------
    results_path : Path
        Path to results directory.
    muscle_types : list of str
        List of muscle types to search for.

    Returns
    -------
    dict
        Nested dictionary: {muscle: {gamma: {force: (filepath, DataFrame)}}}
    """
    all_data = defaultdict(lambda: defaultdict(dict))

    for muscle in muscle_types:
        pattern = f"{muscle}_gamma*_isi_cv_data_*_*.csv"
        files = list(results_path.glob(pattern))

        for file in files:
            parsed = parse_filename(file.name)
            if parsed is None:
                continue

            muscle_name, gamma_str, force = parsed

            # Load data
            try:
                df = pd.read_csv(file)
                all_data[muscle_name][gamma_str][force] = (file, df)
            except Exception:
                print(f"✗ Failed: {file.name}")

    return dict(all_data)


def generate_muscle_gamma_colors(all_data):
    """
    Generate color mappings for muscle-gamma combinations.

    Parameters
    ----------
    all_data : dict
        Nested dictionary: {muscle: {gamma: {force: (filepath, DataFrame)}}}.

    Returns
    -------
    dict
        Nested dictionary: {muscle: {gamma: hex_color}}.
    """
    color_map = {}

    for muscle in sorted(all_data.keys(), key=muscle_name_to_number):
        base_color = MUSCLE_BASE_COLORS.get(muscle, "#000000")
        gammas = sorted(all_data[muscle].keys(), key=gamma_str_to_number)
        n_gammas = len(gammas)

        # Generate shades for this muscle
        shades = generate_color_shades(base_color, n_gammas)

        # Map each gamma to a shade
        color_map[muscle] = {gamma: shades[i] for i, gamma in enumerate(gammas)}

    return color_map


def get_muscle_colors(recruitment_order, base_color):
    """
    Generate colors with recruitment order gradient (early=full color, late=white).

    Parameters
    ----------
    recruitment_order : np.ndarray
        Array of recruitment indices (typically MU_ID).
    base_color : str
        Hex color for the muscle-gamma combination.

    Returns
    -------
    np.ndarray
        Array of RGBA colors with shape (N, 4).
    """
    from matplotlib.colors import LinearSegmentedColormap

    rgb = hex_to_rgb(base_color)
    colors_list = [rgb, (1.0, 1.0, 1.0)]  # Base color to white
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors_list, N=n_bins)

    # Normalize recruitment order to [0, 1]
    norm_recruitment = (recruitment_order - recruitment_order.min()) / (
        recruitment_order.max() - recruitment_order.min() + 1e-10
    )

    # Reverse: early recruited (0) → full color (1.0), late recruited (1) → white (0.0)
    color_values = 1.0 - norm_recruitment

    # Sample from colormap
    colors = cmap(color_values)

    return colors


##############################################################################
# Plotting Function
##############################################################################


def plot_individual_mini(df, muscle, gamma, force, base_color, output_path):
    """
    Create a small individual plot for one muscle-gamma-force combination.

    Parameters
    ----------
    df : pd.DataFrame
        Data for this combination.
    muscle : str
        Muscle name.
    gamma : str
        Gamma value string.
    force : int
        Force level.
    base_color : str
        Hex color for this muscle-gamma combination.
    output_path : Path
        Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(2, 2))

    # Extract recruitment order
    if "MU_ID" in df.columns:
        recruitment_order = df["MU_ID"].values
    else:
        recruitment_order = np.arange(len(df))

    # Get colors with recruitment gradient
    colors = get_muscle_colors(recruitment_order, base_color)

    # Plot
    ax.scatter(
        df["CV_ISI"],
        df["mean_firing_rate_Hz"],
        s=50,
        alpha=0.8,
        c=colors,
        edgecolors="black",
        linewidth=0.6,
        marker="o",
        zorder=2,
    )

    # Format plot
    ax.set_xlim(0, 1.0)
    ax.set_ylim(3.5, 35)
    ax.tick_params(axis="both", labelsize=10)
    sns.despine(ax=ax, offset=10, trim=True)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


##############################################################################
# Main Execution
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate individual mini plots for each muscle-gamma-force combination"
    )
    parser.add_argument(
        "--muscles",
        type=str,
        nargs="+",
        required=True,
        help="Muscle types to plot (e.g., THIRTY TWENTYFIVE TWENTY)",
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
    parser.add_argument(
        "--exclude-gammas",
        type=str,
        nargs="*",
        default=[],
        help="Gamma values to exclude (e.g., gamma50 gamma30 or 50 30)",
    )
    parser.add_argument(
        "--min-firing-rate",
        type=float,
        default=4.0,
        help="Minimum firing rate (Hz) to include motor units (default: 4.0 Hz)",
    )

    args = parser.parse_args()

    print(f"Individual Mini Plots: {', '.join(args.muscles)}")

    # Auto-detect all data
    all_data = auto_detect_all_data(args.results_path, args.muscles)

    if not all_data:
        print("❌ No simulation data found.")
        exit(1)

    # Filter out excluded gammas
    if args.exclude_gammas:
        exclude_set = set()
        for gamma in args.exclude_gammas:
            if gamma.startswith("gamma"):
                exclude_set.add(gamma)
            else:
                exclude_set.add(f"gamma{gamma}")

        for muscle in list(all_data.keys()):
            for gamma in list(all_data[muscle].keys()):
                if gamma in exclude_set:
                    del all_data[muscle][gamma]

            if not all_data[muscle]:
                del all_data[muscle]

        if not all_data:
            print("❌ No data remaining after excluding gammas.")
            exit(1)

    # Filter out motor units below minimum firing rate
    if args.min_firing_rate > 0:
        total_mus_before = 0
        total_mus_after = 0

        for muscle in list(all_data.keys()):
            for gamma in list(all_data[muscle].keys()):
                for force in list(all_data[muscle][gamma].keys()):
                    filepath, df = all_data[muscle][gamma][force]
                    total_mus_before += len(df)

                    df_filtered = df[df["mean_firing_rate_Hz"] >= args.min_firing_rate]
                    total_mus_after += len(df_filtered)

                    if len(df_filtered) > 0:
                        all_data[muscle][gamma][force] = (filepath, df_filtered)
                    else:
                        del all_data[muscle][gamma][force]

                if not all_data[muscle][gamma]:
                    del all_data[muscle][gamma]

            if not all_data[muscle]:
                del all_data[muscle]

        print(f"Filtered MUs: {total_mus_before} → {total_mus_after}")

        if not all_data:
            print("❌ No data remaining after filtering by firing rate.")
            exit(1)

    # Generate color mappings
    muscle_gamma_colors = generate_muscle_gamma_colors(all_data)

    # Create output directory
    output_dir = args.results_path / "mini_plots"
    output_dir.mkdir(exist_ok=True)

    # Generate individual plots
    total_plots = 0
    gamma_dirs = set()

    for muscle in sorted(all_data.keys(), key=muscle_name_to_number):
        for gamma in sorted(all_data[muscle].keys(), key=gamma_str_to_number):
            # Create gamma subdirectory
            gamma_dir = output_dir / gamma
            gamma_dir.mkdir(exist_ok=True)
            gamma_dirs.add(gamma)

            for force in sorted(all_data[muscle][gamma].keys()):
                filepath, df = all_data[muscle][gamma][force]
                base_color = muscle_gamma_colors[muscle][gamma]

                # Generate output filename (without gamma in name since it's in the directory)
                output_file = gamma_dir / f"{muscle}_{force}.{args.output_format}"

                # Create plot
                plot_individual_mini(df, muscle, gamma, force, base_color, output_file)
                total_plots += 1

    print(f"Created {total_plots} mini plots in {output_dir} ({len(gamma_dirs)} gamma folders) ✅")


if __name__ == "__main__":
    main()

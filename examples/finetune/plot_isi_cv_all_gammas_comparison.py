r"""
Multi-Muscle, Multi-Gamma ISI and CV Statistics Comparison Plot
================================================================

This script loads and visualizes ISI/CV data from multiple muscle types across
multiple gamma values and force levels (all auto-detected) and compares them
against experimental data.

Visual encoding:
- Color: Muscle type (THIRTY=red, TWENTYFIVE=blue, etc.)
- Color shading: Different gamma values (light to dark shades of base color)
- Marker shape: Force level/MVC percentage
- Experimental data: Gray convex hulls

Usage:
------
python plot_isi_cv_all_gammas_comparison.py \
    --muscles THIRTY TWENTYFIVE TWENTY FIFTEEN TEN FIVE \
    --output-format jpg \
    --exclude-gammas 50 30 \  # Optional: exclude specific gamma values
    --min-firing-rate 4.0      # Optional: exclude MUs below 4 Hz (default: 4.0)
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
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import colorsys

##############################################################################
# Configure Matplotlib Style
# ---------------------------

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 10)

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

# Base colors for each muscle type
MUSCLE_BASE_COLORS = {
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
# Color Utility Functions
# -----------------------


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
# Data Loading Functions
# -----------------------


##############################################################################
# Sorting Helper Functions
# ------------------------


def muscle_name_to_number(muscle_name):
    """
    Convert muscle name to its numerical value for sorting.

    Parameters
    ----------
    muscle_name : str
        Muscle name (e.g., "FIVE", "TEN", "FIFTEEN").

    Returns
    -------
    int
        Numerical value of the muscle name.
    """
    return {
        "FIVE": 5,
        "TEN": 10,
        "FIFTEEN": 15,
        "TWENTY": 20,
        "TWENTYFIVE": 25,
        "THIRTY": 30,
    }.get(muscle_name, 0)


def gamma_str_to_number(gamma_str):
    """
    Convert gamma string to numerical value for sorting.

    Parameters
    ----------
    gamma_str : str
        Gamma string (e.g., "gamma0.5", "gamma1.0", "gamma15").

    Returns
    -------
    float
        Numerical value of the gamma.
    """
    # Remove "gamma" prefix and extract first number
    gamma_value = gamma_str.replace("gamma", "")

    # Handle range format like "0.1-0.5" by taking the first number
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
        List of muscle types to search for (e.g., ["THIRTY", "TWENTYFIVE"]).

    Returns
    -------
    dict
        Nested dictionary: {muscle: {gamma: {force: (filepath, DataFrame)}}}
    """
    all_data = defaultdict(lambda: defaultdict(dict))

    # Search for all CSV files matching the pattern (flexible middle muscle identifier)
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
            except Exception as e:
                print(f"✗ Failed: {file.name}")

    return dict(all_data)


def load_experimental_data(csv_path):
    """
    Load experimental ISI statistics from CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to ISI_statistics.csv file.

    Returns
    -------
    pd.DataFrame or None
        Experimental data with columns including Muscle, Force Level, FR mean, ISI CV.
    """
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"✗ Experimental data not found")
        return None


##############################################################################
# Marker and Color Generation
# ----------------------------


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
        force_markers[force] = AVAILABLE_MARKERS[i % len(AVAILABLE_MARKERS)]

    return force_markers


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
    # Create a custom colormap from base color to white
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
# Geometric Coverage (Asymmetric)
# --------------------------------


def calculate_geometric_coverage(shape_exp, shape_sim, x_range):
    """
    Calculate what percentage of experimental shape area is covered by simulated shape.

    This asymmetric geometric metric answers: "What fraction of the experimental
    normalized shape's area falls within/under the simulated normalized shape?"

    Coverage = ∫ min(shape_exp, shape_sim) dx / ∫ shape_exp dx

    Where:
    - Numerator = intersection area (where both shapes overlap)
    - Denominator = total area under experimental shape

    Parameters
    ----------
    shape_exp : np.ndarray
        Normalized experimental shape curve (normalized to max=1.0).
    shape_sim : np.ndarray
        Normalized simulated shape curve (normalized to max=1.0).
    x_range : np.ndarray
        X-axis values where shapes are evaluated.

    Returns
    -------
    float
        Coverage percentage (0-100).
        - 100% = All experimental area covered by simulated
        - 90%+ = Almost all experimental area covered (excellent)
        - 70-90% = Most experimental area covered (good)
        - <70% = Significant experimental area not covered

    Notes
    -----
    - This metric is asymmetric: measures coverage of experimental by simulated
    - High values indicate simulated shape spatially encompasses experimental
    - Works on normalized shapes (max=1.0), so focuses on geometric coverage
      rather than probability mass
    - No arbitrary thresholds - pure area calculation
    """
    # Calculate total area under experimental shape
    area_exp = np.trapz(shape_exp, x_range)

    # Calculate intersection area (overlap)
    intersection = np.trapz(np.minimum(shape_exp, shape_sim), x_range)

    # Calculate coverage percentage
    if area_exp > 0:
        coverage = intersection / area_exp
    else:
        coverage = 0.0

    # Return as percentage
    return coverage * 100


##############################################################################
# Plotting Function
# -----------------


def plot_cv_vs_fr_all_gammas(all_data, exp_data, muscle_gamma_colors, force_markers):
    """
    Create CV vs FR plot with all muscle-gamma-force combinations.

    Parameters
    ----------
    all_data : dict
        Nested dictionary: {muscle: {gamma: {force: (filepath, DataFrame)}}}}.
    exp_data : pd.DataFrame or None
        Experimental ISI statistics.
    muscle_gamma_colors : dict
        Color mappings: {muscle: {gamma: hex_color}}.
    force_markers : dict
        Force level to marker shape mapping.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis objects.
    """
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde

    # Create figure with GridSpec layout for marginal distributions
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    # Main scatter plot (bottom-left, 3x3 grid)
    ax = fig.add_subplot(gs[1:4, 0:3])

    # Top marginal plot (CV distribution)
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax)

    # Right marginal plot (FR distribution)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax)

    # Data structures for marginal distributions
    # Both experimental and simulated: aggregate all
    exp_cv_all = []
    exp_fr_all = []
    sim_cv_all = []
    sim_fr_all = []

    # 1. Plot experimental data (convex hulls + scatter)
    if exp_data is not None:
        muscles = exp_data["Muscle"].unique()
        for muscle in muscles:
            muscle_data = exp_data[exp_data["Muscle"] == muscle]
            cv_data = muscle_data["ISI CV"].values
            fr_data = muscle_data["FR mean"].values

            # Collect for marginal distributions (aggregate all)
            exp_cv_all.extend(cv_data)
            exp_fr_all.extend(fr_data)

            # Draw convex hull
            if len(cv_data) > 2:
                points = np.column_stack([cv_data, fr_data])
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    polygon = Polygon(
                        hull_points,
                        facecolor=EXP_COLORS.get(muscle, "#808080"),
                        alpha=0.6,
                        edgecolor=EXP_COLORS.get(muscle, "#808080"),
                        linewidth=2.0,
                        linestyle="-",
                        zorder=0,
                    )
                    ax.add_patch(polygon)
                except Exception:
                    pass

            # Scatter points
            ax.scatter(
                cv_data,
                fr_data,
                s=60,
                alpha=1.0,
                color=EXP_COLORS.get(muscle, "#808080"),
                edgecolors="white",
                linewidth=1.0,
                marker="x",
                zorder=1,
            )

    # 2. Plot simulated data for each muscle-gamma-force combination
    for muscle in sorted(all_data.keys(), key=muscle_name_to_number):
        for gamma in sorted(all_data[muscle].keys(), key=gamma_str_to_number):
            for force in sorted(all_data[muscle][gamma].keys()):
                filepath, df = all_data[muscle][gamma][force]

                # Get color for this muscle-gamma combination
                color = muscle_gamma_colors[muscle][gamma]

                # Get marker for this force level
                marker = force_markers[force]

                # Extract recruitment order
                if "MU_ID" in df.columns:
                    recruitment_order = df["MU_ID"].values
                else:
                    recruitment_order = np.arange(len(df))

                # Get colors with recruitment gradient
                colors = get_muscle_colors(recruitment_order, color)

                # Collect for marginal distributions
                sim_cv_all.extend(df["CV_ISI"].values)
                sim_fr_all.extend(df["mean_firing_rate_Hz"].values)

                # Plot
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

    # 3. Create marginal KDE distributions
    # Convert lists to numpy arrays
    exp_cv_all = np.array(exp_cv_all)
    exp_fr_all = np.array(exp_fr_all)
    sim_cv_all = np.array(sim_cv_all)
    sim_fr_all = np.array(sim_fr_all)

    # Define colors for marginal plots
    exp_color = "#808080"  # Gray for experimental data (matching main plot)
    sim_color = "#2E86AB"  # Blue for simulated data

    # Top marginal plot - CV distribution
    cv_coverage = None
    # Plot experimental data (aggregated)
    if len(exp_cv_all) > 1:
        kde_exp_cv = gaussian_kde(exp_cv_all)
        cv_range = np.linspace(0, 1.0, 500)
        kde_exp_cv_values = kde_exp_cv(cv_range)

        # Normalize to max value of 1.0 for display
        kde_exp_cv_values_norm = kde_exp_cv_values / kde_exp_cv_values.max()
        ax_top.plot(
            cv_range,
            kde_exp_cv_values_norm,
            color=exp_color,
            linewidth=2.5,
            label="Experimental",
            alpha=0.8,
            linestyle="--",
        )
        ax_top.fill_between(
            cv_range, kde_exp_cv_values_norm, alpha=0.3, color=exp_color
        )

    # Plot simulated data (aggregated)
    if len(sim_cv_all) > 1:
        kde_sim_cv = gaussian_kde(sim_cv_all)
        cv_range = np.linspace(0, 1.0, 500)
        kde_sim_cv_values = kde_sim_cv(cv_range)

        # Normalize to max value of 1.0 for display
        kde_sim_cv_values_norm = kde_sim_cv_values / kde_sim_cv_values.max()
        ax_top.plot(
            cv_range,
            kde_sim_cv_values_norm,
            color=sim_color,
            linewidth=2.5,
            label="Simulated",
            alpha=0.8,
            linestyle="-",
        )
        ax_top.fill_between(
            cv_range, kde_sim_cv_values_norm, alpha=0.3, color=sim_color
        )

        # Calculate geometric coverage if both distributions exist
        if len(exp_cv_all) > 1:
            cv_coverage = calculate_geometric_coverage(
                kde_exp_cv_values_norm, kde_sim_cv_values_norm, cv_range
            )

    # Right marginal plot - FR distribution
    fr_coverage = None
    # Plot experimental data (aggregated)
    if len(exp_fr_all) > 1:
        kde_exp_fr = gaussian_kde(exp_fr_all)
        fr_range = np.linspace(3.5, 35, 500)
        kde_exp_fr_values = kde_exp_fr(fr_range)

        # Normalize to max value of 1.0 for display
        kde_exp_fr_values_norm = kde_exp_fr_values / kde_exp_fr_values.max()
        ax_right.plot(
            kde_exp_fr_values_norm,
            fr_range,
            color=exp_color,
            linewidth=2.5,
            label="Experimental",
            alpha=0.8,
            linestyle="--",
        )
        ax_right.fill_betweenx(
            fr_range, kde_exp_fr_values_norm, alpha=0.3, color=exp_color
        )

    # Plot simulated data (aggregated)
    if len(sim_fr_all) > 1:
        kde_sim_fr = gaussian_kde(sim_fr_all)
        fr_range = np.linspace(3.5, 35, 500)
        kde_sim_fr_values = kde_sim_fr(fr_range)

        # Normalize to max value of 1.0 for display
        kde_sim_fr_values_norm = kde_sim_fr_values / kde_sim_fr_values.max()
        ax_right.plot(
            kde_sim_fr_values_norm,
            fr_range,
            color=sim_color,
            linewidth=2.5,
            label="Simulated",
            alpha=0.8,
            linestyle="-",
        )
        ax_right.fill_betweenx(
            fr_range, kde_sim_fr_values_norm, alpha=0.3, color=sim_color
        )

        # Calculate geometric coverage if both distributions exist
        if len(exp_fr_all) > 1:
            fr_coverage = calculate_geometric_coverage(
                kde_exp_fr_values_norm, kde_sim_fr_values_norm, fr_range
            )

    # 4. Create comprehensive legend with color bars for gamma values

    # Custom legend handler for gradient bars
    class GradientPatchHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            width, height = handlebox.width, handlebox.height

            # Get gradient colors from the handle
            colors = orig_handle.get_facecolor()
            if not isinstance(colors, list):
                colors = [colors]

            n_colors = len(colors)
            patch_width = width / n_colors

            patches = []
            for i, color in enumerate(colors):
                patch = Rectangle(
                    (x0 + i * patch_width, y0),
                    patch_width,
                    height,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3,
                    transform=handlebox.get_transform(),
                )
                handlebox.add_artist(patch)
                patches.append(patch)

            return patches[0]

    # Create custom Patch class that holds gradient colors
    class GradientPatch(mpatches.Patch):
        def __init__(self, colors, **kwargs):
            self.gradient_colors = colors
            # Use the first color as the base facecolor for the parent Patch
            if "facecolor" not in kwargs:
                kwargs["facecolor"] = colors[0] if colors else "gray"
            super().__init__(**kwargs)

        def get_facecolor(self):
            return self.gradient_colors

    legend_elements = []

    # Section 1: Muscle Types with color bars showing gamma gradient
    legend_elements.append(
        Line2D([0], [0], color="none", label="Muscle Types (γ gradient):", marker="")
    )

    for muscle in sorted(all_data.keys(), key=muscle_name_to_number):
        gammas = sorted(all_data[muscle].keys(), key=gamma_str_to_number)
        colors = [muscle_gamma_colors[muscle][gamma] for gamma in gammas]

        # Get gamma range for label
        gamma_values = [g.replace("gamma", "") for g in gammas]
        if len(gamma_values) > 1:
            gamma_range = f"γ {gamma_values[0]} → {gamma_values[-1]}"
        else:
            gamma_range = f"γ {gamma_values[0]}"

        # Create gradient patch
        gradient_patch = GradientPatch(
            colors, edgecolor="black", linewidth=0.5, label=f"  {muscle}: {gamma_range}"
        )
        legend_elements.append(gradient_patch)

    # Add spacing
    legend_elements.append(Line2D([0], [0], color="none", label=" ", marker=""))

    # Section 2: Force levels
    legend_elements.append(
        Line2D([0], [0], color="none", label="Force Levels:", marker="")
    )
    for force in sorted(force_markers.keys()):
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
                label=f"  {force}%",
                linewidth=0,
            )
        )

    # Add legend with custom handler
    legend = ax.legend(
        handles=legend_elements,
        handler_map={GradientPatch: GradientPatchHandler()},
        frameon=True,
        fontsize=8,
        loc="upper right",
        ncol=1,
    )

    # 5. Format main plot
    ax.set_xlabel("Coefficient of Variation (CV)", fontsize=12)
    ax.set_ylabel("Mean Firing Rate (pps)", fontsize=12)
    # ax.set_xscale("log")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(3.5, 35)
    ax.set_title("ISI Statistics - Multi-Muscle, Multi-Gamma Comparison", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    sns.despine(ax=ax, offset=10, trim=True)

    # 6. Format marginal axes
    # Top marginal (CV distribution)
    ax_top.set_ylabel("Norm. Density", fontsize=10)
    ax_top.set_ylim(0, 1.05)  # Normalized to [0, 1]
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=10)
    ax_top.legend(fontsize=7, loc="upper right", frameon=True)
    sns.despine(ax=ax_top, offset=10, trim=True, bottom=True)

    # Add coverage text annotation for CV
    if cv_coverage is not None:
        ax_top.text(
            0.02,
            0.95,
            f"Coverage: {cv_coverage:.1f}%",
            transform=ax_top.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    # Right marginal (FR distribution)
    ax_right.set_xlabel("Norm. Density", fontsize=10)
    ax_right.set_xlim(0, 1.05)  # Normalized to [0, 1]
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=10)
    sns.despine(ax=ax_right, offset=10, trim=True, left=True)

    # Add coverage text annotation for FR
    if fr_coverage is not None:
        ax_right.text(
            0.95,
            0.02,
            f"Coverage: {fr_coverage:.1f}%",
            transform=ax_right.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8,
                edgecolor="black",
                linewidth=1.5,
            ),
        )

    return fig, ax, cv_coverage, fr_coverage


##############################################################################
# Main Execution
# --------------


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot ISI/CV comparison for all muscle-gamma-force combinations"
    )
    parser.add_argument(
        "--muscles",
        type=str,
        nargs="+",
        required=True,
        help="Muscle types to compare (e.g., THIRTY TWENTYFIVE TWENTY)",
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
        help="Gamma values to exclude from plot (e.g., gamma50 gamma30 or 50 30)",
    )
    parser.add_argument(
        "--min-firing-rate",
        type=float,
        default=4.0,
        help="Minimum firing rate (Hz) to include motor units (default: 4.0 Hz)",
    )

    args = parser.parse_args()

    print(f"Multi-Muscle ISI/CV Comparison: {', '.join(args.muscles)}")

    # Auto-detect all data
    all_data = auto_detect_all_data(args.results_path, args.muscles)

    if not all_data:
        print("❌ No simulation data found.")
        exit(1)

    # Filter out excluded gammas
    if args.exclude_gammas:
        # Normalize excluded gamma names (handle both "gamma50" and "50" formats)
        exclude_set = set()
        for gamma in args.exclude_gammas:
            if gamma.startswith("gamma"):
                exclude_set.add(gamma)
            else:
                exclude_set.add(f"gamma{gamma}")

        # Filter the data
        for muscle in list(all_data.keys()):
            for gamma in list(all_data[muscle].keys()):
                if gamma in exclude_set:
                    del all_data[muscle][gamma]

            # Remove muscle if no gammas left
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

                    # Filter dataframe by firing rate
                    df_filtered = df[df["mean_firing_rate_Hz"] >= args.min_firing_rate]
                    total_mus_after += len(df_filtered)

                    if len(df_filtered) > 0:
                        # Update with filtered dataframe
                        all_data[muscle][gamma][force] = (filepath, df_filtered)
                    else:
                        # Remove empty combination
                        del all_data[muscle][gamma][force]

                # Remove gamma if no forces left
                if not all_data[muscle][gamma]:
                    del all_data[muscle][gamma]

            # Remove muscle if no gammas left
            if not all_data[muscle]:
                del all_data[muscle]

        print(f"Filtered MUs: {total_mus_before} → {total_mus_after}")

        if not all_data:
            print("❌ No data remaining after filtering by firing rate.")
            exit(1)

    # Generate color mappings for muscle-gamma combinations
    muscle_gamma_colors = generate_muscle_gamma_colors(all_data)

    # Print gamma values being used
    all_gammas = set()
    for muscle_data in all_data.values():
        all_gammas.update(muscle_data.keys())
    gamma_list = sorted(all_gammas, key=gamma_str_to_number)
    print(f"Gammas: {', '.join(gamma_list)}")

    # Collect all force levels
    all_force_levels = set()
    for muscle_data in all_data.values():
        for gamma_data in muscle_data.values():
            all_force_levels.update(gamma_data.keys())

    # Generate force markers
    force_markers = generate_force_markers(all_force_levels)

    # Load experimental data
    exp_csv_path = Path(__file__).parent / "ISI_statistics.csv"
    exp_data = load_experimental_data(exp_csv_path)

    # Create plot
    _, _, cv_coverage, fr_coverage = plot_cv_vs_fr_all_gammas(
        all_data, exp_data, muscle_gamma_colors, force_markers
    )

    # Print geometric coverage statistics
    if cv_coverage is not None and fr_coverage is not None:
        print(f"Coverage: CV={cv_coverage:.1f}% | FR={fr_coverage:.1f}%")

    # Generate output filename
    muscle_str = "_".join(args.muscles)
    output_file = (
        args.results_path / f"isi_cv_all_gammas_{muscle_str}.{args.output_format}"
    )

    # Save with appropriate quality (bbox_inches="tight" handles layout)
    if args.output_format in ["jpg", "jpeg"]:
        plt.savefig(
            output_file, dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95}
        )
    else:
        plt.savefig(output_file, dpi=300, bbox_inches="tight", transparent=True)

    print(f"Saved: {output_file}")

    # Print summary statistics
    total_motor_units = 0
    total_combinations = 0

    for muscle in sorted(all_data.keys(), key=muscle_name_to_number):
        for gamma in sorted(all_data[muscle].keys(), key=gamma_str_to_number):
            for force in sorted(all_data[muscle][gamma].keys()):
                filepath, df = all_data[muscle][gamma][force]
                total_motor_units += len(df)
                total_combinations += 1

    print(f"Total: {total_motor_units} MUs | {len(all_data)} muscles | {total_combinations} combinations ✅")


if __name__ == "__main__":
    main()

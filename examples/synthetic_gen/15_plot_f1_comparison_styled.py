r"""
Styled F1 Score Comparison: iEMG vs sEMG
=========================================

Creates a publication-ready F1 score comparison plot using science/nature
matplotlib styles with thick lines and clean aesthetics.

This script extracts Panel A from the full comparison figure and presents
it as a standalone, wide figure suitable for presentations or publications.

Usage:
------
python examples/synthetic_gen/15_plot_f1_comparison_styled.py \
    --iemg-dir results/synthetic_gen/iemg_mu_76_77_78_79_80_snr1 \
    --semg-dir results/synthetic_gen/semg_mu_41_42_43_plus18_snr10 \
    --output results/synthetic_gen/plots/f1_comparison_styled.png
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
import seaborn as sns
import scienceplots  # noqa

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

##############################################################################
# Configuration
##############################################################################

# Default data directories
DEFAULT_IEMG_DIR = "results/synthetic_gen/iemg_mu_76_77_78_79_80_snr1"
DEFAULT_SEMG_DIR = "results/synthetic_gen/semg_mu_41_42_43_plus18_snr10"
DEFAULT_OUTPUT = "results/synthetic_gen/plots/f1_comparison_styled.png"

# Visual style constants
COLOR_IEMG = "#1f77b4"  # Blue for intramuscular
COLOR_SEMG = "#ff7f0e"  # Orange for surface

##############################################################################
# Data Loading Functions
##############################################################################


def load_agreement_data(directory):
    """
    Load spike train agreement metrics from CSV file.

    Parameters
    ----------
    directory : Path
        Directory containing spike_train_agreement.csv.

    Returns
    -------
    pd.DataFrame
        Agreement metrics for all units.
    """
    csv_path = directory / "spike_train_agreement.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Agreement CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


##############################################################################
# Plotting Function
##############################################################################


def plot_f1_comparison_styled(iemg_df, semg_df, output_path, width=10, height=6):
    """
    Create a styled F1 score comparison plot.

    Parameters
    ----------
    iemg_df : pd.DataFrame
        iEMG agreement dataframe.
    semg_df : pd.DataFrame
        sEMG agreement dataframe.
    output_path : Path
        Path to save the plot.
    width : float, optional
        Figure width in inches (default: 10).
    height : float, optional
        Figure height in inches (default: 6).
    """
    # Extract matched units only
    iemg_matched = iemg_df[iemg_df["matched"] == True]["f1_score"].values
    semg_matched = semg_df[semg_df["matched"] == True]["f1_score"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Prepare data for box plot
    data = [iemg_matched, semg_matched]
    positions = [1, 2]
    labels = ["iEMG", "sEMG"]
    colors = [COLOR_IEMG, COLOR_SEMG]

    # Create box plots with thick lines
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=3),
        boxprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(2)

    # Overlay individual points with jitter
    np.random.seed(42)
    for i, (values, pos, color) in enumerate(zip(data, positions, colors)):
        jitter = np.random.normal(0, 0.04, size=len(values))
        ax.scatter(
            pos + jitter,
            values,
            alpha=0.8,
            s=150,
            color=color,
            edgecolors="black",
            linewidth=1.5,
            zorder=3,
            marker="o",
        )

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=20, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=16, width=2, length=6)

    # Add horizontal grid lines
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=1.5)
    ax.set_axisbelow(True)

    # Add mean lines with thicker style
    for values, pos, color in zip(data, positions, colors):
        mean_val = np.mean(values)
        ax.hlines(
            mean_val, pos - 0.35, pos + 0.35, colors=color, linewidth=4, linestyle="-", zorder=4
        )

    # Add statistics text annotations
    iemg_mean = np.mean(iemg_matched)
    iemg_std = np.std(iemg_matched)
    semg_mean = np.mean(semg_matched)
    semg_std = np.std(semg_matched)

    # Text boxes with statistics
    ax.text(
        1,
        0.05,
        f"u = {iemg_mean:.3f}\nσ = {iemg_std:.3f}\nn = {len(iemg_matched)}",
        ha="center",
        va="bottom",
        fontsize=14,
        bbox=dict(
            boxstyle="round", facecolor=COLOR_IEMG, alpha=0.2, edgecolor="black", linewidth=1.5
        ),
    )

    ax.text(
        2,
        0.05,
        f"u = {semg_mean:.3f}\nσ = {semg_std:.3f}\nn = {len(semg_matched)}",
        ha="center",
        va="bottom",
        fontsize=14,
        bbox=dict(
            boxstyle="round", facecolor=COLOR_SEMG, alpha=0.2, edgecolor="black", linewidth=1.5
        ),
    )

    # Remove spines using seaborn
    sns.despine(ax=ax, offset=10, trim=True)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


##############################################################################
# Main Execution
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create styled F1 score comparison plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--iemg-dir",
        type=Path,
        default=Path(DEFAULT_IEMG_DIR),
        help=f"Directory containing iEMG results (default: {DEFAULT_IEMG_DIR})",
    )

    parser.add_argument(
        "--semg-dir",
        type=Path,
        default=Path(DEFAULT_SEMG_DIR),
        help=f"Directory containing sEMG results (default: {DEFAULT_SEMG_DIR})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output path for figure (default: {DEFAULT_OUTPUT})",
    )

    parser.add_argument(
        "--width", type=float, default=10, help="Figure width in inches (default: 10)"
    )

    parser.add_argument(
        "--height", type=float, default=6, help="Figure height in inches (default: 6)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "svg", "pdf"],
        help="Output format (default: png)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Styled F1 Score Comparison: iEMG vs sEMG")
    print("=" * 80)

    # Update output path with format
    if args.format != "png":
        args.output = args.output.with_suffix(f".{args.format}")

    # Validate input directories
    if not args.iemg_dir.exists():
        raise FileNotFoundError(f"iEMG directory not found: {args.iemg_dir}")
    if not args.semg_dir.exists():
        raise FileNotFoundError(f"sEMG directory not found: {args.semg_dir}")

    # Load data
    print(f"\n📂 Loading data...")
    print(f"  iEMG: {args.iemg_dir}")
    iemg_agreement = load_agreement_data(args.iemg_dir)

    print(f"  sEMG: {args.semg_dir}")
    semg_agreement = load_agreement_data(args.semg_dir)

    # Create plot
    print(f'\n🎨 Creating styled plot ({args.width}" × {args.height}")...')
    plot_f1_comparison_styled(
        iemg_agreement, semg_agreement, args.output, width=args.width, height=args.height
    )

    print(f"\n💾 Saved: {args.output}")
    print("✅ Done!")


if __name__ == "__main__":
    main()

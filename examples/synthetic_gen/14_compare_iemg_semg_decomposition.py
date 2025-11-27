r"""
iEMG vs sEMG Decomposition Comparison
======================================

Creates a tall, narrow visualization comparing intramuscular EMG (iEMG) and
surface EMG (sEMG) decomposition performance across multiple metrics:
- F1 Score distribution
- PNR (Pulse-to-Noise Ratio) distribution
- Match rate statistics
- Precision vs Sensitivity scatter
- Summary statistics table

The visualization highlights the fundamental differences in decomposition
difficulty and performance between recording modalities.

Usage:
------
python examples/synthetic_gen/14_compare_iemg_semg_decomposition.py \
    --iemg-dir results/synthetic_gen/iemg_mu_76_77_78_79_80_snr1 \
    --semg-dir results/synthetic_gen/semg_mu_41_42_43_plus18_snr10 \
    --output results/synthetic_gen/plots/iemg_vs_semg_comparison.png
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##############################################################################
# Configuration
##############################################################################

# Default data directories
DEFAULT_IEMG_DIR = "results/synthetic_gen/iemg_mu_76_77_78_79_80_snr1"
DEFAULT_SEMG_DIR = "results/synthetic_gen/semg_mu_41_42_43_plus18_snr10"
DEFAULT_OUTPUT = "results/synthetic_gen/plots/iemg_vs_semg_comparison.png"

# Visual style constants
COLOR_IEMG = '#1f77b4'  # Blue for intramuscular
COLOR_SEMG = '#ff7f0e'  # Orange for surface
COLOR_MATCHED = '#2ca02c'  # Green
COLOR_UNMATCHED = '#d62728'  # Red

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


def load_pnr_data(directory):
    """
    Load PNR (Pulse-to-Noise Ratio) values from CSV file.

    Parameters
    ----------
    directory : Path
        Directory containing plots/pnr_values.csv.

    Returns
    -------
    pd.DataFrame
        PNR values for all units.
    """
    csv_path = directory / "plots" / "pnr_values.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"PNR CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def merge_agreement_and_pnr(agreement_df, pnr_df):
    """
    Merge agreement and PNR dataframes on motor_unit_index.

    Parameters
    ----------
    agreement_df : pd.DataFrame
        Agreement metrics with motor_unit_index column.
    pnr_df : pd.DataFrame
        PNR values with motor_unit_index column.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with both metrics.
    """
    merged = pd.merge(
        agreement_df,
        pnr_df,
        on='motor_unit_index',
        how='left'
    )
    return merged


def compute_summary_stats(df):
    """
    Compute summary statistics from agreement dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Agreement dataframe with matched column and metrics.

    Returns
    -------
    dict
        Summary statistics including means, stds, match rates.
    """
    matched_df = df[df['matched'] == True]

    stats = {
        'n_total': len(df),
        'n_matched': len(matched_df),
        'n_unmatched': len(df) - len(matched_df),
        'match_rate': len(matched_df) / len(df) * 100 if len(df) > 0 else 0,
        'f1_mean': matched_df['f1_score'].mean() if len(matched_df) > 0 else np.nan,
        'f1_std': matched_df['f1_score'].std() if len(matched_df) > 0 else np.nan,
        'sensitivity_mean': matched_df['sensitivity_recall'].mean() if len(matched_df) > 0 else np.nan,
        'sensitivity_std': matched_df['sensitivity_recall'].std() if len(matched_df) > 0 else np.nan,
        'precision_mean': matched_df['precision'].mean() if len(matched_df) > 0 else np.nan,
        'precision_std': matched_df['precision'].std() if len(matched_df) > 0 else np.nan,
    }

    return stats


def compute_pnr_stats(df):
    """
    Compute PNR summary statistics from merged dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with pnr_demuse_dB column.

    Returns
    -------
    dict
        PNR summary statistics.
    """
    # Only use matched units with valid PNR values
    matched_df = df[df['matched'] == True]

    if 'pnr_demuse_dB' in matched_df.columns:
        valid_pnr = matched_df['pnr_demuse_dB'].dropna()
        if len(valid_pnr) > 0:
            return {
                'pnr_mean': valid_pnr.mean(),
                'pnr_std': valid_pnr.std(),
                'pnr_min': valid_pnr.min(),
                'pnr_max': valid_pnr.max(),
            }

    return {
        'pnr_mean': np.nan,
        'pnr_std': np.nan,
        'pnr_min': np.nan,
        'pnr_max': np.nan,
    }


##############################################################################
# Plotting Functions
##############################################################################


def plot_panel1_f1_comparison(ax, iemg_df, semg_df):
    """
    Plot F1 score comparison (box plots with individual points).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    iemg_df : pd.DataFrame
        iEMG agreement dataframe.
    semg_df : pd.DataFrame
        sEMG agreement dataframe.
    """
    # Extract matched units only
    iemg_matched = iemg_df[iemg_df['matched'] == True]['f1_score'].values
    semg_matched = semg_df[semg_df['matched'] == True]['f1_score'].values

    # Prepare data for box plot
    data = [iemg_matched, semg_matched]
    positions = [1, 2]
    labels = ['iEMG', 'sEMG']
    colors = [COLOR_IEMG, COLOR_SEMG]

    # Create box plots
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                     showfliers=False, medianprops=dict(color='black', linewidth=2))

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points with jitter
    np.random.seed(42)
    for i, (values, pos, color) in enumerate(zip(data, positions, colors)):
        jitter = np.random.normal(0, 0.04, size=len(values))
        ax.scatter(pos + jitter, values, alpha=0.7, s=60, color=color,
                  edgecolors='black', linewidths=0.5, zorder=3)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_title('Panel A: Decomposition Accuracy', fontsize=13, fontweight='bold', pad=10)

    # Add mean lines
    for values, pos, color in zip(data, positions, colors):
        mean_val = np.mean(values)
        ax.hlines(mean_val, pos - 0.3, pos + 0.3, colors=color,
                 linewidth=2.5, linestyle='-', zorder=4)


def plot_panel2_pnr_comparison(ax, iemg_merged_df, semg_merged_df):
    """
    Plot PNR (DEMUSE) comparison with violin plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    iemg_merged_df : pd.DataFrame
        iEMG merged dataframe with PNR values.
    semg_merged_df : pd.DataFrame
        sEMG merged dataframe with PNR values.
    """
    # Extract matched units with valid PNR
    iemg_pnr = iemg_merged_df[iemg_merged_df['matched'] == True]['pnr_demuse_dB'].dropna().values
    semg_pnr = semg_merged_df[semg_merged_df['matched'] == True]['pnr_demuse_dB'].dropna().values

    # Prepare data
    data = [iemg_pnr, semg_pnr]
    positions = [1, 2]
    labels = ['iEMG', 'sEMG']
    colors = [COLOR_IEMG, COLOR_SEMG]

    # Create violin plots
    parts = ax.violinplot(data, positions=positions, widths=0.6,
                          showmeans=True, showextrema=True, showmedians=False)

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)

    # Style mean and extrema lines
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cbars'].set_color('black')
    parts['cmaxes'].set_color('black')
    parts['cmins'].set_color('black')

    # Overlay individual points
    np.random.seed(42)
    for values, pos, color in zip(data, positions, colors):
        jitter = np.random.normal(0, 0.04, size=len(values))
        ax.scatter(pos + jitter, values, alpha=0.5, s=40, color=color,
                  edgecolors='black', linewidths=0.5, zorder=3)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('PNR DEMUSE (dB)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title('Panel B: Signal Quality', fontsize=13, fontweight='bold', pad=10)


def plot_panel3_match_rate(ax, iemg_stats, semg_stats):
    """
    Plot match rate comparison as horizontal stacked bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    iemg_stats : dict
        iEMG summary statistics.
    semg_stats : dict
        sEMG summary statistics.
    """
    categories = ['iEMG', 'sEMG']
    matched = [iemg_stats['n_matched'], semg_stats['n_matched']]
    unmatched = [iemg_stats['n_unmatched'], semg_stats['n_unmatched']]

    y_pos = np.arange(len(categories))

    # Create stacked horizontal bars
    ax.barh(y_pos, matched, color=COLOR_MATCHED, alpha=0.8, label='Matched', edgecolor='black')
    ax.barh(y_pos, unmatched, left=matched, color=COLOR_UNMATCHED, alpha=0.8,
            label='Unmatched', edgecolor='black')

    # Add percentage labels
    for i, (m, u) in enumerate(zip(matched, unmatched)):
        total = m + u
        match_pct = m / total * 100 if total > 0 else 0
        unmatch_pct = u / total * 100 if total > 0 else 0

        # Matched label
        if m > 0:
            ax.text(m / 2, i, f'{m}\n({match_pct:.0f}%)',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Unmatched label
        if u > 0:
            ax.text(m + u / 2, i, f'{u}\n({unmatch_pct:.0f}%)',
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlabel('Number of Units', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title('Panel C: Match Rate', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')


def plot_panel4_precision_sensitivity(ax, iemg_df, semg_df):
    """
    Plot Precision vs Sensitivity scatter plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    iemg_df : pd.DataFrame
        iEMG agreement dataframe.
    semg_df : pd.DataFrame
        sEMG agreement dataframe.
    """
    # Extract matched units
    iemg_matched = iemg_df[iemg_df['matched'] == True]
    semg_matched = semg_df[semg_df['matched'] == True]

    # Plot diagonal reference line (perfect agreement)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='Perfect agreement')

    # Scatter plots
    ax.scatter(iemg_matched['sensitivity_recall'], iemg_matched['precision'],
              s=120, alpha=0.8, color=COLOR_IEMG, edgecolors='black',
              linewidths=1.5, label='iEMG', marker='o', zorder=3)

    ax.scatter(semg_matched['sensitivity_recall'], semg_matched['precision'],
              s=120, alpha=0.8, color=COLOR_SEMG, edgecolors='black',
              linewidths=1.5, label='sEMG', marker='s', zorder=3)

    # Formatting
    ax.set_xlabel('Sensitivity (Recall)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.set_title('Panel D: Precision-Sensitivity Trade-off', fontsize=13, fontweight='bold', pad=10)
    ax.set_aspect('equal', adjustable='box')


def plot_panel5_summary_table(ax, iemg_stats, semg_stats, iemg_pnr_stats, semg_pnr_stats):
    """
    Plot summary statistics as a formatted table.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    iemg_stats : dict
        iEMG summary statistics.
    semg_stats : dict
        sEMG summary statistics.
    iemg_pnr_stats : dict
        iEMG PNR statistics.
    semg_pnr_stats : dict
        sEMG PNR statistics.
    """
    ax.axis('off')

    # Prepare table data
    table_data = [
        ['Metric', 'iEMG', 'sEMG'],
        ['─' * 30, '─' * 20, '─' * 20],
        ['Total Units', f"{iemg_stats['n_total']}", f"{semg_stats['n_total']}"],
        ['Matched Units', f"{iemg_stats['n_matched']}", f"{semg_stats['n_matched']}"],
        ['Match Rate (%)', f"{iemg_stats['match_rate']:.1f}", f"{semg_stats['match_rate']:.1f}"],
        ['', '', ''],
        ['F1 Score', f"{iemg_stats['f1_mean']:.3f} ± {iemg_stats['f1_std']:.3f}",
         f"{semg_stats['f1_mean']:.3f} ± {semg_stats['f1_std']:.3f}"],
        ['Sensitivity', f"{iemg_stats['sensitivity_mean']:.3f} ± {iemg_stats['sensitivity_std']:.3f}",
         f"{semg_stats['sensitivity_mean']:.3f} ± {semg_stats['sensitivity_std']:.3f}"],
        ['Precision', f"{iemg_stats['precision_mean']:.3f} ± {iemg_stats['precision_std']:.3f}",
         f"{semg_stats['precision_mean']:.3f} ± {semg_stats['precision_std']:.3f}"],
        ['', '', ''],
        ['PNR DEMUSE (dB)', f"{iemg_pnr_stats['pnr_mean']:.2f} ± {iemg_pnr_stats['pnr_std']:.2f}",
         f"{semg_pnr_stats['pnr_mean']:.2f} ± {semg_pnr_stats['pnr_std']:.2f}"],
    ]

    # Create table
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.3, 0.3])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#404040')
        cell.set_text_props(weight='bold', color='white', fontsize=11)

    # Color metric column
    for i in range(len(table_data)):
        cell = table[(i, 0)]
        if i > 1 and table_data[i][0] != '':
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')

    # Color data cells
    for i in [1, 2]:
        for j in range(1, 3):
            cell = table[(i, j)]
            cell.set_facecolor([COLOR_IEMG if j == 1 else COLOR_SEMG][0])
            cell.set_alpha(0.2)

    ax.set_title('Panel E: Summary Statistics', fontsize=13, fontweight='bold', pad=20)


##############################################################################
# Main Visualization Function
##############################################################################


def create_comparison_figure(iemg_dir, semg_dir, output_path):
    """
    Create complete comparison figure with all 5 panels.

    Parameters
    ----------
    iemg_dir : Path
        Directory containing iEMG results.
    semg_dir : Path
        Directory containing sEMG results.
    output_path : Path
        Output path for saving the figure.
    """
    print("=" * 80)
    print("iEMG vs sEMG Decomposition Comparison")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    print(f"  iEMG: {iemg_dir}")
    iemg_agreement = load_agreement_data(iemg_dir)
    iemg_pnr = load_pnr_data(iemg_dir)
    iemg_merged = merge_agreement_and_pnr(iemg_agreement, iemg_pnr)

    print(f"  sEMG: {semg_dir}")
    semg_agreement = load_agreement_data(semg_dir)
    semg_pnr = load_pnr_data(semg_dir)
    semg_merged = merge_agreement_and_pnr(semg_agreement, semg_pnr)

    # Compute summary statistics
    print("\n📊 Computing summary statistics...")
    iemg_stats = compute_summary_stats(iemg_agreement)
    semg_stats = compute_summary_stats(semg_agreement)
    iemg_pnr_stats = compute_pnr_stats(iemg_merged)
    semg_pnr_stats = compute_pnr_stats(semg_merged)

    print(f"\n  iEMG: {iemg_stats['n_matched']}/{iemg_stats['n_total']} matched "
          f"({iemg_stats['match_rate']:.1f}%), F1={iemg_stats['f1_mean']:.3f}±{iemg_stats['f1_std']:.3f}")
    print(f"  sEMG: {semg_stats['n_matched']}/{semg_stats['n_total']} matched "
          f"({semg_stats['match_rate']:.1f}%), F1={semg_stats['f1_mean']:.3f}±{semg_stats['f1_std']:.3f}")

    # Create figure with 5 vertical panels
    print("\n🎨 Creating visualization...")
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.0)

    fig = plt.figure(figsize=(6, 18))

    # Create grid: 5 rows, 1 column
    gs = fig.add_gridspec(5, 1, hspace=0.4, top=0.97, bottom=0.03, left=0.15, right=0.95)

    # Panel 1: F1 Score comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_panel1_f1_comparison(ax1, iemg_agreement, semg_agreement)

    # Panel 2: PNR comparison
    ax2 = fig.add_subplot(gs[1, 0])
    plot_panel2_pnr_comparison(ax2, iemg_merged, semg_merged)

    # Panel 3: Match rate
    ax3 = fig.add_subplot(gs[2, 0])
    plot_panel3_match_rate(ax3, iemg_stats, semg_stats)

    # Panel 4: Precision vs Sensitivity
    ax4 = fig.add_subplot(gs[3, 0])
    plot_panel4_precision_sensitivity(ax4, iemg_agreement, semg_agreement)

    # Panel 5: Summary statistics table
    ax5 = fig.add_subplot(gs[4, 0])
    plot_panel5_summary_table(ax5, iemg_stats, semg_stats, iemg_pnr_stats, semg_pnr_stats)

    # Save figure
    print(f"\n💾 Saving figure to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved successfully!")

    plt.close(fig)


##############################################################################
# Main Entry Point
##############################################################################


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Compare iEMG and sEMG decomposition performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--iemg-dir',
        type=Path,
        default=Path(DEFAULT_IEMG_DIR),
        help=f"Directory containing iEMG results (default: {DEFAULT_IEMG_DIR})"
    )

    parser.add_argument(
        '--semg-dir',
        type=Path,
        default=Path(DEFAULT_SEMG_DIR),
        help=f"Directory containing sEMG results (default: {DEFAULT_SEMG_DIR})"
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output path for comparison figure (default: {DEFAULT_OUTPUT})"
    )

    args = parser.parse_args()

    # Validate input directories
    if not args.iemg_dir.exists():
        raise FileNotFoundError(f"iEMG directory not found: {args.iemg_dir}")
    if not args.semg_dir.exists():
        raise FileNotFoundError(f"sEMG directory not found: {args.semg_dir}")

    # Create comparison figure
    create_comparison_figure(args.iemg_dir, args.semg_dir, args.output)


if __name__ == "__main__":
    main()

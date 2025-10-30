"""
Quick ISI/CV Comparison Plot for Plateau-Only Filtering
=========================================================

This script plots all available ISI/CV data extracted with plateau-only filtering.
"""

import os
os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# Configure plotting style
plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Results directory
RESULTS_PATH = Path("./results")

# Color mapping for different study prefixes
STUDY_COLORS = {
    "FIFTEEN": "#9467bd",  # Purple
    "TEN": "#ff7f0e",       # Orange
    "FIVE": "#8c564b",      # Brown
    "TWENTY": "#2ca02c",    # Green
    "TWENTYFIVE": "#1f77b4", # Blue
    "THIRTY": "#d62728",    # Red
}

# Marker mapping for MVC levels
MVC_MARKERS = {
    5: "o",
    15: "s",
    30: "^",
    50: "D",
    75: "v",
    90: "p",
}

print("=" * 70)
print("ISI/CV Plateau-Only Filtering - Comparison Plot")
print("=" * 70)
print()

# Load experimental data
exp_data_path = Path("ISI_statistics.csv")
exp_data = None
if exp_data_path.exists():
    exp_data = pd.read_csv(exp_data_path)
    print(f"✓ Loaded experimental data: {len(exp_data)} records")
    print(f"  Muscles: {exp_data['Muscle'].unique()}")
    print(f"  Force levels: {sorted(exp_data['Force Level'].unique())}")
else:
    print("⚠️  No experimental data found")

print()

# Load all simulation data
all_sim_data = []
csv_files = list(RESULTS_PATH.glob("*_isi_cv_data_*.csv"))

print(f"📁 Found {len(csv_files)} CSV files")
print()

for csv_file in csv_files:
    # Parse filename: PREFIX_gamma{X}-{Y}_isi_cv_data_VLVM_{MVC}.csv
    filename = csv_file.stem
    parts = filename.split("_")

    try:
        # Extract study prefix (everything before 'gamma')
        study_prefix = parts[0]

        # Extract gamma range
        gamma_part = [p for p in parts if p.startswith("gamma")][0]
        gamma_range = gamma_part.replace("gamma", "")

        # Extract MVC level (last part)
        mvc_level = int(parts[-1])

        # Load data
        df = pd.read_csv(csv_file)

        if len(df) > 0:
            df["study_prefix"] = study_prefix
            df["gamma_range"] = gamma_range
            df["mvc_level"] = mvc_level
            all_sim_data.append(df)
            print(f"  ✓ {study_prefix:12s} | gamma{gamma_range:10s} | {mvc_level:2d}% | {len(df):3d} units")

    except Exception as e:
        print(f"  ✗ Failed to parse {csv_file.name}: {e}")

if not all_sim_data:
    print("\n❌ No simulation data found!")
    exit(1)

# Combine all simulation data
sim_df = pd.concat(all_sim_data, ignore_index=True)
print()
print(f"✓ Total simulation data points: {len(sim_df)}")
print(f"  Study prefixes: {sorted(sim_df['study_prefix'].unique())}")
print(f"  MVC levels: {sorted(sim_df['mvc_level'].unique())}")
print(f"  Gamma ranges: {sorted(sim_df['gamma_range'].unique())}")
print()

# Create plot
fig, ax = plt.subplots(figsize=(14, 10))

# Plot experimental data with convex hulls (if available)
if exp_data is not None:
    exp_muscles = exp_data["Muscle"].unique()

    for muscle in exp_muscles:
        muscle_data = exp_data[exp_data["Muscle"] == muscle]

        # Get points for convex hull
        points = muscle_data[["FR mean", "ISI CV"]].values

        # Only create hull if we have enough points
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]

                # Close the polygon
                hull_points = np.vstack([hull_points, hull_points[0]])

                # Plot convex hull
                poly = Polygon(
                    hull_points,
                    fill=True,
                    facecolor="#d0d0d0",
                    edgecolor="#808080",
                    alpha=0.3,
                    linewidth=2,
                    label=f"Experimental {muscle}" if muscle == exp_muscles[0] else None,
                    zorder=1,
                )
                ax.add_patch(poly)
            except:
                pass  # Skip if hull fails

# Plot simulation data
for study_prefix in sorted(sim_df["study_prefix"].unique()):
    study_data = sim_df[sim_df["study_prefix"] == study_prefix]
    color = STUDY_COLORS.get(study_prefix, "#000000")

    for mvc_level in sorted(study_data["mvc_level"].unique()):
        mvc_data = study_data[study_data["mvc_level"] == mvc_level]
        marker = MVC_MARKERS.get(mvc_level, "x")

        # Plot all points for this combination
        ax.scatter(
            mvc_data["mean_firing_rate_Hz"],
            mvc_data["CV_ISI"],
            marker=marker,
            c=color,
            s=60,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label=f"{study_prefix} {mvc_level}% MVC" if study_prefix == sorted(sim_df["study_prefix"].unique())[0] and mvc_level == sorted(study_data["mvc_level"].unique())[0] else None,
            zorder=3,
        )

# Formatting
ax.set_xlabel("Mean Firing Rate (Hz)", fontsize=14, fontweight="bold")
ax.set_ylabel("Coefficient of Variation (CV of ISI)", fontsize=14, fontweight="bold")
ax.set_title(
    "ISI and CV Statistics: Simulation vs Experimental Data\n(Plateau-Only Filtering Applied)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Set reasonable axis limits
ax.set_xlim(0, max(20, sim_df["mean_firing_rate_Hz"].quantile(0.95)))
ax.set_ylim(0, max(1.5, sim_df["CV_ISI"].quantile(0.95)))

# Grid
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Create custom legend
from matplotlib.lines import Line2D

legend_elements = []

# Add experimental data
if exp_data is not None:
    legend_elements.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d0d0d0",
               markeredgecolor="#808080", markersize=10, label="Experimental (convex hull)")
    )

# Add study prefixes
for study_prefix in sorted(sim_df["study_prefix"].unique()):
    color = STUDY_COLORS.get(study_prefix, "#000000")
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
               markeredgecolor="black", markersize=8, label=study_prefix)
    )

# Add MVC levels
legend_elements.append(Line2D([0], [0], color="w", label=""))  # Spacer
legend_elements.append(Line2D([0], [0], color="w", label="MVC Levels:"))

for mvc_level in sorted(sim_df["mvc_level"].unique()):
    marker = MVC_MARKERS.get(mvc_level, "x")
    legend_elements.append(
        Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=8, label=f"{mvc_level}%")
    )

ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    fontsize=9,
    ncol=1,
)

# Save figure
output_file = "isi_cv_plateau_comparison.png"
plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Saved plot: {output_file}")
print()
print("=" * 70)

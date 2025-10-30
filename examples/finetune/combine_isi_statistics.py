#!/usr/bin/env python
"""
Combine all ISI statistics CSV files from experimental_results into a single file.

This script reads all *_stats.csv files from the experimental_results directory
and combines them into a single CSV file called isi_statistics.csv.
"""

import pandas as pd
from pathlib import Path
import glob

# ============================================================================
# Configuration
# ============================================================================
results_folder = "./examples/finetune/experimental_results"
output_filename = "isi_statistics.csv"


# ============================================================================
# Main processing
# ============================================================================
def main():
    # Find all stats CSV files
    stats_files = sorted(glob.glob(f"{results_folder}/*_stats.csv"))

    if not stats_files:
        print(f"No stats CSV files found in {results_folder}/")
        return

    print(f"Found {len(stats_files)} stats files:")
    for f in stats_files:
        print(f"  - {Path(f).name}")

    # Read and combine all CSV files
    dfs = []
    for file in stats_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"✓ Loaded {Path(file).name}: {len(df)} rows")
        except Exception as e:
            print(f"✗ Error loading {Path(file).name}: {e}")

    if not dfs:
        print("No valid CSV files could be loaded.")
        return

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to output file
    output_path = Path(results_folder) / output_filename
    combined_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Successfully combined {len(dfs)} files into {output_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"{'=' * 60}")

    # Show summary statistics
    print("\nSummary by Muscle:")
    print(combined_df.groupby("Muscle").size())

    print("\nSummary by Force Level:")
    print(combined_df.groupby("Force Level").size())

    print("\nSummary by Subject:")
    print(combined_df.groupby("subject_id").size())

    # Show first few rows
    print(f"\nFirst 5 rows of combined data:")
    print(combined_df.head().to_string(index=False))


if __name__ == "__main__":
    main()

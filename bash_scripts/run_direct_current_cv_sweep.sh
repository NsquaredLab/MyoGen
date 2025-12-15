#!/bin/bash
# Parameter sweep for direct current injection to find optimal CoV
# This script tests different noise levels to achieve target CoV

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/examples/finetune" || exit 1

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Configuration
MUSCLE="VLVM"
MVC_LEVEL=30
STUDY_PREFIX="${MUSCLE}_"
SEED=42
TARGET_FR_MEAN=16.8
TARGET_FR_STD=2.5

# Test different noise levels (controls CoV)
NOISE_LEVELS=(0.0 0.01 0.05 0.1 0.2 0.5 1.0)

# Current parameters (adjust these to match firing rate targets)
BASE_CURRENT=5.0
CURRENT_RANGE=10.0

echo "========================================================================"
echo "DIRECT CURRENT INJECTION - CoV PARAMETER SWEEP"
echo "========================================================================"
echo ""
echo "Testing ${#NOISE_LEVELS[@]} noise levels to optimize CoV"
echo "Target: FR = ${TARGET_FR_MEAN} ± ${TARGET_FR_STD} Hz"
echo ""

# Create results directory
mkdir -p results/direct_current_sweep

# Run simulations for each noise level
for NOISE in "${NOISE_LEVELS[@]}"; do
    echo "------------------------------------------------------------------------"
    echo "Testing noise std = ${NOISE} nA"
    echo "------------------------------------------------------------------------"

    python extract_isi_cv_direct_current.py \
        --muscle "$MUSCLE" \
        --mvc-level "$MVC_LEVEL" \
        --study-prefix "${STUDY_PREFIX}noise${NOISE}_" \
        --seed "$SEED" \
        --target-fr-mean "$TARGET_FR_MEAN" \
        --target-fr-std "$TARGET_FR_STD" \
        --current-noise-std "$NOISE" \
        --base-current "$BASE_CURRENT" \
        --current-range "$CURRENT_RANGE"

    echo ""
done

echo "========================================================================"
echo "SWEEP COMPLETE - Analyzing results..."
echo "========================================================================"

# Create comparison plot (if we have results)
python -c "
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

noise_levels = [float(x) for x in '$NOISE_LEVELS'.strip('()').split()]
muscle = '$MUSCLE'
mvc_level = $MVC_LEVEL

results = []
for noise in noise_levels:
    csv_file = Path(f'./results/${STUDY_PREFIX}noise{noise}_isi_cv_direct_{muscle}_{mvc_level}.csv')
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if len(df) > 0:
            results.append({
                'noise_std': noise,
                'mean_fr': df['mean_firing_rate_Hz'].mean(),
                'std_fr': df['mean_firing_rate_Hz'].std(),
                'mean_cv': df['CV_ISI'].mean(),
                'std_cv': df['CV_ISI'].std(),
                'n_active': len(df)
            })

if results:
    df_results = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Firing rate vs noise
    axes[0].errorbar(df_results['noise_std'], df_results['mean_fr'],
                     yerr=df_results['std_fr'], marker='o', capsize=5, label='Mean FR ± std')
    axes[0].axhline($TARGET_FR_MEAN, color='r', linestyle='--', label='Target FR')
    axes[0].fill_between([0, df_results['noise_std'].max()],
                          $TARGET_FR_MEAN - $TARGET_FR_STD,
                          $TARGET_FR_MEAN + $TARGET_FR_STD,
                          alpha=0.2, color='r', label='Target range')
    axes[0].set_xlabel('Current Noise Std (nA)')
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('Firing Rate vs Current Noise')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: CV vs noise
    axes[1].errorbar(df_results['noise_std'], df_results['mean_cv'],
                     yerr=df_results['std_cv'], marker='s', capsize=5, color='purple', label='Mean CV ± std')
    axes[1].set_xlabel('Current Noise Std (nA)')
    axes[1].set_ylabel('CV ISI')
    axes[1].set_title('Coefficient of Variation vs Current Noise')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./results/direct_current_sweep/noise_sweep_summary.png', dpi=150)
    print('✅ Summary plot saved to: results/direct_current_sweep/noise_sweep_summary.png')

    # Print summary table
    print('\n' + '=' * 80)
    print('PARAMETER SWEEP SUMMARY')
    print('=' * 80)
    print(df_results.to_string(index=False))
    print('')

    # Find best match
    df_results['fr_error'] = abs(df_results['mean_fr'] - $TARGET_FR_MEAN)
    best_idx = df_results['fr_error'].idxmin()
    best = df_results.iloc[best_idx]

    print('\nBest match to target firing rate:')
    print(f'  Noise std:    {best[\"noise_std\"]:.3f} nA')
    print(f'  Firing rate:  {best[\"mean_fr\"]:.2f} ± {best[\"std_fr\"]:.2f} Hz')
    print(f'  CV:           {best[\"mean_cv\"]:.3f} ± {best[\"std_cv\"]:.3f}')
    print(f'  Active MUs:   {int(best[\"n_active\"])}/{100}')

    # Save summary
    df_results.to_csv('./results/direct_current_sweep/noise_sweep_results.csv', index=False)
    print('\n✅ Results table saved to: results/direct_current_sweep/noise_sweep_results.csv')
else:
    print('No results found!')
"

echo ""
echo "All done!"

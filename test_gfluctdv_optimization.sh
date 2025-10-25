#!/usr/bin/env bash
# Quick test of Gfluctdv optimization feature
# Runs 1 trial to verify functionality

set -euo pipefail

echo "Testing Gfluctdv optimization with 1 trial..."
echo "=============================================="

./bash_scripts/run_dd_optimization.sh TEST \
    --gamma-shape-min 0.5 \
    --gamma-shape-max 0.75 \
    --enable-gfluctdv \
    --gfluctdv-noise-min 1e-5 \
    --gfluctdv-noise-max 2e-5

echo ""
echo "Test completed! Check results in:"
echo "  ./results/dd_optimization/TEST_gfluctdv_gamma0.5-0.75_optuna_dd_optimization.db"
echo "  ./results/dd_optimization/TEST_gfluctdv_gamma0.5-0.75_dd_optimized_params.json"

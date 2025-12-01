#!/usr/bin/env bash
# Run descending drive optimization for different muscles
#
# Usage: ./bash_scripts/run_dd_optimization.sh [MUSCLE_NAME] [OPTIONS]
#
# Options:
#   --gamma-shape VALUE           Override gamma shape value (default: from config)
#   --enable-gfluctdv             Enable Gfluctdv noise mechanism for motor neurons
#   --gfluctdv-noise-min VALUE    Minimum Gfluctdv noise amplitude in S/cm² (default: 5e-6)
#   --gfluctdv-noise-max VALUE    Maximum Gfluctdv noise amplitude in S/cm² (default: 3e-5)
#
# Examples:
#   ./bash_scripts/run_dd_optimization.sh VLVM
#   ./bash_scripts/run_dd_optimization.sh FDI --gamma-shape 5.0
#   ./bash_scripts/run_dd_optimization.sh THIRTY --gamma-shape 2.5 --enable-gfluctdv
#
# Run in parallel:
#   ./bash_scripts/run_dd_optimization.sh VLVM &
#   ./bash_scripts/run_dd_optimization.sh FDI &
#   ./bash_scripts/run_dd_optimization.sh TA &
#   wait

set -euo pipefail

# =============================================================================
# CONFIGURATION - Edit these parameters for each muscle
# =============================================================================

# 30 - 30 fr mean
THIRTY_TARGET_FR_MEAN=30
THIRTY_TARGET_FR_STD=4.5
THIRTY_TARGET_CONN_PROB=0.30
THIRTY_TARGET_N_DD_NEURONS=400
THIRTY_N_TRIALS=100
THIRTY_N_DD_NEURONS_MIN=100
THIRTY_N_DD_NEURONS_MAX=1000
THIRTY_N_MOTOR_UNITS=100
THIRTY_GAMMA_SHAPE=3.0

# 25 - 25 fr mean
TWENTYFIVE_TARGET_FR_MEAN=25
TWENTYFIVE_TARGET_FR_STD=3.75
TWENTYFIVE_TARGET_CONN_PROB=0.30
TWENTYFIVE_TARGET_N_DD_NEURONS=400
TWENTYFIVE_N_TRIALS=100
TWENTYFIVE_N_DD_NEURONS_MIN=100
TWENTYFIVE_N_DD_NEURONS_MAX=1000
TWENTYFIVE_N_MOTOR_UNITS=100
TWENTYFIVE_GAMMA_SHAPE=3.0

# 20 - 20 fr mean
TWENTY_TARGET_FR_MEAN=20
TWENTY_TARGET_FR_STD=3.0
TWENTY_TARGET_CONN_PROB=0.30
TWENTY_TARGET_N_DD_NEURONS=400
TWENTY_N_TRIALS=100
TWENTY_N_DD_NEURONS_MIN=100
TWENTY_N_DD_NEURONS_MAX=1000
TWENTY_N_MOTOR_UNITS=100
TWENTY_GAMMA_SHAPE=3.0

# 15 - 15 fr mean
FIFTEEN_TARGET_FR_MEAN=15
FIFTEEN_TARGET_FR_STD=2.25
FIFTEEN_TARGET_CONN_PROB=0.30
FIFTEEN_TARGET_N_DD_NEURONS=400
FIFTEEN_N_TRIALS=100
FIFTEEN_N_DD_NEURONS_MIN=100
FIFTEEN_N_DD_NEURONS_MAX=1000
FIFTEEN_N_MOTOR_UNITS=100
FIFTEEN_GAMMA_SHAPE=3.0

# 10 - 10 fr mean
TEN_TARGET_FR_MEAN=10
TEN_TARGET_FR_STD=1.5
TEN_TARGET_CONN_PROB=0.30
TEN_TARGET_N_DD_NEURONS=400
TEN_N_TRIALS=100
TEN_N_DD_NEURONS_MIN=100
TEN_N_DD_NEURONS_MAX=1000
TEN_N_MOTOR_UNITS=100
TEN_GAMMA_SHAPE=3.0

# 5 - 5 fr mean
FIVE_TARGET_FR_MEAN=5
FIVE_TARGET_FR_STD=0.75
FIVE_TARGET_CONN_PROB=0.30
FIVE_TARGET_N_DD_NEURONS=400
FIVE_N_TRIALS=100
FIVE_N_DD_NEURONS_MIN=100
FIVE_N_DD_NEURONS_MAX=1000
FIVE_N_MOTOR_UNITS=100
FIVE_GAMMA_SHAPE=3.0


# TEST - Quick test (10 trials)
TEST_TARGET_FR_MEAN=16.82
TEST_TARGET_FR_STD=2.5
TEST_TARGET_CONN_PROB=0.30
TEST_TARGET_N_DD_NEURONS=400
TEST_N_TRIALS=10
TEST_N_DD_NEURONS_MIN=100
TEST_N_DD_NEURONS_MAX=1000
TEST_N_MOTOR_UNITS=50
TEST_GAMMA_SHAPE=3.0

# =============================================================================
# Main
# =============================================================================

MUSCLE_INPUT="${1:-VLVM}"
# Extract muscle name (part before first underscore or entire string)
MUSCLE="${MUSCLE_INPUT%%_*}"
MUSCLE="${MUSCLE^^}"  # Convert to uppercase
shift || true  # Remove first argument, continue if no more args

# Parse optional arguments
GAMMA_OVERRIDE=""
ENABLE_GFLUCTDV_FLAG=""
GFLUCTDV_NOISE_MIN_OVERRIDE=""
GFLUCTDV_NOISE_MAX_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gamma-shape)
            GAMMA_OVERRIDE="$2"
            shift 2
            ;;
        --enable-gfluctdv)
            ENABLE_GFLUCTDV_FLAG="--enable-gfluctdv"
            shift
            ;;
        --gfluctdv-noise-min)
            GFLUCTDV_NOISE_MIN_OVERRIDE="$2"
            shift 2
            ;;
        --gfluctdv-noise-max)
            GFLUCTDV_NOISE_MAX_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Get parameters for this muscle
FR_MEAN="${MUSCLE}_TARGET_FR_MEAN"
FR_STD="${MUSCLE}_TARGET_FR_STD"
CONN_PROB="${MUSCLE}_TARGET_CONN_PROB"
N_DD="${MUSCLE}_TARGET_N_DD_NEURONS"
TRIALS="${MUSCLE}_N_TRIALS"
DD_MIN="${MUSCLE}_N_DD_NEURONS_MIN"
DD_MAX="${MUSCLE}_N_DD_NEURONS_MAX"
N_MU="${MUSCLE}_N_MOTOR_UNITS"
GAMMA="${MUSCLE}_GAMMA_SHAPE"

# Apply override if provided
if [[ -n "$GAMMA_OVERRIDE" ]]; then
    FINAL_GAMMA="$GAMMA_OVERRIDE"
else
    FINAL_GAMMA="${!GAMMA}"
fi

# Build study prefix with gamma shape and gfluctdv status
GFLUCTDV_SUFFIX=""
if [[ -n "$ENABLE_GFLUCTDV_FLAG" ]]; then
    GFLUCTDV_SUFFIX="gfluctdv_"
fi
STUDY_PREFIX="${MUSCLE}_${GFLUCTDV_SUFFIX}gamma${FINAL_GAMMA}_"

# Build python command with optional Gfluctdv parameters
PYTHON_CMD=(
    python "$PROJECT_ROOT/examples/finetune/optimize_dd_for_target_firing_rate.py"
    --study-prefix "${STUDY_PREFIX}"
    --target-fr-mean "${!FR_MEAN}"
    --target-fr-std "${!FR_STD}"
    --target-conn-prob "${!CONN_PROB}"
    --target-n-dd-neurons "${!N_DD}"
    --n-trials "${!TRIALS}"
    --n-dd-neurons-min "${!DD_MIN}"
    --n-dd-neurons-max "${!DD_MAX}"
    --n-motor-units "${!N_MU}"
    --gamma-shape "${FINAL_GAMMA}"
)

# Add Gfluctdv flags if enabled
if [[ -n "$ENABLE_GFLUCTDV_FLAG" ]]; then
    PYTHON_CMD+=("$ENABLE_GFLUCTDV_FLAG")
fi

if [[ -n "$GFLUCTDV_NOISE_MIN_OVERRIDE" ]]; then
    PYTHON_CMD+=(--gfluctdv-noise-min "$GFLUCTDV_NOISE_MIN_OVERRIDE")
fi

if [[ -n "$GFLUCTDV_NOISE_MAX_OVERRIDE" ]]; then
    PYTHON_CMD+=(--gfluctdv-noise-max "$GFLUCTDV_NOISE_MAX_OVERRIDE")
fi

# Run optimization
"${PYTHON_CMD[@]}"

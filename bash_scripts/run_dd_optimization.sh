#!/usr/bin/env bash
# Run descending drive optimization for different muscles
#
# Usage: ./bash_scripts/run_dd_optimization.sh [MUSCLE_NAME] [OPTIONS]
#
# Options:
#   --gamma-shape-min VALUE       Override gamma shape minimum (default: from config)
#   --gamma-shape-max VALUE       Override gamma shape maximum (default: from config)
#   --enable-gfluctdv             Enable Gfluctdv noise mechanism for motor neurons
#   --gfluctdv-noise-min VALUE    Minimum Gfluctdv noise amplitude in S/cm² (default: 5e-6)
#   --gfluctdv-noise-max VALUE    Maximum Gfluctdv noise amplitude in S/cm² (default: 3e-5)
#
# Examples:
#   ./bash_scripts/run_dd_optimization.sh VLVM
#   ./bash_scripts/run_dd_optimization.sh FDI --gamma-shape-min 5.0 --gamma-shape-max 8.0
#   ./bash_scripts/run_dd_optimization.sh THIRTY --enable-gfluctdv --gfluctdv-noise-min 1e-5 --gfluctdv-noise-max 5e-5
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

# VLVM - Vastus Lateralis/Medialis
VLVM_TARGET_FR_MEAN=16.5
VLVM_TARGET_FR_STD=5.0
VLVM_TARGET_CONN_PROB=0.30
VLVM_TARGET_N_DD_NEURONS=400
VLVM_N_TRIALS=100
VLVM_N_DD_NEURONS_MIN=100
VLVM_N_DD_NEURONS_MAX=2500
VLVM_N_MOTOR_UNITS=100
VLVM_GAMMA_SHAPE_MIN=3.0
VLVM_GAMMA_SHAPE_MAX=10.0

# FDI - First Dorsal Interosseous
FDI_TARGET_FR_MEAN=23.01
FDI_TARGET_FR_STD=6
FDI_TARGET_CONN_PROB=0.30
FDI_TARGET_N_DD_NEURONS=400
FDI_N_TRIALS=500
FDI_N_DD_NEURONS_MIN=100
FDI_N_DD_NEURONS_MAX=2500
FDI_N_MOTOR_UNITS=120
FDI_GAMMA_SHAPE_MIN=3.0
FDI_GAMMA_SHAPE_MAX=10.0

# TA - Tibialis Anterior
TA_TARGET_FR_MEAN=28.57
TA_TARGET_FR_STD=5
TA_TARGET_CONN_PROB=0.30
TA_TARGET_N_DD_NEURONS=400
TA_N_TRIALS=500
TA_N_DD_NEURONS_MIN=100
TA_N_DD_NEURONS_MAX=1000
TA_N_MOTOR_UNITS=110
TA_GAMMA_SHAPE_MIN=3.0
TA_GAMMA_SHAPE_MAX=10.0

# 30 - 30 fr mean
THIRTY_TARGET_FR_MEAN=30
THIRTY_TARGET_FR_STD=4.5
THIRTY_TARGET_CONN_PROB=0.30
THIRTY_TARGET_N_DD_NEURONS=400
THIRTY_N_TRIALS=500
THIRTY_N_DD_NEURONS_MIN=100
THIRTY_N_DD_NEURONS_MAX=1000
THIRTY_N_MOTOR_UNITS=100
THIRTY_GAMMA_SHAPE_MIN=3.0
THIRTY_GAMMA_SHAPE_MAX=10.0

# 25 - 25 fr mean
TWENTYFIVE_TARGET_FR_MEAN=25
TWENTYFIVE_TARGET_FR_STD=3.75
TWENTYFIVE_TARGET_CONN_PROB=0.30
TWENTYFIVE_TARGET_N_DD_NEURONS=400
TWENTYFIVE_N_TRIALS=500
TWENTYFIVE_N_DD_NEURONS_MIN=100
TWENTYFIVE_N_DD_NEURONS_MAX=1000
TWENTYFIVE_N_MOTOR_UNITS=100
TWENTYFIVE_GAMMA_SHAPE_MIN=3.0
TWENTYFIVE_GAMMA_SHAPE_MAX=10.0

# 20 - 20 fr mean
TWENTY_TARGET_FR_MEAN=20
TWENTY_TARGET_FR_STD=3.0
TWENTY_TARGET_CONN_PROB=0.30
TWENTY_TARGET_N_DD_NEURONS=400
TWENTY_N_TRIALS=500
TWENTY_N_DD_NEURONS_MIN=100
TWENTY_N_DD_NEURONS_MAX=1000
TWENTY_N_MOTOR_UNITS=100
TWENTY_GAMMA_SHAPE_MIN=3.0
TWENTY_GAMMA_SHAPE_MAX=10.0

# 15 - 15 fr mean
FIFTEEN_TARGET_FR_MEAN=15
FIFTEEN_TARGET_FR_STD=2.25
FIFTEEN_TARGET_CONN_PROB=0.30
FIFTEEN_TARGET_N_DD_NEURONS=400
FIFTEEN_N_TRIALS=500
FIFTEEN_N_DD_NEURONS_MIN=100
FIFTEEN_N_DD_NEURONS_MAX=1000
FIFTEEN_N_MOTOR_UNITS=100
FIFTEEN_GAMMA_SHAPE_MIN=3.0
FIFTEEN_GAMMA_SHAPE_MAX=10.0

# 10 - 10 fr mean
TEN_TARGET_FR_MEAN=10
TEN_TARGET_FR_STD=1.5
TEN_TARGET_CONN_PROB=0.30
TEN_TARGET_N_DD_NEURONS=400
TEN_N_TRIALS=500
TEN_N_DD_NEURONS_MIN=100
TEN_N_DD_NEURONS_MAX=1000
TEN_N_MOTOR_UNITS=100
TEN_GAMMA_SHAPE_MIN=3.0
TEN_GAMMA_SHAPE_MAX=10.0

# 5 - 5 fr mean
FIVE_TARGET_FR_MEAN=5
FIVE_TARGET_FR_STD=0.75
FIVE_TARGET_CONN_PROB=0.30
FIVE_TARGET_N_DD_NEURONS=400
FIVE_N_TRIALS=500
FIVE_N_DD_NEURONS_MIN=100
FIVE_N_DD_NEURONS_MAX=1000
FIVE_N_MOTOR_UNITS=100
FIVE_GAMMA_SHAPE_MIN=3.0
FIVE_GAMMA_SHAPE_MAX=10.0


# TEST - Quick test (10 trials)
TEST_TARGET_FR_MEAN=16.82
TEST_TARGET_FR_STD=2.5
TEST_TARGET_CONN_PROB=0.30
TEST_TARGET_N_DD_NEURONS=400
TEST_N_TRIALS=10
TEST_N_DD_NEURONS_MIN=100
TEST_N_DD_NEURONS_MAX=1000
TEST_N_MOTOR_UNITS=50
TEST_GAMMA_SHAPE_MIN=3.0
TEST_GAMMA_SHAPE_MAX=10.0

# =============================================================================
# Main
# =============================================================================

MUSCLE_INPUT="${1:-VLVM}"
# Extract muscle name (part before first underscore or entire string)
MUSCLE="${MUSCLE_INPUT%%_*}"
MUSCLE="${MUSCLE^^}"  # Convert to uppercase
shift || true  # Remove first argument, continue if no more args

# Parse optional arguments
GAMMA_MIN_OVERRIDE=""
GAMMA_MAX_OVERRIDE=""
ENABLE_GFLUCTDV_FLAG=""
GFLUCTDV_NOISE_MIN_OVERRIDE=""
GFLUCTDV_NOISE_MAX_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gamma-shape-min)
            GAMMA_MIN_OVERRIDE="$2"
            shift 2
            ;;
        --gamma-shape-max)
            GAMMA_MAX_OVERRIDE="$2"
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
GAMMA_MIN="${MUSCLE}_GAMMA_SHAPE_MIN"
GAMMA_MAX="${MUSCLE}_GAMMA_SHAPE_MAX"

# Apply overrides if provided
if [[ -n "$GAMMA_MIN_OVERRIDE" ]]; then
    FINAL_GAMMA_MIN="$GAMMA_MIN_OVERRIDE"
else
    FINAL_GAMMA_MIN="${!GAMMA_MIN}"
fi

if [[ -n "$GAMMA_MAX_OVERRIDE" ]]; then
    FINAL_GAMMA_MAX="$GAMMA_MAX_OVERRIDE"
else
    FINAL_GAMMA_MAX="${!GAMMA_MAX}"
fi

# Build study prefix with gamma shape range and gfluctdv status
GFLUCTDV_SUFFIX=""
if [[ -n "$ENABLE_GFLUCTDV_FLAG" ]]; then
    GFLUCTDV_SUFFIX="gfluctdv_"
fi
STUDY_PREFIX="${MUSCLE}_${GFLUCTDV_SUFFIX}gamma${FINAL_GAMMA_MIN}-${FINAL_GAMMA_MAX}_"

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
    --gamma-shape-min "${FINAL_GAMMA_MIN}"
    --gamma-shape-max "${FINAL_GAMMA_MAX}"
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

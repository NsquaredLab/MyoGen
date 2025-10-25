#!/usr/bin/env bash
# Run force optimization for different target force levels
#
# Usage: ./bash_scripts/run_force_optimization.sh [STUDY_PREFIX] [FORCE_PCT]
#
# Examples:
#   ./bash_scripts/run_force_optimization.sh VLVM 30
#   ./bash_scripts/run_force_optimization.sh THIRTY_gamma2.0-3.0 50
#
# Run multiple force levels in parallel:
#   for muscle in THIRTY_gamma2.0-3.0 TWENTY_gamma2.0-3.0; do
#     for force in 5 15 30 50; do
#       ./bash_scripts/run_force_optimization.sh "$muscle" "$force" &
#     done
#   done
#   wait

set -euo pipefail

# =============================================================================
# CONFIGURATION - Edit these parameters for each muscle
# =============================================================================

# VLVM - Vastus Lateralis/Medialis
VLVM_N_TRIALS=100
VLVM_DEFAULT_FORCE_PCT=30

# FDI - First Dorsal Interosseous
FDI_N_TRIALS=100
FDI_DEFAULT_FORCE_PCT=30

# TA - Tibialis Anterior
TA_N_TRIALS=100
TA_DEFAULT_FORCE_PCT=30

# 30 - 30 fr mean
THIRTY_N_TRIALS=100
THIRTY_DEFAULT_FORCE_PCT=30

# 25 - 25 fr mean
TWENTYFIVE_N_TRIALS=100
TWENTYFIVE_DEFAULT_FORCE_PCT=30

# 20 - 20 fr mean
TWENTY_N_TRIALS=100
TWENTY_DEFAULT_FORCE_PCT=30

# 15 - 15 fr mean
FIFTEEN_N_TRIALS=100
FIFTEEN_DEFAULT_FORCE_PCT=30

# 10 - 10 fr mean
TEN_N_TRIALS=100
TEN_DEFAULT_FORCE_PCT=30

# 5 - 5 fr mean
FIVE_N_TRIALS=100
FIVE_DEFAULT_FORCE_PCT=30

# TEST - Quick test (10 trials)
TEST_N_TRIALS=10
TEST_DEFAULT_FORCE_PCT=30

# =============================================================================
# Main
# =============================================================================

STUDY_PREFIX_INPUT="${1:-VLVM}"
FORCE_PCT="${2:-}"

# Extract muscle name (part before first underscore) for config lookup
MUSCLE="${STUDY_PREFIX_INPUT%%_*}"
MUSCLE="${MUSCLE^^}"  # Convert to uppercase

# Build full study prefix with trailing underscore
if [[ "$STUDY_PREFIX_INPUT" != *_ ]]; then
    STUDY_PREFIX="${STUDY_PREFIX_INPUT}_"
else
    STUDY_PREFIX="${STUDY_PREFIX_INPUT}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Get parameters for this muscle
TRIALS="${MUSCLE}_N_TRIALS"
DEFAULT_FORCE="${MUSCLE}_DEFAULT_FORCE_PCT"

# Use provided force percentage or default
if [[ -z "$FORCE_PCT" ]]; then
    FORCE_PCT="${!DEFAULT_FORCE}"
fi

echo "================================================================================"
echo "FORCE OPTIMIZATION: ${STUDY_PREFIX_INPUT} @ ${FORCE_PCT}% of baseline"
echo "================================================================================"
echo "  Trials: ${!TRIALS}"
echo "  Study prefix: ${STUDY_PREFIX}"
echo ""

# Set environment variables for headless operation
export MPLBACKEND=Agg
unset DISPLAY 2>/dev/null || true

# Run optimization
python "$PROJECT_ROOT/examples/finetune/optimize_dd_for_target_force.py" \
    --study-prefix "${STUDY_PREFIX}" \
    --target-force-pct "${FORCE_PCT}" \
    --n-trials "${!TRIALS}"

echo ""
echo "================================================================================"
echo "FORCE OPTIMIZATION COMPLETE: ${STUDY_PREFIX_INPUT} @ ${FORCE_PCT}%"
echo "================================================================================"

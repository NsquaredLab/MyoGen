#!/usr/bin/env bash
# Extract ISI and CV data from motor neuron spike trains
#
# Usage: ./bash_scripts/run_extract_isi_cv.sh [STUDY_PREFIX] [MVC_LEVEL]
#
# Examples:
#   ./bash_scripts/run_extract_isi_cv.sh VLVM 30
#   ./bash_scripts/run_extract_isi_cv.sh THIRTY_gamma2.0-3.0 50
#
# Run in parallel:
#   for muscle in THIRTY_gamma2.0-3.0 TWENTY_gamma2.0-3.0; do
#     for force in 5 15 30 50; do
#       ./bash_scripts/run_extract_isi_cv.sh "$muscle" "$force" &
#     done
#   done
#   wait

set -euo pipefail

# =============================================================================
# Main
# =============================================================================

STUDY_PREFIX_INPUT="${1:-VLVM}"
MVC_LEVEL="${2:-30}"

# Extract muscle name (part before first underscore) for display purposes
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

echo "================================================================================"
echo "EXTRACT ISI/CV DATA: ${STUDY_PREFIX_INPUT} @ ${MVC_LEVEL}% MVC"
echo "================================================================================"
echo "  Muscle: ${MUSCLE}"
echo "  MVC Level: ${MVC_LEVEL}%"
echo "  Study prefix: ${STUDY_PREFIX}"
echo ""

# Set environment variables for headless operation
export MPLBACKEND=Agg
unset DISPLAY 2>/dev/null || true

# Run extraction script with command-line arguments
python "$PROJECT_ROOT/examples/finetune/extract_isi_and_cv_per_ramps.py" \
    --muscle "${MUSCLE}" \
    --mvc-level "${MVC_LEVEL}" \
    --study-prefix "${STUDY_PREFIX}"

echo ""
echo "================================================================================"
echo "EXTRACTION COMPLETE: ${STUDY_PREFIX_INPUT} @ ${MVC_LEVEL}%"
echo "================================================================================"
echo "Output files saved to: examples/finetune/results/"

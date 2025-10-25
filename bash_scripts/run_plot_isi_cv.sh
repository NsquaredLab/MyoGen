#!/usr/bin/env bash
# Plot ISI/CV comparison across multiple MVC levels
#
# Usage: ./bash_scripts/run_plot_isi_cv.sh [MUSCLE_NAME] [MVC_LEVELS...]
#
# Examples:
#   ./bash_scripts/run_plot_isi_cv.sh VLVM 5 15 30
#   ./bash_scripts/run_plot_isi_cv.sh FDI 10 20 30
#   ./bash_scripts/run_plot_isi_cv.sh TA 5 15 30 50

set -euo pipefail

# =============================================================================
# Main
# =============================================================================

MUSCLE="${1:-VLVM}"
MUSCLE="${MUSCLE^^}"  # Convert to uppercase
shift || true  # Remove first argument

# Remaining arguments are MVC levels
MVC_LEVELS="${@:-5 15 30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "================================================================================"
echo "PLOT ISI/CV COMPARISON: ${MUSCLE}"
echo "================================================================================"
echo "  Muscle: ${MUSCLE}"
echo "  MVC Levels: ${MVC_LEVELS}"
echo "  Study prefix: ${MUSCLE}_"
echo ""

# Set environment variables for headless operation
export MPLBACKEND=Agg
unset DISPLAY 2>/dev/null || true

# Run plotting script
python "$PROJECT_ROOT/examples/finetune/plot_isi_cv_comparison.py" \
    --muscle "${MUSCLE}" \
    --mvc-levels ${MVC_LEVELS} \
    --study-prefix "${MUSCLE}_" \
    --output-format jpg

echo ""
echo "================================================================================"
echo "PLOT COMPLETE: ${MUSCLE}"
echo "================================================================================"
echo "Output saved to: examples/finetune/results/"

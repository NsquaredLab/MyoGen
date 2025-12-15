#!/usr/bin/env bash
# Validate force output from optimized descending drive parameters
#
# Usage: ./bash_scripts/run_compute_force_mvc.sh [STUDY_PREFIX]
#
# Examples:
#   ./bash_scripts/run_compute_force_mvc.sh VLVM
#   ./bash_scripts/run_compute_force_mvc.sh THIRTY_gamma2.0-3.0
#
# Run in parallel:
#   for fr in THIRTY_gamma2.0-3.0 TWENTY_gamma2.0-3.0; do
#     ./bash_scripts/run_compute_force_mvc.sh "$fr" &
#   done
#   wait

set -euo pipefail

STUDY_PREFIX_INPUT="${1:-VLVM}"

# If input doesn't end with underscore, add it
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

# Run force validation
python "$PROJECT_ROOT/examples/finetune/compute_force_from_optimized_dd.py" \
    --study-prefix "${STUDY_PREFIX}"

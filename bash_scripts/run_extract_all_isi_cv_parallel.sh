#!/bin/bash
#
# Extract ISI and CV data for all force optimization results (PARALLEL VERSION)
# This script processes all optimization result files in parallel using GNU parallel
# or simple background jobs.
#

set -e

# Configuration
OPTIMIZATION_DIR="/home/oj98yqyk/code/simulators/MyoGen/results/force_optimization"
FINETUNE_DIR="/home/oj98yqyk/code/simulators/MyoGen/examples/finetune"
RESULTS_DIR="${FINETUNE_DIR}/results"
MAX_PARALLEL_JOBS=8  # Adjust based on CPU cores available
MUSCLE="VLVM"

# Change to finetune directory
cd ${FINETUNE_DIR}

# Create results directory
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "ISI/CV Extraction (PARALLEL MODE)"
echo "=========================================="
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo ""

# Function to process a single configuration
process_config() {
    local OPTIM_FILE=$1
    local FILENAME=$(basename ${OPTIM_FILE})

    # Parse study prefix and MVC level
    local STUDY_PREFIX=$(echo ${FILENAME} | sed 's/_dd_optimized_params_force_.*//')
    local MVC_LEVEL=$(echo ${FILENAME} | sed 's/.*_force_//' | sed 's/pct.json//')

    # Check if output already exists
    local OUTPUT_FILE="${RESULTS_DIR}/${STUDY_PREFIX}_isi_cv_data_${MUSCLE}_${MVC_LEVEL}.csv"

    if [ -f "${OUTPUT_FILE}" ]; then
        echo "[SKIP] ${STUDY_PREFIX} @ ${MVC_LEVEL}% (exists)"
        return 0
    fi

    echo "[RUN] ${STUDY_PREFIX} @ ${MVC_LEVEL}%"

    # Run extraction (suppress output)
    cd ${FINETUNE_DIR}
    uv run python extract_isi_and_cv_per_ramps.py \
        --muscle ${MUSCLE} \
        --mvc-level ${MVC_LEVEL} \
        --study-prefix "${STUDY_PREFIX}_" \
        --seed 42 \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "[✓] ${STUDY_PREFIX} @ ${MVC_LEVEL}%"
    else
        echo "[✗] ${STUDY_PREFIX} @ ${MVC_LEVEL}% FAILED"
        return 1
    fi
}

export -f process_config
export FINETUNE_DIR RESULTS_DIR MUSCLE

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for processing"
    echo ""

    # Use GNU parallel
    ls ${OPTIMIZATION_DIR}/*_dd_optimized_params_force_*pct.json | \
        parallel -j ${MAX_PARALLEL_JOBS} --bar process_config {}
else
    echo "GNU parallel not found, using simple background jobs"
    echo ""

    # Fallback to simple background jobs with job control
    RUNNING_JOBS=0
    TOTAL_FILES=$(ls ${OPTIMIZATION_DIR}/*_dd_optimized_params_force_*pct.json | wc -l)
    CURRENT=0

    for OPTIM_FILE in ${OPTIMIZATION_DIR}/*_dd_optimized_params_force_*pct.json; do
        CURRENT=$((CURRENT + 1))

        # Wait if we've reached max parallel jobs
        while [ ${RUNNING_JOBS} -ge ${MAX_PARALLEL_JOBS} ]; do
            # Check how many background jobs are still running
            RUNNING_JOBS=$(jobs -r | wc -l)
            sleep 1
        done

        # Start job in background
        process_config "${OPTIM_FILE}" &
        RUNNING_JOBS=$((RUNNING_JOBS + 1))

        # Progress indicator
        if [ $((CURRENT % 10)) -eq 0 ]; then
            echo "[Progress] ${CURRENT}/${TOTAL_FILES} submitted"
        fi
    done

    # Wait for all remaining jobs to complete
    echo ""
    echo "Waiting for remaining jobs to complete..."
    wait
fi

echo ""
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="

# Count output files
OUTPUT_COUNT=$(ls ${RESULTS_DIR}/*isi_cv_data*.csv 2>/dev/null | wc -l)
echo "Total ISI/CV files created: ${OUTPUT_COUNT}"
echo "Results saved to: ${RESULTS_DIR}/"

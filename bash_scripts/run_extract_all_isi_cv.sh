#!/bin/bash
#
# Extract ISI and CV data for all force optimization results
# This script processes all optimization result files and extracts
# ISI/CV statistics using only the plateau phase.
#

set -e

# Change to the finetune examples directory
cd /home/oj98yqyk/code/simulators/MyoGen/examples/finetune

# Create results directory if it doesn't exist
mkdir -p results

# Get list of all optimization result files
OPTIMIZATION_DIR="/home/oj98yqyk/code/simulators/MyoGen/results/force_optimization"

# Count total files for progress
TOTAL_FILES=$(ls ${OPTIMIZATION_DIR}/*_dd_optimized_params_force_*pct.json 2>/dev/null | wc -l)
CURRENT=0

echo "=========================================="
echo "ISI/CV Extraction for All Results"
echo "=========================================="
echo "Total configurations to process: ${TOTAL_FILES}"
echo ""

# Process each optimization result file
for OPTIM_FILE in ${OPTIMIZATION_DIR}/*_dd_optimized_params_force_*pct.json; do
    CURRENT=$((CURRENT + 1))

    # Extract filename without path
    FILENAME=$(basename ${OPTIM_FILE})

    # Parse study prefix and MVC level from filename
    # Format: PREFIX_dd_optimized_params_force_XXpct.json
    # Extract everything before "_dd_optimized_params_force_"
    STUDY_PREFIX=$(echo ${FILENAME} | sed 's/_dd_optimized_params_force_.*//')

    # Extract MVC level (number before "pct.json")
    MVC_LEVEL=$(echo ${FILENAME} | sed 's/.*_force_//' | sed 's/pct.json//')

    # Determine muscle name (default to VLVM)
    MUSCLE="VLVM"

    echo "[${CURRENT}/${TOTAL_FILES}] Processing: ${STUDY_PREFIX} @ ${MVC_LEVEL}% MVC"

    # Check if output file already exists
    OUTPUT_FILE="results/${STUDY_PREFIX}_isi_cv_data_${MUSCLE}_${MVC_LEVEL}.csv"

    if [ -f "${OUTPUT_FILE}" ]; then
        echo "  ⏭️  Skipping (output already exists)"
        continue
    fi

    # Run the extraction script
    echo "  ▶️  Running extraction..."

    uv run python extract_isi_and_cv_per_ramps.py \
        --muscle ${MUSCLE} \
        --mvc-level ${MVC_LEVEL} \
        --study-prefix "${STUDY_PREFIX}_" \
        --seed 42 \
        > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✅ Success"
    else
        echo "  ❌ Failed"
    fi

    echo ""
done

echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo "Processed ${TOTAL_FILES} configurations"
echo "Results saved to: $(pwd)/results/"
echo ""

# Count output files
OUTPUT_COUNT=$(ls results/*isi_cv_data*.csv 2>/dev/null | wc -l)
echo "Total ISI/CV files created: ${OUTPUT_COUNT}"

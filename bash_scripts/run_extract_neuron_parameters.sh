#!/bin/bash
#
# Extract motor neuron parameters in parallel for different recruitment models
# This script processes motor unit pools using GNU parallel or background jobs.
#

set -e

# Configuration
PROJECT_ROOT="/home/oj98yqyk/code/simulators/MyoGen"
EXAMPLE_SCRIPT="${PROJECT_ROOT}/examples/basic/09_extract_neuron_parameters.py"
MAX_PARALLEL_JOBS=8  # Default, can be overridden

# Default parameters
MODEL="all"
BIOPHYSICAL_MODEL="NERLab"
N_MOTOR_UNITS=100
SEED=42

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --n-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --biophysical-model)
            BIOPHYSICAL_MODEL="$2"
            shift 2
            ;;
        --n-motor-units)
            N_MOTOR_UNITS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL               Recruitment model: fuglevand, deluca, konstantin, combined, or all (default: all)"
            echo "  --n-jobs N                  Number of parallel jobs (default: 8)"
            echo "  --biophysical-model MODEL   Biophysical model: NERLab or Powers2017 (default: NERLab)"
            echo "  --n-motor-units N           Number of motor units (default: 100)"
            echo "  --seed N                    Random seed (default: 42)"
            echo "  -h, --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run all models with 10 parallel jobs"
            echo "  $0 --model all --n-jobs 10"
            echo ""
            echo "  # Run single model"
            echo "  $0 --model fuglevand"
            echo ""
            echo "  # Run with Powers2017 model"
            echo "  $0 --model all --biophysical-model Powers2017"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Ensure we're in the project root
cd "${PROJECT_ROOT}"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found at .venv/bin/activate"
    exit 1
fi

echo "=========================================="
echo "Motor Neuron Parameter Extraction"
echo "=========================================="
echo "Model(s): ${MODEL}"
echo "Biophysical model: ${BIOPHYSICAL_MODEL}"
echo "Number of motor units: ${N_MOTOR_UNITS}"
echo "Parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Random seed: ${SEED}"
echo ""

# Define model configurations
# Format: "model_name:slope" (slope only for deluca and combined)
if [ "${MODEL}" = "all" ]; then
    MODEL_CONFIGS=("fuglevand" "deluca:5" "deluca:25" "konstantin" "combined:5" "combined:25")
else
    # Single model
    if [ "${MODEL}" = "deluca" ] || [ "${MODEL}" = "combined" ]; then
        # Default to testing both slopes for these models
        MODEL_CONFIGS=("${MODEL}:5" "${MODEL}:25")
    else
        MODEL_CONFIGS=("${MODEL}")
    fi
fi

# Function to process a single cell
process_cell() {
    local CELL_INDEX=$1
    local MODEL_NAME=$2
    local SLOPE=$3

    # Build command
    CMD="uv run python ${EXAMPLE_SCRIPT} \
        --model ${MODEL_NAME} \
        --cell-index ${CELL_INDEX} \
        --save-individual \
        --biophysical-model ${BIOPHYSICAL_MODEL} \
        --n-motor-units ${N_MOTOR_UNITS} \
        --seed ${SEED}"

    # Add slope parameter if applicable
    if [ -n "${SLOPE}" ]; then
        CMD="${CMD} --deluca-slope ${SLOPE}"
    fi

    # Run silently
    ${CMD} > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        return 0
    else
        echo "[✗] Cell ${CELL_INDEX} failed"
        return 1
    fi
}

export -f process_cell
export EXAMPLE_SCRIPT BIOPHYSICAL_MODEL N_MOTOR_UNITS SEED PROJECT_ROOT

# Process each model configuration
for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
    # Parse model name and slope
    MODEL_NAME=$(echo ${MODEL_CONFIG} | cut -d: -f1)
    SLOPE=$(echo ${MODEL_CONFIG} | cut -d: -f2 -s)

    # Display configuration
    if [ -n "${SLOPE}" ]; then
        echo "Processing: ${MODEL_NAME} (slope=${SLOPE})"
        DISPLAY_NAME="${MODEL_NAME}_s${SLOPE}"
    else
        echo "Processing: ${MODEL_NAME}"
        DISPLAY_NAME="${MODEL_NAME}"
    fi
    echo "----------------------------------------"

    # Generate cell indices (0 to N_MOTOR_UNITS-1)
    CELL_INDICES=$(seq 0 $((N_MOTOR_UNITS - 1)))

    # Check if GNU parallel is available
    if command -v parallel &> /dev/null; then
        echo "Using GNU parallel for processing"

        # Process with GNU parallel
        echo "${CELL_INDICES}" | tr ' ' '\n' | \
            parallel -j ${MAX_PARALLEL_JOBS} --bar \
                process_cell {} ${MODEL_NAME} ${SLOPE}

        PARALLEL_EXIT=$?
    else
        echo "GNU parallel not found, using background jobs"

        # Fallback to background jobs with job control
        RUNNING_JOBS=0
        TOTAL_CELLS=${N_MOTOR_UNITS}
        CURRENT=0
        FAILED_COUNT=0

        for CELL_INDEX in ${CELL_INDICES}; do
            CURRENT=$((CURRENT + 1))

            # Wait if we've reached max parallel jobs
            while [ ${RUNNING_JOBS} -ge ${MAX_PARALLEL_JOBS} ]; do
                # Check how many background jobs are still running
                RUNNING_JOBS=$(jobs -r | wc -l)
                sleep 0.1
            done

            # Start job in background
            process_cell ${CELL_INDEX} ${MODEL_NAME} ${SLOPE} &
            RUNNING_JOBS=$((RUNNING_JOBS + 1))

            # Progress indicator
            if [ $((CURRENT % 10)) -eq 0 ]; then
                PERCENT=$((CURRENT * 100 / TOTAL_CELLS))
                echo "[Progress] ${CURRENT}/${TOTAL_CELLS} cells (${PERCENT}%)"
            fi
        done

        # Wait for all remaining jobs to complete
        echo "Waiting for remaining jobs to complete..."
        wait
        PARALLEL_EXIT=$?
    fi

    # Check if processing succeeded
    if [ ${PARALLEL_EXIT} -ne 0 ]; then
        echo "[WARNING] Some cells failed during processing"
    fi

    echo ""
    echo "Aggregating results for ${DISPLAY_NAME}..."

    # Aggregate results
    AGGREGATE_CMD="uv run python ${EXAMPLE_SCRIPT} \
        --model ${MODEL_NAME} \
        --aggregate-only \
        --biophysical-model ${BIOPHYSICAL_MODEL} \
        --n-motor-units ${N_MOTOR_UNITS} \
        --seed ${SEED}"

    if [ -n "${SLOPE}" ]; then
        AGGREGATE_CMD="${AGGREGATE_CMD} --deluca-slope ${SLOPE}"
    fi

    ${AGGREGATE_CMD}

    if [ $? -eq 0 ]; then
        echo "[✓] ${DISPLAY_NAME} complete"
    else
        echo "[✗] ${DISPLAY_NAME} aggregation failed"
    fi

    echo ""
done

echo "=========================================="
echo "All Extractions Complete!"
echo "=========================================="

# Count output files
CSV_COUNT=$(ls ${PROJECT_ROOT}/examples/basic/results/parameters_*_${BIOPHYSICAL_MODEL}.csv 2>/dev/null | wc -l)
echo "Total parameter files created: ${CSV_COUNT}"
echo "Results saved to: ${PROJECT_ROOT}/examples/basic/results/"
echo ""
echo "To visualize results, run:"
echo "  python ${EXAMPLE_SCRIPT} --model <model_name>"

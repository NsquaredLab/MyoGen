#!/bin/bash
# Parameter scan script for combined/deluca models with varying slopes
# This script runs parameter extraction across a range of slope values

set -e  # Exit on error

# Default parameters
MODEL="combined"
BIOPHYSICAL_MODEL="NERLab"
N_MOTOR_UNITS=100
SEED=42
N_JOBS=-2  # Use all CPUs except 1
SLOPE_START=0.1
SLOPE_END=25.0
SLOPE_STEP=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --slope-start)
            SLOPE_START="$2"
            shift 2
            ;;
        --slope-end)
            SLOPE_END="$2"
            shift 2
            ;;
        --slope-step)
            SLOPE_STEP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run parameter extraction across a range of slope values"
            echo ""
            echo "Options:"
            echo "  --model MODEL              Model to use (deluca or combined, default: combined)"
            echo "  --biophysical-model MODEL  Biophysical model (NERLab or Powers2017, default: NERLab)"
            echo "  --n-motor-units N          Number of motor units (default: 100)"
            echo "  --seed SEED                Random seed (default: 42)"
            echo "  --n-jobs N                 Number of parallel jobs (default: -2, all CPUs except 1)"
            echo "  --slope-start VALUE        Starting slope value (default: 0.1)"
            echo "  --slope-end VALUE          Ending slope value (default: 25.0)"
            echo "  --slope-step VALUE         Step size for slope scan (default: 0.1)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Scan slopes from 0.1 to 25 with step 0.5"
            echo "  $0 --slope-start 0.1 --slope-end 25.0 --slope-step 0.5"
            echo ""
            echo "  # Scan slopes for deluca model with 50 motor units"
            echo "  $0 --model deluca --n-motor-units 50 --slope-start 1 --slope-end 10 --slope-step 1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate model
if [[ "$MODEL" != "deluca" && "$MODEL" != "combined" ]]; then
    echo "Error: Model must be 'deluca' or 'combined'"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Parameter Scan Configuration"
echo "=========================================="
echo "Model: $MODEL"
echo "Biophysical Model: $BIOPHYSICAL_MODEL"
echo "Number of Motor Units: $N_MOTOR_UNITS"
echo "Random Seed: $SEED"
echo "Parallel Jobs: $N_JOBS"
echo "Slope Range: $SLOPE_START to $SLOPE_END (step: $SLOPE_STEP)"
echo "=========================================="
echo ""

# Generate slope values using Python
echo "Generating slope values..."
SLOPES=$(python3 -c "import numpy as np; slopes = np.arange($SLOPE_START, $SLOPE_END + $SLOPE_STEP/2, $SLOPE_STEP); print(' '.join(map(str, slopes)))")

# Count total number of slopes
N_SLOPES=$(echo $SLOPES | wc -w)
echo "Running scan for $N_SLOPES slope values"
echo ""

# Run extraction for each slope value
CURRENT=0
for SLOPE in $SLOPES; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$N_SLOPES] Processing slope = $SLOPE"

    # Run the extraction
    uv run python examples/basic/09_extract_neuron_parameters.py \
        --model "$MODEL" \
        --deluca-slope "$SLOPE" \
        --biophysical-model "$BIOPHYSICAL_MODEL" \
        --n-motor-units "$N_MOTOR_UNITS" \
        --seed "$SEED" \
        --n-jobs "$N_JOBS"

    echo "  ✓ Completed slope = $SLOPE"
    echo ""
done

echo "=========================================="
echo "Scan Complete!"
echo "=========================================="
echo "Processed $N_SLOPES slope values"
echo "Results saved in ./results/"
echo ""
echo "Generated files:"
echo "  - parameters_${MODEL}_s*_${BIOPHYSICAL_MODEL}.csv"
echo "  - parameters_${MODEL}_s*_${BIOPHYSICAL_MODEL}.pkl"
echo "  - parameters_main_${MODEL}_s*.png"
echo "=========================================="

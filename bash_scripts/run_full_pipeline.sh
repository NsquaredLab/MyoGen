#!/usr/bin/env bash
# Full optimization and analysis pipeline
#
# Usage: ./bash_scripts/run_full_pipeline.sh --gamma-shape VALUE [OPTIONS]
#
# Options:
#   --gamma-shape VALUE         Gamma shape value (required, controls DD input variability)
#   --enable-gfluctdv           Enable Gfluctdv noise mechanism
#   --clear-optuna-cache        Delete existing optuna databases to force fresh optimization
#   --skip-dd-optimization      Skip DD optimization (use existing results)
#   --skip-force-computation    Skip force computation at MVC
#   --skip-force-optimization   Skip force optimization
#   --skip-optuna-optimization  Skip all optuna-based optimizations (DD + Force)
#   --skip-isi-extraction       Skip ISI/CV extraction
#   --skip-plotting             Skip final plotting
#   --output-format FORMAT      Plot output format (jpg, png, pdf, svg; default: jpg)
#
# Examples:
#   # Full pipeline with regular CV input
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 3.0
#
#   # High CV input with Gfluctdv enabled
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 0.5 --enable-gfluctdv
#
#   # Clear optuna cache for gamma=3.0 and re-optimize from scratch
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 3.0 --clear-optuna-cache
#
#   # Skip DD optimization if already done
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 2.5 --skip-dd-optimization
#
#   # Skip all optuna optimizations (DD + Force)
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 3.0 --skip-optuna-optimization
#
#   # Custom output format
#   ./bash_scripts/run_full_pipeline.sh --gamma-shape 3.0 --output-format pdf

set -euo pipefail

# =============================================================================
# Parse arguments
# =============================================================================

GAMMA_SHAPE=""
ENABLE_GFLUCTDV_FLAG=""
CLEAR_OPTUNA_CACHE=false
SKIP_DD_OPT=false
SKIP_FORCE_COMP=false
SKIP_FORCE_OPT=false
SKIP_ISI_EXTRACT=false
SKIP_PLOT=false
OUTPUT_FORMAT="jpg"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gamma-shape)
            GAMMA_SHAPE="$2"
            shift 2
            ;;
        --enable-gfluctdv)
            ENABLE_GFLUCTDV_FLAG="--enable-gfluctdv"
            shift
            ;;
        --clear-optuna-cache)
            CLEAR_OPTUNA_CACHE=true
            shift
            ;;
        --skip-dd-optimization)
            SKIP_DD_OPT=true
            shift
            ;;
        --skip-force-computation)
            SKIP_FORCE_COMP=true
            shift
            ;;
        --skip-force-optimization)
            SKIP_FORCE_OPT=true
            shift
            ;;
        --skip-optuna-optimization)
            SKIP_DD_OPT=true
            SKIP_FORCE_OPT=true
            shift
            ;;
        --skip-isi-extraction)
            SKIP_ISI_EXTRACT=true
            shift
            ;;
        --skip-plotting)
            SKIP_PLOT=true
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --gamma-shape VALUE [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$GAMMA_SHAPE" ]]; then
    echo "Error: --gamma-shape is required"
    echo "Usage: $0 --gamma-shape VALUE [OPTIONS]"
    exit 1
fi

# =============================================================================
# Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Build gamma suffix for muscle names
GFLUCTDV_SUFFIX=""
if [[ -n "$ENABLE_GFLUCTDV_FLAG" ]]; then
    GFLUCTDV_SUFFIX="gfluctdv_"
fi
GAMMA_SUFFIX="${GFLUCTDV_SUFFIX}gamma${GAMMA_SHAPE}"

# Define firing rate targets
FIRING_RATES=(THIRTY TWENTYFIVE TWENTY FIFTEEN TEN FIVE)

# Define force levels for optimization
FORCE_LEVELS=(5 15 30 50 75 90)

echo "========================================================================"
echo "MyoGen Full Optimization Pipeline"
echo "========================================================================"
echo "Gamma shape: ${GAMMA_SHAPE}"
echo "Gfluctdv enabled: ${ENABLE_GFLUCTDV_FLAG:-No}"
echo "Clear optuna cache: ${CLEAR_OPTUNA_CACHE}"
echo "Output format: ${OUTPUT_FORMAT}"
echo "Muscle naming pattern: {FR}_${GAMMA_SUFFIX}"
echo "========================================================================"
echo ""

# =============================================================================
# Clear Optuna Cache (if requested)
# =============================================================================

if [[ "$CLEAR_OPTUNA_CACHE" == true ]]; then
    echo "Clearing Optuna cache..."
    echo "------------------------------------------------------------"

    RESULTS_DIR="$PROJECT_ROOT/results/dd_optimization"

    # Clear DD optimization databases
    for fr in "${FIRING_RATES[@]}"; do
        muscle="${fr}_${GAMMA_SUFFIX}"
        db_file="${RESULTS_DIR}/${muscle}_optuna_dd_optimization.db"
        if [[ -f "$db_file" ]]; then
            echo "  Deleting: $db_file"
            rm -f "$db_file"
        fi
    done

    # Clear force optimization databases
    for fr in "${FIRING_RATES[@]}"; do
        muscle="${fr}_${GAMMA_SUFFIX}"
        for force in "${FORCE_LEVELS[@]}"; do
            db_file="${RESULTS_DIR}/${muscle}_optuna_force_${force}pct.db"
            if [[ -f "$db_file" ]]; then
                echo "  Deleting: $db_file"
                rm -f "$db_file"
            fi
        done
    done

    echo "  ✓ Optuna cache cleared"
    echo ""
fi

# =============================================================================
# Step 1: Descending Drive Optimization
# =============================================================================

if [[ "$SKIP_DD_OPT" == false ]]; then
    echo "Step 1/5: Running DD optimization for all firing rates..."
    echo "------------------------------------------------------------"

    for fr in "${FIRING_RATES[@]}"; do
        echo "  Starting: $fr"
        ./bash_scripts/run_dd_optimization.sh "$fr" \
            --gamma-shape "$GAMMA_SHAPE" \
            $ENABLE_GFLUCTDV_FLAG &
    done

    echo "  Waiting for all DD optimizations to complete..."
    wait
    echo "  ✓ DD optimization complete"
    echo ""
else
    echo "Step 1/5: Skipping DD optimization (using existing results)"
    echo ""
fi

# =============================================================================
# Step 2: Force Computation at MVC
# =============================================================================

if [[ "$SKIP_FORCE_COMP" == false ]]; then
    echo "Step 2/5: Computing force at MVC for all muscles..."
    echo "------------------------------------------------------------"

    for fr in "${FIRING_RATES[@]}"; do
        muscle="${fr}_${GAMMA_SUFFIX}"
        echo "  Starting: $muscle"
        ./bash_scripts/run_compute_force_mvc.sh "$muscle" &
    done

    echo "  Waiting for all force computations to complete..."
    wait
    echo "  ✓ Force computation complete"
    echo ""
else
    echo "Step 2/5: Skipping force computation (using existing results)"
    echo ""
fi

# =============================================================================
# Step 3: Force Optimization
# =============================================================================

if [[ "$SKIP_FORCE_OPT" == false ]]; then
    echo "Step 3/5: Running force optimization for all muscles and force levels..."
    echo "------------------------------------------------------------"

    total_jobs=$((${#FIRING_RATES[@]} * ${#FORCE_LEVELS[@]}))
    current_job=0

    for fr in "${FIRING_RATES[@]}"; do
        muscle="${fr}_${GAMMA_SUFFIX}"
        for force in "${FORCE_LEVELS[@]}"; do
            current_job=$((current_job + 1))
            echo "  Starting: $muscle at ${force}% MVC ($current_job/$total_jobs)"
            ./bash_scripts/run_force_optimization.sh "$muscle" "$force" &
        done
    done

    echo "  Waiting for all force optimizations to complete..."
    wait
    echo "  ✓ Force optimization complete"
    echo ""
else
    echo "Step 3/5: Skipping force optimization (using existing results)"
    echo ""
fi

# =============================================================================
# Step 4: ISI and CV Extraction
# =============================================================================

if [[ "$SKIP_ISI_EXTRACT" == false ]]; then
    echo "Step 4/5: Extracting ISI and CV statistics..."
    echo "------------------------------------------------------------"

    total_jobs=$((${#FIRING_RATES[@]} * ${#FORCE_LEVELS[@]}))
    current_job=0

    for fr in "${FIRING_RATES[@]}"; do
        muscle="${fr}_${GAMMA_SUFFIX}"
        for force in "${FORCE_LEVELS[@]}"; do
            current_job=$((current_job + 1))
            echo "  Starting: $muscle at ${force}% MVC ($current_job/$total_jobs)"
            ./bash_scripts/run_extract_isi_cv.sh "$muscle" "$force" &
        done
    done

    echo "  Waiting for all ISI/CV extractions to complete..."
    wait
    echo "  ✓ ISI/CV extraction complete"
    echo ""
else
    echo "Step 4/5: Skipping ISI/CV extraction (using existing results)"
    echo ""
fi

# =============================================================================
# Step 5: Multi-Muscle Comparison Plot
# =============================================================================

if [[ "$SKIP_PLOT" == false ]]; then
    echo "Step 5/5: Generating multi-muscle comparison plots..."
    echo "------------------------------------------------------------"

    # Build muscle list with gamma suffix
    MUSCLE_LIST=()
    for fr in "${FIRING_RATES[@]}"; do
        MUSCLE_LIST+=("${fr}_${GAMMA_SUFFIX}")
    done

    echo "  Muscles: ${MUSCLE_LIST[*]}"
    echo "  Output format: ${OUTPUT_FORMAT}"

    python "$PROJECT_ROOT/examples/finetune/plot_isi_cv_multi_muscle_comparison.py" \
        --muscles "${MUSCLE_LIST[@]}" \
        --results-path "$PROJECT_ROOT/results" \
        --output-format "$OUTPUT_FORMAT"

    echo "  ✓ Plotting complete"
    echo ""
else
    echo "Step 5/5: Skipping plotting"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================

echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"
echo "Gamma shape: ${GAMMA_SHAPE}"
echo "Muscle suffix: ${GAMMA_SUFFIX}"
echo ""
echo "Results location:"
echo "  DD optimization:  ./results/dd_optimization/"
echo "  Force data:       ./results/dd_optimization/"
echo "  ISI/CV data:      ./results/ISI_statistics/"
echo "  Plots:            ./results/figures/"
echo ""
echo "To re-run specific steps, use --skip-* flags"
echo "Example: $0 --gamma-shape $GAMMA_SHAPE --skip-dd-optimization"
echo "Or skip all optuna optimizations: $0 --gamma-shape $GAMMA_SHAPE --skip-optuna-optimization"
echo "========================================================================"

#!/bin/bash
# Simulate Surface EMG with Custom Parameters
# ============================================
#
# This script provides an easy command-line interface for simulating
# surface EMG with configurable motor unit selection and SNR.
#
# Usage Examples:
#   ./simulate_semg.sh --mus "0,5,10,15,20" --snr 5
#   ./simulate_semg.sh --mus all --snr 10
#   ./simulate_semg.sh --mus "0-20" --snr 3
#   ./simulate_semg.sh --mus "0-100-10" --snr 1
#
#   # Negative indices (use = syntax to avoid bash interpretation issues)
#   ./simulate_semg.sh --mus="-1" --snr 5               # Last active MU (largest)
#   ./simulate_semg.sh --mus="-5,-1" --snr 5            # 5th from last and last
#   ./simulate_semg.sh --mus="-10--1" --snr 5           # Last 10 active MUs
#   ./simulate_semg.sh --mus="-5,-4,-3,-2,-1" --snr 5   # Last 5 active MUs
#
#   ./simulate_semg.sh --mus "0,5,10" --snr 5 --no-plot
#   ./simulate_semg.sh --mus "0,5,10" --snr 5 --peak-hz 80 --plateau-time 8000
#   ./simulate_semg.sh --mus "0,5,10" --snr 5 --num-rows 8 --num-cols 8
#
# Parameters:
#   --mus     Motor units to simulate:
#             "all"          = All motor units
#             "0,5,10"       = Specific indices [0, 5, 10]
#             "0-20"         = Range from 0 to 20
#             "0-100-10"     = Every 10th MU from 0 to 100
#             "-1"           = Last active MU (largest that fired)
#             "-5,-1"        = 5th from last and last active MUs
#             "-10--1"       = Last 10 active MUs
#             Note: For negative indices, use --mus="-1" syntax (with =)
#
#   --snr     Signal-to-noise ratio in dB (default: 5)
#
#   --no-plot Skip plotting for faster execution
#
# Electrode Grid Parameters:
#   --num-rows    Number of electrode rows (default: 5)
#   --num-cols    Number of electrode columns (default: 5)
#
# Trapezoid Drive Pattern Parameters (optional - generates custom pattern if any provided):
#   --sim-time        Simulation time in ms (default: 13000)
#   --timestep        Time step in ms (default: 0.1)
#   --rise-time       Ramp-up duration in ms (default: 500)
#   --plateau-time    Plateau duration in ms (default: 10000)
#   --fall-time       Ramp-down duration in ms (default: 500)
#   --rest-before     Initial rest in ms (default: 1000)
#   --rest-after      Final rest in ms (default: 1000)
#   --baseline-hz     Baseline drive in Hz (default: 0.0)
#   --peak-hz         Peak drive in Hz (default: 65.0)
#   --noise-std       Noise std deviation in Hz (default: 1.0)
#   --no-noise        Disable noise in trapezoid
#
# Output Directory:
#   results/synthetic_gen/semg_mu_X_snrY/          (for default 5×5 grid)
#   results/synthetic_gen/semg_mu_X_snrY_RxC/      (for custom R×C grid)
#     ├── signals.pkl            - EMG signals
#     ├── decomp.pkl             - Decomposition package
#     ├── emg_plot.png           - EMG visualization (optional)
#     ├── muap_spatial_patterns.png - MUAP spatial patterns (optional)
#     ├── muap_temporal.png      - MUAP temporal waveforms (optional)
#     └── trapezoid_drive.pkl    - Custom trapezoid (if generated)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -f "${SCRIPT_DIR}/../../.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/../../.venv/bin/activate"
fi

# Run the Python simulation script
python "${SCRIPT_DIR}/run_semg.py" "$@"

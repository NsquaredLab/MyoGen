#!/bin/bash
# Simulate Intramuscular EMG with Custom Parameters
# ==================================================
#
# This script provides an easy command-line interface for simulating
# intramuscular EMG with configurable motor unit selection and SNR.
#
# Usage Examples:
#   ./simulate_iemg.sh --mus "0,5,10,15,20" --snr 20
#   ./simulate_iemg.sh --mus all --snr 15
#   ./simulate_iemg.sh --mus "0-20" --snr 10
#   ./simulate_iemg.sh --mus "0-100-10" --snr 25
#
#   # Negative indices (use = syntax to avoid bash interpretation issues)
#   ./simulate_iemg.sh --mus="-1" --snr 20              # Last active MU (largest)
#   ./simulate_iemg.sh --mus="-5,-1" --snr 20           # 5th from last and last
#   ./simulate_iemg.sh --mus="-10--1" --snr 20          # Last 10 active MUs
#   ./simulate_iemg.sh --mus="-5,-4,-3,-2,-1" --snr 5   # Last 5 active MUs
#
#   ./simulate_iemg.sh --mus "0,5,10" --snr 20 --no-plot
#   ./simulate_iemg.sh --mus "0,5,10" --snr 20 --peak-hz 80 --plateau-time 8000
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
#   --snr     Signal-to-noise ratio in dB (default: 20)
#
#   --no-plot Skip plotting for faster execution
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
#   results/synthetic_gen/iemg_mu_X_snrY/
#     ├── signals.pkl         - EMG signals
#     ├── decomp.pkl          - Decomposition package
#     ├── emg_plot.png        - EMG visualization (optional)
#     ├── muap_templates.png  - MUAP templates (optional)
#     └── trapezoid_drive.pkl - Custom trapezoid (if generated)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -f "${SCRIPT_DIR}/../../.venv/bin/activate" ]; then
    source "${SCRIPT_DIR}/../../.venv/bin/activate"
fi

# Run the Python simulation script
python "${SCRIPT_DIR}/run_iemg.py" "$@"

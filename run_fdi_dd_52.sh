#!/usr/bin/env bash
# On calib/fdi-dd-main (github/main v0.13.0 + sEMG fixes). FDI descending-drive calibration, 120 MUs, recruitment ending at 0.52 excitation
# (De Luca et al. 1982: no new FDI units above ~52 % MVC). Approved by Devon 2026-11-01.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$(cd "$here" && pwd -W);c:\\Neuron\\lib\\python"
cd "$here/examples/finetune"
"$here/../../../.venv/Scripts/python.exe" 01_optimize_dd_multilevel.py --target-muscle fdi --n-mu 120 \
  --n-trials 100 --parallel --n-workers 6 --fresh --threshold-mode fuglevand --recruitment-range 50 \
  --max-recruitment 0.52 --target-model de_luca \
  --de-luca-params '{"fr_onset_hz": 8.9, "fr_peak_early_hz": 69.0, "fr_peak_late_hz": 44.0, "steepness": 1.55}' \
  --results-tag fuglevand50_rec52_deluca82_main

#!/usr/bin/env bash
# Run the Roeleveld arms in parallel: bash run_arms.sh <outdir>   (PY selects the interpreter)
set -u
OUT=${1:?outdir}; PY=${PY:-python}; mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
G=$(dirname "$0")/gen.py
run() { [ -s "$OUT/$1.npz" ] || $PY "$G" "$OUT/$1.npz" "$2" > "$OUT/$1.log" 2>&1; echo "$1 exit $?"; }
run III      '{"skin_sigma": 1.0, "rhos_mm": [0]}' &
run IIb      '{"skin_sigma": 0.05, "rhos_mm": [0]}' &
run s1       '{"skin_sigma": 1.0, "rhos_mm": [2.5, 5, 7.5, 12.5]}' &
run s0.15    '{"skin_sigma": 0.15}' &
run s4.88e-4 '{"skin_sigma": 0.000488}' &
wait

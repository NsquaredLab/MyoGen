#!/usr/bin/env bash
# Fall-off arms for replicate-style end-plate scatter: one process per skin
# conductivity and territory radius. bash run_scatter_arms.sh <outdir>
set -u
OUT=${1:?outdir}; PY=${PY:-python}; mkdir -p "$OUT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
G=$(dirname "$0")/gen.py
GRID='"z_rows_mm": [12, 36], "arcs_mm": [0, 12, 24, 36], "point_fibres": 1'
run() { [ -s "$OUT/$1.npz" ] || $PY "$G" "$OUT/$1.npz" "$2" > "$OUT/$1.log" 2>&1; echo "$1 exit $?"; }
for s in 1.0 0.15 0.000488; do
  for rho in 0 2.5 5 7.5 12.5; do
    run "sc_s${s}_r${rho}" "{\"skin_sigma\": $s, \"rhos_mm\": [$rho], $GRID}" &
  done
done
wait

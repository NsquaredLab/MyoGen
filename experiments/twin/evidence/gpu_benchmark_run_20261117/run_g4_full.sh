#!/bin/bash
# Full G4 Stage A/B on CPU at the checked-out commit (optional rerun; smoke matched CPU vs GPU).
set -u
cd ~/code/MyoGen-jaxley-fig2/experiments/twin
D=results/gpu_benchmark_run/checks/cpu_full
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
ls evidence > $D/.before
echo "=== g4_full $(date)"
../../.venv/bin/python g4_descent.py --iters 10 > $D/g4_full.log 2>&1; echo "g4_full exit $?"
for f in $(ls evidence | grep -vxFf $D/.before); do mv "evidence/$f" "$D/g4_full__$f"; done
echo "=== DONE $(date)"

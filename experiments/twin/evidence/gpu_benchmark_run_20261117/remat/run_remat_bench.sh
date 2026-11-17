#!/bin/bash
# Checkpointed GPU gradient benchmark; starts after the G4 rerun (PID 769587) exits,
# so nothing else runs on the machine.
set -u
cd ~/code/MyoGen-jaxley-remat/experiments/twin
while kill -0 769587 2>/dev/null; do sleep 30; done
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== remat bench $(date) $(git rev-parse HEAD)"
nvidia-smi > results/gpu_benchmark_remat_nvidia_smi_start.txt
~/code/MyoGen-jaxley-fig2/.venv/bin/python gpu_benchmark.py --devices gpu \
  --sweeps batch pool stress --pool-sizes 10 100 1000 --ops surrogate_value_and_grad \
  --remat-segment 140 --timeout-s 3600 --out-dir results/gpu_benchmark_remat \
  > results/gpu_benchmark_remat.log 2>&1
echo "remat bench exit $?"
echo "=== DONE $(date)"

#!/bin/bash
# Checkpointed GPU gradient at the pool sizes the first checkpointed run skipped.
set -u
cd ~/code/MyoGen-jaxley-remat/experiments/twin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "=== remat pool2 $(date) $(git rev-parse HEAD)"
nvidia-smi > results/gpu_benchmark_remat_pool2_nvidia_smi_start.txt
~/code/MyoGen-jaxley-fig2/.venv/bin/python gpu_benchmark.py --devices gpu --sweeps pool \
  --pool-sizes 1 50 250 500 --ops surrogate_value_and_grad --remat-segment 140 \
  --timeout-s 3600 --out-dir results/gpu_benchmark_remat_pool2 > results/gpu_benchmark_remat_pool2.log 2>&1
echo "exit $?"
echo "=== DONE $(date)"

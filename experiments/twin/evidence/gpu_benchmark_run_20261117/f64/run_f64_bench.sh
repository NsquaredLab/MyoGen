#!/bin/bash
# Float64 (mixed) checkpointed gradient timings, GPU then CPU, nothing else on the machine.
# Checkpointing on both devices: the plain float64 CPU gradient at K 1000 would need ~170 GB RAM.
set -u
cd ~/code/MyoGen-jaxley-remat/experiments/twin
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/code/MyoGen-jaxley-fig2/.venv/bin/python
echo "=== f64 bench $(date) $(git rev-parse HEAD)"
nvidia-smi > results/gpu_benchmark_f64_nvidia_smi_start.txt
$PY gpu_benchmark.py --devices gpu --sweeps pool batch stress --ops surrogate_value_and_grad \
  --precision float64 --remat-segment 140 --timeout-s 3600 --out-dir results/gpu_benchmark_f64_gpu \
  > results/gpu_benchmark_f64_gpu.log 2>&1
echo "gpu exit $? $(date)"
$PY gpu_benchmark.py --devices cpu --sweeps pool batch --ops surrogate_value_and_grad \
  --precision float64 --remat-segment 140 --timeout-s 3600 --out-dir results/gpu_benchmark_f64_cpu \
  > results/gpu_benchmark_f64_cpu.log 2>&1
echo "cpu exit $? $(date)"
nvidia-smi > results/gpu_benchmark_f64_nvidia_smi_end.txt
echo "=== DONE $(date)"

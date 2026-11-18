#!/bin/bash
# CPU at 2000 and 5000 MN: float32 forward, then float64 (mixed) checkpointed gradient. Nothing else on the machine.
# Timeout 12 h per configuration: the 5000 MN gradient is extrapolated to ~50 min per call x 10 calls plus a ~30 min build.
set -u
cd ~/code/MyoGen-jaxley-remat/experiments/twin
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
PY=~/code/MyoGen-jaxley-fig2/.venv/bin/python
echo "=== cpu stress $(date) $(git rev-parse HEAD)"
{ uptime; free -g; } > results/cpu_stress_state_start.txt
$PY gpu_benchmark.py --devices cpu --sweeps stress --ops hard_forward \
  --timeout-s 43200 --out-dir results/gpu_benchmark_cpu_stress_fwd > results/gpu_benchmark_cpu_stress_fwd.log 2>&1
echo "fwd exit $? $(date)"
$PY gpu_benchmark.py --devices cpu --sweeps stress --ops surrogate_value_and_grad \
  --precision float64 --remat-segment 140 --timeout-s 43200 \
  --out-dir results/gpu_benchmark_cpu_stress_f64 > results/gpu_benchmark_cpu_stress_f64.log 2>&1
echo "grad exit $? $(date)"
{ uptime; free -g; } > results/cpu_stress_state_end.txt
echo "=== DONE $(date)"

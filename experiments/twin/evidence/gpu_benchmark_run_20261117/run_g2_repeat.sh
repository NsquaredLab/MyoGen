#!/bin/bash
# Repeat float32 G2 at the same commit to separate device dependence from run-to-run noise.
set -u
cd ~/code/MyoGen-jaxley-fig2/experiments/twin
D=results/gpu_benchmark_run/checks/repeat
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for spec in gpu:1 gpu:2 cpu:1; do
  dev=${spec%%:*}; n=${spec##*:}
  if [ $dev = cpu ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  ls evidence | grep "^gradient_repair_f32_" > $D/.before
  echo "=== $dev g2_f32 rep$n $(date)"
  ../../.venv/bin/python gradient_repair.py --tests g2 > $D/${dev}_g2_f32_rep$n.log 2>&1; echo "exit $?"
  for f in $(ls evidence | grep "^gradient_repair_f32_" | grep -vxFf $D/.before); do mv "evidence/$f" "$D/${dev}_g2_f32_rep${n}__$f"; done
done
echo "=== DONE $(date)"

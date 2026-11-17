#!/bin/bash
# Benchmark + reproducibility checks at the checked-out commit. Stages run in order;
# the benchmarks run alone on the machine.
set -u
cd ~/code/MyoGen-jaxley-fig2/experiments/twin
P=../../.venv/bin/python
OUT=results/gpu_benchmark_run
mkdir -p $OUT/checks/cpu $OUT/checks/gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
git rev-parse HEAD > $OUT/commit.txt; nvidia-smi > $OUT/nvidia_smi_start.txt
echo "=== bench $(date)"
$P gpu_benchmark.py --timeout-s 3600 --out-dir $OUT/bench > $OUT/bench.log 2>&1; echo "bench exit $?"
if $P -c "import json,sys; r=[x for x in json.load(open(\"$OUT/bench/benchmark.json\"))[\"rows\"] if x[\"sweep\"]==\"pool\" and x[\"device\"]==\"gpu\"]; sys.exit(0 if r and all(x[\"status\"]==\"ok\" for x in r) else 1)"; then
  echo "=== stress $(date)"
  $P gpu_benchmark.py --sweeps stress --devices gpu --timeout-s 3600 --out-dir $OUT/stress > $OUT/stress.log 2>&1; echo "stress exit $?"
else
  echo "=== stress skipped: GPU pool sweep not fully ok"
fi
for dev in cpu gpu; do
  if [ $dev = cpu ]; then export JAX_PLATFORMS=cpu; else unset JAX_PLATFORMS; fi
  D=$OUT/checks/$dev
  run() {
    name=$1; shift
    echo "=== $dev $name $(date)"
    ls evidence > $D/.before
    $P "$@" > $D/$name.log 2>&1; echo "$dev $name exit $?"
    for f in $(ls evidence | grep -vxFf $D/.before); do mv "evidence/$f" "$D/${name}__$f"; done
  }
  run g2_f32 gradient_repair.py --tests g2
  run g2_x64 gradient_repair.py --tests g2 --x64
  run g1_x64_bconfig gradient_repair.py --tests g1 --x64 --regime b_config
  run g4_smoke g4_descent.py --smoke --iters 2
done
nvidia-smi > $OUT/nvidia_smi_end.txt
echo "=== ALL DONE $(date)"

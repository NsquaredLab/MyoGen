#!/usr/bin/env bash
# Simulate sweep arms of one scenario, three at a time, skipping finished ones.
# Usage: bash run_arms.sh <limb|fdi> <outdir> [arm ...]   (all arms of the scenario if none given)
# PY selects the interpreter (default: python).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
scenario=$1; out=$2; shift 2
PY=${PY:-python}
mkdir -p "$out"
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
  mapfile -t arms < <($PY -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))[sys.argv[2]]))" \
    "$here/arms.json" "$scenario" | tr -d '\r')
fi
run_one() {
  local a=$1
  [ -f "$out/arm_$a.npz" ] && { echo "$a exists"; return; }
  cfg=$($PY -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]))" \
    "$here/arms.json" "$scenario" "$a")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 \
    $PY "$here/gen_arm.py" "$scenario" "$a" "$out/arm_$a.npz" "$cfg" 6 > "$out/log_$a.txt" 2>&1 \
    || { echo "$a FAILED (see $out/log_$a.txt)"; return; }
  tail -1 "$out/log_$a.txt"
}
export -f run_one; export out scenario PY here
printf '%s\n' "${arms[@]}" | xargs -P 3 -I{} bash -c 'run_one "$@"' _ {}

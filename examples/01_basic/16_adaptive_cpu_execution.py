"""Public CPU execution, repeated recordings, and explicit cache restart.

Inputs are trusted local files produced by the existing public examples.
Example: --modality iemg --muscle results/muscle_model.pkl --spikes recordings.pkl
Use --existing to run without an execution policy.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import joblib
import quantities as pq

import myogen
from myogen import simulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=["iemg", "semg"], required=True)
    parser.add_argument("--muscle", type=Path, required=True)
    parser.add_argument("--spikes", type=Path, nargs="*", default=[])
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--all-mus", action="store_true", help="Surface only: evaluate every configured MU")
    parser.add_argument("--memory-mib", type=float)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--rng-state", type=Path,
                        help="Restore pre-MUAP JSON RNG state if present; otherwise save it before simulation")
    args = parser.parse_args()
    muscle = joblib.load(args.muscle)
    if args.modality == "iemg":
        electrode = simulator.IntramuscularElectrodeArray(
            num_electrodes=4, inter_electrode_distance__mm=2 * pq.mm,
            position__mm=(0 * pq.mm, 0 * pq.mm, 15 * pq.mm), differentiation_mode="consecutive")
        emg = simulator.IntramuscularEMG(muscle, electrode)
        assemble = emg.simulate_intramuscular_emg
    else:
        electrode = simulator.SurfaceElectrodeArray(
            num_rows=5, num_cols=5, inter_electrode_distance__mm=5 * pq.mm,
            electrode_radius__mm=5 * pq.mm, differentiation_mode="monopolar",
            bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm)
        emg = simulator.SurfaceEMG(muscle, [electrode],
                                   MUs_to_simulate=None if args.all_mus else [0, 1, 2, 3])
        assemble = emg.simulate_surface_emg
    if args.rng_state:
        if args.rng_state.exists():
            saved = json.loads(args.rng_state.read_text())
            # iEMG also derives NMJ clustering sub-seeds from the public seed.
            # A generator state alone is insufficient after a nondefault seed.
            myogen.set_random_seed(saved["seed"])
            myogen.get_random_generator().bit_generator.state = saved["generator_state"]
        else:
            args.rng_state.parent.mkdir(parents=True, exist_ok=True)
            args.rng_state.write_text(json.dumps(dict(
                seed=myogen.get_random_seed(),
                generator_state=deepcopy(myogen.get_random_generator().bit_generator.state)),
                indent=2), encoding="utf-8")
    options = {} if args.existing else {"execution": simulator.ExecutionPolicy(
        memory_budget_mib=args.memory_mib, cache_dir=args.cache_dir)}
    emg.simulate_muaps(n_jobs=2, **options)
    if not args.existing:
        print(dict(emg.execution_report))
    for path in args.spikes:
        recording = assemble(joblib.load(path), **options)
        print(path, recording)
        if not args.existing:
            print(dict(emg.execution_report))


if __name__ == "__main__":
    main()

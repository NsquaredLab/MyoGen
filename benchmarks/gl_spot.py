"""450 actual fibres across all public MUs; fresh corrected GitLab references."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from gl_common import ROOT, OUT, BASE, setup, dump_json, sha


def metrics(ref, actual):
    import numpy as np
    assert ref.shape == actual.shape and ref.dtype == actual.dtype
    assert np.all(np.isfinite(ref)) and np.all(np.isfinite(actual))
    peak = np.max(np.abs(ref),axis=-1)
    delta = np.max(np.abs(ref-actual),axis=-1)
    gate = 1e-12+1e-9*peak
    assert np.all(delta <= gate), float(np.max(delta/gate))
    return dict(max_abs=float(delta.max()), max_gate_fraction=float((delta/gate).max()),
                relative_l2=float(np.linalg.norm(ref-actual)/max(np.linalg.norm(ref),1e-300)),
                peak_sample_mismatches=int(np.count_nonzero(np.argmax(np.abs(ref),axis=-1)!=np.argmax(np.abs(actual),axis=-1))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    args=parser.parse_args()
    environment=setup()
    import joblib
    import numpy as np
    from scipy.signal import resample
    from threadpoolctl import threadpool_limits
    import myogen
    from myogen.simulator.core.emg.surface.simulate_fiber import _simulate_fiber_v2_python as optimized
    from myogen.simulator.core.emg.runtime.surface_preparation import SurfacePreparation
    from gl_reference import surface_reference
    original=surface_reference()
    inputs=joblib.load(args.inputs/"sp/inputs.joblib")
    plan=json.loads((args.inputs/"sp/plan.json").read_text())
    assert sha(args.inputs/"sp/inputs.joblib")==plan["inputs_sha256"]
    selected=plan["sample"]
    assert len(selected)==450 and len(inputs["rows"])==58755
    folder=OUT/"spot"
    folder.mkdir(exist_ok=False)
    report=dict(base=BASE, environment=environment, input_hash=sha(args.inputs/"sp/inputs.joblib"),
                muscle_hash=sha(args.inputs/"muscle.joblib"), sample=selected,
                sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"myogen/simulator/core/emg").rglob("*.py")},
                harness_sha256=sha(Path(__file__)), status="running", results=[])
    started=time.monotonic()
    before=deepcopy(myogen.get_random_generator().bit_generator.state)
    A=B=None
    prep=SurfacePreparation()
    with threadpool_limits(limits=1):
        for number,index in enumerate(selected):
            if time.monotonic()-started > 3600:
                raise TimeoutError("One-hour bounded spot-check budget reached")
            kwargs=dict(inputs["common"])
            kwargs.update(zip(("R","L1","L2","zi","theta_offset"),inputs["rows"][index]))
            begin=time.perf_counter()
            ref,A,B=original(**kwargs,A_matrix=A,B_incomplete=B)
            ref_seconds=time.perf_counter()-begin
            begin=time.perf_counter()
            actual,_,_=optimized(**kwargs,_prepared=prep)
            opt_seconds=time.perf_counter()-begin
            raw=metrics(ref,actual)
            final=metrics(resample(ref,inputs["samples"],axis=2),resample(actual,inputs["samples"],axis=2))
            np.savez_compressed(folder/f"{index}.npz",reference=ref,optimized=actual)
            report["results"].append(dict(index=index,raw=raw,resampled=final,
                reference_seconds=ref_seconds,optimized_seconds=opt_seconds))
            dump_json(folder/"result.json",report)
            if (number+1)%25==0:
                print(f"{number+1}/450; {time.monotonic()-started:.1f}s; worst gate fraction "+
                      str(max(item["raw"]["max_gate_fraction"] for item in report["results"])),flush=True)
    assert before == myogen.get_random_generator().bit_generator.state
    report.update(status="complete",rng_unchanged=True,elapsed=time.monotonic()-started,
                  maximum_backward_residual=prep.response.maximum_backward_residual)
    dump_json(folder/"result.json",report)

if __name__=="__main__":
    main()

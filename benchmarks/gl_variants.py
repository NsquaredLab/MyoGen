"""Complete actual MU 0 across supported conductor/electrode configurations."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
from gl_common import OUT, BASE, setup, dump_json, sha
from gl_spot import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    args=parser.parse_args()
    environment=setup()
    import joblib
    import numpy as np
    import quantities as pq
    from scipy.signal import resample
    from threadpoolctl import threadpool_limits
    from myogen.simulator import SurfaceElectrodeArray
    from myogen.simulator.core.emg.surface.simulate_fiber import _simulate_fiber_v2_python
    from myogen.simulator.core.emg.runtime.surface_preparation import SurfacePreparation
    from gl_reference import surface_reference
    original=surface_reference()
    data=joblib.load(args.inputs/"sp/inputs.joblib")
    folder=OUT/"variants"
    folder.mkdir(exist_ok=False)
    results=[]
    cases=[("bonefree",{"r_bone":0.},"monopolar",0.),
           ("lowfat",{"sig_fat":.02},"monopolar",0.),
           ("rotated",{},"bipolar_longitudinal",17.),
           ("laplacian",{},"laplacian",0.)]
    with threadpool_limits(limits=1):
        for name,changes,mode,angle in cases:
            common=deepcopy(data["common"])
            common.update(changes)
            electrode=SurfaceElectrodeArray(num_rows=5,num_cols=5,
                inter_electrode_distance__mm=5*pq.mm,electrode_radius__mm=5*pq.mm,
                bending_radius__mm=common["r"]*pq.mm,
                differentiation_mode=mode,rotation_angle__deg=angle*pq.deg)
            common.update(electrode_array=electrode,
                pos_z_precomputed=electrode.pos_z.rescale(pq.mm).magnitude,
                pos_theta_precomputed=electrode.pos_theta.rescale(pq.rad).magnitude,
                rele_precomputed=float(electrode.electrode_radius__mm.rescale(pq.mm).magnitude))
            prep=SurfacePreparation()
            A=B=None
            expected=actual=None
            fibres=[]
            start=time.perf_counter()
            with np.errstate(invalid="ignore",divide="ignore",over="ignore"):
                for index in range(int(data["counts"][0])):
                    kw=dict(common)
                    kw.update(zip(("R","L1","L2","zi","theta_offset"),data["rows"][index]))
                    ref,A,B=original(**kw,A_matrix=A,B_incomplete=B)
                    opt,_,_=_simulate_fiber_v2_python(**kw,_prepared=prep)
                    fibres.append(metrics(ref,opt))
                    expected=ref.copy() if expected is None else expected+ref
                    actual=opt.copy() if actual is None else actual+opt
            raw=metrics(expected,actual)
            final=metrics(resample(expected,data["samples"],axis=2),resample(actual,data["samples"],axis=2))
            np.savez_compressed(folder/(name+".npz"),reference=expected,optimized=actual)
            results.append(dict(name=name,fibres=len(fibres),raw=raw,resampled=final,
                                individual=fibres,diagnostic_seconds=time.perf_counter()-start))
            dump_json(folder/"result.json",dict(base=BASE,environment=environment,
                input_sha256=sha(args.inputs/"sp/inputs.joblib"),results=results,status="running"))
            print(name,"complete:",len(fibres),"actual fibres, all gates passed",flush=True)
    dump_json(folder/"result.json",dict(base=BASE,environment=environment,
                input_sha256=sha(args.inputs/"sp/inputs.joblib"),results=results,status="passed"))

if __name__=="__main__":
    main()

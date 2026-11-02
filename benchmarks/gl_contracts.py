"""Realized-input contract checks and real-MU allocation/cache recovery."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import joblib
from gl_common import ROOT, OUT, setup, restore, owner as build_owner, dump_json, sha
setup()
import myogen
from myogen import simulator
from myogen.simulator.core.emg.runtime import semg
from myogen.simulator.core.emg.runtime.cache import fingerprint
from myogen.simulator.core.emg.surface import surface_emg
from gl_compare import compare
from threadpoolctl import threadpool_limits
VARIABLES=("R","L1","L2","zi","theta_offset")
OMIT=("A_matrix","B_incomplete","_prepared")
RunFailure=RuntimeError
INPUTS=None

def restored_rng():
    restore(INPUTS)

def owner(muscle, selected=None):
    return build_owner(muscle,"s",selected)

def input_id(kwargs):
    values = {k: v for k, v in kwargs.items() if k not in (*OMIT, "electrode_array")}
    return fingerprint(values, vars(kwargs["electrode_array"]))

def capture(muscle, optimized):
    """Input/metadata contract only: no numerical signal evidence is produced."""
    restored_rng()
    obj = owner(muscle)
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers, dtype=int)
    rows, identities, common, finished = [], [], {}, []
    module = semg if optimized else surface_emg
    original, sampling = module._simulate_fiber_v2_python, module.resample
    dummy = np.zeros((5, 5, int(obj._internal_sampling_points)))
    common_key = [None]
    def observe(**kwargs):
        fixed = {k: v for k, v in kwargs.items() if k not in (*OMIT, *VARIABLES)}
        fixed_key = input_id(fixed)
        if common_key[0] is None:
            common_key[0] = fixed_key
            common.update(deepcopy(fixed))
        if fixed_key != common_key[0]:
            raise RunFailure("Unexpected varying input; extend capture before sampling")
        rows.append([kwargs[k] for k in VARIABLES])
        identities.append(input_id(kwargs))
        return dummy, None, None
    def boundary(value, *args, **kwargs):
        mu = len(finished)
        if len(rows) != int(counts[:mu+1].sum()):
            raise RunFailure("Capture fibre ordering/count mismatch")
        finished.append(mu)
        return sampling(value, *args, **kwargs)
    module._simulate_fiber_v2_python, module.resample = observe, boundary
    try:
        options = {"execution": simulator.ExecutionPolicy()} if optimized else {}
        template = obj.simulate_muaps(n_jobs=1, use_gpu=False, verbose=False, **options)
    finally:
        module._simulate_fiber_v2_python, module.resample = original, sampling
    assert finished == list(range(100)) and len(rows) == 58755
    assert all("FAILED" not in s.name for s in template.groups[0].segments)
    return dict(rows=np.asarray(rows), ids=np.asarray(identities), common=common,
                counts=counts, template=template, samples=obj._sampling_points_in_t_and_z_domains,
                rng=deepcopy(myogen.get_random_generator().bit_generator.state))

def main():
    global INPUTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    INPUTS=parser.parse_args().inputs
    folder=OUT/"contracts"
    folder.mkdir(exist_ok=False)
    muscle=joblib.load(INPUTS/"muscle.joblib")
    report=dict(scope="Zero-return capture establishes input/metadata contracts only; actual waveforms checked separately.",
                input_sha256=sha(INPUTS/"muscle.joblib"))
    with threadpool_limits(limits=1):
        ref,opt=capture(muscle,False),capture(muscle,True)
        saved=joblib.load(INPUTS/"sp/inputs.joblib")
        for key in ("rows","ids","counts"):
            np.testing.assert_array_equal(ref[key],opt[key])
            np.testing.assert_array_equal(ref[key],saved[key])
        assert ref["rng"] == opt["rng"] == saved["rng"]
        report["surface_input_contract"]=dict(identities=58755,mu_order=100,rng="exact",
                                               metadata=compare(ref["template"],opt["template"]))
        dump_json(folder/"result.json",report)
        print("All 58755 surface input identities and RNG match.",flush=True)

        # iEMG raw inputs are compared before evaluating selected real MUs.
        restored_rng()
        reference=build_owner(muscle,"i",[0,25,50])
        reference._initialize_motor_units(verbose=False)
        reference._simulate_neuromuscular_junctions(verbose=False)
        expected_rng=deepcopy(myogen.get_random_generator().bit_generator.state)
        restored_rng()
        candidate=build_owner(muscle,"i",[0,25,50])
        actual=candidate.simulate_muaps(n_jobs=1,verbose=False,execution=simulator.ExecutionPolicy())
        for left,right in zip(reference._motor_units,candidate._motor_units):
            for key,value in vars(left).items():
                if isinstance(value,np.ndarray):
                    np.testing.assert_array_equal(value,vars(right)[key])
        assert expected_rng == myogen.get_random_generator().bit_generator.state
        report["iemg_input_contract"]=dict(mus=100,fibres=58755,rng="exact",
                 waveform=compare(joblib.load(OUT/"ir/out0.joblib"),actual))
        print("All iEMG realized arrays and RNG match.",flush=True)

        # Inject failure within a four-fibre batch; partial contributions must not commit.
        restored_rng()
        model=owner(muscle,[0,1,2,3])
        original=semg._simulate_fiber_v2_python
        calls=[0]
        def fail_once(**kwargs):
            calls[0]+=1
            if calls[0]==3:
                raise MemoryError("injected third-fibre allocation failure")
            return original(**kwargs)
        semg._simulate_fiber_v2_python=fail_once
        try:
            actual=model.simulate_muaps(n_jobs=1,verbose=False,use_gpu=False,
                                        execution=simulator.ExecutionPolicy())
        finally:
            semg._simulate_fiber_v2_python=original
        report["retry"]=dict(comparison=compare(joblib.load(OUT/"sr/out0.joblib"),actual),
                             execution=model.execution_report)
        assert model.execution_report["allocation_retries"]==1
        assert model.execution_report["completed_work"]==88
        print("Third-fibre retry passed full-MU waveform gate.",flush=True)

        # Persist all four completed MUs, then hit them on a restored realization.
        policy=simulator.ExecutionPolicy(cache_dir=folder/"cache")
        restored_rng()
        obj=owner(muscle,[0,1,2,3])
        first=obj.simulate_muaps(n_jobs=1,verbose=False,use_gpu=False,execution=policy)
        assert obj.execution_report["cache_hits"]==0
        after_first=deepcopy(myogen.get_random_generator().bit_generator.state)
        restored_rng()
        resumed=owner(muscle,[0,1,2,3])
        second=resumed.simulate_muaps(n_jobs=1,verbose=False,use_gpu=False,execution=policy)
        assert resumed.execution_report["cache_hits"]==4
        assert myogen.get_random_generator().bit_generator.state==after_first
        report["cache_resume"]=compare(first,second)
        # Do not restore RNG: this must create a new realization and miss old entries.
        third=resumed.simulate_muaps(n_jobs=1,verbose=False,use_gpu=False,execution=policy)
        assert resumed.execution_report["cache_hits"]==0
        assert myogen.get_random_generator().bit_generator.state != after_first
        assert any(not np.array_equal(a.analogsignals[0].magnitude,b.analogsignals[0].magnitude)
                   for a,b in zip(second.groups[0].segments,third.groups[0].segments))
        report.update(cache_hits_after_restore=4,new_realization_hits=0,status="passed")
        dump_json(folder/"result.json",report)
        print("Persistent-cache reuse and fresh-call RNG passed.",flush=True)

if __name__=="__main__":
    main()

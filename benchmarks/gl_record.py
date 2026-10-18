"""Recording synthesis from fresh corrected templates and saved real spike trains."""
import argparse
from copy import deepcopy
import time
from pathlib import Path
import threading
from gl_common import OUT, setup, restore, owner, dump_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    args=parser.parse_args()
    environment=setup()
    import joblib
    import numpy as np
    import quantities as pq
    import psutil
    from neo import Block, Segment, SpikeTrain
    from threadpoolctl import threadpool_limits
    from myogen import simulator
    from myogen.utils.binning import bin_spike_trains
    from myogen.simulator.core.emg.runtime.assembly import occupancy
    from myogen.simulator.core.emg.runtime.session import ExecutionSession
    from gl_compare import compare
    folder=OUT/"record"
    folder.mkdir(exist_ok=False)
    muscle=joblib.load(args.inputs/"muscle.joblib")
    spikes=joblib.load(args.inputs/"fr/o0/spikes.joblib")
    long=Block()
    for pool in spikes.segments:
        segment=Segment(name=pool.name)
        for st in pool.spiketrains:
            duration=st.t_stop-st.t_start
            times=np.concatenate([(st+offset*duration).rescale(pq.s).magnitude for offset in range(15)])
            train=SpikeTrain(times*pq.s,t_start=st.t_start,t_stop=st.t_stop+14*duration)
            train.sampling_period=st.sampling_period
            segment.spiketrains.append(train)
        long.segments.append(segment)
    report=dict(environment=environment,spikes_sha256=sha(args.inputs/"fr/o0/spikes.joblib"),
                long_note="60-second repetition of real spike schedules; numerical duration stress, not new physiology.",
                results=[])
    def timed(call):
        peak=[psutil.Process().memory_info().rss]
        stop=threading.Event()
        def monitor():
            while not stop.wait(.05):
                peak[0]=max(peak[0],psutil.Process().memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True)
        thread.start()
        try:
            start=time.perf_counter()
            value=call()
            seconds=time.perf_counter()-start
        finally:
            stop.set()
            thread.join()
        return value,dict(seconds=seconds,peak_sampled_rss=peak[0])
    with threadpool_limits(limits=1):
        # Verify all occupied bins, including repeated events collapsing to True.
        for trains in (spikes,long):
            obj=owner(muscle,"i",[0,25,50])
            with ExecutionSession(obj,simulator.ExecutionPolicy(),"bin_contract") as session:
                actual,dt=occupancy(trains,session)
                for index,pool in enumerate(trains.segments):
                    np.testing.assert_array_equal(actual[index],bin_spike_trains(pool.spiketrains,
                                                        bin_size=pool.spiketrains[0].sampling_period))
        report["spike_bins"]="exact, both durations and all 100 neurons in both pools"
        for kind,label,selected in (("i","ir",[0,25,50]),("s","sr",[0,1,2,3])):
            for duration,trains in ((4,spikes),(60,long)):
                restore(args.inputs)
                model=owner(muscle,kind,selected)
                model._muaps__Block=joblib.load(OUT/label/"out0.joblib")
                method=model.simulate_intramuscular_emg if kind=="i" else model.simulate_surface_emg
                ref,rt=timed(lambda:method(trains,verbose=False))
                opt,ot=timed(lambda:method(trains,verbose=False,execution=simulator.ExecutionPolicy()))
                parity=compare(ref,opt,destination=folder/f"{kind}{duration}")
                assert parity["timing_mismatches"] == parity["exact_reference_peak_ties"], "Non-tied peak changed"
                item=dict(kind=kind,duration=duration,selected=selected,reference=rt,optimized=ot,
                          comparison=parity,execution=model.execution_report)
                joblib.dump(ref,folder/f"{kind}{duration}-ref.joblib")
                joblib.dump(opt,folder/f"{kind}{duration}-opt.joblib")
                if duration==4:
                    repetitions=[]
                    for _ in range(10):
                        candidate,measure=timed(lambda:method(trains,verbose=False,execution=simulator.ExecutionPolicy()))
                        compare(ref,candidate)
                        measure["cache_hits"]=model.execution_report["cache_hits"]
                        repetitions.append(measure)
                    item["repeated"]=repetitions
                report["results"].append(item)
                dump_json(folder/"result.json",report)
                print(kind,duration,"s; reference",rt["seconds"],"optimized",ot["seconds"],flush=True)
                del ref,opt,model
        report["status"]="passed"
        dump_json(folder/"result.json",report)

if __name__=="__main__":
    main()

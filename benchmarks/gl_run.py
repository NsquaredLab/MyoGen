"""Bounded CPU profiling and paired measurements against corrected GitLab main.

Inputs are the trusted saved complete public muscle and real neural recordings.
Selected source MUs retain every assigned fibre. Old sEMG waveforms are never
used as the corrected-source reference.
"""
import argparse
from copy import deepcopy
import cProfile
import json
from pathlib import Path
import pstats
import threading
import time

from gl_common import ROOT, OUT, BASE, setup, restore, owner, dump_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--kind", choices=["i", "s"], default="s")
    parser.add_argument("--mus", default="0,1,2,3")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--templates", type=Path, help="Assembly only; a saved template Block")
    parser.add_argument("--repeats", type=int, default=1, help="Recording calls per process")
    args = parser.parse_args()
    if not args.label.isalnum() or args.repeats < 1:
        raise ValueError("Use alphanumeric labels and positive repeats")
    environment = setup()
    import joblib
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits, threadpool_info
    import myogen
    from myogen import simulator
    from gl_compare import compare
    folder = OUT / args.label
    folder.mkdir(parents=True, exist_ok=False)
    selected = [int(v) for v in args.mus.split(",")] if args.mus else None
    muscle = joblib.load(args.inputs/"muscle.joblib")
    counts = muscle.resulting_number_of_innervated_fibers
    assert len(counts) == 100 and sum(counts) == 58755
    restore(args.inputs)
    model = owner(muscle, args.kind, selected)
    options = dict(verbose=False)
    if args.optimized:
        options["execution"] = simulator.ExecutionPolicy()
    if args.templates:
        model._muaps__Block = joblib.load(args.templates)
        spikes = joblib.load(args.inputs/"fr/o0/spikes.joblib")
        fn = model.simulate_surface_emg if args.kind == "s" else model.simulate_intramuscular_emg
        call = lambda: fn(spikes, **options)
    else:
        options["n_jobs"] = 1
        if args.kind == "s":
            options["use_gpu"] = False
        call = lambda: model.simulate_muaps(**options)

    result = dict(base=BASE, optimized=args.optimized, profiled=args.profile,
        stage=args.kind+("-assembly" if args.templates else "-source"),
        selected=selected, selected_fibres=int(sum(counts if selected is None else counts[selected])),
        population=100, fibres=58755, environment=environment,
        fixture_sha256=sha(args.inputs/"muscle.joblib"), before_rng=deepcopy(myogen.get_random_generator().bit_generator.state),
        source_sha256={str(p.relative_to(ROOT)): sha(p)
                       for p in (ROOT/"myogen/simulator/core/emg").rglob("*.py")},
        status="running", calls=[])
    dump_json(folder/"result.json", result)
    peak = [0]
    stop = threading.Event()
    def monitor():
        process = psutil.Process()
        while not stop.wait(.2):
            peak[0] = max(peak[0], process.memory_info().rss)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with threadpool_limits(limits=1):
            result["native_pools"] = threadpool_info()
            for repetition in range(args.repeats):
                profile = cProfile.Profile() if args.profile else None
                started = time.perf_counter()
                block = profile.runcall(call) if profile else call()
                elapsed = time.perf_counter()-started
                entry = dict(seconds=elapsed, rng=deepcopy(myogen.get_random_generator().bit_generator.state))
                if args.optimized:
                    entry["execution_report"] = model.execution_report
                joblib.dump(block, folder/f"out{repetition}.joblib")
                if args.reference:
                    reference_folder = OUT / args.reference
                    expected = joblib.load(reference_folder/"out0.joblib")
                    entry["comparison"] = compare(expected, block, destination=folder/f"comparison{repetition}")
                    old = json.loads((reference_folder/"result.json").read_text())
                    assert old["calls"][0]["rng"] == entry["rng"], "final RNG differs"
                    assert old["fixture_sha256"] == result["fixture_sha256"]
                if profile:
                    profile.dump_stats(folder/"profile.prof")
                    with (folder/"profile.txt").open("w", encoding="utf-8") as stream:
                        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(40)
                        pstats.Stats(profile, stream=stream).sort_stats("tottime").print_stats(30)
                result["calls"].append(entry)
                print(args.label, result["stage"], round(elapsed, 3), "s", flush=True)
                dump_json(folder/"result.json", result)
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        thread.join()
        result["peak_sampled_rss"] = peak[0]
        dump_json(folder/"result.json", result)


if __name__ == "__main__":
    main()

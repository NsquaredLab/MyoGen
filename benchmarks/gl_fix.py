"""Bounded post-review checks against saved corrected-GitLab references.

Reuse the complete public input population, evaluating only the previously
approved complete MUs. No reference source generation or population benchmark.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from gl_common import OUT, ROOT, setup, restore, owner, dump_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--label", default="fix")
    args = parser.parse_args()
    if not args.label.isalnum():
        raise ValueError("Use a short alphanumeric artifact label")
    environment = setup()
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits
    import myogen
    from myogen.simulator import ExecutionPolicy
    from myogen.utils.binning import bin_spike_trains
    from myogen.simulator.core.emg.runtime.assembly import occupancy
    from myogen.simulator.core.emg.runtime.session import ExecutionSession
    from gl_compare import compare, segments

    folder = OUT / args.label
    folder.mkdir(exist_ok=False)
    muscle = joblib.load(args.inputs / "muscle.joblib")
    counts = muscle.resulting_number_of_innervated_fibers
    assert len(counts) == 100 and sum(counts) == 58755
    spikes = joblib.load(args.inputs / "fr/o0/spikes.joblib")
    report = dict(status="running", environment=environment,
        fixture_sha256=sha(args.inputs / "muscle.joblib"),
        spikes_sha256=sha(args.inputs / "fr/o0/spikes.joblib"),
        source_sha256={str(p.relative_to(ROOT)): sha(p)
            for p in (ROOT / "myogen/simulator/core/emg/runtime").glob("*.py")}, results=[])
    dump_json(folder / "result.json", report)
    try:
        with threadpool_limits(limits=1):
            for kind, label, selected in (("i", "ir", [0, 25, 50]), ("s", "sr", [0, 1, 2, 3])):
                baseline = json.loads((OUT / label / "result.json").read_text())
                assert baseline["fixture_sha256"] == report["fixture_sha256"]
                reference = joblib.load(OUT / label / "out0.joblib")
                restore(args.inputs)
                model = owner(muscle, kind, selected)
                policy = ExecutionPolicy(cache_dir=folder / kind)
                options = dict(n_jobs=2, verbose=False, execution=policy)
                if kind == "s":
                    options["use_gpu"] = False
                actual = model.simulate_muaps(**options)
                after = deepcopy(myogen.get_random_generator().bit_generator.state)
                assert after == baseline["calls"][0]["rng"]
                source = compare(reference, actual, destination=folder / (kind + "-source"))
                item = dict(kind=kind, selected=selected, fibres=int(sum(counts[selected])),
                    source=source, source_execution=model.execution_report, final_rng="exact")
                restore(args.inputs)
                resumed = owner(muscle, kind, selected)
                cached = resumed.simulate_muaps(**options)
                assert resumed.execution_report["cache_hits"] == len(selected)
                assert myogen.get_random_generator().bit_generator.state == after
                for (_, a), (_, b) in zip(segments(actual), segments(cached)):
                    np.testing.assert_array_equal(a.analogsignals[0].magnitude, b.analogsignals[0].magnitude)
                item["cache_hits"] = len(selected)
                item["cache_values"] = "bitwise exact"
                item["cache_metadata"] = compare(actual, cached)
                with ExecutionSession(model, policy, "bin_contract", native_pools=False) as session:
                    bins, _ = occupancy(spikes, session)
                    for index, pool in enumerate(spikes.segments):
                        np.testing.assert_array_equal(bins[index], bin_spike_trains(
                            pool.spiketrains, bin_size=pool.spiketrains[0].sampling_period))
                item["spike_bins"] = "exact"
                method = resumed.simulate_intramuscular_emg if kind == "i" else resumed.simulate_surface_emg
                expected = joblib.load(OUT / "record" / (kind + "4-ref.joblib"))
                item["recordings"] = []
                for repetition in range(2):
                    signal = method(spikes, verbose=False, execution=policy)
                    result = compare(expected, signal, destination=folder / f"{kind}-record{repetition}")
                    assert result["timing_mismatches"] == result["exact_reference_peak_ties"]
                    item["recordings"].append(result)
                assert myogen.get_random_generator().bit_generator.state == after
                report["results"].append(item)
                dump_json(folder / "result.json", report)
                print(kind, item["fibres"], "fibres: source, cache, bins, recording and RNG passed", flush=True)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        dump_json(folder / "result.json", report)


if __name__ == "__main__":
    main()

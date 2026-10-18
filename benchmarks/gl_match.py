"""Matched optimized sEMG timings: three complete MUs, never full population.

Run this same harness against both checkouts. Numerical execution is unmodified;
a separate zero-return replay verifies realized inputs only, after the timer.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from gl_common import ROOT, OUT, owner, restore, sha, topology, dump_json

MUS = [0, 25, 50]
ORDER = ["old1", "new1", "new2", "old2", "old3", "new3"]
EXPECTED = {"old": "174a71404eceae963e31ad027b642f72572c72f3",
            "new": "0f26b54d59067eb3a6b30ec4de144c6cd0c69f6d"}


def revision(root):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + root.as_posix(), "-C", str(root),
         "rev-parse", "HEAD"], text=True).strip()



def check_source_revision(root, branch):
    """Allow documentation/benchmark-only descendants of the measured sources."""
    head = revision(root)
    command = ["git", "-c", "safe.directory=" + root.as_posix(), "-C", str(root)]
    subprocess.run(command + ["merge-base", "--is-ancestor", EXPECTED[branch], head],
                   check=True, capture_output=True)
    subprocess.run(command + ["diff", "--exit-code", EXPECTED[branch], "--",
                              "myogen", "pyproject.toml"],
                   check=True, capture_output=True)
    return head

def source_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted((root / "myogen").rglob("*.py"))}


def child(args):
    import psutil
    process = psutil.Process()
    allowed = process.cpu_affinity()
    cores = sorted(topology(), key=lambda c: (-c["efficiency_class"], c["logical"][0]))
    affinity = [next(i for i in c["logical"] if i in allowed)
                for c in cores if any(i in allowed for i in c["logical"])][:4]
    assert affinity == [0, 2, 4, 6], affinity
    process.cpu_affinity(affinity)
    sys.modules["cupy"] = None
    import joblib
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    import myogen
    from myogen import simulator
    from myogen.simulator.core.emg.runtime import semg
    from myogen.simulator.core.emg.runtime.cache import fingerprint

    assert Path(myogen.__file__).resolve().is_relative_to(args.root.resolve())
    check_source_revision(args.root, args.label[:3])
    sources = source_hashes(args.root)
    folder = args.output / args.label
    folder.mkdir(parents=True, exist_ok=False)
    muscle = joblib.load(args.inputs / "muscle.joblib")
    counts = np.asarray(muscle.resulting_number_of_innervated_fibers, dtype=int)
    assert len(counts) == 100 and sum(counts) == 58755
    assert int(sum(counts[MUS])) == 470
    policy = simulator.ExecutionPolicy(max_workers=1)
    restore(args.inputs)
    model = owner(muscle, "s", MUS)
    initial_rng = deepcopy(myogen.get_random_generator().bit_generator.state)
    result = dict(status="running", label=args.label, root=str(args.root),
                  head=revision(args.root), harness_sha256=sha(__file__),
                  source_sha256=sources, python=sys.executable,
                  imported=myogen.__file__, affinity=affinity,
                  mus=MUS, counts=counts[MUS].tolist(), fibres=470,
                  population=100, total_fibres=58755, native_threads=1,
                  n_jobs=1, max_workers=1, gpu=False, persistent_cache=False,
                  physical_bytes=psutil.virtual_memory().total,
                  available_before=psutil.virtual_memory().available,
                  fixture_sha256=sha(args.inputs / "muscle.joblib"),
                  rng_fixture_sha256=sha(args.inputs / "realization-rng.json"),
                  initial_rng=initial_rng,
                  timestamp=datetime.now(timezone.utc).isoformat())
    dump_json(folder / "result.json", result)
    try:
        with threadpool_limits(limits=1):
            result["native_pools_before"] = threadpool_info()
            assert all(p["num_threads"] == 1 for p in result["native_pools_before"])
            begin_cpu = process.cpu_times()
            started = time.perf_counter()
            block = model.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False,
                                         execution=policy)
            result["seconds"] = time.perf_counter() - started
            end_cpu = process.cpu_times()
            result["cpu_seconds"] = end_cpu.user + end_cpu.system - begin_cpu.user - begin_cpu.system
            result["rng"] = deepcopy(myogen.get_random_generator().bit_generator.state)
            result["execution_report"] = dict(model.execution_report)
            result["native_pools_after"] = threadpool_info()
            assert all(p["num_threads"] == 1 for p in result["native_pools_after"])
            assert model.execution_report["completed_work"] == 470
            assert tuple(model.execution_report["worker_count_changes"]) == (1,)
            assert model.execution_report["cache_hits"] == 0
            assert model.execution_report["allocation_retries"] == 0
            assert process.cpu_affinity() == affinity

            # Save selected real MU outputs, plus a digest of ALL output slots and
            # complete metadata. All validation and serialization are untimed.
            arrays, metadata, selected_arrays = [], [], {}
            for group in block.groups:
                for segment in group.segments:
                    for signal in segment.analogsignals:
                        a = signal.magnitude
                        assert np.all(np.isfinite(a))
                        arrays.append(a)
                        metadata.append(dict(group=group.name, segment=segment.name,
                            group_annotations=group.annotations, segment_annotations=segment.annotations,
                            shape=a.shape, dtype=a.dtype.str, units=str(signal.units),
                            sampling_rate=str(signal.sampling_rate), t_start=str(signal.t_start),
                            t_stop=str(signal.t_stop), name=signal.name, annotations=signal.annotations,
                            array_annotations=dict(signal.array_annotations)))
                        if segment.name in [f"MUAP_{mu}" for mu in MUS]:
                            selected_arrays[segment.name] = a
            assert len(selected_arrays) == 3 and len(metadata) == 100
            result["waveform_sha256"] = fingerprint(arrays)
            result["metadata_sha256"] = fingerprint(metadata)
            dump_json(folder / "metadata.json", metadata)
            np.savez_compressed(folder / "templates.npz", **selected_arrays)

            # Contract-only zero-return replay: exact arguments/order, not an
            # additional numerical source run and not waveform-accuracy evidence.
            restore(args.inputs)
            capture_model = owner(muscle, "s", MUS)
            original = semg._simulate_fiber_v2_python
            ids = []
            dummy = np.zeros((5, 5, int(capture_model._internal_sampling_points)))
            def capture(**kwargs):
                values = {k: v for k, v in kwargs.items()
                          if k not in ("_prepared", "electrode_array")}
                ids.append(fingerprint(values, vars(kwargs["electrode_array"])))
                return dummy, None, None
            semg._simulate_fiber_v2_python = capture
            try:
                capture_model.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False,
                                            execution=policy)
            finally:
                semg._simulate_fiber_v2_python = original
            assert len(ids) == 470
            assert result["rng"] == myogen.get_random_generator().bit_generator.state
            result["realized_inputs_sha256"] = fingerprint(ids)
            dump_json(folder / "input_ids.json", ids)
            result["capture_scope"] = "Contract-only; zero-return replay outside timed numerical call."
        assert revision(args.root) == result["head"]
        assert source_hashes(args.root) == sources
        result["available_after"] = psutil.virtual_memory().available
        result["status"] = "complete"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        dump_json(folder / "result.json", result)
    print(args.label, f'{result["seconds"]:.6f} seconds', flush=True)


def orchestrate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    data = []
    for label in ORDER:
        root = args.old if label.startswith("old") else ROOT
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), str(root / ".runtime-deps"), env.get("PYTHONPATH", "")])
        env.update(MYOGEN_DISABLE_GPU="1", NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
                   "--root", str(root), "--label", label, "--inputs", str(args.inputs),
                   "--output", str(args.output)]
        print("Starting", label, flush=True)
        with (args.output / (label + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        item = json.loads((args.output / label / "result.json").read_text())
        assert item["status"] == "complete"
        data.append(item)
        print(label, round(item["seconds"], 3), "s", flush=True)
    for field in ("fixture_sha256", "rng_fixture_sha256", "initial_rng", "rng",
                  "realized_inputs_sha256", "metadata_sha256", "affinity", "counts"):
        assert all(item[field] == data[0][field] for item in data), field
    for branch in ("old", "new"):
        subset = [item for item in data if item["label"].startswith(branch)]
        assert len({item["waveform_sha256"] for item in subset}) == 1, branch
        assert len({json.dumps(item["source_sha256"], sort_keys=True) for item in subset}) == 1
    times = {branch: [item["seconds"] for item in data if item["label"].startswith(branch)]
             for branch in ("old", "new")}
    medians = {branch: statistics.median(values) for branch, values in times.items()}
    summary = dict(status="complete", scope="Optimized source MUAP generation only; 470 real fibres.",
                   order=ORDER, times=times, medians=medians,
                   new_over_old=medians["new"] / medians["old"],
                   new_runtime_reduction_percent=100*(1-medians["new"]/medians["old"]),
                   old_over_new_speed_ratio=medians["old"] / medians["new"],
                   input_rng_metadata="Exact across all six runs",
                   repeat_waveforms="Bitwise exact within each branch",
                   cross_branch_waveforms="Different conductor formulations; no equivalence claim",
                   runs=data)
    dump_json(args.output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT / "m471")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.child:
        child(args)
    else:
        assert args.old is not None
        orchestrate(args)

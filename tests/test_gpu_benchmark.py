"""Reporting layer of experiments/twin/gpu_benchmark.py and the plotting
script that reads its evidence. No simulation runs here."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "experiments" / "twin"
sys.path.insert(0, str(TWIN))

import gpu_benchmark as gb  # noqa: E402


def _row(**kw):
    base = dict(sweep="pool", device="cpu", precision="float32", op="hard_forward",
                workload="step", theta="default", n_mn=10, batch_size=1, dt_ms=0.1,
                status="ok")
    return {**base, **kw}


def test_every_timed_call_is_synchronized_before_the_clock_stops():
    events = []

    def clock():
        events.append("clock")
        return float(len(events))

    def sync(x):
        events.append("sync")
        return x

    timing, _ = gb.time_op(lambda: events.append("call"), (), n_warmup=2, n_rep=7,
                           sync=sync, clock=clock)
    timed = ["clock", "call", "sync", "clock"]
    assert events == timed + ["call", "sync"] * 2 + timed * 7


def test_compile_and_steady_state_are_stored_separately():
    timing, _ = gb.time_op(lambda: None, (), sync=lambda x: x)
    assert set(timing) >= {"first_call_s", "reps_s", "median_s", "iqr_s", "n_warmup"}
    assert len(timing["reps_s"]) == 7 and timing["n_warmup"] == 2


def test_protocol_minimums_are_enforced():
    with pytest.raises(ValueError):
        gb.time_op(lambda: None, (), n_warmup=2, n_rep=3, sync=lambda x: x)


def test_cpu_and_gpu_rows_cannot_overwrite_each_other():
    rows = []
    gb.add_row(rows, _row(device="cpu"))
    gb.add_row(rows, _row(device="gpu"))
    with pytest.raises(ValueError):
        gb.add_row(rows, _row(device="gpu", median_s=1.0))
    assert [r["device"] for r in rows] == ["cpu", "gpu"]


def test_precision_is_recorded_and_mixed_graphs_are_labelled_mixed():
    args = argparse.Namespace(devices=["cpu"], sweeps=["pool", "precision"],
                              precision="float32", pool_sizes=[1], batch_sizes=[1],
                              batch_n_mn=10, n_dd=100, dt_ms=0.1, t_on=1.0, seed=0,
                              n_warmup=2, n_rep=7, theta="auto")
    cfgs = gb.configs(args)
    assert all(c["precision"] in ("float32", "float64") for c in cfgs)
    assert {c["precision"] for c in cfgs if c["sweep"] == "precision"} == {"float32",
                                                                           "float64"}
    assert gb.float_label({"a": "float64", "b": "int32"}) == ("float64", [])
    assert gb.float_label({"a": "float32"}) == ("float32", ["a"])
    assert gb.float_label({"a": "float64", "b": "float32"}) == ("mixed", ["b"])


def test_graph_scan_overrides_leaf_dtypes():
    assert gb.graph_label({"a": "float64"}, {"float64": 9}) == ("float64", [])
    label, f32 = gb.graph_label({"a": "float64"}, {"float64": 9, "float32": 2})
    assert label == "mixed" and "2 float32 values inside the graph" in f32


def test_k1_uses_default_parameters_and_batches_use_draws():
    args = argparse.Namespace(devices=["cpu"], sweeps=["batch"], precision="float32",
                              pool_sizes=[1], batch_sizes=[1, 4], batch_n_mn=10,
                              n_dd=100, dt_ms=0.1, t_on=1.0, seed=0, n_warmup=2,
                              n_rep=7, theta="auto")
    assert [(c["batch_size"], c["theta"]) for c in gb.configs(args)][:2] == [
        (1, "default"), (4, "perturbed")]


def test_device_fallback_is_a_failure():
    assert gb.device_status("gpu", "cpu") == "device_fallback"
    assert gb.device_status("gpu", "gpu") == "ok"


def test_failure_classification():
    assert gb.classify_failure(1, "XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory") == "oom"
    assert gb.classify_failure(1, "CUDA_ERROR_OUT_OF_MEMORY") == "oom"
    assert gb.classify_failure(-9, "") == "oom"
    assert gb.classify_failure(1, "ValueError: zoom factor") == "failed"
    assert gb.classify_failure(1, "ValueError: shapes") == "failed"


def test_timeout_produces_a_row():
    cfg = dict(_row(), n_warmup=2, n_rep=7, n_dd=100, t_pre_s=0.2, t_on_s=1.0,
               t_post_s=0.2, level_hz=40.0, seed=0)
    row = gb.run_config(cfg, timeout_s=0.01)
    assert row["status"] == "timeout" and row["device"] == "cpu"


def test_failed_oom_and_skipped_rows_survive_in_json_and_csv(tmp_path):
    rows = []
    gb.add_row(rows, _row(n_mn=1, median_s=1.0, sims_per_s=1.0, reps_s=[1.0] * 7))
    gb.add_row(rows, _row(n_mn=500, status="oom"))
    gb.add_row(rows, _row(n_mn=1000, status="not_attempted"))
    gb.add_row(rows, _row(n_mn=1, device="gpu", status="timeout"))
    gb.write_outputs(tmp_path, {"manifest": {}, "rows": rows})
    saved = json.loads((tmp_path / "benchmark.json").read_text())["rows"]
    assert [r["status"] for r in saved] == ["ok", "oom", "not_attempted", "timeout"]
    with open(tmp_path / "benchmark_reps.csv") as f:
        tidy = list(csv.DictReader(f))
    assert len(tidy) == 7 + 3
    assert {r["status"] for r in tidy} == {"ok", "oom", "not_attempted", "timeout"}


def test_manifest_schema():
    args = argparse.Namespace(pool_sizes=[1], batch_sizes=[1], batch_n_mn=10)
    m = gb.manifest(args, {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
    assert set(m) >= {"command_line", "args", "child_environment",
                      "environment_variables", "git", "hardware", "software",
                      "source", "protocol"}
    assert set(m["hardware"]) >= {"os", "kernel", "cpu_model", "cpu_physical_cores",
                                  "cpu_logical_cores", "ram_bytes",
                                  "gpu_name_vram_driver_cc", "cuda_version_driver"}
    assert set(m["software"]) >= {"python", "jax", "jaxlib", "jaxley", "numpy"}
    assert m["git"]["sha"]


def _evidence(tmp_path):
    bench = {"manifest": {"git": {"sha": "x", "dirty": False}}, "rows": [
        _row(device=d, op=op, n_mn=n, mn_steps_per_s=1e6 * n, median_s=1.0)
        for d in ("cpu", "gpu") for op in ("hard_forward", "surrogate_value_and_grad")
        for n in (1, 10)] + [_row(device="gpu", n_mn=1000, status="oom")] + [
        _row(sweep="batch", device=d, op=op, batch_size=k, sims_per_s=k, median_s=1.0)
        for d in ("cpu", "gpu") for op in ("hard_forward", "surrogate_value_and_grad")
        for k in (1, 4)]}
    g2 = lambda x64: {"x64": x64, "manifest": {}, "g2": {  # noqa: E731
        "per_coordinate": {"gCaL": {"agree": 8, "wrong": 1, "unresolved": 1},
                           "w_exc": {"agree": 9, "wrong": 1, "unresolved": 0}},
        "rows": [{"variant": "beta=5.0", "classification": ["agree", "agree"],
                  "leaf_dtype": "float64", "grad_finite": True}]}}
    g1 = {"x64": True, "regime": "b_config", "manifest": {}, "g1": {"rows": [
        {"dt_ms": dt, "horizon_s": h, "n_spikes": 5, "verdict": v,
         "rel_rev_vs_fdconv": None, "cosine": None}
        for dt, h, v in ((0.1, 2.0, "FD_UNCONVERGED"), (0.025, 2.0, "PASS"),
                         (0.1, 4.0, "FAIL"), (0.025, 4.0, "FD_UNCONVERGED"))]}}
    gate = {"joint_accepted_fraction": 0.8, "coord_accepted_fraction": {"gCaL": 1, "w": 0.5},
            "nonfinite": 0, "joint_pass": True, "per_start": {"S1": {
                "joint_acc": 4, "joint_tot": 5, "coord_acc": {"gCaL": 2, "w": 1},
                "coord_tot": {"gCaL": 2, "w": 2}}}}
    g4 = {"manifest": {}, "stageA": {"gate": {"surrogate": gate, "crn_fd": gate},
                                     "rows": [{"case": {}, "starts": {"S1": {
                                         "cosine_sur_vs_fd": -0.99}}}]}}
    paths = {}
    for name, d in (("bench", bench), ("f32", g2(False)), ("f64", g2(True)),
                    ("g1", g1), ("g4", g4)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(d))
    return paths


def test_plot_reads_only_saved_evidence(tmp_path):
    p = _evidence(tmp_path)
    code = f"""
import sys, json
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import plot_computational_gradient_scaling as plot
src = plot.build_source_data([{str(p['bench'])!r}],
                             {{"Float32": {str(p['f32'])!r}, "Float64": {str(p['f64'])!r}}},
                             {str(p['g1'])!r}, {str(p['g4'])!r})
fig = plot.draw(src)
banned = [m for m in ("jax", "jaxley", "poolbuild", "gpu_benchmark") if m in sys.modules]
assert not banned, banned
assert src["panels"]["B"]["Float64"]["agree"] == 17
assert src["panels"]["B"]["Float64"]["classes"]["gCaL"] == ["agree"]
assert [r["class"] for r in src["panels"]["C"]] == ["Unresolved", "Pass", "Failed", "Unresolved"]
assert src["panels"]["A"]["pool_forward"]["series"]["gpu"][-1]["status"] == "oom"
assert [p["x"] for p in src["panels"]["A"]["batch_forward"]["series"]["cpu"]] == [1, 4]
assert src["panels"]["D"]["estimators"]["surrogate"]["per_start"]["S1"]["joint"] == 0.8
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_batch_speedup_uses_the_default_theta_k1_row():
    rows = [_row(sweep="batch", batch_size=1, theta="default", sims_per_s=2.0, median_s=0.5),
            _row(sweep="batch", batch_size=4, theta="perturbed", sims_per_s=6.0, median_s=0.67)]
    d = gb.derive(rows)[1]
    assert d["speedup_vs_k1"] == 3.0 and d["parallel_efficiency"] == 0.75


def test_checkpointed_rows_are_separate_and_ops_filter():
    rows = []
    gb.add_row(rows, _row(op="surrogate_value_and_grad"))
    gb.add_row(rows, _row(op="surrogate_value_and_grad", remat_segment=140))
    args = argparse.Namespace(devices=["gpu"], sweeps=["batch"], precision="float32",
                              pool_sizes=[1], batch_sizes=[1, 4], batch_n_mn=10,
                              n_dd=100, dt_ms=0.1, t_on=1.0, seed=0, n_warmup=2,
                              n_rep=7, theta="auto", ops=["surrogate_value_and_grad"],
                              remat_segment=140)
    cfgs = gb.configs(args)
    assert {c["op"] for c in cfgs} == {"surrogate_value_and_grad"}
    assert all(c["remat_segment"] == 140 for c in cfgs)

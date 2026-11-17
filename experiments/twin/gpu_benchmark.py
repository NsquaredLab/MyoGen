"""CPU/GPU benchmark of the pool twin for the computational-scaling figure.

Two operations are timed and always labelled separately:

* ``hard_forward``: the scientific forward model (hard spikes, no gradient).
* ``surrogate_value_and_grad``: surrogate-spike ``value_and_grad``, used only
  to PROPOSE parameter updates. It is not the derivative of the hard-spike
  objective.

Sweeps (step protocol of ``phase2_benchmark.py``: dt 0.1 ms, 0.2 + 1.0 + 0.2 s,
40 Hz, 100 drive sources, seed 0; fixed before any result was seen):

* ``pool``: n_mn in POOL_SIZES at batch K = 1;
* ``batch``: K in BATCH_SIZES at n_mn = 10 (``jax.vmap`` over K parameter sets);
* ``stress``: n_mn 2000 and 5000 at K = 1 (optional, not headline);
* ``precision``: float32 vs float64 on the G4 Stage-A configuration
  (``g4_descent.Case``: n_mn 8, dt 0.025 ms, full fit ramp, K = 1), timing the
  hard-forward loss and the surrogate gradient. Each row carries a dtype audit
  of the graph; a graph with any float32 component is labelled ``mixed``.

Parameters: K = 1 runs ``default_theta()`` (the calibrated set; the pool
fires); K > 1 runs ``phase2_benchmark.batched_thetas`` (K random 0.5-2x draws,
some silent). Deviation from phase2_benchmark, whose first draw (its whole
K = 1 workload) is a silent pool with an exactly zero gradient.

Every configuration runs in a fresh subprocess (device chosen with
``JAX_PLATFORMS``; ``XLA_PYTHON_CLIENT_PREALLOCATE=false``) under a timeout. The
first call (compile + first execution) is timed apart from steady state; then
``n_warmup`` untimed calls and ``n_rep`` timed calls, each synchronized with
``jax.block_until_ready`` and timed with ``time.perf_counter``. A row is kept for
every configuration: ``ok``, ``timeout``, ``oom``, ``failed``,
``device_fallback`` (counts as a failure) or ``not_attempted`` (a smaller size
of the same series already timed out or ran out of memory). The JSON and the
tidy CSV are rewritten after every row, so a crash loses nothing.

``phase2_benchmark.py`` and its command line are unchanged.

Usage::

    python gpu_benchmark.py                                  # pool, batch, precision on cpu then gpu
    python gpu_benchmark.py --sweeps stress --devices gpu
    python gpu_benchmark.py --sweeps pool --pool-sizes 1 10 --n-rep 7 --timeout-s 600
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / "evidence"
RESULT_TAG = "GPU_BENCHMARK_RESULT "

POOL_SIZES = [1, 10, 50, 100, 250, 500, 1000]
BATCH_SIZES = [1, 4, 16, 64, 256, 1000]
STRESS_SIZES = [2000, 5000]
STEP = dict(dt_ms=0.1, t_pre_s=0.2, t_on_s=1.0, t_post_s=0.2, level_hz=40.0,
            n_dd=100, seed=0)
G4_CASE = dict(conn_seed=0, event_offset=0, theta_true=(1.5, 0.011),
               start=(1.0, 0.009))           # g4_descent case 0, start S1
OPS = {"step": ("hard_forward", "surrogate_value_and_grad"),
       "g4": ("hard_forward_loss", "surrogate_grad")}
PLATFORM_ENV = {"cpu": "cpu", "gpu": "cuda"}
FAILED = ("timeout", "oom")                  # larger sizes of the series are skipped
KEY = ("sweep", "device", "precision", "op", "workload", "theta", "n_mn",
       "batch_size", "dt_ms", "remat_segment")


# --------------------------------------------------------------------------- #
# Timing and bookkeeping (pure; unit-tested)
# --------------------------------------------------------------------------- #

def summarize(reps) -> dict:
    r = np.asarray(reps, dtype=float)
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    return {"median_s": float(med), "min_s": float(r.min()), "max_s": float(r.max()),
            "q1_s": float(q1), "q3_s": float(q3), "iqr_s": float(q3 - q1)}


def time_op(fn, args, n_warmup=2, n_rep=7, sync=None, clock=time.perf_counter,
            after_first=None):
    """First call (compile + first run), ``n_warmup`` untimed calls, then
    ``n_rep`` timed calls; every call is synchronized before the clock stops.
    ``after_first`` runs (untimed) after the first call, e.g. a memory probe."""
    if n_warmup < 2 or n_rep < 7:
        raise ValueError("protocol needs >= 2 warm-up and >= 7 timed repetitions")
    if sync is None:
        import jax
        sync = jax.block_until_ready
    t0 = clock()
    out = sync(fn(*args))
    first = clock() - t0
    if after_first is not None:
        after_first()
    for _ in range(n_warmup):
        sync(fn(*args))
    reps = []
    for _ in range(n_rep):
        t0 = clock()
        out = sync(fn(*args))
        reps.append(clock() - t0)
    return {"first_call_s": first, "n_warmup": n_warmup, "reps_s": reps,
            **summarize(reps)}, out


def row_key(row) -> tuple:
    return tuple(row.get(k) for k in KEY)


def add_row(rows: list, row: dict) -> None:
    """Append-only: a configuration (device included) is never overwritten."""
    if any(row_key(r) == row_key(row) for r in rows):
        raise ValueError(f"duplicate configuration {row_key(row)}")
    rows.append(row)


def device_status(requested: str, actual: str) -> str:
    return "ok" if requested == actual else "device_fallback"


def classify_failure(returncode: int, stderr: str) -> str:
    text = (stderr or "").lower()
    if any(s in text for s in ("resource_exhausted", "out of memory", "out_of_memory",
                               "outofmemory")):
        return "oom"
    if returncode in (-9, 137):              # killed by the kernel OOM killer
        return "oom"
    return "failed"


def float_label(dtypes: dict) -> tuple[str, list[str]]:
    """'float32' / 'float64' / 'mixed' over every floating component."""
    floats = {k: v for k, v in dtypes.items() if v.startswith("float")}
    f32 = sorted(k for k, v in floats.items() if v != "float64")
    if not f32:
        return "float64", []
    return ("float32" if len(f32) == len(floats) else "mixed"), f32


def derive(rows: list) -> list:
    """Speed-up vs K = 1, parallel efficiency and gradient/forward ratio."""
    ok = {row_key(r): r for r in rows if r["status"] == "ok"}
    out = []
    for r in rows:
        d = {"key": list(row_key(r))}
        if r["status"] == "ok":
            # under --theta auto the K = 1 row runs the default theta, K > 1 the draws
            base = (ok.get(row_key({**r, "batch_size": 1}))
                    or ok.get(row_key({**r, "batch_size": 1, "theta": "default"})))
            if base and r["sweep"] == "batch":
                d["speedup_vs_k1"] = r["sims_per_s"] / base["sims_per_s"]
                d["parallel_efficiency"] = d["speedup_vs_k1"] / r["batch_size"]
            fwd_op, grad_op = OPS[r["workload"]]
            if r["op"] == grad_op:
                fwd = ok.get(row_key({**r, "op": fwd_op}))
                if fwd:
                    d["grad_over_forward_time"] = r["median_s"] / fwd["median_s"]
        out.append(d)
    return out


def tidy_rows(rows: list) -> list:
    """One CSV row per device, precision, op, n_mn, batch size and replicate."""
    cols = ("sweep", "device", "precision", "precision_label", "op", "workload",
            "theta", "n_mn", "batch_size", "dt_ms", "remat_segment", "status")
    out = []
    for r in rows:
        base = {c: r.get(c) for c in cols}
        reps = r.get("reps_s") or [None]
        for i, t in enumerate(reps):
            out.append({**base, "replicate": i if t is not None else None,
                        "time_s": t, "first_call_s": r.get("first_call_s")})
    return out


def write_outputs(out_dir: Path, report: dict) -> None:
    report["derived"] = derive(report["rows"])
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=1, default=str))
    rows = tidy_rows(report["rows"])
    with open(out_dir / "benchmark_reps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Child: build one workload and time it (imports JAX)
# --------------------------------------------------------------------------- #

def _dtypes(tree, prefix) -> dict:
    import jax
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return {prefix + jax.tree_util.keystr(p): str(np.dtype(x.dtype)) for p, x in leaves}


def build_step(cfg):
    import jax
    import jax.numpy as jnp
    import observables as obs
    import protocols
    from phase2_benchmark import batched_thetas
    from poolbuild import PoolTwin, TwinSpec

    twin = PoolTwin(TwinSpec(n_mn=cfg["n_mn"], n_dd=cfg["n_dd"], dt_ms=cfg["dt_ms"],
                             seed=cfg["seed"], remat_segment=cfg.get("remat_segment")))
    _, cmd = protocols.step(cfg["dt_ms"], t_pre_s=cfg["t_pre_s"], t_on_s=cfg["t_on_s"],
                            t_post_s=cfg["t_post_s"], level=cfg["level_hz"])
    drives = {"dd": jnp.asarray(cmd), "inh": jnp.zeros(len(cmd), jnp.float32)}
    key = jax.random.PRNGKey(cfg["seed"])
    if cfg["theta"] == "default":           # the calibrated parameter set, K copies
        thetas = {k: jnp.full((cfg["batch_size"],), v)
                  for k, v in twin.default_theta().items()}
    else:                                   # phase2_benchmark's K random draws
        thetas = batched_thetas(twin, cfg["batch_size"],
                                np.random.default_rng(cfg["seed"]))

    def rates(th):
        out = twin._run(th, drives, key, spike_mode="surrogate")
        return obs.smoothed_rates(out["spikes"], cfg["dt_ms"], tau_ms=150.0)

    def loss_fn(th):                                   # as phase2_benchmark
        return jnp.mean(rates(th) ** 2)

    if cfg["op"] == "hard_forward":
        fn = jax.jit(jax.vmap(lambda th: twin._run(th, drives, key, "hard")["spikes"]))
    else:
        fn = jax.jit(jax.vmap(jax.value_and_grad(loss_fn)))
    th0 = jax.tree_util.tree_map(lambda x: x[0], thetas)
    audit = {**_dtypes(th0, "theta"), **_dtypes(drives, "drives"),
             "exc_connectivity": str(np.dtype(twin._exc_mat.dtype)),
             "inh_connectivity": str(np.dtype(twin._inh_mat.dtype)),
             "voltage": str(np.dtype(jax.eval_shape(
                 lambda th: twin._run(th, drives, key, "hard")["v"], th0).dtype)),
             "rates": str(np.dtype(jax.eval_shape(rates, th0).dtype)),
             "loss": str(np.dtype(jax.eval_shape(loss_fn, th0).dtype)),
             **_dtypes(jax.eval_shape(jax.grad(loss_fn), th0), "grad")}
    n_steps = len(cmd)
    return fn, (thetas,), audit, n_steps, n_steps * cfg["dt_ms"] * 1e-3


def build_g4(cfg):
    import jax
    import jax.numpy as jnp
    from g4_descent import DT, Case, binned

    case = Case(G4_CASE["conn_seed"], G4_CASE["event_offset"], G4_CASE["theta_true"])
    vec = jnp.asarray(np.asarray(G4_CASE["start"]))
    fn = case.hard_loss if cfg["op"] == "hard_forward_loss" else case._sur_grad
    w = case.windows[0]
    th = case.theta_of(vec)
    run = lambda v: case.twin._run_events_carry(case.theta_of(v), w, None, "hard")[0]
    sp = lambda v: case.twin._run_events_carry(case.theta_of(v), w, None, "surrogate")[0]["spikes"]
    audit = {**_dtypes(th, "theta"),
             **_dtypes({k: w[k] for k in ("exc", "inh")}, "events"),
             **_dtypes({k: w[k] for k in ("exc_conn", "inh_conn")}, "connectivity"),
             **_dtypes(case.twin._states0, "state"),
             "voltage": str(np.dtype(jax.eval_shape(run, vec)["v"].dtype)),
             "rates": str(np.dtype(jax.eval_shape(lambda v: binned(sp(v)), vec).dtype)),
             "observed_rates": str(np.dtype(case.obs_rates.dtype)),
             "loss": str(np.dtype(jax.eval_shape(case.hard_loss, vec).dtype)),
             "grad": str(np.dtype(jax.eval_shape(case._sur_grad, vec).dtype))}
    n_steps = sum(int(x["exc"].shape[0]) for x in case.windows)
    assert cfg["dt_ms"] == DT
    return fn, (vec,), audit, n_steps, n_steps * DT * 1e-3


def jaxpr_float_counts(fn, *args) -> dict:
    """Floating dtypes of every value in the traced graph (nested jaxprs of
    scan / jit / custom_jvp included): finds casts the leaf audit cannot see."""
    import jax
    counts = {}

    def walk(jaxpr):
        for v in [*jaxpr.invars, *jaxpr.constvars,
                  *(o for e in jaxpr.eqns for o in e.outvars)]:
            dt = str(getattr(v.aval, "dtype", ""))
            if dt.startswith("float"):
                counts[dt] = counts.get(dt, 0) + 1
        for e in jaxpr.eqns:
            for p in e.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", sub)
                    if hasattr(inner, "eqns"):
                        walk(inner)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


def graph_label(audit: dict, jaxpr_counts: dict) -> tuple[str, list[str]]:
    """Leaf audit plus graph scan: float64 only if no float32 value anywhere."""
    label, f32 = float_label(audit)
    n32 = jaxpr_counts.get("float32", 0)
    if n32 and label == "float64":
        return "mixed", f32 + [f"{n32} float32 values inside the graph"]
    if n32:
        f32 = f32 + [f"{n32} float32 values inside the graph"]
    if label == "float32" and jaxpr_counts.get("float64", 0):
        label = "mixed"
    return label, f32


def audit_gradient_checks(precision: str) -> dict:
    """Graph dtypes of the gradient_repair G1 (b_config) and G2 objectives as
    they run under ``--x64`` (float64) or without it (float32)."""
    import jax
    import jax.numpy as jnp
    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
    from gradient_repair import _g2_setup, _make_loss

    out = {"precision_flag": precision}
    loss, n_spk = _make_loss("b_config", 0.1, 2.0)
    s0 = jnp.array([1.0, 1.0])
    out["g1_b_config"] = {"loss_hard": jaxpr_float_counts(loss, s0),
                          "grad": jaxpr_float_counts(jax.grad(loss), s0),
                          "n_spikes_dt0.1_T2s": n_spk}
    twin, rates, target = _g2_setup(0.1)
    leaf = jnp.float64 if precision == "float64" else jnp.float32
    z = jnp.asarray((1.0, 0.009), dtype=leaf)
    sur = lambda z: jnp.mean((rates((z[0], z[1]), "surrogate") - target) ** 2)  # noqa: E731
    hard = lambda z: jnp.mean((rates((z[0], z[1]), "hard") - target) ** 2)  # noqa: E731
    out["g2"] = {"surrogate_grad": jaxpr_float_counts(jax.grad(sur), z),
                 "hard_loss_crn_fd": jaxpr_float_counts(hard, z),
                 "target_dtype": str(np.dtype(target.dtype))}
    return out


def output_summary(out) -> dict:
    import jax
    leaves = [np.asarray(x, dtype=np.float64) for x in jax.tree_util.tree_leaves(out)]
    flat = np.concatenate([x.ravel() for x in leaves])
    return {"all_finite": bool(np.isfinite(flat).all()), "n_values": int(flat.size),
            "sum": float(np.nansum(flat)), "abs_max": float(np.nanmax(np.abs(flat))),
            "sha256_16": hashlib.sha256(b"".join(x.tobytes() for x in leaves)).hexdigest()[:16],
            "leaves": [{"shape": list(x.shape), "sum": float(np.nansum(x)),
                        "abs_max": float(np.nanmax(np.abs(x))) if x.size else 0.0}
                       for x in leaves]}


def memory() -> dict:
    import jax
    import psutil
    stats = jax.devices()[0].memory_stats() or {}
    return {"process_rss_bytes": psutil.Process().memory_info().rss,
            "device_bytes_in_use": stats.get("bytes_in_use"),
            "device_peak_bytes_in_use": stats.get("peak_bytes_in_use")}


def child(cfg: dict) -> dict:
    import jax
    if cfg["precision"] == "float64":
        jax.config.update("jax_enable_x64", True)
    actual = jax.devices()[0].platform
    mem = {"before_build": memory()}
    t0 = time.perf_counter()
    build = build_g4 if cfg["workload"] == "g4" else build_step
    fn, args, audit, n_steps, sim_s = build(cfg)
    build_s = time.perf_counter() - t0
    jaxpr_counts = jaxpr_float_counts(fn, *args)
    mem["before_compile"] = memory()
    timing, out = time_op(fn, args, cfg["n_warmup"], cfg["n_rep"],
                          after_first=lambda: mem.update(after_compile=memory()))
    mem["after_steady_state"] = memory()
    out_platforms = sorted({d.platform for x in jax.tree_util.tree_leaves(out)
                            for d in x.devices()})
    label, f32 = graph_label(audit, jaxpr_counts)
    k, med = cfg["batch_size"], timing["median_s"]
    summary = output_summary(out)
    return {"status": device_status(cfg["device"], actual) if out_platforms == [actual]
            else "device_fallback",
            "jax_platform": actual, "output_platforms": out_platforms,
            "device_kind": jax.devices()[0].device_kind,
            "build_s": build_s, **timing, "n_steps": n_steps, "sim_duration_s": sim_s,
            "sims_per_s": k / med, "bio_s_per_wall_s": k * sim_s / med,
            "mn_steps_per_s": k * cfg["n_mn"] * n_steps / med,
            "memory": mem,
            "memory_note": "device_* from jax memory_stats() (XLA allocator; peak is "
                           "process-wide since start, compilation included); "
                           "process_rss_bytes from psutil (observed, not peak)",
            "dtype_audit": audit, "jaxpr_float_counts": jaxpr_counts,
            "precision_label": label, "float32_components": f32,
            # first output leaf: hard spikes (forward) or loss (value_and_grad)
            "pool_fired": (summary["leaves"][0]["sum"] > 0
                           if cfg["workload"] == "step" else None),
            "output_summary": summary}


# --------------------------------------------------------------------------- #
# Parent: configurations, subprocesses, manifest
# --------------------------------------------------------------------------- #

def configs(args) -> list:
    base = dict(n_warmup=args.n_warmup, n_rep=args.n_rep)
    step = dict(STEP, dt_ms=args.dt_ms, t_on_s=args.t_on, n_dd=args.n_dd, seed=args.seed)
    out = []
    for dev in args.devices:
        for sweep in args.sweeps:
            if sweep == "precision":
                for prec in ("float32", "float64"):
                    for op in OPS["g4"]:
                        out.append(dict(base, sweep=sweep, device=dev, precision=prec,
                                        op=op, workload="g4", theta="g4_start_S1",
                                        n_mn=8, batch_size=1, dt_ms=0.025, seed=0))
                continue
            sizes = {"pool": args.pool_sizes, "batch": args.batch_sizes,
                     "stress": STRESS_SIZES}[sweep]
            for op in OPS["step"]:
                if op not in getattr(args, "ops", OPS["step"]):
                    continue
                for s in sizes:
                    n_mn, k = (args.batch_n_mn, s) if sweep == "batch" else (s, 1)
                    theta = args.theta if args.theta != "auto" else (
                        "default" if k == 1 else "perturbed")
                    out.append(dict(base, **step, sweep=sweep, device=dev,
                                    precision=args.precision, op=op, workload="step",
                                    theta=theta, n_mn=n_mn, batch_size=k,
                                    remat_segment=getattr(args, "remat_segment", None)))
    return out


def run_config(cfg: dict, timeout_s: float) -> dict:
    env = dict(os.environ, JAX_PLATFORMS=PLATFORM_ENV[cfg["device"]],
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(Path(__file__)), "--child", json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout_s, env=env,
                           cwd=HERE)
    except subprocess.TimeoutExpired as e:
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {**cfg, "status": "timeout", "timeout_s": timeout_s,
                "wall_s": time.time() - t0, "stderr_tail": _tail(err)}
    found = [l for l in p.stdout.splitlines() if l.startswith(RESULT_TAG)]
    if p.returncode == 0 and found:
        return {**cfg, **json.loads(found[-1][len(RESULT_TAG):]),
                "wall_s": time.time() - t0}
    return {**cfg, "status": classify_failure(p.returncode, p.stderr),
            "returncode": p.returncode, "wall_s": time.time() - t0,
            "stderr_tail": _tail(p.stderr)}


def _tail(stderr: str) -> str:
    """Last 2000 characters, home directory as ``~`` (evidence carries no user paths)."""
    return stderr.replace(str(Path.home()), "~")[-2000:]


def _cmd(*a) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                     # noqa: BLE001 (manifest must not fail)
        return f"unavailable: {type(e).__name__}"


def manifest(args, child_env: dict) -> dict:
    import psutil
    from sensitivity_checks import _manifest    # source hashes (parent stays on CPU)

    smi = _cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader")
    header = _cmd("nvidia-smi")
    cuda = next((l.split("CUDA Version:")[1].strip(" |") for l in header.splitlines()
                 if "CUDA Version:" in l), None)
    os_release = dict(l.split("=", 1) for l in Path("/etc/os-release").read_text()
                      .splitlines() if "=" in l) if Path("/etc/os-release").exists() else {}
    cpu = next((l.split(":", 1)[1].strip() for l in _cmd("lscpu").splitlines()
                if l.startswith("Model name")), platform.processor())
    pkgs = {}
    for name in ("jax", "jaxlib", "jaxley", "numpy", "jax-cuda12-plugin",
                 "jax-cuda12-pjrt", "psutil"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    src = _manifest()
    src.pop("device", None)
    src.pop("precision", None)
    return {
        "command_line": sys.argv, "args": vars(args),
        "child_environment": child_env,
        "environment_variables": {k: v for k, v in os.environ.items()
                                  if k.startswith(("JAX_", "XLA_", "CUDA_", "NVIDIA_",
                                                   "TF_", "OMP_", "MKL_"))},
        "git": {"sha": _cmd("git", "-C", str(REPO), "rev-parse", "HEAD"),
                "branch": _cmd("git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_cmd("git", "-C", str(REPO), "status", "--porcelain"))},
        "hardware": {
            "os": os_release.get("PRETTY_NAME", platform.platform()).strip('"'),
            "kernel": platform.release(), "wsl": "microsoft" in platform.release().lower(),
            "cpu_model": cpu, "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu_name_vram_driver_cc": smi, "cuda_version_driver": cuda},
        "software": {"python": platform.python_version(), **pkgs},
        "source": src,
        "protocol": {"step": STEP, "g4_case": G4_CASE, "pool_sizes": args.pool_sizes,
                     "batch_sizes": args.batch_sizes, "batch_n_mn": args.batch_n_mn,
                     "stress_sizes": STRESS_SIZES, "timing": "first call = compile + "
                     "first execution; then n_warmup untimed; then n_rep timed with "
                     "block_until_ready after each; perf_counter"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--audit", choices=["float32", "float64"],
                    help="print the graph-dtype audit of the G1/G2 objectives and exit")
    ap.add_argument("--theta", choices=["auto", "default", "perturbed"], default="auto",
                    help="parameters of step workloads: auto = default_theta() at K = 1, "
                         "phase2_benchmark's random draws at K > 1")
    ap.add_argument("--sweeps", nargs="+", default=["pool", "batch", "precision"],
                    choices=["pool", "batch", "precision", "stress"])
    ap.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    ap.add_argument("--precision", choices=["float32", "float64"], default="float32",
                    help="precision of the pool/batch/stress sweeps")
    ap.add_argument("--ops", nargs="+", default=list(OPS["step"]), choices=OPS["step"],
                    help="operations of the pool/batch/stress sweeps")
    ap.add_argument("--remat-segment", type=int, default=None,
                    help="gradient checkpointing: steps per jax.checkpoint block of the "
                         "simulation scan (must divide the step count); default off")
    ap.add_argument("--pool-sizes", type=int, nargs="+", default=POOL_SIZES)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    ap.add_argument("--batch-n-mn", type=int, default=10)
    ap.add_argument("--n-dd", type=int, default=STEP["n_dd"])
    ap.add_argument("--dt-ms", type=float, default=STEP["dt_ms"])
    ap.add_argument("--t-on", type=float, default=STEP["t_on_s"])
    ap.add_argument("--seed", type=int, default=STEP["seed"])
    ap.add_argument("--n-warmup", type=int, default=2)
    ap.add_argument("--n-rep", type=int, default=7)
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.audit:
        sys.path.insert(0, str(HERE))
        print(json.dumps(audit_gradient_checks(args.audit), indent=1))
        return
    if args.child:
        sys.path.insert(0, str(HERE))
        print(RESULT_TAG + json.dumps(child(json.loads(args.child)), default=str),
              flush=True)
        return

    os.environ["JAX_PLATFORMS"] = "cpu"        # the parent never touches the GPU
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or HERE / "results" / f"gpu_benchmark_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    child_env = {"JAX_PLATFORMS": PLATFORM_ENV, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    report = {"manifest": manifest(args, child_env), "rows": []}
    broken = set()
    for cfg in configs(args):
        series = (cfg["sweep"], cfg["device"], cfg["precision"], cfg["op"])
        if series in broken:
            row = {**cfg, "status": "not_attempted",
                   "reason": "a smaller size of this series timed out or ran out of memory"}
        else:
            print(f"{cfg['sweep']:9} {cfg['device']} {cfg['precision']} {cfg['op']:24} "
                  f"n_mn {cfg['n_mn']:5} K {cfg['batch_size']:5} ...", end=" ", flush=True)
            row = run_config(cfg, args.timeout_s)
            print(row["status"], f"{row.get('median_s', float('nan')):.4g} s", flush=True)
            if row["status"] in FAILED:
                broken.add(series)
        add_row(report["rows"], row)
        write_outputs(out_dir, report)
    EVIDENCE.mkdir(exist_ok=True)
    shutil.copy(out_dir / "benchmark.json", EVIDENCE / f"gpu_benchmark_{stamp}.json")
    print(f"saved -> {out_dir} (+ evidence/gpu_benchmark_{stamp}.json)")


if __name__ == "__main__":
    main()

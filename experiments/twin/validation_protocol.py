"""Paired NEURON ↔ Jaxley validation protocol — immutable artifact + runners.

The strict Phase-1 validation (roadmap) requires that both simulators receive
EXACTLY the same inputs. This module creates one frozen protocol artifact
containing everything either backend needs:

* presynaptic event matrices (generated once, stored — never regenerated),
* connectivity matrices,
* synaptic constants (site, tau_rise, tau_decay, E_exc, E_inh, weights),
* cell parameters (recruitment thresholds, gCaL/gNaP scales),
* deterministic current-injection waveforms (step levels sized to the
  measured rheobase, 5–30 nA) for the intrinsic single-cell comparison,
* dt, durations, and a content hash for provenance.

Runners (both implemented):

* :func:`run_jaxley` — ``PoolTwin.simulate_events`` / ``simulate_current``.
* :func:`run_neuron` — NEURON backend; VecStim → NetCon fan-in onto the
  cells' dendritic Exp2Syn, IClamp at the declared site, backward Euler
  (``h.secondorder = 0``, matching Jaxley's default solver), spike detection
  at +50 mV (NERLab 1952-HH frame).

Matched-construction notes (verified against the code):

* The NEURON pool interpolates per-cell parameters on RAW thresholds while
  the Jaxley pool normalizes them first — ``run_neuron`` therefore passes
  min-max NORMALIZED thresholds so both backends build the same cells.
* The NEURON pool's YAML initial voltage (−67 mV) is wrong for the NERLab
  frame — ``initial_voltage__mV=0.0`` is passed explicitly.
* NEURON records sample 0 as the initial condition and Jaxley's scan emits
  post-step values: NEURON traces are sliced ``[1:n_steps+1]``.
* The artifact's ``w_exc_uS`` is used VERBATIM in both backends. The default
  0.009 µS was calibrated for the legacy 5 ms surrogate kernel; with the
  NEURON Exp2Syn constants (0.2/0.3 ms) it is largely subthreshold — a
  valid matched-input comparison (subthreshold voltage diff), just not a
  physiological firing regime. Pass ``w_exc_uS=0.15`` to ``make_protocol``
  for an ex03-equivalent firing comparison. State which one you ran.

E_inh convention: the artifact stores the number actually used (−75 =
NEURON literal, −15 = frame-corrected driving force) and the choice is
recorded with the result.

Usage::

    python validation_protocol.py                    # build artifact + Jaxley side
    python validation_protocol.py --backend both     # paired run + comparison
    python validation_protocol.py --backend both --w-exc 0.15   # firing regime
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

RESULTS = Path(__file__).parent / "results"
EVIDENCE = Path(__file__).parent / "evidence"
ARTIFACT_VERSION = 2  # v2: current steps 5/15/30 nA (rheobase-scaled)

# PROCESS-ISOLATION NOTE: loading JAX and NEURON's compiled mechanisms in the
# same process segfaults (verified: jaxley-then-neuron → SIGSEGV; each alone
# is fine). Therefore jax/poolbuild imports are deferred into the jaxley
# runner, the NEURON runner imports no jax, and `--backend both` orchestrates
# the two backends as separate subprocesses.


def fuglevand_thresholds(n: int, recruitment_range: float = 30.0) -> np.ndarray:
    """Local copy (poolbuild has the canonical one) so building an artifact
    does not pull jax into a NEURON process."""
    i = np.arange(1, n + 1)
    return np.exp(i * np.log(recruitment_range) / n) / recruitment_range


# --------------------------------------------------------------------------- #
# Artifact creation
# --------------------------------------------------------------------------- #

def _renewal_events(rng, rates_hz, dt_ms, order: int) -> np.ndarray:
    """Frozen gamma/Erlang renewal events, (n_steps, n_src), 0/1 float32."""
    n_steps, n_src = rates_hz.shape
    yi = np.zeros(n_src)
    thres = rng.gamma(order, 1.0 / order, size=n_src)
    out = np.zeros((n_steps, n_src), dtype=np.float32)
    for t in range(n_steps):
        yi += rates_hz[t] * dt_ms * 1e-3
        fired = yi >= thres
        if fired.any():
            out[t, fired] = 1.0
            yi[fired] = 0.0
            thres[fired] = rng.gamma(order, 1.0 / order, size=int(fired.sum()))
    return out


def make_protocol(
    n_mn: int = 8,
    n_exc: int = 100,
    n_inh: int = 40,
    dt_ms: float = 0.1,
    connect_p: float = 0.5,
    t_rise_s: float = 7.5,
    t_pre_s: float = 1.0,
    t_post_s: float = 3.0,
    peak_hz: float = 60.0,
    inh_rate_hz: float = 0.0,
    renewal_order: int = 5,
    syn_site: str = "dendrite",
    tau_rise_ms: float = 0.2,
    tau_decay_ms: float = 0.3,
    e_exc_mV: float = 70.0,
    e_inh_mV: float = -15.0,
    w_exc_uS: float = 0.009,
    w_inh_uS: float = 0.009,
    gcal_scale: float = 1.0,
    gnap_scale: float = 1.0,
    step_currents_nA=(5.0, 15.0, 30.0),
    seed: int = 0,
) -> dict:
    """Build the frozen protocol artifact (pure data, no simulator objects)."""
    rng = np.random.default_rng(seed)

    n_ramp = int(round((t_pre_s + 2 * t_rise_s + t_post_s) * 1000 / dt_ms))
    t = np.arange(n_ramp) * dt_ms * 1e-3
    cmd = np.interp(t, [0, t_pre_s, t_pre_s + t_rise_s, t_pre_s + 2 * t_rise_s,
                        t[-1]], [0, 0, peak_hz, 0, 0]).astype(np.float32)
    exc_events = _renewal_events(
        rng, np.tile(cmd[:, None], (1, n_exc)), dt_ms, renewal_order)
    inh_events = _renewal_events(
        rng, np.full((n_ramp, n_inh), inh_rate_hz, dtype=np.float32),
        dt_ms, renewal_order)

    exc_conn = (rng.random((n_mn, n_exc)) < connect_p).astype(np.float32)
    inh_conn = (rng.random((n_mn, n_inh)) < connect_p).astype(np.float32)

    # Deterministic current steps (1 s pre, 4 s on, 2 s post), rheobase-scaled.
    n_step = int(round(7.0 * 1000 / dt_ms))
    i_steps = {}
    for amp in step_currents_nA:
        w = np.zeros(n_step, dtype=np.float32)
        w[int(1.0 * 1000 / dt_ms): int(5.0 * 1000 / dt_ms)] = amp
        i_steps[float(amp)] = w

    art = {
        "version": ARTIFACT_VERSION,
        "seed": seed,
        "dt_ms": dt_ms,
        "n_mn": n_mn,
        "thresholds": fuglevand_thresholds(n_mn),
        "ramp_command_hz": cmd,
        "exc_events": exc_events,
        "inh_events": inh_events,
        "exc_conn": exc_conn,
        "inh_conn": inh_conn,
        "synapse": {
            "site": syn_site,
            "tau_rise_ms": tau_rise_ms,
            "tau_decay_ms": tau_decay_ms,
            "e_exc_mV": e_exc_mV,
            "e_inh_mV": e_inh_mV,
            "w_exc_uS": w_exc_uS,
            "w_inh_uS": w_inh_uS,
            "live_driving_force": True,
            "e_inh_convention": (
                "literal NEURON value" if e_inh_mV == -75.0
                else "frame-corrected driving force"),
        },
        "cell": {"gcal_scale": gcal_scale, "gnap_scale": gnap_scale},
        "current_steps_nA": i_steps,
    }
    art["content_hash"] = _hash_artifact(art)
    return art


def _hash_artifact(art: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(art):
        if key == "content_hash":
            continue
        val = art[key]
        h.update(key.encode())
        if isinstance(val, np.ndarray):
            h.update(val.tobytes())
        elif isinstance(val, dict):
            h.update(json.dumps(
                {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in sorted(val.items())}, sort_keys=True).encode())
        else:
            h.update(json.dumps(val, default=float).encode())
    return h.hexdigest()[:16]


def save_protocol(art: dict, path: Path) -> None:
    joblib.dump(art, path)


def load_protocol(path: Path) -> dict:
    art = joblib.load(path)
    expected = art.get("content_hash")
    actual = _hash_artifact(art)
    if expected != actual:
        raise ValueError(
            f"protocol artifact hash mismatch ({expected} != {actual}) — "
            "the artifact was modified after creation")
    return art


# --------------------------------------------------------------------------- #
# Jaxley runner
# --------------------------------------------------------------------------- #

def _twin_from_artifact(art: dict):
    from poolbuild import PoolTwin, TwinSpec  # deferred: keeps jax out of
    # NEURON-only processes (see process-isolation note above)

    syn = art["synapse"]
    return PoolTwin(TwinSpec(
        n_mn=art["n_mn"],
        n_dd=art["exc_events"].shape[1],
        n_inh=art["inh_events"].shape[1],
        dt_ms=art["dt_ms"],
        seed=art["seed"],
        syn_site=syn["site"],
        live_driving_force=syn["live_driving_force"],
        syn_tau_rise_ms=syn["tau_rise_ms"],
        tau_syn_ms=syn["tau_decay_ms"],
        e_exc_mV=syn["e_exc_mV"],
        e_inh_mV=syn["e_inh_mV"],
        w_exc_uS=syn["w_exc_uS"],
        w_inh_uS=syn["w_inh_uS"],
    ))


def run_jaxley(art: dict) -> dict:
    """Run the full artifact on the Jaxley twin. Returns numpy result dict."""
    import jax.numpy as jnp

    twin = _twin_from_artifact(art)
    theta = twin.default_theta()
    theta["gCaL_scale"] = jnp.float32(art["cell"]["gcal_scale"])
    theta["gNaP_scale"] = jnp.float32(art["cell"]["gnap_scale"])

    out = twin.simulate_events(
        theta,
        art["exc_events"],
        inh_events=art["inh_events"],
        exc_connectivity=art["exc_conn"],
        inh_connectivity=art["inh_conn"],
        spike_mode="hard",
    )
    res = {
        "backend": "jaxley",
        "content_hash": art["content_hash"],
        "pool": {k: np.asarray(v) for k, v in out.items()},
        "current_steps": {},
    }
    for amp, wave in art["current_steps_nA"].items():
        i_ext = np.tile(np.asarray(wave)[:, None], (1, art["n_mn"]))
        o = twin.simulate_current(theta, i_ext, spike_mode="hard")
        res["current_steps"][amp] = {k: np.asarray(v) for k, v in o.items()}
    return res


# --------------------------------------------------------------------------- #
# NEURON runner
# --------------------------------------------------------------------------- #

def _neuron_pool(art: dict):
    """Build the matched NEURON NERLab pool (normalized thresholds,
    v_init = 0, artifact PIC scales). Returns (h, pool)."""
    from myogen.utils.nmodl import load_nmodl_mechanisms

    load_nmodl_mechanisms(quiet=True)
    from neuron import h

    h.load_file("stdrun.hoc")
    from myogen.simulator.neuron import cells as ncells
    from myogen.simulator.neuron.populations import AlphaMN__Pool

    ncells.reset_cell_id_counters()
    rt = np.asarray(art["thresholds"], dtype=float)
    # NEURON's special_interp uses RAW thresholds; Jaxley normalizes first.
    # Pass normalized thresholds so both backends build identical cells.
    rt_norm = (rt - rt.min()) / (rt.max() - rt.min() + 1e-10)
    pool = AlphaMN__Pool(
        recruitment_thresholds__array=rt_norm,
        model="NERLab",
        mode="active",
        initial_voltage__mV=0.0,   # YAML default −67 is wrong for NERLab
        spike_threshold__mV=50.0,
    )
    for c in pool:
        c.dend[0].gcaLbar_caL *= art["cell"]["gcal_scale"]
        c.soma.gnapbar_napp *= art["cell"]["gnap_scale"]
    return h, pool


def _neuron_record(h, pool):
    keep, vrec, dvrec, spk = [], [], [], []
    for c in pool:
        vrec.append(h.Vector().record(c.soma(0.5)._ref_v))
        dvrec.append(h.Vector().record(c.dend[0](0.5)._ref_v))
        sv = h.Vector()
        nc = h.NetCon(c.soma(0.5)._ref_v, None, sec=c.soma)
        nc.threshold = 50.0
        nc.record(sv)
        keep.append(nc)
        spk.append(sv)
    return keep, vrec, dvrec, spk


def _neuron_advance(h, pool, dt_ms: float, n_steps: int):
    h.dt = dt_ms
    h.secondorder = 0  # backward Euler ≡ jaxley's default solver
    for sec, v in zip(*pool.get_initialization_data()):
        sec.v = v
    h.finitialize()
    for _ in range(n_steps):  # exact count — `while h.t < tstop` over-steps
        h.fadvance()


def _collect(vrec, dvrec, spk, dt_ms, n_steps, n_mn):
    # NEURON sample 0 is the initial condition; Jaxley emits post-step values.
    v = np.stack([np.asarray(r.as_numpy())[1:n_steps + 1] for r in vrec], axis=1)
    vd = np.stack([np.asarray(r.as_numpy())[1:n_steps + 1] for r in dvrec], axis=1)
    spikes = np.zeros((n_steps, n_mn), dtype=np.float32)
    for i, sv in enumerate(spk):
        for t_ms in np.asarray(sv.as_numpy()):
            k = min(int(round(t_ms / dt_ms)), n_steps - 1)
            spikes[k, i] = 1.0
    return {"v": v, "v_dend": vd, "spikes": spikes}


def run_neuron(art: dict) -> dict:
    """Run the full artifact on the NEURON backend (matched construction)."""
    dt = art["dt_ms"]
    n_mn = art["n_mn"]
    syn_p = art["synapse"]

    # ---- pool/synaptic phase
    h, pool = _neuron_pool(art)
    keep = []
    vstims = []
    syn_ncs = []
    for name, events, conn, syn_idx, w, e in (
            ("exc", art["exc_events"], art["exc_conn"], 0,
             syn_p["w_exc_uS"], syn_p["e_exc_mV"]),
            ("inh", art["inh_events"], art["inh_conn"], 1,
             syn_p["w_inh_uS"], syn_p["e_inh_mV"])):
        stims = []
        for s in range(events.shape[1]):
            times = np.flatnonzero(events[:, s]) * dt
            tv = h.Vector(times.astype(float))
            vs = h.VecStim()
            vs.play(tv)
            stims.append((vs, tv))  # keep tv alive — GC trap
        vstims.append(stims)
        for i, c in enumerate(pool):
            syn = c.synapse__list[syn_idx]  # 0=exc, 1=inh; both at dend(0.5)
            syn.tau1 = syn_p["tau_rise_ms"]
            syn.tau2 = syn_p["tau_decay_ms"]
            syn.e = e
            for s in np.flatnonzero(conn[i]):
                nc = h.NetCon(stims[s][0], syn)
                nc.weight[0] = w
                nc.delay = 0.0  # matched to the zero-delay Jaxley event path
                keep.append(nc)
                syn_ncs.append(nc)
    rec_keep, vrec, dvrec, spk = _neuron_record(h, pool)
    n_steps = art["exc_events"].shape[0]
    _neuron_advance(h, pool, dt, n_steps)
    res = {
        "backend": "neuron",
        "content_hash": art["content_hash"],
        "pool": _collect(vrec, dvrec, spk, dt, n_steps, n_mn),
        "current_steps": {},
    }

    # ---- intrinsic current steps, SAME pool (building a second pool while
    # the first is alive segfaults NEURON at finitialize — duplicate hoc
    # identities after reset_cell_id_counters). Disable the synaptic input
    # in place instead: zero all NetCon weights and empty the event vectors.
    for nc in syn_ncs:
        nc.weight[0] = 0.0
    for stims in vstims:
        for _vs, tv in stims:
            tv.resize(0)  # VecStim with empty vector = silent (verified)
    ics = []
    for c in pool:
        seg = (c.dend[0](0.5) if syn_p["site"] == "dendrite"
               else c.soma(0.5))
        ic = h.IClamp(seg)
        ic.delay = 0.0
        ic.dur = 1e9
        ic.amp = 0.0
        ics.append(ic)
    for amp, wave in art["current_steps_nA"].items():
        players = []
        for ic in ics:
            tvec = h.Vector(np.arange(len(wave)) * dt)
            ivec = h.Vector(np.asarray(wave, dtype=float))
            ivec.play(ic._ref_amp, tvec, 1)
            players.append((tvec, ivec))
        for sv in spk:  # NetCon.record vectors are not cleared by finitialize
            sv.resize(0)
        n = len(wave)
        _neuron_advance(h, pool, dt, n)
        res["current_steps"][amp] = _collect(vrec, dvrec, spk, dt, n, n_mn)
        for _tvec, ivec in players:
            ivec.play_remove()  # avoid stacking play sources on the same ref
    return res


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def _match_spikes(a: np.ndarray, b: np.ndarray, tol_s: float) -> int:
    i = j = n = 0
    while i < len(a) and j < len(b):
        d = a[i] - b[j]
        if abs(d) <= tol_s:
            n += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return n


def compare_backends(res_j: dict, res_n: dict, dt_ms: float) -> dict:
    """Backend-diff metrics for every phase of the artifact."""
    assert res_j["content_hash"] == res_n["content_hash"]

    def diff(pj, pn):
        out = {}
        for key in ("v", "v_dend"):
            d = pj[key] - pn[key]
            out[f"{key}_rmse_mV"] = float(np.sqrt((d ** 2).mean()))
            out[f"{key}_max_abs_mV"] = float(np.abs(d).max())
        sj, sn = pj["spikes"], pn["spikes"]
        out["spikes_jaxley_per_mn"] = sj.sum(axis=0).astype(int).tolist()
        out["spikes_neuron_per_mn"] = sn.sum(axis=0).astype(int).tolist()
        tot_j, tot_n = int(sj.sum()), int(sn.sum())
        matched = sum(
            _match_spikes(np.flatnonzero(sj[:, i]) * dt_ms * 1e-3,
                          np.flatnonzero(sn[:, i]) * dt_ms * 1e-3, 5e-3)
            for i in range(sj.shape[1]))
        out["spike_match_pm5ms"] = {
            "matched": matched, "jaxley_total": tot_j, "neuron_total": tot_n}
        return out

    rep = {"content_hash": res_j["content_hash"],
           "pool": diff(res_j["pool"], res_n["pool"]),
           "current_steps": {amp: diff(res_j["current_steps"][amp],
                                       res_n["current_steps"][amp])
                             for amp in res_j["current_steps"]}}
    return rep


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path,
                    default=RESULTS / "val_protocol.pkl")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--backend", choices=["jaxley", "neuron", "both"],
                    default="jaxley")
    ap.add_argument("--n-mn", type=int, default=8)
    ap.add_argument("--e-inh", type=float, default=-15.0)
    ap.add_argument("--w-exc", type=float, default=0.009,
                    help="0.009 = matched-input (subthreshold) diff; "
                         "0.15 = ex03-equivalent firing regime")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    art = None
    if args.artifact.exists() and not args.rebuild:
        art = load_protocol(args.artifact)
        if (art.get("version") != ARTIFACT_VERSION
                or art["synapse"]["w_exc_uS"] != args.w_exc
                or art["synapse"]["e_inh_mV"] != args.e_inh
                or art["n_mn"] != args.n_mn):
            art = None  # stale → rebuild
    if art is None:
        art = make_protocol(n_mn=args.n_mn, e_inh_mV=args.e_inh,
                            w_exc_uS=args.w_exc)
        save_protocol(art, args.artifact)
        print(f"built artifact v{ARTIFACT_VERSION} (hash "
              f"{art['content_hash']}, w_exc {args.w_exc} µS, e_inh "
              f"{args.e_inh} = {art['synapse']['e_inh_convention']})")
    else:
        print(f"loaded artifact {args.artifact} (hash {art['content_hash']})")

    results = {}
    if args.backend == "both":
        # JAX and NEURON cannot share a process (verified SIGSEGV) —
        # run each backend as its own subprocess, then compare the pkls.
        import subprocess
        import sys

        for backend in ("jaxley", "neuron"):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--backend", backend, "--artifact", str(args.artifact),
                   "--n-mn", str(args.n_mn), "--e-inh", str(args.e_inh),
                   "--w-exc", str(args.w_exc)]
            print(f"--- subprocess: {backend} ---", flush=True)
            proc = subprocess.run(cmd, cwd=Path(__file__).parent,
                                  capture_output=True, text=True)
            for line in (proc.stdout + proc.stderr).splitlines():
                if not any(s in line for s in ("DISPLAY", "graphics", "Added")):
                    print("   ", line)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{backend} subprocess failed (exit {proc.returncode})")
            results[backend] = joblib.load(
                RESULTS / f"val_{backend}_{art['content_hash']}.pkl")
    elif args.backend in ("jaxley", "neuron"):
        runner = run_jaxley if args.backend == "jaxley" else run_neuron
        t0 = time.time()
        res = runner(art)
        results[args.backend] = res
        sp = res["pool"]["spikes"]
        print(f"[{args.backend}] pool: {int((sp.sum(axis=0) > 0).sum())}/"
              f"{art['n_mn']} active, {int(sp.sum()):,} spikes "
              f"({time.time()-t0:.0f}s)", flush=True)
        for amp, o in res["current_steps"].items():
            print(f"    step {amp:4.1f} nA → spikes/MN "
                  f"{np.asarray(o['spikes']).sum(axis=0).astype(int)}",
                  flush=True)
        joblib.dump(res, RESULTS / f"val_{args.backend}_{art['content_hash']}.pkl")

    if args.backend == "both":
        rep = compare_backends(results["jaxley"], results["neuron"],
                               art["dt_ms"])
        # manifest import deferred: sensitivity_checks pulls jax, which is
        # safe in THIS parent process (it never imports neuron)
        from sensitivity_checks import _manifest

        payload = json.dumps(
            {"manifest": {
                **_manifest(),
                "validation_protocol_sha256_16": hashlib.sha256(
                    Path(__file__).read_bytes()).hexdigest()[:16],
                "args": vars(args),
             },
             "artifact": {k: art[k] for k in
                          ("version", "seed", "dt_ms", "n_mn", "synapse",
                           "cell", "content_hash")},
             "comparison": rep}, indent=1, default=float)
        out = RESULTS / f"val_comparison_{art['content_hash']}.json"
        out.write_text(payload)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (EVIDENCE / f"val_comparison_{art['content_hash']}_{stamp}.json"
         ).write_text(payload)
        print("\n=== NEURON ↔ Jaxley paired comparison ===")
        p = rep["pool"]
        print(f"pool phase: soma V rmse {p['v_rmse_mV']:.3f} mV "
              f"(max {p['v_max_abs_mV']:.2f}); dend rmse "
              f"{p['v_dend_rmse_mV']:.3f} mV")
        print(f"  spikes: jaxley {p['spike_match_pm5ms']['jaxley_total']} vs "
              f"neuron {p['spike_match_pm5ms']['neuron_total']}, matched "
              f"±5 ms: {p['spike_match_pm5ms']['matched']}")
        for amp, d in rep["current_steps"].items():
            print(f"step {amp:4.1f} nA: soma rmse {d['v_rmse_mV']:7.3f} mV | "
                  f"spikes j{d['spikes_jaxley_per_mn']} vs "
                  f"n{d['spikes_neuron_per_mn']}")
        print(f"saved → {out} (+ evidence copy)")


if __name__ == "__main__":
    main()

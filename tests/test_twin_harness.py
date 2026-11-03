"""Tests for the digital-twin harness (experiments/twin/).

Covers the pure-function layer (protocols, observables incl. the ΔF
estimator on synthetic spike trains with known ground truth) and the
PoolTwin build invariants (seed determinism, synapse-spec validation,
PIC-parameter scaling masks). Simulation behaviour itself is exercised by
the phase scripts, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TWIN = Path(__file__).resolve().parents[1] / "experiments" / "twin"
sys.path.insert(0, str(TWIN))

import protocols  # noqa: E402
import observables as obs  # noqa: E402
from poolbuild import PoolTwin, TwinSpec, fuglevand_thresholds  # noqa: E402


# --------------------------------------------------------------------------- #
# protocols / thresholds
# --------------------------------------------------------------------------- #

def test_fuglevand_thresholds_monotonic_normalized():
    rt = fuglevand_thresholds(25, recruitment_range=30.0)
    assert rt.shape == (25,)
    assert np.all(np.diff(rt) > 0)
    assert np.isclose(rt[-1], 1.0)
    assert np.isclose(rt[-1] / rt[0], 30.0, rtol=0.2)


def test_triangular_ramp_shape_and_peak():
    dt = 0.5
    t, cmd = protocols.triangular_ramp(dt, t_rise_s=2.0, t_pre_s=1.0,
                                       t_post_s=1.0, peak=40.0)
    assert len(t) == len(cmd)
    assert cmd.dtype == np.float32
    assert np.isclose(cmd.max(), 40.0, atol=0.5)
    i_peak = int(np.argmax(cmd))
    assert abs(t[i_peak] - 3.0) < 0.05  # pre 1 s + rise 2 s
    assert cmd[0] == 0.0 and cmd[-1] == 0.0


def test_trapezoid_plateau():
    dt = 0.5
    _, cmd = protocols.trapezoid(dt, t_rise_s=1.0, t_hold_s=2.0, t_fall_s=1.0,
                                 t_pre_s=0.5, t_post_s=0.5, peak=10.0)
    n_plateau = int(np.sum(cmd >= 10.0 - 1e-6))
    assert abs(n_plateau - int(2.0 * 1000 / dt)) <= 2


# --------------------------------------------------------------------------- #
# ΔF estimator on synthetic data with known ground truth
# --------------------------------------------------------------------------- #

def _spikes_from_rate(rate_fn, t_on, t_off, duration_s, dt_ms):
    """Deterministic spike train: integrate instantaneous rate to spike times."""
    n = int(duration_s * 1000 / dt_ms)
    spikes = np.zeros(n, dtype=np.float32)
    acc, t = 0.0, t_on
    dt_s = dt_ms * 1e-3
    for i in range(int(t_on / dt_s), min(int(t_off / dt_s), n)):
        acc += rate_fn(i * dt_s) * dt_s
        if acc >= 1.0:
            spikes[i] = 1.0
            acc -= 1.0
    return spikes


def _synthetic_ramp_pool(extra_control: bool = False):
    """Control MN(s) tracking a triangular drive; test MN with hysteresis.

    Drive: 0→20 Hz at t=15 s→0 at t=30 s. Control rate = drive (min 4 Hz
    while firing, 2–28 s). Test MN fires 10–24 s (recruited while drive
    rising at 13.3 Hz, derecruited while falling at 8 Hz).
    True ΔF = f_ctrl(10 s) − f_ctrl(24 s) = 13.3 − 8 = 5.3 Hz.
    """
    dt_ms, dur = 1.0, 30.0
    drive = lambda t: 20.0 * (t / 15.0 if t <= 15.0 else max(0.0, 2.0 - t / 15.0))
    ctrl_rate = lambda t: max(4.0, drive(t))
    cols = [_spikes_from_rate(ctrl_rate, 2.0, 28.0, dur, dt_ms)]
    if extra_control:
        cols.append(_spikes_from_rate(lambda t: max(4.0, 0.9 * drive(t)),
                                      2.5, 27.5, dur, dt_ms))
    cols.append(_spikes_from_rate(lambda t: max(3.0, 0.8 * drive(t)),
                                  10.0, 24.0, dur, dt_ms))
    return np.stack(cols, axis=1), dt_ms


def test_delta_f_recovers_known_value():
    spikes, dt_ms = _synthetic_ramp_pool()
    res = obs.delta_f_pairs(spikes, dt_ms)
    assert len(res.delta_f) == 1
    assert len(res.delta_f_per_test) == 1
    # true ΔF ≈ 5.3 Hz; allow smoothing/discretization slack
    assert res.mean == pytest.approx(5.3, abs=1.5)
    assert res.mean > 0


def test_delta_f_per_test_averaging():
    spikes, dt_ms = _synthetic_ramp_pool(extra_control=True)
    res = obs.delta_f_pairs(spikes, dt_ms)
    assert len(res.delta_f) == 2  # two controls, one test unit
    assert len(res.delta_f_per_test) == 1  # averaged within the test unit
    assert res.mean == pytest.approx(res.delta_f.mean(), abs=1e-9)


def test_delta_f_rejects_short_outlast_control():
    # Control derecruited only 0.5 s after the test MN → must be rejected
    # by the min_control_outlast_s=1.5 safeguard.
    dt_ms, dur = 1.0, 30.0
    drive = lambda t: 20.0 * (t / 15.0 if t <= 15.0 else max(0.0, 2.0 - t / 15.0))
    ctrl = _spikes_from_rate(lambda t: max(4.0, drive(t)), 2.0, 24.5, dur, dt_ms)
    test = _spikes_from_rate(lambda t: max(3.0, 0.8 * drive(t)), 10.0, 24.0,
                             dur, dt_ms)
    res = obs.delta_f_pairs(np.stack([ctrl, test], axis=1), dt_ms)
    assert len(res.delta_f) == 0


def test_recruitment_order_matches_onsets():
    dt_ms, dur = 1.0, 10.0
    a = _spikes_from_rate(lambda t: 10.0, 1.0, 9.0, dur, dt_ms)
    b = _spikes_from_rate(lambda t: 10.0, 3.0, 9.0, dur, dt_ms)
    order = obs.recruitment_order(np.stack([b, a], axis=1), dt_ms)
    assert list(order) == [1, 0]


# --------------------------------------------------------------------------- #
# PoolTwin build invariants (slow-ish: builds real jaxley cells)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_twin():
    return PoolTwin(TwinSpec(n_mn=3, n_dd=5, n_inh=5, seed=0))


def test_spec_validation_rejects_bad_options():
    with pytest.raises(ValueError, match="syn_site"):
        PoolTwin(TwinSpec(n_mn=2, syn_site="axon"))
    with pytest.raises(ValueError, match="syn_tau_rise_ms"):
        PoolTwin(TwinSpec(n_mn=2, syn_tau_rise_ms=10.0, tau_syn_ms=5.0))


def test_connectivity_seed_determinism(tiny_twin):
    same = PoolTwin(TwinSpec(n_mn=3, n_dd=5, n_inh=5, seed=0))
    other = PoolTwin(TwinSpec(n_mn=3, n_dd=5, n_inh=5, seed=1))
    assert np.array_equal(np.asarray(tiny_twin._exc_mat), np.asarray(same._exc_mat))
    assert not np.array_equal(np.asarray(tiny_twin._exc_mat),
                              np.asarray(other._exc_mat))


def test_delta_f_rejects_flat_test_profile():
    # Test unit firing at a CONSTANT rate while the control modulates: the
    # common-modulation check (flat profile or low/NaN correlation) must
    # reject the pair.
    dt_ms, dur = 1.0, 30.0
    drive = lambda t: 20.0 * (t / 15.0 if t <= 15.0 else max(0.0, 2.0 - t / 15.0))
    ctrl = _spikes_from_rate(lambda t: max(4.0, drive(t)), 2.0, 28.0, dur, dt_ms)
    test = _spikes_from_rate(lambda t: 8.0, 10.0, 24.0, dur, dt_ms)
    res = obs.delta_f_pairs(np.stack([ctrl, test], axis=1), dt_ms)
    assert len(res.delta_f) == 0


def test_pic_scaling_masks(tiny_twin):
    import jax.numpy as jnp

    theta = tiny_twin.default_theta()
    theta["gCaL_scale"] = jnp.float32(2.0)
    scaled = tiny_twin._scaled_jaxley_params(theta)
    base = np.asarray(tiny_twin._all_params["caL_gcaLbar"])
    out = np.asarray(scaled["caL_gcaLbar"])
    mask = np.asarray(tiny_twin._pic_masks["caL_gcaLbar"])
    assert mask.any()  # dendrite compartments carry caL
    assert np.allclose(out[mask], 2.0 * base[mask])
    # unmasked (sentinel/NaN) entries untouched
    untouched = ~mask & np.isfinite(base)
    assert np.allclose(out[untouched], base[untouched])


# --------------------------------------------------------------------------- #
# New synapse modes: Exp2Syn kernel, event API, placement, live driving force
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def exp2_twin():
    """Dendritic Exp2Syn twin, fixed driving force."""
    return PoolTwin(TwinSpec(n_mn=2, n_dd=3, n_inh=3, seed=0,
                             syn_site="dendrite", syn_tau_rise_ms=0.5))


@pytest.fixture(scope="module")
def exp2_live_twin():
    """Same as exp2_twin but with voltage-dependent driving force."""
    return PoolTwin(TwinSpec(n_mn=2, n_dd=3, n_inh=3, seed=0,
                             syn_site="dendrite", syn_tau_rise_ms=0.5,
                             live_driving_force=True))


def test_exp2syn_kernel_peak_time_and_decay(exp2_twin):
    import jax.numpy as jnp

    spec = exp2_twin.spec
    t1, t2 = spec.syn_tau_rise_ms, spec.tau_syn_ms
    syn = exp2_twin._syn_init()
    zero = jnp.zeros(spec.n_mn, dtype=jnp.float32)
    gs = []
    for i in range(400):  # 40 ms impulse response
        inc = jnp.full(spec.n_mn, 1.0 if i == 0 else 0.0, dtype=jnp.float32)
        syn, g_exc, _ = exp2_twin._syn_step(syn, inc, zero)
        gs.append(float(g_exc[0]))
    gs = np.array(gs)
    # peak-normalized: unit-weight event peaks at ~1.0 at the analytic t_peak
    tp = (t1 * t2 / (t2 - t1)) * np.log(t2 / t1)
    i_peak = int(np.argmax(gs))
    assert abs((i_peak + 1) * spec.dt_ms - tp) <= 2 * spec.dt_ms
    assert gs[i_peak] == pytest.approx(1.0, rel=0.02)
    # late decay is governed by tau_decay: one tau2 drops the value by e
    i0 = int(20.0 / spec.dt_ms)
    i1 = i0 + int(t2 / spec.dt_ms)
    assert gs[i1] / gs[i0] == pytest.approx(np.exp(-1.0), rel=0.05)


def test_events_api_current_signs(tiny_twin):
    theta = tiny_twin.default_theta()
    n_steps, n_mn = 600, tiny_twin.spec.n_mn
    conn = np.ones((n_mn, 1), dtype=np.float32)
    silent = np.zeros((n_steps, 1), dtype=np.float32)
    burst = silent.copy()
    burst[50:60, 0] = 1.0  # 10 events

    v_base = np.asarray(tiny_twin.simulate_events(
        theta, silent, exc_connectivity=conn)["v"])
    v_exc = np.asarray(tiny_twin.simulate_events(
        theta, burst, exc_connectivity=conn)["v"])
    v_inh = np.asarray(tiny_twin.simulate_events(
        theta, silent, inh_events=burst, exc_connectivity=conn,
        inh_connectivity=conn)["v"])

    d_exc = v_exc - v_base
    d_inh = v_inh - v_base
    assert d_exc.max() > 1e-4 and d_exc.min() > -1e-5  # pure depolarization
    assert d_inh.min() < -1e-5 and d_inh.max() < 1e-5  # pure hyperpolarization


def test_events_api_shape_validation(tiny_twin):
    theta = tiny_twin.default_theta()
    with pytest.raises(ValueError, match="exc_connectivity"):
        tiny_twin.simulate_events(theta, np.zeros((10, 2), dtype=np.float32))


def test_stimulus_placement_differs_between_sites(tiny_twin, exp2_twin):
    a = np.asarray(tiny_twin._external_inds["i"]).ravel()[: tiny_twin.spec.n_mn]
    b = np.asarray(exp2_twin._external_inds["i"]).ravel()[: exp2_twin.spec.n_mn]
    assert not np.array_equal(a[: len(b)], b)


def test_live_driving_force_changes_dynamics(exp2_twin, exp2_live_twin):
    theta = exp2_twin.default_theta()
    n_steps = 600
    conn = np.ones((exp2_twin.spec.n_mn, 1), dtype=np.float32)
    burst = np.zeros((n_steps, 1), dtype=np.float32)
    burst[50:80, 0] = 1.0
    v_fixed = np.asarray(exp2_twin.simulate_events(
        theta, burst, exc_connectivity=conn)["v"])
    v_live = np.asarray(exp2_live_twin.simulate_events(
        theta, burst, exc_connectivity=conn)["v"])
    assert np.max(np.abs(v_fixed - v_live)) > 1e-6
    # live DF weakens depolarizing current as V rises → smaller peak response
    assert v_live.max() <= v_fixed.max() + 1e-6


def test_hard_surrogate_forward_equivalence_new_modes(exp2_live_twin):
    import jax
    import jax.numpy as jnp

    _, cmd = protocols.step(exp2_live_twin.spec.dt_ms, t_pre_s=0.05,
                            t_on_s=0.3, t_post_s=0.05, level=60.0)
    drives = {"dd": jnp.asarray(cmd),
              "inh": jnp.zeros(len(cmd), dtype=jnp.float32)}
    theta = exp2_live_twin.default_theta()
    key = jax.random.PRNGKey(0)
    hard = exp2_live_twin.simulate(theta, drives, key, spike_mode="hard")
    sur = exp2_live_twin.simulate(theta, drives, key, spike_mode="surrogate")
    assert np.allclose(np.asarray(hard["v"]), np.asarray(sur["v"]))
    assert np.allclose(np.asarray(hard["spikes"]), np.asarray(sur["spikes"]))


# --------------------------------------------------------------------------- #
# Sensitivity-diagnostic regression tests (Deliverable A report semantics)
# --------------------------------------------------------------------------- #

def test_safe_norm_no_float32_overflow():
    import sensitivity_checks as sc

    g = np.array([1.4e28, 4.7e28], dtype=np.float32)  # the T4 1 s components
    n = sc._safe_norm(g)
    assert np.isfinite(n) and n > 1e28  # float32 norm would report inf


def test_manifest_schema():
    import sensitivity_checks as sc

    m = sc._manifest()
    assert {"git_revision", "versions", "device", "precision",
            "seed_taxonomy", "objectives", "scope"} <= set(m)


def test_t5_reports_joint_and_coordinate_separately():
    import sensitivity_checks as sc

    # 2.0 s fixture produces a FIRING observation (≈8 spikes), so the test
    # exercises coordinate semantics, not just the report schema.
    res = sc.t5_hard_forward_descent(
        n_mn=2, duration_s=2.0, conn_seeds=(0,), event_seeds=(0,),
        true_points=((1.5, 0.011),), lambdas=(0.1,))
    assert {"accepted_step_fraction", "coordinate_agreement", "cases",
            "n_failures", "n_evaluated",
            "n_independent_event_realizations"} <= set(res)
    assert res["n_evaluated"] >= 1  # non-vacuous: at least one case evaluated
    for name in ("gCaL_scale", "w_exc"):
        assert {"n_resolved", "n_unresolved", "agreement_rate"} <= set(
            res["coordinate_agreement"][name])
    n_probed = 0
    for c in res["cases"]:
        if "joint_steps" in c:  # evaluated case → both result kinds present
            assert "coordinate_probes" in c
            for probe in c["coordinate_probes"].values():
                n_probed += 1
                assert {"true_descent_dir", "surrogate_dir", "agrees",
                        "hard_loss_baseline"} <= set(probe)
                # baseline-aware semantics: a resolved descent direction must
                # actually improve on the baseline loss, never be "less bad"
                if probe["agrees"] is not None:
                    assert probe["true_descent_dir"] != 0
                    assert (min(probe["hard_loss_up"], probe["hard_loss_down"])
                            < probe["hard_loss_baseline"])
    assert n_probed >= 2  # both coordinates probed in the evaluated case


# --------------------------------------------------------------------------- #
# Deliverable B regression tests
# --------------------------------------------------------------------------- #

def test_limb_segmentation():
    import deliverable_b as db

    cmd = np.concatenate([np.zeros(10), np.linspace(0, 10, 20),
                          np.full(5, 10.0), np.linspace(10, 0, 20),
                          np.zeros(10)])
    limbs = db.segment_limbs(cmd)
    masks = np.stack([limbs[k] for k in ("pre", "rising", "peak",
                                         "falling", "post")])
    assert masks.sum(axis=0).max() == 1  # disjoint
    assert limbs["pre"][:9].all() and limbs["post"][-9:].all()
    assert limbs["peak"].sum() >= 5  # plateau included
    assert limbs["rising"].any() and limbs["falling"].any()
    i_peak = int(np.argmax(cmd))
    assert not limbs["rising"][i_peak:].any()
    assert not limbs["falling"][:i_peak].any()


def test_truth_fit_loss_is_zero_same_model():
    import jax.numpy as jnp
    import deliverable_b as db

    commands = db.make_commands(0.1, dur_scale=0.3)
    twin, events = db.build_twin_and_events(2, 0, 0, commands, 0.1)
    rates_fn = db.make_rates_fn(twin, events)
    theta = np.array([1.6, 0.0105])
    obs = {p: rates_fn(theta, p) for p in ("fit_ramp", "fit_trap")}
    tot = 0.0
    for p, o in obs.items():
        r = rates_fn(theta, p)
        tot += float(jnp.mean((r - o) ** 2))
    assert tot == 0.0  # identical model + identical frozen events


def test_event_matrix_hash_stable():
    import deliverable_b as db

    a = np.zeros((10, 3), dtype=np.float32)
    a[2, 1] = 1.0
    h1 = db._sha16(a)
    assert h1 == db._sha16(a.copy())
    b = a.copy()
    b[3, 0] = 1.0
    assert h1 != db._sha16(b)


def test_binned_rates_counts_and_jitter_invariance():
    import jax.numpy as jnp
    import deliverable_b as db

    sp = np.zeros((1000, 2), dtype=np.float32)  # 100 ms at dt 0.1 → 2 bins
    sp[10, 0] = 1.0
    sp[20, 0] = 1.0  # two spikes in bin 0 of unit 0
    r = np.asarray(db.binned_rates_jnp(jnp.asarray(sp), 0.1,
                                       bin_ms=50.0, smooth_ms=50.0))
    assert r.shape == (2, 2)
    assert r[0, 0] == 2 / 0.05  # 40 Hz
    assert r[1, 0] == 0.0 and r[:, 1].sum() == 0.0
    # spike-time jitter WITHIN a bin must not change the observable at all —
    # this is the property that kills the chaotic-decorrelation floor
    sp2 = np.zeros_like(sp)
    sp2[14, 0] = 1.0
    sp2[33, 0] = 1.0
    r2 = np.asarray(db.binned_rates_jnp(jnp.asarray(sp2), 0.1,
                                        bin_ms=50.0, smooth_ms=50.0))
    assert np.allclose(r, r2)


def test_axis_with_truth_contains_truth_sorted_unique():
    import deliverable_b as db

    ax = db.axis_with_truth(0.25, 4.0, 21, 1.6)
    assert 1.6 in ax
    assert np.all(np.diff(ax) > 0)
    assert len(ax) == len(np.unique(ax))
    # truth already on a grid point → no duplicate inserted
    ax2 = db.axis_with_truth(1.0, 4.0, 3, 2.0)
    assert 2.0 in ax2 and len(ax2) == len(np.unique(ax2))


def test_run_tag_separates_smoke_from_full():
    import argparse
    import deliverable_b as db

    full = argparse.Namespace(dense_n=21, replicates=3, dur_scale=1.0, n_mn=8)
    smoke = argparse.Namespace(dense_n=5, replicates=3, dur_scale=1.0, n_mn=8)
    assert db.run_tag(full) == "full"
    assert db.run_tag(smoke) == "smoke"


# --------------------------------------------------------------------------- #
# Deliverable C nesting + validation-pipeline regression tests
# --------------------------------------------------------------------------- #

def test_nested_profiles_nesting_property():
    import deliverable_c as dc

    g_axis = np.geomspace(0.5, 2.0, 9)
    b_axis = np.geomspace(0.005, 0.02, 9)

    def losses_at(g, b):
        # protocols preferring DIFFERENT amplitudes → unknown < known strictly
        l1 = (b - 0.008) ** 2 * 1e4 + (g - 1.0) ** 2
        l2 = (b - 0.015) ** 2 * 1e4 + (g - 1.0) ** 2
        return (l1, l2)

    known, unknown = dc.compute_nested_profiles(losses_at, g_axis, b_axis, 2)
    assert np.all(unknown <= known + dc.NESTING_TOL)
    assert np.any(unknown < known - 1e-6)  # strictly better somewhere

    def losses_same(g, b):
        v = (b - 0.01) ** 2 * 1e4 + (g - 1.0) ** 2
        return (v, v)

    k2, u2 = dc.compute_nested_profiles(losses_same, g_axis, b_axis, 2)
    # identical protocols → releasing amplitudes gains nothing
    assert np.allclose(k2, u2, atol=dc.NESTING_TOL)


def test_match_spikes_greedy():
    from validation_protocol import _match_spikes

    a = np.array([0.1, 0.2, 0.3])
    assert _match_spikes(a, a.copy(), 1e-3) == 3
    b = a + 0.004
    assert _match_spikes(a, b, 5e-3) == 3
    assert _match_spikes(a, b, 1e-3) == 0
    # greedy one-to-one: two a-spikes cannot share one b-spike
    assert _match_spikes(np.array([0.1, 0.101]), np.array([0.1]), 5e-3) == 1


def test_collect_alignment_semantics():
    from validation_protocol import _collect

    class FakeVec:
        def __init__(self, arr):
            self._a = np.asarray(arr, dtype=float)

        def as_numpy(self):
            return self._a

    n_steps, dt = 5, 0.1
    # NEURON records n_steps+1 samples; sample 0 is the initial condition and
    # must be dropped so neuron[k] aligns with jaxley's post-step sample k
    vrec = [FakeVec([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])]
    dvrec = [FakeVec([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])]
    spk = [FakeVec([0.2, 0.41])]  # ms → steps 2 and 4
    out = _collect(vrec, dvrec, spk, dt, n_steps, 1)
    assert out["v"].shape == (n_steps, 1)
    assert out["v"][0, 0] == 1.0  # initial condition dropped
    assert out["spikes"][2, 0] == 1.0 and out["spikes"][4, 0] == 1.0
    assert out["spikes"].sum() == 2


def test_run_neuron_subprocess_smoke():
    # run_neuron must live in a jax-free process (JAX+NEURON segfault when
    # mixed); this also regression-tests the single-pool phase structure.
    import subprocess
    import sys

    code = (
        "import validation_protocol as vp\n"
        "art = vp.make_protocol(n_mn=2, t_rise_s=0.4, t_pre_s=0.1,"
        " t_post_s=0.2, step_currents_nA=(15.0,))\n"
        "res = vp.run_neuron(art)\n"
        "print('STEP_SPIKES', int(res['current_steps'][15.0]['spikes'].sum()))\n"
        "print('POOL_SHAPE', res['pool']['v'].shape[0],"
        " res['pool']['v'].shape[1])\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(TWIN),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    n_spk = int(out.split("STEP_SPIKES")[1].split()[0])
    assert n_spk > 10  # 15 nA drives the small cells
    # pool phase length = (0.1 pre + 2×0.4 ramp + 0.2 post) s / 0.1 ms
    assert "POOL_SHAPE 11000 2" in out


# --------------------------------------------------------------------------- #
# Gradient-repair diagnostic helpers (G1-G3 v2)
# --------------------------------------------------------------------------- #

def test_fd_sweep_finds_convergence_plateau():
    import gradient_repair as gr

    f = lambda s: float(s[0] ** 2 + 3 * s[1])
    _, c0 = gr.fd_sweep_converged(f, np.array([1.0, 1.0]), 0)
    _, c1 = gr.fd_sweep_converged(f, np.array([1.0, 1.0]), 1)
    assert c0 == pytest.approx(2.0, rel=1e-5)
    assert c1 == pytest.approx(3.0, rel=1e-5)


def test_fd_sweep_reports_no_plateau_on_discontinuity():
    import gradient_repair as gr

    step = lambda s: 1.0 if s[0] > 1.0 else 0.0  # FD slope ∝ 1/h, no plateau
    _, c = gr.fd_sweep_converged(step, np.array([1.0]), 0)
    assert c is None


def test_classify_direction_zero_is_unresolved_not_wrong():
    import gradient_repair as gr

    assert gr.classify_direction([1.0, -1.0, 1.0], [2.0, 3.0, 0.0]) == \
        ["agree", "wrong", "unresolved"]


def test_interp_time_gradient_matches_analytic():
    import jax
    import jax.numpy as jnp
    import gradient_repair as gr

    ramp = jnp.arange(10.0)
    k, c = 4, 4.5

    def ts(theta):  # v(θ) = θ·t ⇒ crossing time c/θ ms ⇒ d/dθ = −c/θ² ·1e-3 s
        return gr.interp_time(theta * ramp, k, c, 1.0)

    g = float(jax.grad(ts)(1.0))
    assert g == pytest.approx(-c * 1e-3, rel=1e-6)


def test_manifest_precision_and_script_hashes_honest():
    import jax
    from sensitivity_checks import _manifest

    m = _manifest()
    if jax.config.jax_enable_x64:
        assert "float64" in m["precision"]
    else:
        assert "float32" in m["precision"]
    assert "gradient_repair.py" in m["source_sha256_16"]
    assert "deliverable_b.py" in m["source_sha256_16"]


def test_zero_insert_events_preserves_physical_times():
    import gradient_repair as gr

    ev = np.zeros((5, 2), dtype=np.float32)
    ev[2, 0] = 1.0
    ev[4, 1] = 1.0
    fine = gr.zero_insert_events(ev, 4)
    assert fine.shape == (20, 2)
    assert fine[8, 0] == 1.0 and fine[16, 1] == 1.0  # k·factor
    assert fine.sum() == ev.sum()


def test_detect_plateau_is_ad_independent():
    import gradient_repair as gr

    rows = [(1e-1, 5.0), (1e-2, 2.2), (1e-3, 2.01), (1e-4, 2.0)]
    assert gr.detect_plateau(rows) == pytest.approx(2.0)
    assert gr.detect_plateau([(1e-1, 1.0), (1e-2, 10.0), (1e-3, 100.0)]) is None


def test_chained_windows_equal_uninterrupted_run():
    import jax.numpy as jnp
    import g4_descent as g4

    twin = PoolTwin(TwinSpec(n_mn=2, n_dd=5, n_inh=2, seed=0))
    n = 4000
    rng = np.random.default_rng(0)
    ev = {"exc": jnp.asarray((rng.random((n, 5)) < 0.002).astype(np.float64)),
          "inh": jnp.zeros((n, 2)),
          "exc_conn": jnp.ones((2, 5)), "inh_conn": jnp.zeros((2, 2))}
    th = twin.default_theta()
    full = np.asarray(twin.simulate_events(
        th, ev["exc"], inh_events=np.asarray(ev["inh"]),
        exc_connectivity=np.asarray(ev["exc_conn"]),
        inh_connectivity=np.asarray(ev["inh_conn"]),
        spike_mode="hard")["spikes"])
    wins = g4.split_windows(ev, 4)
    chained = np.asarray(g4.chained_spikes(twin, th, wins, "hard"))
    # state-preserving windowing must change NOTHING about the hard forward
    assert chained.shape == full.shape
    assert np.array_equal(chained, full)


def test_g4_case_events_are_float64_under_x64():
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    import g4_descent as g4

    _twin, ev = g4.build_case(0, 0, dur_scale=0.25)
    assert str(ev["fit_ramp"]["exc"].dtype) == "float64"
    assert str(ev["fit_ramp"]["exc_conn"].dtype) == "float64"

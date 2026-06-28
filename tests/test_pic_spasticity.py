import sys
from pathlib import Path

import numpy as np
import pytest

# _pic_protocols lives next to the clinical example; make it importable
_CLINICAL = Path(__file__).resolve().parents[1] / "examples" / "04_clinical"
sys.path.insert(0, str(_CLINICAL))

import _pic_protocols as pic  # noqa: E402

NAP_CEILING = 0.00215  # S/cm^2 — type-S baseline (~0.00043) x5, the verified-safe max


def test_nap_scaling_caps_largest_cell_and_keeps_it_spiking():
    """A uniform NaP x5 must not push the largest (type-F) cell past the safe
    ceiling, and that cell must still fire (not depolarization-block)."""
    pool = pic.build_single_cell_pool(gamma=1.5)
    baselines = [seg.gnapbar_napp for cell in pool for seg in cell.soma]
    pic.scale_nap(pool, factor=5.0, ceiling=NAP_CEILING)
    scaled = [seg.gnapbar_napp for cell in pool for seg in cell.soma]
    assert max(scaled) <= NAP_CEILING + 1e-9
    for b, s in zip(baselines, scaled):
        assert s == pytest.approx(min(b * 5.0, NAP_CEILING), rel=1e-6)
    # 6 nA dendritic drive: the large type-F dendrite needs more current than
    # the type-S cell to reach repetitive firing (4 nA only elicits a single
    # spike on this cell — it is under-driven, not depol-blocked; higher NaP
    # monotonically increases its firing, confirming the ceiling is safe).
    n = pic.count_spikes_under_step(pool[-1], pool, amp_nA=6.0, dur_ms=500.0)
    assert n >= 3, f"largest cell blocked (only {n} spikes)"


def test_ramp_hysteresis_control_vs_sci():
    """Control: symmetric recruit/derecruit (dI ~ 0). SCI (gamma+NaP):
    positive hysteresis (I_off << I_on) = self-sustained firing."""
    ctrl = pic.ramp_hysteresis(gamma=0.2, nap_factor=1.0, imax_nA=12.0)
    sci = pic.ramp_hysteresis(gamma=1.5, nap_factor=5.0, imax_nA=12.0,
                              nap_ceiling=NAP_CEILING)
    assert abs(ctrl["i_on"] - ctrl["i_off"]) < 0.5            # ~symmetric
    # SCI hysteresis is large and reproducible (dI ~ 0.92 nA: I_on ~ 1.02,
    # I_off ~ 0.10); an order of magnitude above the symmetric control (~0.09).
    assert sci["i_on"] - sci["i_off"] > 0.8                   # strong hysteresis
    assert sci["i_on"] < ctrl["i_on"]                         # amplification
    assert sci["i_off"] < 0.5                                 # firing persists to ~0


def test_after_discharge_control_vs_sci():
    """Brief dendritic pulse then input -> hold. Control: firing stops at offset.
    SCI: firing persists after offset (self-sustained / after-discharge)."""
    ctrl = pic.after_discharge(gamma=0.2, nap_factor=1.0)
    sci = pic.after_discharge(gamma=1.5, nap_factor=5.0, nap_ceiling=NAP_CEILING)
    assert ctrl["n_after"] <= 1, "control should not self-sustain"
    assert sci["n_after"] >= 5, f"SCI should show after-discharge (got {sci['n_after']})"


@pytest.mark.parametrize("n_mu", [12])  # reduced scale for speed
def test_pool_spasm_sci_only(n_mu):
    """Same brief low-MVC command. Control: discharge stops after command.
    SCI: a subset keeps firing after command offset (spasm)."""
    from myogen import set_random_seed
    set_random_seed(42)
    command, t_off_s, total_s = pic.brief_command_drive(n_points=20000)
    ctrl = pic.run_pool(command, n_mu=n_mu, gamma=0.2, nap_factor=1.0,
                        total_s=total_s)
    sci = pic.run_pool(command, n_mu=n_mu, gamma=1.5, nap_factor=5.0,
                       nap_ceiling=NAP_CEILING, total_s=total_s)
    ctrl_after = pic.spikes_after(ctrl, t_off_s)
    sci_after = pic.spikes_after(sci, t_off_s)
    assert sci_after > ctrl_after
    assert sci_after >= 10, f"expected a sustained discharge in SCI (got {sci_after})"


def test_iemg_synthesis_returns_signal_and_reports_tail_ratio(tmp_path):
    from myogen import set_random_seed
    set_random_seed(42)
    # Validates synthesis plumbing + a non-vacuous tail window. total_s (5 s)
    # outlasts the command offset (~3.5 s) so the tail RMS is computed over real
    # post-command signal, not an empty slice.
    command, t_off_s, total_s = pic.brief_command_drive(total_s=5.0)
    sci = pic.run_pool(command, n_mu=12, gamma=1.5, nap_factor=5.0,
                       nap_ceiling=NAP_CEILING, total_s=total_s)
    out = pic.synthesize_iemg(sci, n_mu=12, snr_dB=None, t_off_s=t_off_s)  # noiseless
    assert out["iemg"].shape[0] > 0
    assert np.any(out["times"] > t_off_s), "tail window must be non-empty"
    assert np.isfinite(out["tail_ratio"]) and out["tail_ratio"] >= 0.0


def test_single_cell_ca_current_manuscript_regime():
    """Manuscript regression: at the SCI regime (gamma=1.3 = +160% from the 0.5
    baseline, nap_factor=5) the single-cell SUSTAINED L-type Ca current is ~10 nA,
    within the 5-15 nA range cited for spinal MN PICs (paper_nature.tex SCI
    section). The manuscript claims the *sustained* current, so we measure the
    plateau over the self-sustained window (after the pulse, before inhibition),
    not the transient peak. Guards the ~10 nA claim against model drift."""
    mech = pic.single_cell_pic_mechanism(gamma=1.3, nap_factor=5.0)
    t = np.asarray(mech["t"])
    ica_nA = -np.asarray(mech["pic_nA"])                 # inward L-type Ca current
    lo, hi = mech["t_pulse"][1], mech["t_inhib"][0]      # self-sustained plateau
    sustained = float(np.median(ica_nA[(t >= lo) & (t < hi)]))
    assert 5.0 <= sustained <= 15.0, f"sustained Ca {sustained:.1f} nA outside 5-15 nA"
    assert 8.0 <= sustained <= 13.0, (
        f"sustained Ca {sustained:.1f} nA drifted from the manuscript ~10 nA")


def test_manuscript_regime_pool_self_sustains_vs_baseline():
    """Manuscript regression at the EXACT SCI parameterization (gamma 0.5->1.3,
    nap_factor 1->5), distinct from the stronger 0.2/1.5 stress test above. With
    a truly-silent post-offset command, the baseline PIC stops firing while the
    up-regulated PIC self-sustains the discharge -- the paper's core claim.
    Reduced scale (n_mu=12, total_s=6) for CI speed; exact CV2 figures are
    regenerated by 01_sci_iemg_mechanistic.py, not pinned here."""
    from myogen import set_random_seed
    set_random_seed(42)            # seed BEFORE building the drive (deterministic command)
    command, t_off_s, total_s = pic.brief_command_drive(total_s=6.0, n_points=12000)
    set_random_seed(42)            # reseed -> identical substrate for both pools
    baseline = pic.run_pool(command, n_mu=12, gamma=0.5, nap_factor=1.0,
                            total_s=total_s)
    set_random_seed(42)
    sci = pic.run_pool(command, n_mu=12, gamma=1.3, nap_factor=5.0,
                       nap_ceiling=NAP_CEILING, total_s=total_s)
    base_after = pic.spikes_after(baseline, t_off_s)
    sci_after = pic.spikes_after(sci, t_off_s)
    assert base_after <= 1, (
        f"baseline PIC must NOT self-sustain after offset (got {base_after})")
    assert sci_after > base_after, (
        f"SCI ({sci_after}) should out-fire baseline ({base_after}) post-offset")
    assert sci_after >= 10, f"up-regulated PIC should self-sustain (got {sci_after})"


def test_cyclic_voluntary_drive_returns_to_zero_at_troughs():
    """The shared command must return to ~0 at troughs and reach peak at peaks
    (so 'loss of derecruitment' in SCI is emergent, not built into the input)."""
    sig, total_s = pic.cyclic_voluntary_drive(peak_pps=45.0, freq_hz=0.5,
                                              total_s=8.0, n_points=8000)
    mag = np.asarray(sig.magnitude).ravel()
    t = np.linspace(0.0, total_s, len(mag), endpoint=False)
    trough = mag[(t > 1.9) & (t < 2.1)].mean()   # trough at t=2 s
    peak = mag[(t > 0.9) & (t < 1.1)].mean()      # peak at t=1 s
    assert total_s == pytest.approx(8.0)
    assert trough < 5.0, f"trough not ~0 (got {trough:.1f})"
    assert peak > 38.0, f"peak not ~45 (got {peak:.1f})"

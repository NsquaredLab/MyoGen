"""Reusable PIC protocol functions for the clinical spasticity example.

NERLab alpha-MN model. Single-cell drive is injected into dend[0] (where the
caL Ca PIC lives); somatic NaP (gnapbar_napp) is scaled post-construction.
NERLab rests at ~0 mV; spikes are detected relative to the relaxed vhold.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from neuron import h

from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.simulator.neuron.populations import AlphaMN__Pool

h.load_file("stdrun.hoc")
load_nmodl_mechanisms()

V_INIT, DT, CELSIUS = -67.0, 0.0125, 36.0

# import the known-good clamp helpers from example 10 (module name starts with a digit)
_EX10 = (Path(__file__).resolve().parents[1] / "01_basic"
         / "10_extract_neuron_parameters.py")
_spec = importlib.util.spec_from_file_location("_ex10", _EX10)
_ex10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ex10)
get_vhold__mV = _ex10.get_vhold__mV
fi0_multicompartment = _ex10.fi0_multicompartment

# a few low recruitment thresholds -> smallest is type-S (strongest PIC)
_SINGLE_RT = np.array([0.05, 0.3, 0.6])


def build_single_cell_pool(gamma: float) -> AlphaMN__Pool:
    """A small NERLab pool; pool[0] is the type-S cell, pool[-1] the largest."""
    return AlphaMN__Pool(
        recruitment_thresholds__array=_SINGLE_RT, model="NERLab", gamma=gamma
    )


def scale_nap(pool: AlphaMN__Pool, factor: float, ceiling: float) -> None:
    """Scale somatic gnapbar_napp by `factor`, capped at `ceiling` (S/cm^2)."""
    for cell in pool:
        for seg in cell.soma:
            seg.gnapbar_napp = min(seg.gnapbar_napp * factor, ceiling)


def _sections(cell):
    return [cell.soma] + list(cell.dend)


def count_spikes_under_step(cell, pool, amp_nA: float, dur_ms: float) -> int:
    """Inject a constant dendritic step into `cell` and count somatic spikes."""
    secs = _sections(cell)
    vhold = get_vhold__mV(cell, secs, [V_INIT] * len(secs))
    h.tstop = dur_ms
    h.dt = DT
    h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vhold] * len(secs)))
    stim = h.IClamp(cell.dend[0](0.5))
    stim.delay, stim.dur, stim.amp = 0, dur_ms, amp_nA
    v = h.Vector(); v.record(cell.soma(0.5)._ref_v)
    t = h.Vector(); t.record(h._ref_t)
    h.finitialize(); h.run()
    return _count_crossings(np.array(v), np.array(t), vhold + 20.0)


def _count_crossings(v, t, vthr, refr_ms=3.0):
    above = v > vthr
    cross = np.where((~above[:-1]) & (above[1:]))[0] + 1
    n, last = 0, -1e9
    for c in cross:
        if t[c] - last > refr_ms:
            n += 1; last = t[c]
    return n


def ramp_hysteresis(gamma, nap_factor=1.0, imax_nA=12.0,
                    t_up_ms=8000.0, t_down_ms=8000.0, nap_ceiling=0.00215):
    """Slow triangular dendritic current ramp (0 -> imax -> 0). Returns dict with
    soma Vm trace, injected current, and recruitment/derecruitment currents."""
    pool = build_single_cell_pool(gamma=gamma)
    if nap_factor != 1.0:
        scale_nap(pool, nap_factor, nap_ceiling)
    cell = pool[0]
    secs = _sections(cell)
    vhold = get_vhold__mV(cell, secs, [V_INIT] * len(secs))
    vthr = vhold + 20.0
    tstop = t_up_ms + t_down_ms
    h.tstop = tstop; h.dt = DT; h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vhold] * len(secs)))
    tpts = np.arange(0.0, tstop + 1.0, 1.0)
    iwave = imax_nA * np.minimum(np.clip(tpts / t_up_ms, 0, 1),
                                 np.clip((tstop - tpts) / t_down_ms, 0, 1))
    stim = h.IClamp(cell.dend[0](0.5)); stim.delay = 0; stim.dur = tstop
    ivec = h.Vector(iwave); tvec = h.Vector(tpts)
    ivec.play(stim._ref_amp, tvec, True)
    v = h.Vector(); v.record(cell.soma(0.5)._ref_v)
    t = h.Vector(); t.record(h._ref_t)
    h.finitialize(); h.run()
    v, t = np.array(v), np.array(t)
    icur = imax_nA * np.minimum(np.clip(t / t_up_ms, 0, 1),
                                np.clip((tstop - t) / t_down_ms, 0, 1))
    above = v > vthr
    sp = np.where((~above[:-1]) & (above[1:]))[0] + 1
    spk_t, spk_i = t[sp], icur[sp]
    half = tstop / 2.0
    i_on = float(spk_i[spk_t <= half].min()) if np.any(spk_t <= half) else np.nan
    i_off = float(spk_i[spk_t > half].min()) if np.any(spk_t > half) else np.nan
    return dict(t=t, v=v, icur=icur, vhold=vhold, i_on=i_on, i_off=i_off)


def after_discharge(gamma, nap_factor=1.0, hold_nA=0.6, pulse_nA=3.0,
                    hold_ms=1000.0, pulse_ms=600.0, tail_ms=2000.0,
                    nap_ceiling=0.00215):
    """Subthreshold dendritic hold, a brief suprathreshold pulse, then back to
    hold-only. Counts somatic spikes in the tail window (after pulse offset)."""
    pool = build_single_cell_pool(gamma=gamma)
    if nap_factor != 1.0:
        scale_nap(pool, nap_factor, nap_ceiling)
    cell = pool[0]
    secs = _sections(cell)
    vhold = get_vhold__mV(cell, secs, [V_INIT] * len(secs))
    vthr = vhold + 20.0
    tstop = hold_ms + pulse_ms + tail_ms
    h.tstop = tstop; h.dt = DT; h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vhold] * len(secs)))
    hold = h.IClamp(cell.dend[0](0.5)); hold.delay = 0; hold.dur = tstop; hold.amp = hold_nA
    pulse = h.IClamp(cell.dend[0](0.5)); pulse.delay = hold_ms; pulse.dur = pulse_ms
    pulse.amp = pulse_nA - hold_nA
    v = h.Vector(); v.record(cell.soma(0.5)._ref_v)
    t = h.Vector(); t.record(h._ref_t)
    h.finitialize(); h.run()
    v, t = np.array(v), np.array(t)
    above = v > vthr
    sp = np.where((~above[:-1]) & (above[1:]))[0] + 1
    spk_t = t[sp]
    offset = hold_ms + pulse_ms
    n_after = int(np.sum(spk_t > offset + 20.0))  # ignore the offset transient
    return dict(t=t, v=v, vhold=vhold, n_after=n_after, offset_ms=offset)


import itertools

import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain

from myogen import get_random_generator, simulator
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import DescendingDrive__Pool
from myogen.utils.types import pps

_POOL_TIMESTEP = 0.1 * pq.ms


def brief_command_drive(peak_pps=22.0, rise_s=1.0, plateau_s=1.5,
                        total_s=10.0, n_points=100000):
    """Low-MVC raised-cosine command: rise, brief plateau, fall to zero, then
    silence. Returns (AnalogSignal in pps, command_offset_s, total_s)."""
    t = np.linspace(0.0, total_s, n_points, endpoint=False)
    off = rise_s + plateau_s + rise_s
    cmd = np.zeros_like(t)
    up = t < rise_s
    cmd[up] = (peak_pps / 2.0) * (1.0 - np.cos(np.pi * t[up] / rise_s))
    plat = (t >= rise_s) & (t < rise_s + plateau_s)
    cmd[plat] = peak_pps
    down = (t >= rise_s + plateau_s) & (t < off)
    td = t[down] - (rise_s + plateau_s)
    cmd[down] = (peak_pps / 2.0) * (1.0 + np.cos(np.pi * td / rise_s))
    noise = np.clip(get_random_generator().normal(0, 1.0, size=n_points), 0, None)
    sig = AnalogSignal((cmd + noise) * pps,
                       sampling_period=(total_s / n_points) * pq.s)
    return sig, off, total_s


def run_pool(command, n_mu, gamma, nap_factor=1.0, nap_ceiling=0.00215,
             total_s=10.0):
    """Build a NERLab pool with PIC knobs, drive it with `command`, return the
    motor-neuron spike-train Block. Separate NEURON run per call."""
    rt, _ = simulator.RecruitmentThresholds(
        N=n_mu, recruitment_range__ratio=100, deluca__slope=5,
        konstantin__max_threshold__ratio=1.0, mode="combined")
    h.secondorder = 2
    mn_pool = AlphaMN__Pool(recruitment_thresholds__array=rt, model="NERLab",
                            gamma=gamma)
    if nap_factor != 1.0:
        scale_nap(mn_pool, nap_factor, nap_ceiling)
    dd_pool = DescendingDrive__Pool(n=100, poisson_batch_size=5,
                                    timestep__ms=_POOL_TIMESTEP)
    net = Network({"DD": dd_pool, "aMN": mn_pool})
    net.connect(source="DD", target="aMN", probability=0.5, weight__uS=0.15 * pq.uS)
    net.connect_from_external(source="cortical_input", target="DD",
                              weight__uS=1.0 * pq.uS)
    dd_netcons = net.get_netcons("cortical_input", "DD")
    sim_time = total_s * 1000.0 * pq.ms
    h.dt = _POOL_TIMESTEP
    h.tstop = sim_time
    recs = []
    for cell in mn_pool:
        rec = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(rec)
        recs.append((nc, rec))
    for section, voltage in itertools.chain.from_iterable(
        zip(*pool.get_initialization_data()) for pool in [mn_pool, dd_pool]):
        section.v = voltage
    h.finitialize()
    # Index the drive by simulation TIME (not step count), so the command plays
    # at the correct rate regardless of how finely it was sampled. (Indexing by
    # step is only correct when len(command) == tstop/dt exactly.)
    cmd_dt_ms = float(command.sampling_period.rescale(pq.ms).magnitude)
    while h.t < h.tstop:
        drive_val = command[min(int(h.t / cmd_dt_ms), len(command) - 1)]
        for dd_cell in dd_pool:
            if dd_cell.integrate(drive_val):
                st = h.t + 1
                if st < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(st)
        h.fadvance()
    block = Block(name=f"gamma={gamma},nap={nap_factor}")
    seg = Segment(name="Motor Neurons")
    seg.spiketrains = [
        SpikeTrain((rec.as_numpy() * pq.ms).rescale(pq.s),
                   t_stop=sim_time.rescale(pq.s),
                   sampling_rate=(1 / (h.dt * pq.ms)).rescale(pq.Hz),
                   name=f"MN_{i}")
        for i, (_, rec) in enumerate(recs)]
    block.segments.append(seg)
    return block


def spikes_after(block, t_off_s):
    """Total motor-neuron spikes occurring after `t_off_s` (the command offset)."""
    sts = block.segments[0].spiketrains
    return int(sum(np.sum(st.rescale(pq.s).magnitude > t_off_s) for st in sts))


def cyclic_voluntary_drive(peak_pps=45.0, freq_hz=0.5, total_s=8.0, n_points=None):
    """A repeating voluntary command (0 -> peak -> 0 each cycle), returning to
    zero at every trough. The SAME drive is used for healthy and SCI conditions;
    only the motoneuron PIC state differs. With a healthy PIC the pool derecruits
    at each trough; with an up-regulated PIC it stays firing (loss of
    derecruitment) -- the phenotype the old example imposed via a tonic floor.
    Returns (AnalogSignal in pps, total_s)."""
    if n_points is None:
        n_points = int(total_s * 1000.0 / float(_POOL_TIMESTEP.magnitude))
    t = np.linspace(0.0, total_s, n_points, endpoint=False)
    cmd = (peak_pps / 2.0) * (1.0 - np.cos(2.0 * np.pi * freq_hz * t))
    noise = np.clip(get_random_generator().normal(0, 1.0, size=n_points), 0, None)
    sig = AnalogSignal((cmd + noise) * pps,
                       sampling_period=(total_s / n_points) * pq.s)
    return sig, total_s


def population_rate(block, total_s, bin_s=0.1, smooth_bins=5):
    """Mean per-unit discharge rate (pps) over time across units that fire.
    Returns (bin_centers_s, rate_pps)."""
    sts = [st.rescale(pq.s).magnitude for st in block.segments[0].spiketrains
           if len(st) > 0]
    edges = np.arange(0.0, total_s + bin_s, bin_s)
    centers = edges[:-1] + bin_s / 2.0
    if not sts:
        return centers, np.zeros_like(centers)
    per_unit = np.stack([np.histogram(s, bins=edges)[0] / bin_s for s in sts])
    rate = per_unit.mean(axis=0)
    if smooth_bins > 1:
        rate = np.convolve(rate, np.ones(smooth_bins) / smooth_bins, mode="same")
    return centers, rate


def rate_in_windows(centers, rate, times, half_w=0.25):
    """Mean rate in +/- half_w windows around each time in `times` (pps)."""
    return float(np.mean([rate[(centers > x - half_w) & (centers < x + half_w)].mean()
                          for x in times]))


def make_iemg_simulator(muscle, n_mu):
    electrode = simulator.IntramuscularElectrodeArray(
        num_electrodes=4, inter_electrode_distance__mm=2.0 * pq.mm,
        differentiation_mode="consecutive",
        position__mm=(0.0 * pq.mm, 0.0 * pq.mm, 15.0 * pq.mm))
    sim = simulator.IntramuscularEMG(
        muscle_model=muscle, electrode_array=electrode,
        MUs_to_simulate=list(range(n_mu)))
    sim.simulate_muaps(n_jobs=2)
    return sim


def build_muscle(n_mu):
    rt, _ = simulator.RecruitmentThresholds(
        N=n_mu, recruitment_range__ratio=100, deluca__slope=5,
        konstantin__max_threshold__ratio=1.0, mode="combined")
    muscle = simulator.Muscle(
        recruitment_thresholds=rt, radius_bone__mm=1.0 * pq.mm,
        fiber_density__fibers_per_mm2=400 * pq.mm**-2,
        fat_thickness__mm=10 * pq.mm, autorun=True)
    return muscle, rt


def synthesize_iemg(spike_block, n_mu, iemg_sim=None, muscle=None,
                    snr_dB=None, noise_type="realistic", t_off_s=None):
    """Convolve spike trains with MUAPs into iEMG. If `snr_dB` is None, returns
    the noiseless signal (inspect amplitude before choosing SNR). Reports the
    tail/command-on RMS ratio when `t_off_s` is given."""
    if iemg_sim is None:
        if muscle is None:
            muscle, _ = build_muscle(n_mu)
        iemg_sim = make_iemg_simulator(muscle, n_mu)
    clean_block = iemg_sim.simulate_intramuscular_emg(spike_train__Block=spike_block)
    if snr_dB is None:
        sig = clean_block.segments[0].analogsignals[0]
    else:
        sig = iemg_sim.add_noise(snr__dB=snr_dB,
                                 noise_type=noise_type).segments[0].analogsignals[0]
    arr = np.asarray(sig.magnitude)[:, 0]
    times = np.asarray(sig.times.rescale(pq.s).magnitude)
    if t_off_s is None:
        tail_ratio = float("nan")
    else:
        on = arr[times <= t_off_s]; tail = arr[times > t_off_s]
        on_rms = np.sqrt(np.mean(on**2)) if on.size else 1.0
        tail_rms = np.sqrt(np.mean(tail**2)) if tail.size else 0.0
        tail_ratio = float(tail_rms / on_rms) if on_rms else 0.0
    return dict(iemg=arr, times=times, signal=sig, sim=iemg_sim,
                tail_ratio=tail_ratio)

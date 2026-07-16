"""Reusable PIC protocol functions for the clinical spasticity example.

NERLab alpha-MN model. Single-cell drive is injected into dend[0] (where the
caL Ca PIC lives); somatic NaP (gnapbar_napp) is scaled post-construction.
NERLab rests at ~0 mV; spikes are detected relative to the relaxed vhold.
"""
from __future__ import annotations

import itertools

import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Segment, SpikeTrain
from neuron import h

from myogen import get_random_generator, simulator
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import (
    AlphaMN__Pool,
    DescendingDrive__Pool,
)
from myogen.utils.nmodl import load_nmodl_mechanisms
from myogen.utils.types import pps

h.load_file("stdrun.hoc")
load_nmodl_mechanisms()

V_INIT, DT, CELSIUS = -67.0, 0.0125, 36.0
_POOL_TIMESTEP = 0.1 * pq.ms

# a few low recruitment thresholds -> smallest is type-S (strongest PIC)
_SINGLE_RT = np.array([0.05, 0.3, 0.6])


def fi0_multicompartment(sections, voltages) -> None:
    """Initialise the voltage of each compartment (for an FInitializeHandler)."""
    for sec, v in zip(sections, voltages):
        sec.v = v


def get_vhold__mV(cell, sections, voltages, tstop__ms=500.0) -> float:
    """Relax the cell to rest and return the steady-state somatic voltage (mV)."""
    h.tstop = tstop__ms
    h.dt = DT
    h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(sections, voltages))
    vsoma = h.Vector()
    vsoma.record(cell.soma(0.5)._ref_v)
    h.finitialize()
    h.run()
    return vsoma.to_python()[-1]


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
    v = h.Vector()
    v.record(cell.soma(0.5)._ref_v)
    t = h.Vector()
    t.record(h._ref_t)
    h.finitialize()
    h.run()
    return _count_crossings(np.array(v), np.array(t), vhold + 20.0)


def _count_crossings(v, t, vthr, refr_ms=3.0):
    above = v > vthr
    cross = np.where((~above[:-1]) & (above[1:]))[0] + 1
    n, last = 0, -1e9
    for c in cross:
        if t[c] - last > refr_ms:
            n += 1
            last = t[c]
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
    h.tstop = tstop
    h.dt = DT
    h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vhold] * len(secs)))
    tpts = np.arange(0.0, tstop + 1.0, 1.0)
    iwave = imax_nA * np.minimum(np.clip(tpts / t_up_ms, 0, 1),
                                 np.clip((tstop - tpts) / t_down_ms, 0, 1))
    stim = h.IClamp(cell.dend[0](0.5))
    stim.delay = 0
    stim.dur = tstop
    ivec = h.Vector(iwave)
    tvec = h.Vector(tpts)
    ivec.play(stim._ref_amp, tvec, True)
    v = h.Vector()
    v.record(cell.soma(0.5)._ref_v)
    t = h.Vector()
    t.record(h._ref_t)
    h.finitialize()
    h.run()
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
    h.tstop = tstop
    h.dt = DT
    h.celsius = CELSIUS
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vhold] * len(secs)))
    hold = h.IClamp(cell.dend[0](0.5))
    hold.delay = 0
    hold.dur = tstop
    hold.amp = hold_nA
    pulse = h.IClamp(cell.dend[0](0.5))
    pulse.delay = hold_ms
    pulse.dur = pulse_ms
    pulse.amp = pulse_nA - hold_nA
    v = h.Vector()
    v.record(cell.soma(0.5)._ref_v)
    t = h.Vector()
    t.record(h._ref_t)
    h.finitialize()
    h.run()
    v, t = np.array(v), np.array(t)
    above = v > vthr
    sp = np.where((~above[:-1]) & (above[1:]))[0] + 1
    spk_t = t[sp]
    offset = hold_ms + pulse_ms
    n_after = int(np.sum(spk_t > offset + 20.0))  # ignore the offset transient
    return dict(t=t, v=v, vhold=vhold, n_after=n_after, offset_ms=offset)


def single_cell_pic_mechanism(gamma=1.2, nap_factor=5.0, hold_nA=0.3,
                              pulse_nA=3.0, inhib_nA=-3.0, total_ms=5500.0,
                              t_pulse=(1000.0, 1500.0),
                              t_inhib=(3800.0, 4700.0)):
    """Single cell in the bistable PIC regime, recording soma Vm and the summed
    dendritic Ca PIC current (nA): subthreshold hold -> brief excitatory pulse
    latches the regenerative Ca plateau (self-sustained firing) -> a gentle
    GLOBAL inhibition deactivates the plateau (firing stops, the cell returns to
    the silent branch). This is the cell-level bistability that drives the pool
    spasm; the inhibition illustrates the off-switch the open-loop pool lacks.
    The cell uses gamma + somatic NaP (``nap_factor``). Returns traces in
    seconds / nA."""
    pool = build_single_cell_pool(gamma=gamma)
    if nap_factor != 1.0:
        scale_nap(pool, nap_factor, 0.00215)
    cell = pool[0]
    secs = _sections(cell)
    h.dt = DT
    h.celsius = CELSIUS
    h.tstop = 400.0                                  # relax to rest
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [V_INIT] * len(secs)))
    h.finitialize()
    h.run()
    vh = cell.soma(0.5).v
    h.tstop = total_ms
    _ = h.FInitializeHandler(0, lambda: fi0_multicompartment(secs, [vh] * len(secs)))
    tw = np.arange(0.0, total_ms + 1.0, 1.0)
    exc = np.minimum(tw / 400.0, 1.0) * hold_nA
    exc[(tw >= t_pulse[0]) & (tw < t_pulse[1])] = pulse_nA
    inh = np.zeros_like(tw)
    inh[(tw >= t_inhib[0]) & (tw < t_inhib[1])] = inhib_nA
    tv = h.Vector(tw)
    stim = h.IClamp(cell.dend[0](0.5))
    stim.delay = 0
    stim.dur = total_ms
    av = h.Vector(exc)
    av.play(stim._ref_amp, tv, True)
    keep = [tv, av]                                  # keep play vectors alive
    for sec in secs:                                 # global inhibition soma+dends
        ic = h.IClamp(sec(0.5))
        ic.delay = 0
        ic.dur = total_ms
        iv = h.Vector(inh)
        iv.play(ic._ref_amp, tv, True)
        keep += [ic, iv]
    vm = h.Vector()
    vm.record(cell.soma(0.5)._ref_v)
    t = h.Vector()
    t.record(h._ref_t)
    icas = [(h.Vector(), d) for d in cell.dend]
    for rec, d in icas:
        rec.record(d(0.5)._ref_icaL)
    nap_rec = h.Vector()                              # somatic NaP (napp.inap)
    nap_rec.record(cell.soma(0.5)._ref_inap_napp)
    h.finitialize()
    h.run()
    t = np.array(t)
    vm = np.array(vm)
    pic_nA = np.zeros_like(t)
    for rec, d in icas:
        area_cm2 = np.pi * d.L * d.diam * 1e-8        # um^2 -> cm^2
        pic_nA = pic_nA + np.array(rec) * area_cm2 * 1e6   # mA/cm2 -> nA
    soma_area = np.pi * cell.soma.L * cell.soma.diam * 1e-8
    nap_nA = np.array(nap_rec) * soma_area * 1e6
    inp = np.interp(t, tw, exc) + np.interp(t, tw, inh)
    return dict(t=t / 1000.0, vm=vm, pic_nA=pic_nA, nap_nA=nap_nA, input_nA=inp,
                vhold=vh, t_pulse=tuple(x / 1000.0 for x in t_pulse),
                t_inhib=tuple(x / 1000.0 for x in t_inhib))


def brief_command_drive(peak_pps=22.0, rise_s=1.0, plateau_s=1.5,
                        total_s=10.0, n_points=100000):
    """Low-MVC raised-cosine command: rise, brief plateau, fall to zero, then
    TRUE silence (command and its synaptic noise both zero after offset, so any
    post-offset discharge is paced by the intrinsic PIC, not residual drive).
    Returns (AnalogSignal in pps, command_offset_s, total_s)."""
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
    noise[t >= off] = 0.0          # truly silent after offset: no residual drive
    sig = AnalogSignal((cmd + noise) * pps,
                       sampling_period=(total_s / n_points) * pq.s)
    return sig, off, total_s


def run_pool(command, n_mu, gamma, nap_factor=1.0, nap_ceiling=0.00215,
             total_s=10.0, dd_n=100, mn_noise=0.0, noise_floor=0.3,
             dd_weight__uS=0.15):
    """Build a NERLab motoneuron pool with PIC knobs, drive it with `command`,
    return the motor-neuron spike-train Block. Separate NEURON run per call. The
    somatic ``gnapbar_napp`` is scaled after construction via ``nap_factor``
    (scale_nap), capped at ``nap_ceiling``."""
    rt, _ = simulator.RecruitmentThresholds(
        N=n_mu, recruitment_range__ratio=100, deluca__slope=5,
        konstantin__max_threshold__ratio=1.0, mode="combined")
    h.secondorder = 2
    mn_pool = AlphaMN__Pool(recruitment_thresholds__array=rt, model="NERLab",
                            gamma=gamma)
    if nap_factor != 1.0:
        scale_nap(mn_pool, nap_factor, nap_ceiling)
    spike_threshold = 50.0   # NERLab rests ~0 mV
    # Independent OU (colored) noise current injected into each motoneuron, so
    # firing is realistically irregular (real voluntary CV ~10-20%) rather than
    # clock-like. (The built-in Gfluctdv conductance-noise mechanism does not
    # generate -- its Scop random source stays 0 -- so we inject current.)
    # mn_noise is the PEAK OU current sigma (nA), reached at full descending
    # drive. The amplitude SCALES WITH THE DRIVE envelope (synaptic bombardment
    # tracks input), dropping to `mn_noise * noise_floor` when the drive
    # withdraws -- so a self-sustained spasm, paced by the intrinsic PIC, fires
    # regularly (CV ~5%) while voluntary firing is irregular (CV ~12-15%).
    _noise_keep = []
    if mn_noise > 0:
        from scipy.signal import lfilter
        rng = get_random_generator()
        n_steps = int(total_s * 1000.0)               # 1 ms resolution
        a = np.exp(-1.0 / 20.0)                        # OU tau = 20 ms
        amp_unit = np.sqrt(1.0 - a * a)               # -> unit-variance OU
        # drive envelope on the 1 ms grid, normalised to its peak
        cmd_mag = np.asarray(command.magnitude, dtype=float).ravel()
        cmd_dt = float(command.sampling_period.rescale(pq.ms).magnitude)
        cmd_t = np.arange(len(cmd_mag)) * cmd_dt
        env = np.interp(np.arange(n_steps), cmd_t, cmd_mag)
        peak = env.max() if env.max() > 0 else 1.0
        env = noise_floor + (1.0 - noise_floor) * np.clip(env / peak, 0.0, 1.0)
        tvec_n = h.Vector(np.arange(n_steps, dtype=float))
        for cell in mn_pool:
            ou = lfilter([amp_unit], [1.0, -a], rng.normal(0, 1, n_steps))
            ou = mn_noise * env * ou                  # drive-scaled amplitude
            ic = h.IClamp(cell.soma(0.5))
            ic.delay = 0
            ic.dur = total_s * 1000.0
            iv = h.Vector(ou)
            iv.play(ic._ref_amp, tvec_n, True)
            _noise_keep.append((ic, iv))
        _noise_keep.append(tvec_n)
    dd_pool = DescendingDrive__Pool(n=dd_n, process_type="gamma", shape=5,
                                    timestep__ms=_POOL_TIMESTEP)
    net = Network({"DD": dd_pool, "aMN": mn_pool})
    net.connect(source="DD", target="aMN", probability=0.5,
                weight__uS=dd_weight__uS * pq.uS)
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
        nc.threshold = spike_threshold
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


def population_rate(block, total_s, win_s=0.8, step_s=0.02):
    """Sliding-window mean per-unit discharge rate (pps) across active units.
    `win_s` sets the window (larger = smoother); `step_s` the shift between
    points (smaller = more, overlapping points). Returns (centers_s, rate_pps)."""
    sts = [st.rescale(pq.s).magnitude for st in block.segments[0].spiketrains
           if len(st) > 0]
    centers = np.arange(win_s / 2.0, total_s - win_s / 2.0 + step_s, step_s)
    if not sts:
        return centers, np.zeros_like(centers)
    half = win_s / 2.0
    rate = np.array([
        np.mean([np.count_nonzero((s >= c - half) & (s < c + half)) / win_s
                 for s in sts])
        for c in centers])
    return centers, rate


def rate_in_windows(centers, rate, times, half_w=0.25):
    """Mean rate in +/- half_w windows around each time in `times` (pps)."""
    return float(np.mean([rate[(centers > x - half_w) & (centers < x + half_w)].mean()
                          for x in times]))


def population_cv(block, total_s, win_s=1.0, step_s=0.02, max_isi=0.3,
                  min_isi=3, min_units=2):
    """Sliding-window ISI irregularity (%): per unit, then MEDIAN across units
    (robust to outliers). Uses CV2 -- the mean of 2*|ISI_{i+1}-ISI_i| /
    (ISI_{i+1}+ISI_i) over consecutive intervals -- NOT the plain CV. CV2 is a
    LOCAL measure, insensitive to slow rate drift, so the cyclic voluntary drive
    (which ramps the rate within a window) does not inflate it; for steady firing
    CV2 == plain CV, so the self-sustained spasm value still matches Gorassini
    2004. ISIs longer than `max_isi` (inter-burst gaps) are EXCLUDED so silence
    is not counted as one huge interval. NaN where fewer than `min_units` units
    have >= `min_isi` valid within-burst intervals -> gaps during true silence.
    Returns (centers_s, cv_percent), spanning 0..total_s."""
    sts = [st.rescale("s").magnitude for st in block.segments[0].spiketrains
           if len(st) > 1]
    centers = np.arange(0.0, total_s + step_s, step_s)
    half = win_s / 2.0
    cv = []
    for c in centers:
        unit_cvs = []
        for s in sts:
            sw = s[(s >= c - half) & (s < c + half)]
            if len(sw) >= 2:
                isi = np.diff(sw)
                isi = isi[isi <= max_isi]          # drop inter-burst gap ISIs
                if len(isi) >= min_isi:
                    a, b = isi[:-1], isi[1:]       # consecutive ISI pairs
                    cv2 = np.mean(2.0 * np.abs(b - a) / (a + b))
                    unit_cvs.append(cv2)
        cv.append(100.0 * float(np.median(unit_cvs))
                  if len(unit_cvs) >= min_units else np.nan)
    return centers, np.asarray(cv)


def make_iemg_simulator(muscle, n_mu):
    # IntramuscularEMG returns MONOPOLAR per-contact channels; the figure shows a
    # single illustrative trace. (differentiation_mode configures the electrode's
    # optional differential_matrix tool, which is not auto-applied to the
    # simulated output, so we leave it at its default rather than imply a montage
    # that isn't used here.)
    electrode = simulator.IntramuscularElectrodeArray(
        num_electrodes=4, inter_electrode_distance__mm=2.0 * pq.mm,
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
        on = arr[times <= t_off_s]
        tail = arr[times > t_off_s]
        on_rms = np.sqrt(np.mean(on**2)) if on.size else 1.0
        tail_rms = np.sqrt(np.mean(tail**2)) if tail.size else 0.0
        tail_ratio = float(tail_rms / on_rms) if on_rms else 0.0
    return dict(iemg=arr, times=times, signal=sig, sim=iemg_sim,
                tail_ratio=tail_ratio)

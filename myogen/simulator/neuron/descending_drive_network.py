"""Calibrated descending-drive network: the DD pool, its wiring to the motor neurons, and the drive loop.

This is the drive the FDI calibration (``examples/02_finetune/06_optimize_dd_multilevel.py``) was fitted with and the
``fdi_semg`` profile runs: a gamma-process descending-drive pool, equal fan-in onto every motor neuron with
dendritic-transfer weights, Gfluctdv conductance noise at a fitted scale, and a drive rate in Hz stepped with the
NEURON integration. The optimiser imports these functions, so the calibration and the profile share one authority.
"""
from types import SimpleNamespace

import numpy as np
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen.simulator.neuron.network import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.utils.helper import get_gamma_shape_for_mvc

NETWORK = {"fan_in": "equal", "weight_scaling": "dendritic_transfer"}  # DD -> MN wiring, see build_dd_mn_network
SPIKE_SAMPLING_RATE__HZ = 10240.0  # sampling rate attached to the recorded spike trains


def insert_scaled_gfluctdv(mn_pool, scale):
    """Insert Gfluctdv independent noise on all MN dendrites at given scale."""
    if scale <= 0:
        return
    for cell in mn_pool:
        cell.insert_Gfluctdv()
        for d in cell.dend:
            d.std_e_Gfluctdv *= scale
            d.std_i_Gfluctdv *= scale
            d.g_e0_Gfluctdv *= scale
            d.g_i0_Gfluctdv *= scale


def dendritic_transfer(mn_pool):
    """Fraction of a steady current injected at each MN's DD synapse site that reaches its clamped soma.

    DC transfer impedance (synapse site -> soma) over soma input impedance, from ``h.Impedance`` linearised at
    the initial state (active channels included). Call before ``insert_scaled_gfluctdv``.
    """
    for s, v in zip(*mn_pool.get_initialization_data()):
        s.v = v
    h.finitialize()
    k = []
    for cell in mn_pool:
        syn = next(s for s in cell.synapse__list if s.e >= -40)
        seg = syn.get_segment()
        imp = h.Impedance()
        imp.loc(0.5, sec=cell.soma)
        imp.compute(0, 1)
        k.append(imp.transfer(seg.x, sec=seg.sec) / imp.input(0.5, sec=cell.soma))
    return np.array(k)


def build_dd_mn_network(mn_pool, dd_pool, dd_params):
    """Wire the DD pool to the motor neurons and the cortical input to the DD pool.

    ``dd_params["network"]`` must be ``NETWORK``; parameter files without it (calibrated with random fan-in and
    one weight for all neurons) are refused rather than reinterpreted.

    - Equal fan-in: every MN receives exactly round(conn_probability * dd_neurons) DD inputs, drawn without
      replacement. Modelling assumption (descending drive shared across the pool); it removes the binomial
      +/-13 % input-count scatter that reordered the largest units, whose thresholds are ~3 % apart.
    - Dendritic-transfer weights: weight_i = synaptic_weight * (k_max / k_i) ** p (``dendritic_transfer``), p from
      ``network["weight_exponent"]`` (default 1). With p = 1 each input delivers the same current to a clamped soma
      at low drive whatever the neuron's size, and recruitment order follows rheobase: equal effective synaptic
      current per input, the equal-share case of Heckman & Binder 1991 (J Neurophysiol 65:952) [heckmanbinder1991]. p < 1 gives larger
      neurons less effective current per input (p = 0: one weight for all). Modelling assumption either way;
      measured effective currents are input-system specific (cat data) and none are known for corticospinal
      input to a hand muscle. Not compensated: the Gfluctdv mean-conductance shunt and the driving-force
      saturation at high drive.
    """
    net_cfg = dict(dd_params.get("network") or {})
    p = net_cfg.pop("weight_exponent", 1.0)
    if not isinstance(p, (int, float)) or isinstance(p, bool) or not np.isfinite(p) or p < 0:
        raise ValueError(f"network weight_exponent must be a finite number >= 0, got {p!r}")
    if net_cfg != NETWORK:
        raise ValueError(
            f"dd_params['network'] must be {NETWORK}, got {net_cfg!r}; parameter files from before 2026-11-01 "
            "were calibrated with random fan-in and uniform weights and must be recalibrated."
        )
    n_dd = len(dd_pool)
    n_in = int(round(dd_params["conn_probability"] * n_dd))
    if n_in < 1:
        raise ValueError(f"conn_probability * dd_neurons rounds to {n_in} inputs per motor neuron")
    transfer = dendritic_transfer(mn_pool)
    weights = dd_params["synaptic_weight"] * (transfer.max() / transfer) ** p
    network = Network({"DD": dd_pool, "aMN": mn_pool})
    dd_to_mn = network.connect_fan_in("DD", "aMN", n_inputs=n_in, weights__uS=weights * pq.uS,
                                      delay__ms=1.0 * pq.ms)
    network.connect_from_external(source="cortical_input", target="DD", weight__uS=1.0 * pq.uS)
    return SimpleNamespace(network=network, dd_to_mn=dd_to_mn,
                           cortical_netcons=network.get_netcons("cortical_input", "DD"),
                           transfer=transfer, weights=weights)


def excitation_to_drive_hz(segments, threshold_hz, max_hz, timestep_ms=0.1, noise_seed=42):
    """Piecewise-linear excitation [(duration_s, start, end)] and its DD rate (Hz): threshold_hz + e (max - threshold),
    0.8 threshold_hz at rest, plus 1 % half-normal noise."""
    parts = [np.linspace(a, b, int(round(d * 1000 / timestep_ms)), endpoint=False) for d, a, b in segments]
    e = np.concatenate(parts)
    hz = np.where(e > 0, threshold_hz + e * (max_hz - threshold_hz), 0.8 * threshold_hz)
    hz = hz + np.abs(np.random.default_rng(noise_seed).normal(0, hz.max() * 0.01, size=hz.size))
    return e, hz


def build_pools(thresholds, dd_params, timestep_ms=0.1, config_file="alpha_mn_default.yaml"):
    """Motor neuron pool, DD pool and their wiring (``build_dd_mn_network``), with Gfluctdv at the fitted scale."""
    gamma_shape = get_gamma_shape_for_mvc(100, mvc_shape_value=dd_params["gamma_shape"])
    mn_pool = AlphaMN__Pool(recruitment_thresholds__array=thresholds, config_file=config_file)
    dd_pool = DescendingDrive__Pool(n=dd_params["dd_neurons"], timestep__ms=timestep_ms * pq.ms,
                                    process_type="gamma", shape=gamma_shape)
    wiring = build_dd_mn_network(mn_pool, dd_pool, dd_params)
    insert_scaled_gfluctdv(mn_pool, dd_params.get("gfluctdv_scale", 0.0))
    return SimpleNamespace(mn_pool=mn_pool, dd_pool=dd_pool, wiring=wiring)


def run_descending_drive(thresholds, drive_hz, dd_params, total_ms, timestep_ms=0.1,
                         config_file="alpha_mn_default.yaml"):
    """Run the DD network with a time-varying drive rate (Hz, one value per time step); return the MN spike trains."""
    pools = build_pools(thresholds, dd_params, timestep_ms, config_file)
    mn_pool, dd_pool, dd_netcons = pools.mn_pool, pools.dd_pool, pools.wiring.cortical_netcons

    mn_recorders = []
    for cell in mn_pool:
        rec = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = 50
        nc.record(rec)
        mn_recorders.append(rec)

    h.load_file("stdrun.hoc")
    h.dt = timestep_ms
    h.tstop = total_ms

    for s, v in zip(*mn_pool.get_initialization_data()):
        s.v = v
    for s, v in zip(*dd_pool.get_initialization_data()):
        s.v = v
    h.finitialize()

    n_steps = len(drive_hz)
    step = 0
    while h.t < h.tstop:
        d = drive_hz[min(step, n_steps - 1)]
        for dd_cell in dd_pool:
            if dd_cell.integrate(d):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step += 1

    fs = SPIKE_SAMPLING_RATE__HZ
    block = Block()
    seg = Segment()
    block.segments.append(seg)
    seg.spiketrains = [
        SpikeTrain(rec.as_numpy() / 1000 * pq.s, t_stop=total_ms / 1000.0 * pq.s, sampling_rate=fs * pq.Hz,
                   sampling_period=(1 / fs) * pq.s, name=str(i))
        for i, rec in enumerate(mn_recorders)
    ]
    return block

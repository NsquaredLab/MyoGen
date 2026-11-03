"""The calibrated descending-drive network (moved from examples/finetune/01_optimize_dd_multilevel.py).

The pinned hash was computed with the optimiser's own functions before the move (2026-11-03), so the library
module reproduces the calibrated drive exactly. Gfluctdv is off in the pinned run: its ``normrand`` stream is
NEURON's and not seeded from Python.
"""
import hashlib

import numpy as np
import pytest

from myogen import load_nmodl_mechanisms, set_random_seed
from myogen.simulator import RecruitmentThresholds

# FDI calibration _human59_wexp, best trial 85 (examples/basic/results/..., copied into the fdi_semg card)
DD_PARAMS = {
    "dd_neurons": 147,
    "conn_probability": 0.3596767670514289,
    "dd_drive__Hz": 759.6041875607232,
    "synaptic_weight": 0.05,
    "gamma_shape": 3.0,
    "network": {"fan_in": "equal", "weight_scaling": "dendritic_transfer", "weight_exponent": 0.6660301395551126},
    "gfluctdv_scale": 0.0,
}
THRESHOLD_HZ = 103.2586942465358
PINNED_SHA1 = "b6d0e8e21f601c0521f1e39008198f8e7a015795"  # 2000 spikes


def _thresholds():
    set_random_seed(42)
    thr, _ = RecruitmentThresholds(N=120, recruitment_range__ratio=50, mode="fuglevand")
    thr = np.asarray(thr, dtype=float)
    return thr / thr.max()


def test_run_matches_pinned_spikes():
    from myogen.simulator.neuron.descending_drive_network import excitation_to_drive_hz, run_descending_drive

    load_nmodl_mechanisms()
    thr = _thresholds()
    e, hz = excitation_to_drive_hz([(0.5, 0, 0), (1.0, 0, 0.3)], THRESHOLD_HZ, DD_PARAMS["dd_drive__Hz"])
    block = run_descending_drive(thr, hz, DD_PARAMS, len(e) * 0.1)
    trains = block.segments[0].spiketrains
    digest = hashlib.sha1(b"".join(np.asarray(st.magnitude, float).tobytes() for st in trains)).hexdigest()
    assert sum(len(st) for st in trains) == 2000
    assert digest == PINNED_SHA1


def test_gfluctdv_scale_changes_noise():
    from myogen.simulator.neuron.descending_drive_network import insert_scaled_gfluctdv
    from myogen.simulator.neuron.populations import AlphaMN__Pool

    load_nmodl_mechanisms()
    full = AlphaMN__Pool(recruitment_thresholds__array=np.array([0.5]))
    half = AlphaMN__Pool(recruitment_thresholds__array=np.array([0.5]))
    insert_scaled_gfluctdv(full, 1.0)
    insert_scaled_gfluctdv(half, 0.5)
    assert half[0].dend[0].std_e_Gfluctdv == pytest.approx(0.5 * full[0].dend[0].std_e_Gfluctdv)
    assert half[0].dend[0].g_i0_Gfluctdv == pytest.approx(0.5 * full[0].dend[0].g_i0_Gfluctdv)


def test_network_block_is_required():
    from myogen.simulator.neuron.descending_drive_network import build_dd_mn_network
    from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
    import quantities as pq

    load_nmodl_mechanisms()
    mn = AlphaMN__Pool(recruitment_thresholds__array=np.array([0.5, 1.0]))
    dd = DescendingDrive__Pool(n=4, timestep__ms=0.1 * pq.ms, process_type="gamma", shape=3.0)
    params = {k: v for k, v in DD_PARAMS.items() if k != "network"}
    with pytest.raises(ValueError, match="network"):
        build_dd_mn_network(mn, dd, params)

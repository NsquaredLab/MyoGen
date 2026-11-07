"""The calibrated descending-drive network (moved from the optimiser, examples/02_finetune/06_optimize_dd_multilevel.py).

The stored spikes were computed with the optimiser's own functions before the move (2026-11-03), so the library
module reproduces the calibrated drive exactly. Gfluctdv is off in the pinned run: its ``normrand`` stream is
NEURON's and not seeded from Python.
"""
import numpy as np
import pytest

from myogen import load_nmodl_mechanisms, set_random_seed
from myogen.simulator import RecruitmentThresholds

# FDI calibration _human59_wexp, best trial 85 (copied into the fdi_semg card)
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
# Stored spikes: count per motor unit, the first spike time (s) of six units, and the sum of all spike times (s).
SPIKES_PER_UNIT = [
    22, 23, 21, 23, 22, 23, 21, 22, 21, 23, 22, 22, 21, 21, 23, 22, 22, 22, 21, 22, 21, 21, 21, 21, 22,
    22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 21, 21, 21, 21, 20, 20,
    20, 20, 20, 20, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 20, 19, 19, 19, 18, 18, 19, 18, 18,
    18, 18, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 13, 13, 12, 12, 12, 11, 11, 10,
    10, 9, 8, 8, 7, 6, 5, 4, 5, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0
]
FIRST_SPIKE__S = {0: 0.18880000000010866, 19: 0.2636000000001256, 38: 0.5264000000001691, 57: 0.546200000000151, 76: 0.6731000000000356, 95: 0.9732999999997626}
SPIKE_TIME_SUM__S = 2193.9812999992955


def _thresholds():
    set_random_seed(42)
    thr, _ = RecruitmentThresholds(N=120, recruitment_range__ratio=50, mode="fuglevand")
    thr = np.asarray(thr, dtype=float)
    return thr / thr.max()


def test_run_matches_stored_spikes():
    from myogen.simulator.neuron.descending_drive_network import excitation_to_drive_hz, run_descending_drive

    load_nmodl_mechanisms()
    thr = _thresholds()
    e, hz = excitation_to_drive_hz([(0.5, 0, 0), (1.0, 0, 0.3)], THRESHOLD_HZ, DD_PARAMS["dd_drive__Hz"])
    block = run_descending_drive(thr, hz, DD_PARAMS, len(e) * 0.1)
    trains = block.segments[0].spiketrains
    assert [len(st) for st in trains] == SPIKES_PER_UNIT  # 2000 spikes
    for unit, t0 in FIRST_SPIKE__S.items():
        assert float(trains[unit].rescale("s").magnitude[0]) == pytest.approx(t0, rel=1e-9)
    total = sum(float(np.sum(st.rescale("s").magnitude)) for st in trains)
    assert total == pytest.approx(SPIKE_TIME_SUM__S, rel=1e-9)


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

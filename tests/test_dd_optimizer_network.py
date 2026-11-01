"""DD -> motor neuron network of the multilevel DD optimiser (examples/finetune/01_optimize_dd_multilevel.py).

Every motor neuron gets the same number of DD inputs, and each input's weight is scaled so that it delivers the
same current to a clamped soma whatever the neuron's size (equal effective synaptic current per input, the
equal-share case of Heckman & Binder 1991). Parameter files without a ``network`` block are refused.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import quantities as pq
from neo import Block, Segment, SpikeTrain
from neuron import h

from myogen import load_nmodl_mechanisms, set_random_seed
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool

OPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "finetune" / "01_optimize_dd_multilevel.py"


@pytest.fixture(scope="module")
def opt():
    load_nmodl_mechanisms()
    spec = importlib.util.spec_from_file_location("dd_multilevel_opt", OPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pools(opt, n_dd=40):
    set_random_seed(7)
    thresholds = np.linspace(0.02, 1.0, 12)
    mn = AlphaMN__Pool(recruitment_thresholds__array=thresholds, config_file="alpha_mn_default.yaml")
    dd = DescendingDrive__Pool(n=n_dd, timestep__ms=opt.TIMESTEP_MS * pq.ms, process_type="gamma", shape=3.0)
    return thresholds, mn, dd


def _params(**network):
    return {"dd_neurons": 40, "conn_probability": 0.3, "synaptic_weight": 0.05,
            "network": {"fan_in": "equal", "weight_scaling": "dendritic_transfer", **network}}


def _owner(mn):
    owner = {}
    for i, cell in enumerate(mn):
        for sec in [cell.soma, *cell.dend]:
            owner[sec.hname()] = i
    return owner


def test_every_motor_neuron_gets_the_same_number_of_excitatory_dd_inputs(opt):
    _, mn, dd = _pools(opt)
    net = opt.build_dd_mn_network(mn, dd, _params())
    owner = _owner(mn)
    counts = np.zeros(len(mn), int)
    for nc in net.dd_to_mn:
        syn = nc.syn()
        assert syn.e > -40, "DD input landed on an inhibitory synapse"
        counts[owner[syn.get_segment().sec.hname()]] += 1
    assert np.all(counts == round(0.3 * 40)), counts


def test_input_weights_equalise_the_current_reaching_the_soma(opt):
    _, mn, dd = _pools(opt)
    net = opt.build_dd_mn_network(mn, dd, _params())
    k = np.asarray(net.transfer)
    assert np.all(np.diff(k) < 0), "dendrite-to-soma transfer should fall with neuron size"
    owner = _owner(mn)
    w = np.zeros(len(mn))
    for nc in net.dd_to_mn:
        w[owner[nc.syn().get_segment().sec.hname()]] = nc.weight[0]
    np.testing.assert_allclose(w * k, w[0] * k[0], rtol=1e-9)
    assert w[0] == pytest.approx(0.05), "smallest unit keeps the nominal weight"


def test_parameter_sets_without_a_network_block_are_refused(opt):
    _, mn, dd = _pools(opt)
    params = _params()
    del params["network"]
    with pytest.raises(ValueError, match="network"):
        opt.build_dd_mn_network(mn, dd, params)


def test_last_recruitment_excitation_uses_sustained_firing(opt):
    dt_ms = 1.0
    e = np.linspace(0.0, 1.0, 10001)  # 10 s ramp, 1 ms steps
    t_stop = 10.0 * pq.s
    early = SpikeTrain(np.arange(1.0, 10.0, 0.05) * pq.s, t_stop=t_stop)  # sustained from e = 0.1
    late = SpikeTrain(np.r_[0.5, np.arange(6.0, 10.0, 0.05)] * pq.s, t_stop=t_stop)  # one stray spike, then e = 0.6
    block = Block()
    block.segments.append(Segment())
    block.segments[0].spiketrains = [early, late]
    e_rec = opt.recruitment_excitation(block, e, timestep_ms=dt_ms)
    np.testing.assert_allclose(e_rec, [0.1, 0.6], atol=0.03)

    never = SpikeTrain(np.array([]) * pq.s, t_stop=t_stop)
    block.segments[0].spiketrains = [early, never]
    assert np.isnan(opt.recruitment_excitation(block, e, timestep_ms=dt_ms)[1])


def test_recruitment_times_use_sustained_firing(opt):
    t_stop = 10.0 * pq.s
    early = SpikeTrain(np.arange(1.0, 10.0, 0.05) * pq.s, t_stop=t_stop)
    late = SpikeTrain(np.r_[0.5, np.arange(6.0, 10.0, 0.05)] * pq.s, t_stop=t_stop)
    never = SpikeTrain(np.array([]) * pq.s, t_stop=t_stop)
    block = Block()
    block.segments.append(Segment())
    block.segments[0].spiketrains = [early, late, never]
    t_rec = opt.recruitment_times(block)
    np.testing.assert_allclose(t_rec[:2], [1.0, 6.0])
    assert np.isnan(t_rec[2])


def test_last_recruitment_force_is_read_from_the_smoothed_force_curve(opt):
    fs = 1000.0
    t = np.arange(0, 10.0, 1 / fs)
    force = 50.0 * t  # 0 -> 500 model units; MVC 500
    pct, n_unrec = opt.last_recruitment_pct_mvc(np.array([1.0, 4.4, 2.0]), force, fs, mvc=500.0)
    assert n_unrec == 0
    assert pct == pytest.approx(100 * 50.0 * 4.4 / 500.0, abs=0.1)


def test_unrecruited_units_push_the_ceiling_past_mvc(opt):
    fs = 1000.0
    force = np.full(10000, 100.0)
    pct, n_unrec = opt.last_recruitment_pct_mvc(np.array([1.0, np.nan, np.nan, 2.0]), force, fs, mvc=100.0)
    assert n_unrec == 2
    assert pct == pytest.approx(150.0)


def test_ceiling_target_is_given_in_percent_mvc(opt):
    assert opt.ceiling_penalty(55.0, 55.0) == 0.0
    assert opt.ceiling_penalty(66.0, 55.0) == pytest.approx(opt.CEILING_WEIGHT * 11.0 / 55.0)
    assert opt.ceiling_penalty(80.0, None) == 0.0, "no target: the ceiling term is off"

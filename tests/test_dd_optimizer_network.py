"""DD -> motor neuron network of the multilevel DD optimiser (examples/02_finetune/06_optimize_dd_multilevel.py).

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

OPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "02_finetune" / "06_optimize_dd_multilevel.py"


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


def _weights(opt, net, mn):
    owner = _owner(mn)
    w = np.zeros(len(mn))
    for nc in net.dd_to_mn:
        w[owner[nc.syn().get_segment().sec.hname()]] = nc.weight[0]
    return w


@pytest.mark.parametrize("p", [0.0, 0.5])
def test_weight_exponent_sets_how_much_larger_neurons_are_compensated(opt, p):
    _, mn, dd = _pools(opt)
    net = opt.build_dd_mn_network(mn, dd, _params(weight_exponent=p))
    k = np.asarray(net.transfer)
    w = _weights(opt, net, mn)
    np.testing.assert_allclose(w, 0.05 * (k.max() / k) ** p, rtol=1e-9)


def test_weight_exponent_defaults_to_equal_current_per_input(opt):
    _, mn, dd = _pools(opt)
    a = _weights(opt, opt.build_dd_mn_network(mn, dd, _params()), mn)
    _, mn2, dd2 = _pools(opt)
    b = _weights(opt, opt.build_dd_mn_network(mn2, dd2, _params(weight_exponent=1.0)), mn2)
    np.testing.assert_allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), "1"])
def test_invalid_weight_exponent_is_refused(opt, bad):
    _, mn, dd = _pools(opt)
    with pytest.raises(ValueError, match="weight_exponent must be"):
        opt.build_dd_mn_network(mn, dd, _params(weight_exponent=bad))


def test_network_block_records_the_weight_exponent(opt):
    assert opt.network_block() == opt.NETWORK
    assert opt.network_block(0.4) == {**opt.NETWORK, "weight_exponent": 0.4}


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


def test_ceiling_rank_scores_the_unit_with_k_units_above_it(opt):
    fs = 1000.0
    force = 10.0 * np.arange(0, 10.0, 1 / fs)  # MVC 100 at t = 10 s
    t_rec = np.array([1.0, 6.0, 3.0, 5.0])
    assert opt.last_recruitment_pct_mvc(t_rec, force, fs, mvc=100.0)[0] == pytest.approx(60.0, abs=0.2)
    assert opt.last_recruitment_pct_mvc(t_rec, force, fs, mvc=100.0, rank=1)[0] == pytest.approx(50.0, abs=0.2)


def test_unrecruited_units_count_as_above_the_ranked_unit(opt):
    fs = 1000.0
    force = 10.0 * np.arange(0, 10.0, 1 / fs)
    t_rec = np.array([1.0, np.nan, 3.0, 5.0])
    pct, n_unrec = opt.last_recruitment_pct_mvc(t_rec, force, fs, mvc=100.0, rank=1)
    assert n_unrec == 1
    assert pct == pytest.approx(50.0, abs=0.2), "one unit (unrecruited) above: the highest recruited unit is scored"
    pct, _ = opt.last_recruitment_pct_mvc(np.array([1.0, np.nan, np.nan, 5.0]), force, fs, mvc=100.0, rank=1)
    assert pct == pytest.approx(125.0), "more unrecruited units than the rank allows: past 100 % as before"


def test_rates_are_read_where_the_force_reaches_each_level(opt):
    fs = 1000.0
    force = 10.0 * np.arange(0, 10.0, 1 / fs)  # 40 % MVC at 4 s, 80 % at 8 s
    trains = [np.arange(0.0, 10.0, 1 / 20.0), np.r_[np.arange(0.0, 6.0, 1 / 10.0), np.arange(6.0, 10.0, 1 / 30.0)],
              np.array([9.5])]  # third unit not active at either level
    rates = opt.rates_at_pct_mvc(trains, force, fs, mvc=100.0, levels=(40, 80))
    assert rates[40] == pytest.approx((20 + 10) / 2, abs=1.0)
    assert rates[80] == pytest.approx((20 + 30) / 2, abs=1.0)


def test_high_force_rates_are_scored_against_de_luca_1982(opt):
    assert opt.HIGH_FORCE_RATE_TARGETS == {40: 25.3, 80: 41.4}
    assert opt.high_force_rate_penalty({40: 25.3, 80: 41.4}) == 0.0
    expected = opt.HIGH_FORCE_RATE_WEIGHT * (5.06 / 25.3 + 8.28 / 41.4)
    assert opt.high_force_rate_penalty({40: 20.24, 80: 33.12}) == pytest.approx(expected)


def test_force_model_uses_the_selected_force_frequency_shape(opt):
    from myogen.simulator.core.force.force_utils import FUGLEVAND_FORCE_FREQUENCY, HUMAN_UNIT_FORCE_FREQUENCY
    thresholds = np.linspace(0.02, 1.0, 12)
    assert opt.FORCE_FREQUENCY_SHAPE == FUGLEVAND_FORCE_FREQUENCY
    old = opt.FORCE_FREQUENCY_SHAPE
    try:
        opt.FORCE_FREQUENCY_SHAPE = HUMAN_UNIT_FORCE_FREQUENCY
        assert opt.make_force_model(thresholds).force_frequency_shape == HUMAN_UNIT_FORCE_FREQUENCY
    finally:
        opt.FORCE_FREQUENCY_SHAPE = old

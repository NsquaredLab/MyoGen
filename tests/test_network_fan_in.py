"""Network.connect_fan_in: every target receives the same number of inputs, each with its target's weight."""
import numpy as np
import pytest
import quantities as pq

from myogen import get_random_generator, load_nmodl_mechanisms, set_random_seed
from myogen.simulator.neuron import Network
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool


@pytest.fixture(scope="module", autouse=True)
def mechanisms():
    load_nmodl_mechanisms()


def _network(n_dd=20):
    set_random_seed(3)
    mn = AlphaMN__Pool(recruitment_thresholds__array=np.linspace(0.05, 1.0, 6),
                       config_file="alpha_mn_default.yaml")
    dd = DescendingDrive__Pool(n=n_dd, timestep__ms=0.1 * pq.ms, process_type="gamma", shape=3.0)
    return Network({"DD": dd, "aMN": mn}), mn, dd


def _target(mn, netcon):
    sec = netcon.syn().get_segment().sec.hname()
    return next(i for i, cell in enumerate(mn) if sec in {s.hname() for s in [cell.soma, *cell.dend]})


def test_each_target_gets_n_excitatory_inputs_with_its_own_weight():
    net, mn, dd = _network()
    weights = np.linspace(0.05, 0.02, len(mn))
    netcons = net.connect_fan_in("DD", "aMN", n_inputs=7, weights__uS=weights * pq.uS, delay__ms=1.0 * pq.ms)
    targets = np.array([_target(mn, nc) for nc in netcons])
    np.testing.assert_array_equal(np.bincount(targets, minlength=len(mn)), 7)
    np.testing.assert_allclose([nc.weight[0] for nc in netcons], weights[targets])
    assert all(nc.syn().e > -40 for nc in netcons)
    assert all(nc.delay == 1.0 for nc in netcons)
    assert net.get_netcons("DD", "aMN") == netcons


def test_sources_are_drawn_without_replacement_per_target_in_target_order():
    net, mn, dd = _network()
    state = get_random_generator().bit_generator.state
    netcons = net.connect_fan_in("DD", "aMN", n_inputs=5, weights__uS=0.05 * pq.uS)
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    expected = [list(rng.choice(len(dd), size=5, replace=False)) for _ in mn]
    source = {cell.ns.hname(): j for j, cell in enumerate(dd)}
    drawn = [source[nc.pre().hname()] for nc in netcons]
    assert drawn == [j for row in expected for j in row]
    assert [_target(mn, nc) for nc in netcons] == [i for i in range(len(mn)) for _ in range(5)]


@pytest.mark.parametrize("n_inputs", [0, 21])
def test_impossible_fan_in_is_refused(n_inputs):
    net, _, _ = _network()
    with pytest.raises(ValueError, match="n_inputs"):
        net.connect_fan_in("DD", "aMN", n_inputs=n_inputs, weights__uS=0.05 * pq.uS)

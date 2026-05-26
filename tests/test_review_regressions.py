import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Group, Segment, SpikeTrain


def _single_spike_block() -> Block:
    block = Block()
    segment = Segment(name="Pool_0")
    spiketrain = SpikeTrain([5] * pq.ms, t_start=0 * pq.ms, t_stop=11 * pq.ms)
    spiketrain.sampling_period = 1 * pq.ms
    segment.spiketrains.append(spiketrain)
    block.segments.append(segment)
    return block


def test_nmodl_loader_returns_true_only_for_insertable_mechanisms():
    from neuron import h

    from myogen.utils.nmodl import load_nmodl_mechanisms

    assert load_nmodl_mechanisms(quiet=True, strict=True)
    section = h.Section()
    section.insert("caL")


def test_population_connection_delay_uses_requested_delay_plus_axon_delay():
    from neuron import h

    from myogen.simulator.neuron.network import _connect_one_to_one, _connect_populations

    class Source:
        def __init__(self, global_id: int):
            self.ns = h.NetStim()
            self.global__ID = global_id
            self.axon_delay__ms = 2.0

    class Target:
        def __init__(self):
            self.section = h.Section()
            self.synapse__list = [h.ExpSyn(0.5, sec=self.section)]

    for connect in (_connect_populations, _connect_one_to_one):
        populations = {"source": [Source(0)], "target": [Target()]}
        if connect is _connect_populations:
            netcons = connect(
                populations,
                "source",
                "target",
                1.0,
                synaptic_delay=7.0,
                synaptic_weight=0.25,
            )
        else:
            netcons = connect(
                "source",
                "target",
                populations,
                1.0,
                synaptic_delay=7.0,
                synaptic_weight=0.25,
            )

        assert len(netcons) == 1
        assert netcons[0].delay == 9.0
        assert netcons[0].weight[0] == 0.25


def test_surface_emg_uses_convolution_not_correlation(monkeypatch):
    from myogen.simulator.core.emg.surface import surface_emg as surface_module
    from myogen.simulator.core.emg.surface.surface_emg import SurfaceEMG
    from myogen.utils.neo import create_grid_signal

    monkeypatch.setattr(surface_module, "HAS_CUPY", False)

    simulator = object.__new__(SurfaceEMG)
    simulator._MUs_to_simulate = [0]
    simulator._sampling_frequency__Hz = 1000.0
    simulator._surface_emg__Block = None
    simulator._spike_train__Block = None

    muap_block = Block()
    group = Group(name="ElectrodeArray_0")
    segment = Segment(name="MUAP_0")
    muap = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
    segment.analogsignals.append(
        create_grid_signal(muap * pq.mV, grid_shape=(1, 1), sampling_rate=1000 * pq.Hz)
    )
    group.segments.append(segment)
    muap_block.groups.append(group)
    simulator._muaps__Block = muap_block

    result = SurfaceEMG.simulate_surface_emg(simulator, _single_spike_block(), verbose=False)
    observed = result.groups[0].segments[0].analogsignals[0].magnitude[:, 0]

    expected = np.array([0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=float)
    np.testing.assert_allclose(observed, expected)


def test_intramuscular_emg_uses_convolution_not_correlation(monkeypatch):
    from myogen.simulator.core.emg.intramuscular import (
        intramuscular_emg as intramuscular_module,
    )
    from myogen.simulator.core.emg.intramuscular.intramuscular_emg import (
        IntramuscularEMG,
    )

    monkeypatch.setattr(intramuscular_module, "HAS_CUPY", False)

    simulator = object.__new__(IntramuscularEMG)
    simulator._MUs_to_simulate = [0]
    simulator._sampling_frequency__Hz = 1000.0
    simulator._intramuscular_emg__Block = None
    simulator._spike_train__Block = None

    muap_block = Block()
    segment = Segment(name="MUAP_0")
    segment.analogsignals.append(
        AnalogSignal(
            np.array([[1.0], [2.0], [3.0], [4.0]]) * pq.dimensionless,
            sampling_rate=1000 * pq.Hz,
        )
    )
    muap_block.segments.append(segment)
    simulator._muaps__Block = muap_block

    result = IntramuscularEMG.simulate_intramuscular_emg(
        simulator,
        _single_spike_block(),
        verbose=False,
    )
    observed = result.segments[0].analogsignals[0].magnitude[:, 0]

    expected = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 0, 0, 0])
    np.testing.assert_allclose(observed, expected)


def test_force_model_vectorized_matches_force_model():
    """ForceModelVectorized must produce the same force output as ForceModel.

    The two classes share the IPI/gain/twitch pipeline via ``force_utils`` so the
    optional vectorized path cannot silently drift from the reference model.
    """
    from myogen.simulator.core.force.force_model import ForceModel
    from myogen.simulator.core.force.force_model_vectorized import (
        ForceModelVectorized,
    )

    rng = np.random.default_rng(0)
    n_mus = 4
    recruitment_thresholds = np.linspace(0.1, 1.0, n_mus)
    recording_frequency = 2000.0  # Hz
    t_stop__s = 0.5
    sampling_period = (1.0 / recording_frequency) * pq.s

    segment = Segment(name="Pool_0")
    for _ in range(n_mus):
        spike_times__s = np.sort(rng.uniform(0.0, t_stop__s, size=8))
        spiketrain = SpikeTrain(
            spike_times__s * pq.s, t_start=0 * pq.s, t_stop=t_stop__s * pq.s
        )
        spiketrain.sampling_period = sampling_period
        segment.spiketrains.append(spiketrain)
    block = Block()
    block.segments.append(segment)

    # Both models take a quantity for the recording frequency and must yield
    # the same force for the same spike trains.
    reference = ForceModel(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=recording_frequency * pq.Hz,
    )
    vectorized = ForceModelVectorized(
        recruitment_thresholds=recruitment_thresholds,
        recording_frequency__Hz=recording_frequency * pq.Hz,
    )

    reference_force = reference.generate_force(block, verbose=False)
    vectorized_force = vectorized.generate_force(block, verbose=False)

    assert reference_force.shape == vectorized_force.shape
    np.testing.assert_allclose(
        vectorized_force.magnitude,
        reference_force.magnitude,
        rtol=1e-6,
        atol=1e-9,
    )


def test_to_float_rescales_quantities_to_native_neuron_units():
    """NEURON NetCon values must be rescaled to native units, not raw magnitudes.

    The buggy path took ``float(quantity)`` (NEURON's __float__ coercion), which
    drops the unit and returns the raw magnitude -- so a value in a non-native unit
    would be stored wrong (e.g. 1 s stored as 1.0 ms). These cases all differ
    between the buggy (raw magnitude) and fixed (rescaled) behavior.
    """
    from myogen.simulator.neuron.network import _to_float

    assert _to_float(1.0 * pq.s, pq.ms) == 1000.0  # buggy gave 1.0
    assert _to_float(0.0006 * pq.mS, pq.uS) == 0.6  # buggy gave 0.0006
    assert _to_float(-0.01 * pq.V, pq.mV) == -10.0  # buggy gave -0.01
    assert _to_float(0.6 * pq.uS, pq.uS) == 0.6  # already native
    assert _to_float(7.0, pq.ms) == 7.0  # plain float passes through


def test_default_synaptic_params_rescale_nonnative_delay():
    """A delay given in seconds must be stored in NEURON as milliseconds."""
    from neuron import h

    from myogen.simulator.neuron.network import _apply_default_synaptic_params

    class _Src:  # no axon_delay__ms attribute
        pass

    section = h.Section()
    syn = h.ExpSyn(0.5, sec=section)
    stim = h.NetStim()  # keep a reference: NEURON segfaults if the source is GC'd
    netcon = h.NetCon(stim, syn)

    _apply_default_synaptic_params(netcon, _Src(), synaptic_delay=1.0 * pq.s)

    assert netcon.delay == 1000.0  # 1 s -> 1000 ms; buggy code stored 1.0
    assert netcon.weight[0] == 0.6  # DEFAULT_SYNAPTIC_WEIGHT magnitude (uS)
    assert netcon.threshold == -10.0  # DEFAULT_SPIKE_THRESHOLD magnitude (mV)

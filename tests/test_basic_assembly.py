"""Synthetic contract fixtures for public Neo adapters, not physiological evidence."""
import copy
import numpy as np
import quantities as pq
import pytest
from neo import AnalogSignal, Block, Group, Segment, SpikeTrain

from myogen.simulator import ExecutionPolicy, IntramuscularEMG, SurfaceEMG
from myogen.utils.neo import create_grid_signal


def fixture(kind):
    cls = IntramuscularEMG if kind == "iemg" else SurfaceEMG
    sim = object.__new__(cls)
    sim._MUs_to_simulate = [0, 2]
    sim._sampling_frequency__Hz = 1000.0
    sim._spike_train__Block = None
    sim._intramuscular_emg__Block = None
    sim._surface_emg__Block = None
    templates = Block()
    parent = templates
    if kind == "semg":
        parent = Group(name="ElectrodeArray_0")
        templates.groups.append(parent)
    for mu in range(3):
        segment = Segment(name=f"MUAP_{mu}")
        data = np.random.default_rng(mu).normal(size=(8, 4))
        if kind == "iemg":
            signal = AnalogSignal(data * pq.dimensionless, sampling_rate=1000 * pq.Hz)
        else:
            signal = create_grid_signal(data.reshape(8, 2, 2) * pq.mV,
                                        grid_shape=(2, 2), sampling_rate=1000 * pq.Hz)
        segment.analogsignals.append(signal)
        parent.segments.append(segment)
    sim._muaps__Block = templates
    spikes = Block()
    for pool in range(2):
        segment = Segment(name=f"Pool_{pool}")
        for mu in range(3):
            train = SpikeTrain([0, 0.2, 7, 31] * pq.ms, t_stop=40 * pq.ms)
            train.sampling_period = 1 * pq.ms
            segment.spiketrains.append(train)
        spikes.segments.append(segment)
    return sim, spikes


@pytest.mark.parametrize("kind", ["iemg", "semg"])
def test_public_adapter_preserves_complete_waveforms_and_metadata(kind, monkeypatch):
    from myogen.simulator.core.emg.intramuscular import intramuscular_emg
    from myogen.simulator.core.emg.surface import surface_emg
    monkeypatch.setattr(intramuscular_emg, "HAS_CUPY", False)
    monkeypatch.setattr(surface_emg, "HAS_CUPY", False)
    sim, spikes = fixture(kind)
    call = getattr(sim, "simulate_intramuscular_emg" if kind == "iemg" else "simulate_surface_emg")
    expected = call(spikes, verbose=False)
    actual = call(spikes, verbose=False, execution=ExecutionPolicy())
    e = expected.segments if kind == "iemg" else expected.groups[0].segments
    a = actual.segments if kind == "iemg" else actual.groups[0].segments
    for reference, optimized in zip(e, a):
        x, y = reference.analogsignals[0], optimized.analogsignals[0]
        assert reference.name == optimized.name
        assert x.shape == y.shape
        assert x.units == y.units
        assert x.sampling_rate == y.sampling_rate
        assert x.t_start == y.t_start
        assert x.annotations == y.annotations
        error = np.max(np.abs(x.magnitude - y.magnitude), axis=0)
        assert np.all(error <= 1e-12 + 1e-9 * np.max(np.abs(x.magnitude), axis=0))
    assert sim.execution_report["status"] == "complete"


@pytest.mark.parametrize("kind", ["iemg", "semg"])
def test_preparation_reuse_and_mutation_invalidation(kind):
    sim, spikes = fixture(kind)
    call = getattr(sim, "simulate_intramuscular_emg" if kind == "iemg" else "simulate_surface_emg")
    policy = ExecutionPolicy()
    call(spikes, verbose=False, execution=policy)
    old_keys = set(sim._basic_preparation_cache.entries)
    call(spikes, verbose=False, execution=policy)
    assert sim.execution_report["cache_hits"] > 0
    group = sim._muaps__Block if kind == "iemg" else sim._muaps__Block.groups[0]
    group.segments[0].analogsignals[0].magnitude[0, 0] += 1
    call(spikes, verbose=False, execution=policy)
    assert set(sim._basic_preparation_cache.entries) != old_keys


@pytest.mark.parametrize("kind", ["iemg", "semg"])
def test_serial_assembly_does_not_discover_unused_blas_pools(kind, monkeypatch):
    """Startup/resource contract only; no physiological assertion."""
    from myogen.simulator.core.emg.runtime import session
    def unused():
        raise AssertionError("Assembly uses serial NumPy kernels and explicitly single-worker FFTs")
    monkeypatch.setattr(session, "enter_native_limit", unused)
    sim, spikes = fixture(kind)
    call = getattr(sim, "simulate_intramuscular_emg" if kind == "iemg" else "simulate_surface_emg")
    call(spikes, verbose=False, execution=ExecutionPolicy())
    assert sim.execution_report["status"] == "complete"


def test_surface_preparation_tracks_selection_without_dropping_source_templates():
    """Cache dependency and useful-buffer contract, not model evidence."""
    sim, spikes = fixture("semg")
    policy = ExecutionPolicy()
    sim.simulate_surface_emg(spikes, verbose=False, execution=policy)
    old_keys = set(sim._basic_preparation_cache.entries)
    assert len(old_keys) == 1
    assert next(iter(sim._basic_preparation_cache.entries.values())).shape[0] == 2
    assert len(sim._muaps__Block.groups[0].segments) == 3
    sim._MUs_to_simulate = [1]
    sim.simulate_surface_emg(spikes, verbose=False, execution=policy)
    new_keys = set(sim._basic_preparation_cache.entries) - old_keys
    assert len(new_keys) == 1
    assert sim._basic_preparation_cache.entries[new_keys.pop()].shape[0] == 1
    assert len(sim._muaps__Block.groups[0].segments) == 3


def test_recording_time_axes_and_interpolation_temporary_need_admission():
    """All-zero allocation contract; no signal or model behavior claim."""
    from types import SimpleNamespace
    from myogen.simulator.core.emg.runtime.assembly import assemble
    from myogen.simulator.core.emg.runtime import MemoryBudgetError
    templates, source = Block(), Block()
    segment = Segment(name="MUAP_0")
    segment.analogsignals.append(AnalogSignal(np.zeros((4, 1)), units=pq.dimensionless,
                                              sampling_rate=1e6 * pq.Hz))
    templates.segments.append(segment)
    pool = Segment(name="Pool_0")
    train = SpikeTrain([] * pq.s, t_stop=1 * pq.s)
    train.sampling_period = 1 * pq.us
    pool.spiketrains.append(train)
    source.segments.append(pool)
    owner = SimpleNamespace(_muaps__Block=templates, _MUs_to_simulate=[0],
                            _sampling_frequency__Hz=1e6)
    with pytest.raises(MemoryBudgetError):
        assemble(owner, source, ExecutionPolicy(memory_budget_mib=32), surface=False)
    assert not hasattr(owner, "_intramuscular_emg__Block")


def test_channel_streaming_fits_output_without_recording_sized_copies():
    """All-zero, full-size allocation contract only; no physiology assertion."""
    from types import SimpleNamespace
    from myogen.simulator.core.emg.runtime.assembly import assemble
    templates, source = Block(), Block()
    group = Group(name="ElectrodeArray_0")
    templates.groups.append(group)
    for index in range(100):
        segment = Segment(name=f"MUAP_{index}")
        segment.analogsignals.append(create_grid_signal(np.zeros((32, 5, 5)),
            grid_shape=(5, 5), sampling_rate=2000 * pq.Hz))
        group.segments.append(segment)
    pool = Segment(name="Pool_0")
    for _ in range(100):
        train = SpikeTrain([] * pq.s, t_stop=60 * pq.s)
        train.sampling_period = 1 * pq.ms
        pool.spiketrains.append(train)
    source.segments.append(pool)
    owner = SimpleNamespace(_muaps__Block=templates, _MUs_to_simulate=None,
                            _sampling_frequency__Hz=2000.)
    result = assemble(owner, source, ExecutionPolicy(memory_budget_mib=48), surface=True)
    assert result.groups[0].segments[0].analogsignals[0].shape == (120000, 25)
    assert owner._execution_report["peak_managed_bytes"] < 48 * 1024**2

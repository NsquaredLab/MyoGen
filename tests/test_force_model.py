"""Fuglevand (1993) force and sampling regressions for issue #26."""

import numpy as np
import pytest
import quantities as pq
from neo import Block, Segment, SpikeTrain

from myogen.simulator import ForceModel, ForceModelVectorized
from myogen.simulator.core.force.force_utils import fuglevand_gain


MODELS = (ForceModel, ForceModelVectorized)
THRESHOLDS = np.exp(np.log(30) * np.arange(1, 11) / 10) / 100


def _spike_block(times_by_unit, dt_ms=0.1, t_stop=2.0, t_start=0.0):
    dt = dt_ms / 1000
    segment = Segment()
    for times in times_by_unit:
        times = t_start + np.round((np.asarray(times) - t_start) / dt) * dt
        segment.spiketrains.append(
            SpikeTrain(
                times * pq.s, t_start=t_start * pq.s, t_stop=t_stop * pq.s,
                sampling_rate=(1 / dt) * pq.Hz,
            )
        )
    block = Block()
    block.segments.append(segment)
    return block


def test_fuglevand_gain():
    np.testing.assert_array_equal(fuglevand_gain(np.array([0, 0.1, 0.3, 0.4])), 1)
    np.testing.assert_allclose(fuglevand_gain(np.array([0.4 - 1e-10, 0.4 + 1e-10])), 1,
                               rtol=1e-8)
    r = np.array([0.41, 0.6, 1, 2, 10])
    expected = (1 - np.exp(-2 * r**3)) / r / ((1 - np.exp(-2 * 0.4**3)) / 0.4)
    np.testing.assert_allclose(fuglevand_gain(r), expected, rtol=1e-14)


@pytest.mark.parametrize("model_class", MODELS)
def test_force_first_discharge_and_preceding_interval(model_class):
    model = model_class(np.array([0.1, 1.0]), 10000 * pq.Hz)
    # A repeated spike in one bin is still a single discharge.
    times_by_unit = ([1.015, 1.035, 1.035, 1.195, 1.22], [1.08, 1.23, 1.25])
    block = _spike_block(times_by_unit, t_start=1.0, t_stop=2.0)
    block.segments.append(_spike_block([[], []], t_start=1.0, t_stop=2.0).segments[0])
    force = model.generate_force(block, verbose=False)
    expected = np.zeros(len(force))
    for n, times in enumerate(times_by_unit):
        times = np.unique(times)
        T = model.contraction_times__samples[n] / 10000
        gains = np.r_[1.0, fuglevand_gain(T / np.diff(times))]
        for time, gain in zip(times, gains):
            t = np.maximum(force.times.rescale("s").magnitude - time, 0)
            expected += gain * model.peak_twitch_forces__unitless[n] * (t / T) * np.exp(1 - t / T)
    np.testing.assert_allclose(force.magnitude[:, 0], expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_array_equal(force.magnitude[:, 1], 0)
    assert force.t_start == 1 * pq.s
    assert force.sampling_rate == 10000 * pq.Hz
    assert force.shape == (10000, 2)
    assert force.units == pq.dimensionless


def test_force_duration_not_multiple_of_float_grid():
    # np.arange(0, L * dt, dt) has L + 1 samples for this length.
    block = _spike_block([[0.1, 0.2]] * 10, dt_ms=0.025, t_stop=20940 * 0.025e-3)
    force = ForceModel(THRESHOLDS, 2048 * pq.Hz).generate_force(block, verbose=False)
    assert np.isfinite(force.magnitude).all()


@pytest.mark.parametrize("model_class", MODELS)
def test_force_sampling_invariance(model_class):
    means = []
    for fs in (100, 2048, 10000):
        model = model_class(THRESHOLDS, fs * pq.Hz)
        T = model.contraction_times__samples / fs
        expected = np.sum(np.e * model.peak_twitch_forces__unitless * T * 30 * fuglevand_gain(T * 30))
        for dt_ms in (0.025, 0.1, 0.5):
            block = _spike_block([np.arange(0.01, 2, 1 / 30)] * 10, dt_ms=dt_ms)
            force = model.generate_force(block, verbose=False)
            assert force.shape == (2 * fs, 1)
            # [1, 2) seconds spans exactly 30 firing periods.
            mean = force.magnitude[fs:, 0].mean()
            assert mean == pytest.approx(expected, rel=0.005)
            means.append(mean)
    assert np.ptp(means) / np.mean(means) < 0.005


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("unit", [0, 9], ids=["slow", "fast"])
@pytest.mark.parametrize("rate", [5, 10, 20, 40, 120])
def test_force_matches_closed_form(model_class, unit, rate):
    model = model_class(THRESHOLDS, 10000 * pq.Hz)
    times_by_unit = [[] for _ in THRESHOLDS]
    times_by_unit[unit] = np.arange(0.01, 2, 1 / rate)
    force = model.generate_force(_spike_block(times_by_unit), verbose=False)
    T = model.contraction_times__samples[unit] / 10000
    P = model.peak_twitch_forces__unitless[unit]
    expected = np.e * P * T * rate * fuglevand_gain(T * rate)
    # The final second contains an integer number of firing periods at every rate.
    assert force.magnitude[10000:, 0].mean() == pytest.approx(expected, rel=0.005)


@pytest.mark.parametrize("model_class", MODELS)
def test_force_saturates(model_class):
    model = model_class(THRESHOLDS, 10000 * pq.Hz)
    means = []
    for rate in (50, 100):
        block = _spike_block([np.arange(0.01, 2, 1 / rate)] + [[] for _ in range(9)])
        force = model.generate_force(block, verbose=False)
        means.append(force.magnitude[10000:, 0].mean())
    assert means[1] / means[0] < 1.01
    assert means[1] == pytest.approx(9.05 * model.peak_twitch_forces__unitless[0], rel=0.005)


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("unit", [0, 9], ids=["slow", "fast"])
def test_force_single_spike_has_analytic_tail(model_class, unit):
    model = model_class(THRESHOLDS, 10000 * pq.Hz)
    times_by_unit = [[] for _ in THRESHOLDS]
    times_by_unit[unit] = [0.01]
    force = model.generate_force(_spike_block(times_by_unit), verbose=False)
    T = model.contraction_times__samples[unit] / 10000
    P = model.peak_twitch_forces__unitless[unit]
    t = np.maximum(force.times.rescale("s").magnitude - 0.01, 0)
    expected = P * (t / T) * np.exp(1 - t / T)
    np.testing.assert_allclose(force.magnitude[:, 0], expected, rtol=1e-8, atol=1e-10)
    assert np.all(force.magnitude[t > 5 * T, 0] > 0)
    lengths = np.ceil(10 * model.contraction_times__samples).astype(int)
    assert model.twitch_mat.shape == (max(lengths), len(THRESHOLDS))
    assert [len(twitch) for twitch in model.twitch_list] == list(lengths)

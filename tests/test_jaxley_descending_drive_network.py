"""Regression tests for the profile-matched Jaxley descending-drive path."""

import numpy as np
import pytest

from myogen.simulator.jaxley.descending_drive_network import (
    _exp2_constants,
    equal_fan_in_connectivity,
    excitation_to_drive_hz,
)
from myogen.simulator.jaxley.populations import AlphaMN__Pool


def test_equal_fan_in_is_seeded_and_exact():
    a = equal_fan_in_connectivity(8, 17, 6, seed=42)
    b = equal_fan_in_connectivity(8, 17, 6, seed=42)
    c = equal_fan_in_connectivity(8, 17, 6, seed=43)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a.sum(axis=1), np.full(8, 6.0))
    assert not np.array_equal(a, c)


def test_excitation_mapping_matches_profile_definition():
    excitation, hz = excitation_to_drive_hz(
        [(0.001, 0.0, 0.0), (0.001, 0.5, 0.5)],
        threshold_hz=100.0,
        max_hz=300.0,
        timestep_ms=1.0,
        noise_seed=42,
    )
    np.testing.assert_allclose(excitation, [0.0, 0.5])
    # Noise is half-normal, so rest is at least 0.8*threshold and active drive
    # is at least threshold + excitation*(max-threshold).
    assert hz[0] >= 80.0
    assert hz[1] >= 200.0


def test_exp2syn_constants_are_peak_normalized():
    dt, tau1, tau2 = 0.001, 0.2, 0.3
    a1, a2, norm = _exp2_constants(dt, tau1, tau2)
    t = np.arange(0.0, 3.0, dt)
    waveform = norm * (np.exp(-t / tau2) - np.exp(-t / tau1))
    assert waveform.max() == pytest.approx(1.0, rel=2e-5)
    assert 0 < a1 < a2 < 1


def test_nerlab_raw_threshold_mode_matches_neuron_interpolation_convention():
    thresholds = np.array([0.2, 0.6, 1.0])
    normalized = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        normalize_nerlab_thresholds=True,
    )
    raw = AlphaMN__Pool(
        recruitment_thresholds__array=thresholds,
        normalize_nerlab_thresholds=False,
    )
    norm_diam = normalized.neuron_params["soma_diameter"]
    raw_diam = raw.neuron_params["soma_diameter"]
    assert norm_diam[0] < raw_diam[0]
    assert raw_diam[-1] == pytest.approx(norm_diam[-1])

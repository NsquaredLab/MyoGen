"""Recruitment force (% MVC) from an MVC ramp, and motor-unit CV mapped from it."""

import inspect

import numpy as np
import pytest

from myogen import set_random_seed

from myogen.utils.ramp_analysis import cv_from_recruitment_force, recruitment_force_pct_mvc

FS = 1000.0
FORCE = np.linspace(0.0, 100.0, int(10 * FS) + 1)  # 0 -> 100 % MVC over 10 s


def _train(onset_s, rate_hz=10.0, end_s=10.0):
    return np.arange(onset_s, end_s, 1.0 / rate_hz)


def test_recruitment_force_is_the_force_at_the_first_sustained_firing():
    trains = [_train(1.0), _train(4.25), np.array([]), np.r_[0.5, _train(6.0)]]
    rt = recruitment_force_pct_mvc(trains, FORCE, FS)
    np.testing.assert_allclose(rt[:2], [10.0, 42.5], atol=0.1)
    assert np.isnan(rt[2])
    assert rt[3] == pytest.approx(60.0, abs=0.1)  # an isolated spike at 0.5 s does not count


def test_cv_passes_through_the_del_vecchio_band_means():
    rt = np.array([5.0, 10.0, 15.0, 40.0, 55.0, 65.0, np.nan])
    cv, (a, b) = cv_from_recruitment_force(rt, residual_sd__m_s=0.0)
    assert a + b * np.mean([5.0, 10.0, 15.0]) == pytest.approx(3.89)
    assert a + b * np.mean([55.0, 65.0]) == pytest.approx(5.12)
    np.testing.assert_allclose(cv[:-1], a + b * rt[:-1])
    assert cv[-1] == pytest.approx(a + b * 100.0)  # never recruited on the ramp: placed at 100 % MVC


def test_cv_needs_units_in_both_bands():
    with pytest.raises(ValueError):
        cv_from_recruitment_force(np.array([5.0, 10.0, 40.0]))


def test_cv_scatter_is_the_del_vecchio_residual():
    """Units scatter about the line by sqrt(1 - R^2) * SD: R^2 0.70, SD 0.71 m/s [delvecchio2018 p. 2]."""
    default = inspect.signature(cv_from_recruitment_force).parameters["residual_sd__m_s"].default
    assert default == pytest.approx(0.39, abs=0.005)
    rt = np.r_[np.linspace(0.0, 70.0, 4000), 5.0, 60.0]
    cv, (a, b) = cv_from_recruitment_force(rt, rng=np.random.default_rng(1))
    residual = cv - (a + b * rt)
    assert np.std(residual) == pytest.approx(default, abs=0.02)
    assert np.mean(residual) == pytest.approx(0.0, abs=0.03)


def test_cv_scatter_is_seeded():
    rt = np.array([5.0, 10.0, 15.0, 40.0, 55.0, 65.0])
    first = cv_from_recruitment_force(rt, rng=np.random.default_rng(3))[0]
    np.testing.assert_array_equal(first, cv_from_recruitment_force(rt, rng=np.random.default_rng(3))[0])
    set_random_seed(5)
    global_first = cv_from_recruitment_force(rt)[0]
    set_random_seed(5)
    np.testing.assert_array_equal(global_first, cv_from_recruitment_force(rt)[0])

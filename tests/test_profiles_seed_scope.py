"""Card builders seed only their own build: the caller's generator and seed are given back, nothing is printed."""
import warnings

import numpy as np
import pytest
import quantities as pq

from myogen import get_random_generator, get_random_seed, load_profile, set_random_seed
from myogen.profiles import ProfileNote
from tests.test_profiles_review_fixes import _small_muscle


def _expected_draw(seed):
    return np.random.default_rng(seed).integers(0, 10**9, 5)


def _assert_users_stream(seed):
    assert get_random_seed() == seed
    np.testing.assert_array_equal(get_random_generator().integers(0, 10**9, 5), _expected_draw(seed))


def _semg(card, muscle, **kw):
    from myogen.simulator import SurfaceElectrodeArray

    array = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(12 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=8.3 * pq.mm, differentiation_mode="bipolar_longitudinal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        return card.surface_emg(muscle, [array], sampling_frequency__Hz=2048 * pq.Hz,
                                internal_sampling_frequency__Hz=2048 * pq.Hz, MUs_to_simulate=[4, 11],
                                fiber_dispersion=None, iap_kernel_length__mm=150.0, **kw)


def _muaps(semg):
    from myogen.utils.neo import signal_to_grid

    block = semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
    return np.array([np.asarray(signal_to_grid(s.analogsignals[0].rescale(pq.mV))) for s in block.groups[0].segments])


def test_thresholds_leave_the_users_generator_and_seed():
    set_random_seed(123)
    load_profile().thresholds()
    _assert_users_stream(123)


def test_builders_do_not_print_the_seed(capsys):
    set_random_seed(123)
    capsys.readouterr()
    load_profile().thresholds()
    assert "Random seed set" not in capsys.readouterr().out


def test_surface_emg_build_and_muaps_leave_the_users_generator_and_seed():
    muscle = _small_muscle()
    card = load_profile()
    set_random_seed(123)
    semg = _semg(card, muscle)
    _assert_users_stream(123)
    set_random_seed(123)
    _muaps(semg)
    _assert_users_stream(123)


def test_pool_build_leaves_the_users_generator_and_seed():
    from myogen.utils.nmodl import load_nmodl_mechanisms

    load_nmodl_mechanisms()
    set_random_seed(123)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        pools = load_profile().motor_neuron_pool()
    _assert_users_stream(123)
    assert pools.profile_record["seed"] == 42


def test_surface_emg_seed_keyword_gives_another_realisation_and_is_recorded():
    muscle = _small_muscle()
    card = load_profile()
    default, other = _semg(card, muscle), _semg(card, muscle, seed=8)
    assert default.profile_record["seed"] == card.seeds["surface_emg"]
    assert other.profile_record["seed"] == 8
    assert not any("seed" in path for path in card.overrides())  # a seed is not a card field
    assert not np.array_equal(_muaps(default), _muaps(other))
    np.testing.assert_array_equal(_muaps(default), _muaps(_semg(card, muscle, seed=card.seeds["surface_emg"])))


def test_simulate_seed_keyword_reaches_the_drive_and_is_recorded(monkeypatch):
    import neo

    from myogen.profiles import _builders
    from myogen.simulator.neuron import descending_drive_network as ddn

    seen = {}

    def fake_excitation(segments, threshold_hz, max_hz, timestep_ms=0.1, noise_seed=42):
        seen["noise_seed"] = noise_seed
        return np.zeros(10), np.zeros(10)

    def fake_run(thresholds, drive_hz, dd_params, total_ms, timestep_ms=0.1, config_file="alpha_mn_default.yaml"):
        seen["seed_in_run"] = get_random_seed()
        block = neo.Block()
        block.segments.append(neo.Segment())
        return block

    class _Force:
        def generate_force(self, block, verbose=False):
            return np.full((50, 1), 100.0) * pq.dimensionless

    monkeypatch.setattr(ddn, "excitation_to_drive_hz", fake_excitation)
    monkeypatch.setattr(ddn, "run_descending_drive", fake_run)
    monkeypatch.setattr(_builders.CardBuilders, "force_model", lambda self, **kw: _Force())
    set_random_seed(123)
    res = load_profile().simulate([(1, 0, 25)], seed=5)
    assert seen == {"noise_seed": 5, "seed_in_run": 5}
    assert res.profile["seed"] == 5 and res.spikes.annotations["myogen_profile"]["seed"] == 5
    _assert_users_stream(123)


def test_thresholds_accept_a_seed():
    card = load_profile()
    np.testing.assert_array_equal(card.thresholds(seed=5), card.thresholds())  # fuglevand mode draws nothing
    with pytest.raises(TypeError):
        card.thresholds(seed=1.5)

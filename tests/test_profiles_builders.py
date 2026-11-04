"""Card builders: card values reach the library objects, overrides are noted and recorded, builds are reseeded."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import quantities as pq

from myogen import load_profile
from myogen.profiles import ProfileNote
from myogen.profiles._notes import reset_notes

INPUTS = Path(__file__).resolve().parents[1] / "benchmarks" / "semg_dispersion" / "unit_cv"
ARM_F_FIBRE_HASH = "0799727c717e684107e459c205c9991493a4ca07"
ARM_F_PROPERTY_HASH = "776e5e62a88ec4940614b28e1258d722d2df6a38"


def _notes(rec):
    return [str(w.message) for w in rec if issubclass(w.category, ProfileNote)]


def test_muscle_kwargs_match_card():
    card = load_profile()
    kw = card.muscle_kwargs()
    assert kw["radius__mm"].rescale(pq.mm).magnitude == pytest.approx(6.91)
    assert kw["fiber_density__fibers_per_mm2"].rescale(pq.mm**-2).magnitude == pytest.approx(267.0)
    assert kw["skin_thickness__mm"].rescale(pq.mm).magnitude == pytest.approx(1.2)
    assert kw["fat_thickness__mm"].rescale(pq.mm).magnitude == pytest.approx(1.3)
    assert kw["skin_conductivity__S_per_m"].rescale(pq.S / pq.m).magnitude == pytest.approx(1.0)
    assert tuple(kw["within_unit_fiber_diameter_log_sd"]) == (0.08, 0.13)
    cv = np.asarray(kw["motor_unit_conduction_velocities__m_per_s"])
    assert cv.shape == (120,)
    np.testing.assert_array_equal(cv, np.asarray(card.muscle_kwargs()["motor_unit_conduction_velocities__m_per_s"]))
    assert len(kw["recruitment_thresholds"]) == 120


@pytest.mark.skipif(not INPUTS.exists(), reason="archived card inputs not on this machine")
def test_unit_cvs_are_the_validated_ones():
    validated = json.load(open(INPUTS / "rt_force_cv_scatter.json"))["cv__m_s"]
    cv = np.asarray(load_profile().muscle_kwargs()["motor_unit_conduction_velocities__m_per_s"])
    np.testing.assert_array_equal(cv, validated)


def test_override_equal_to_card_is_not_an_override():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        card.muscle_kwargs(skin_thickness__mm=0.0012 * pq.m)
    assert _notes(rec) == []
    assert card.overrides() == {}


def test_override_prints_note_and_is_recorded():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        kw = card.surface_emg_kwargs(iap_lambda__per_mm=1.0)
        card.surface_emg_kwargs(iap_lambda__per_mm=1.0)
    notes = _notes(rec)
    assert len(notes) == 1 and "0.63" in notes[0]
    assert kw["iap_lambda__per_mm"] == 1.0
    assert card.overrides() == {"surface_emg.iap_lambda__per_mm": 1.0}
    assert card.record() == {"profile": "fdi_semg", "version": 1, "parent": None, "parent_version": None,
                             "overrides": {"surface_emg.iap_lambda__per_mm": 1.0}}


def test_note_names_other_overrides():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        card.muscle_kwargs(skin_thickness__mm=2.0 * pq.mm)
        card.surface_emg_kwargs(iap_lambda__per_mm=1.0)
    assert "muscle.skin_thickness__mm" in _notes(rec)[-1]


def test_calibrated_override_warns_calibration_stale():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        card.thresholds(recruitment_range=30.0)
    assert any("01_optimize_dd_multilevel.py" in n for n in _notes(rec))
    assert card.overrides() == {"thresholds.recruitment_range": 30.0}


def test_unknown_override_raises():
    with pytest.raises(TypeError, match="skin_thicknes__mm"):
        load_profile().muscle_kwargs(skin_thicknes__mm=1.0)


def test_force_model_uses_card_shape():
    card = load_profile()
    fm = card.force_model()
    assert fm.force_frequency_shape == (0.93, 1.29)
    assert fm.profile_record == card.record({})  # no pass-through keywords
    assert load_profile("fdi_semg.published").force_model().force_frequency_shape == (2.0, 3.0)


def test_card_built_objects_do_not_hint():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        load_profile().force_model()
    assert _notes(rec) == []


def test_builds_are_reseeded_and_match_the_validated_fibres():
    """Slow (full 120-unit FDI muscle, twice)."""
    card = load_profile()

    def hashes():
        m = card.muscle(n_jobs=4)
        fibre = hashlib.sha1(b"".join(np.ascontiguousarray(m.resulting_fiber_assignment(i)).tobytes()
                                      for i in range(120))).hexdigest()
        prop = hashlib.sha1(np.ascontiguousarray(m.muscle_fiber_diameters__mm.magnitude).tobytes()
                            + np.ascontiguousarray(m.muscle_fiber_conduction_velocities__mm_per_s.magnitude).tobytes()
                            ).hexdigest()
        return fibre, prop

    first = hashes()
    np.random.default_rng(0).normal(size=10)
    from myogen import set_random_seed
    set_random_seed(999)  # another builder or user code moved the global generator
    assert hashes() == first
    assert first == (ARM_F_FIBRE_HASH, ARM_F_PROPERTY_HASH)


def test_fibre_properties_only_with_muscle_fibres(monkeypatch):
    from myogen.simulator import Muscle
    calls = []
    for step in ("distribute_innervation_centers", "generate_muscle_fiber_centers", "assign_mfs2mns"):
        monkeypatch.setattr(Muscle, step, lambda self, *a, **k: None)
    monkeypatch.setattr(Muscle, "_generate_fiber_properties", lambda self: calls.append(1))
    load_profile().muscle()
    assert calls == [1]
    load_profile("fdi_semg.published").muscle()
    assert calls == [1]  # no dispersion on the variant: surface CVs, no fibre properties


def test_sampling_frequency_is_the_users_choice():
    card = load_profile()
    kw = card.surface_emg_kwargs()
    assert "sampling_frequency__Hz" not in kw and "internal_sampling_frequency__Hz" not in kw
    kw = card.surface_emg_kwargs(sampling_frequency__Hz=10240.0 * pq.Hz)
    assert kw["sampling_frequency__Hz"] == 10240.0 * pq.Hz and "internal_sampling_frequency__Hz" not in kw
    assert card.overrides() == {}  # not a card field: passed through, not recorded

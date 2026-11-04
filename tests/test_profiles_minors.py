"""Deferred minors of the named-profiles review: note text, warning location, note keys, parent version, hints."""
import warnings

import numpy as np
import pytest
import quantities as pq
import yaml

from myogen import load_profile
from myogen.profiles import ProfileNote
from myogen.profiles._card import CARD_DIR
from myogen.profiles._notes import reset_notes


def _notes(rec):
    return [w for w in rec if issubclass(w.category, ProfileNote)]


def _card_without_note(tmp_path, field="thresholds.recruitment_range"):
    raw = yaml.safe_load((CARD_DIR / "fdi_semg.yaml").read_text(encoding="utf-8"))
    section, name = field.split(".")
    del raw["sections"][section][name]["note"]
    assert raw["sections"][section][name]["calibrated"] is True
    path = tmp_path / "no_note.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_profile(path)


def test_calibrated_field_without_note_still_warns(tmp_path):
    reset_notes()
    card = _card_without_note(tmp_path)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        card.thresholds(recruitment_range=30.0)
    texts = [str(w.message) for w in _notes(rec)]
    assert len(texts) == 1
    assert "thresholds.recruitment_range" in texts[0] and "calibration" in texts[0]
    assert "85" in texts[0]  # the card's calibration.best_trial


def test_note_points_at_the_callers_line():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        card.muscle_kwargs(skin_thickness__mm=2.0 * pq.mm)  # muscle_kwargs -> _muscle_kwargs -> _take -> _override
        card.thresholds(recruitment_range=30.0)  # thresholds -> _take -> _override
    notes = _notes(rec)
    assert len(notes) == 2
    assert all(w.filename == __file__ for w in notes), [w.filename for w in notes]


def test_hand_built_hint_points_at_the_callers_line():
    from myogen.simulator import Muscle

    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Muscle(recruitment_thresholds=np.array([0.1, 0.5, 1.0]))
    notes = _notes(rec)
    assert len(notes) == 1 and notes[0].filename == __file__


def test_same_field_on_two_cards_notes_twice():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        load_profile("fdi_semg").thresholds(recruitment_range=30.0)
        load_profile("fdi_semg.published").thresholds(recruitment_range=30.0)
    texts = [str(w.message) for w in _notes(rec)]
    assert len(texts) == 2
    assert texts[0].startswith("fdi_semg:") and texts[1].startswith("fdi_semg.published:")


def test_record_carries_the_parent_version():
    parent = load_profile("fdi_semg")
    assert load_profile("fdi_semg.published").record()["parent_version"] == parent.version
    assert parent.record()["parent_version"] is None


def test_card_built_surface_emg_does_not_hint():
    from myogen.simulator import SurfaceElectrodeArray
    from tests.test_profiles_review_fixes import _small_muscle

    muscle = _small_muscle()
    array = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(12 * pq.mm, 0 * pq.deg),
                                  bending_radius__mm=8.3 * pq.mm, differentiation_mode="bipolar_longitudinal")
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        load_profile().surface_emg(muscle, [array], fiber_dispersion=None)
    assert [str(w.message) for w in _notes(rec) if "no profile used" in str(w.message)] == []


def test_card_built_pools_do_not_hint():
    from myogen.utils.nmodl import load_nmodl_mechanisms

    load_nmodl_mechanisms()
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pools = load_profile().motor_neuron_pool()
    assert [str(w.message) for w in _notes(rec) if "no profile used" in str(w.message)] == []
    assert pools.profile_record["profile"] == "fdi_semg"


@pytest.mark.parametrize("phrase", ["thresholds of example 01"])
def test_example_04_does_not_claim_example_01_thresholds(phrase):
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "examples/01_basic/04_simulate_muscle.py").read_text(encoding="utf-8")
    assert phrase not in text

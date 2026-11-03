"""ProfileNote: card notes once per run, and the one-line hint for components built without a card."""
import warnings

import numpy as np
import pytest

from myogen.profiles import ProfileNote
from myogen.profiles._notes import building_from_card, emit_note, reset_notes
from myogen.simulator import Muscle

THRESHOLDS = np.linspace(0.02, 1.0, 10)


def _notes(record):
    return [w for w in record if issubclass(w.category, ProfileNote)]


def test_note_prints_once_per_key():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        emit_note("surface_emg.iap_lambda__per_mm", "lambda note")
        emit_note("surface_emg.iap_lambda__per_mm", "lambda note")
        emit_note("muscle.skin_thickness__mm", "skin note")
    assert [str(w.message) for w in _notes(rec)] == ["lambda note", "skin note"]


def test_hand_built_muscle_hints_once():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Muscle(recruitment_thresholds=THRESHOLDS)
        Muscle(recruitment_thresholds=THRESHOLDS)
    notes = _notes(rec)
    assert len(notes) == 1
    assert "load_profile" in str(notes[0].message) and "Muscle" in str(notes[0].message)


def test_no_hint_inside_card_build():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with building_from_card():
            Muscle(recruitment_thresholds=THRESHOLDS)
    assert _notes(rec) == []


def test_notes_can_be_silenced():
    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ProfileNote)
        emit_note("a", "text")
        Muscle(recruitment_thresholds=THRESHOLDS)
    assert _notes(rec) == []


def test_notes_can_be_errors():
    reset_notes()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProfileNote)
        with pytest.raises(ProfileNote):
            emit_note("b", "text")

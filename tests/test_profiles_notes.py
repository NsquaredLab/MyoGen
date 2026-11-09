"""ProfileNote: card notes once per run; classes built by hand print nothing."""
import warnings

import numpy as np
import pytest
import quantities as pq

from myogen.profiles import ProfileNote
from myogen.profiles._notes import emit_note, reset_notes
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


def test_hand_built_classes_do_not_hint():
    """Classes built by hand say nothing about profile cards (a card is FDI-specific)."""
    from myogen.simulator import ForceModel

    reset_notes()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Muscle(recruitment_thresholds=THRESHOLDS)
        ForceModel(recruitment_thresholds=THRESHOLDS, recording_frequency__Hz=2048 * pq.Hz)
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

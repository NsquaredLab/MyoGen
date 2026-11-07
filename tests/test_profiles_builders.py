"""Card builders: card values reach the library objects, overrides are noted and recorded, builds are reseeded."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import quantities as pq

from myogen import load_profile
from myogen.profiles import ProfileNote
from myogen.profiles._notes import reset_notes

ROOT = Path(__file__).resolve().parents[1]
# The validated arm-F muscle (benchmarks/semg_dispersion arm delsys5r_wexp_musclertsc_lam0.63): fibres per unit,
# and fibre diameter (mm) and CV (mm/s) at evenly spaced fibre indices, with their sums.
ARM_F_FIBRES_PER_UNIT = [
    22, 21, 27, 29, 30, 26, 27, 28, 28, 29, 30, 32, 28, 33, 31, 43, 49, 43, 46, 42, 47, 45, 57, 52, 43,
    54, 69, 64, 67, 60, 54, 72, 71, 86, 79, 79, 83, 93, 73, 78, 91, 90, 101, 100, 122, 108, 112, 100,
    121, 105, 112, 119, 125, 140, 143, 170, 186, 167, 174, 216, 186, 227, 203, 203, 230, 243, 219, 265,
    262, 265, 302, 284, 269, 282, 317, 331, 307, 330, 347, 340, 351, 364, 393, 370, 422, 399, 422, 418,
    447, 442, 483, 539, 441, 537, 646, 641, 650, 605, 686, 713, 716, 712, 777, 739, 834, 850, 868, 822,
    867, 1058, 1074, 1028, 1087, 1142, 1162, 1264, 1171, 1334, 1319, 1274
]
ARM_F_INDICES = [0, 5721, 11442, 17164, 22885, 28607, 34328, 40050]
ARM_F_DIAMETERS__MM = [
    0.06728408646276396, 0.06839268111349746, 0.047277283638866574, 0.06324227369016999,
    0.06061115896061368, 0.054665281872642656, 0.0429479657676721, 0.04763986418537935
]
ARM_F_CVS__MM_PER_S = [
    4809.070973006686, 4864.50070554336, 3806.142092441618, 4758.021558504406, 4626.46582202659,
    4213.794291232723, 3634.5395187905533, 3946.1445527315545
]
ARM_F_SUMS = (2294.4213299653193, 175695618.33762425)  # diameters (mm), CVs (mm/s)


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


def test_unit_cvs_are_the_validated_ones():
    validated = json.load(open(ROOT / "benchmarks" / "semg_dispersion" / "unit_cv" / "rt_force_cv_scatter.json"))["cv__m_s"]
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
    assert any("06_optimize_dd_multilevel.py" in n for n in _notes(rec))
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

    def fibres():
        m = card.muscle(n_jobs=4)
        d = np.asarray(m.muscle_fiber_diameters__mm.magnitude, dtype=float)
        cv = np.asarray(m.muscle_fiber_conduction_velocities__mm_per_s.magnitude, dtype=float)
        counts = [len(m.resulting_fiber_assignment(i)) for i in range(120)]
        return counts, d, cv

    counts, d, cv = fibres()
    np.random.default_rng(0).normal(size=10)
    from myogen import set_random_seed
    set_random_seed(999)  # another builder or user code moved the global generator
    counts2, d2, cv2 = fibres()
    assert counts2 == counts
    np.testing.assert_array_equal(d2, d)
    np.testing.assert_array_equal(cv2, cv)
    assert counts == ARM_F_FIBRES_PER_UNIT
    np.testing.assert_allclose(d[ARM_F_INDICES], ARM_F_DIAMETERS__MM, rtol=1e-9)
    np.testing.assert_allclose(cv[ARM_F_INDICES], ARM_F_CVS__MM_PER_S, rtol=1e-9)
    np.testing.assert_allclose((d.sum(), cv.sum()), ARM_F_SUMS, rtol=1e-9)


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


@pytest.mark.parametrize("name", ["fdi_semg", "fdi_semg.published"])
def test_cards_use_the_typed_fibre_model(name):
    """The cards were validated with fibre types by motor unit, which Muscle makes opt-in."""
    from myogen.simulator import Muscle

    kw = load_profile(name).muscle_kwargs()
    assert kw["fiber_model"] == "typed"
    Muscle(**kw, autorun=False)  # typed-only arguments are accepted

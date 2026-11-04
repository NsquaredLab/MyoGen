"""Final-review fixes: overrides are applied where they are recorded, carry across builders, and records are exact."""
import copy
import warnings

import numpy as np
import pytest
import quantities as pq
import yaml

from myogen import load_profile
from myogen.profiles import ProfileError, ProfileNote
from myogen.profiles._card import CARD_DIR
from myogen.profiles._notes import reset_notes

CV = "motor_unit_conduction_velocities__m_per_s"


class _StubForce:
    def __init__(self, n):
        self.n = n

    def generate_force(self, block, verbose=False):
        return np.full((50, 1), 100.0) * pq.dimensionless


@pytest.fixture
def stub_drive(monkeypatch):
    """Capture what simulate() hands to the drive, without running NEURON."""
    import neo

    from myogen.profiles import _builders
    from myogen.simulator.neuron import descending_drive_network as ddn

    seen = {}

    def fake_excitation(segments, threshold_hz, max_hz, timestep_ms=0.1, noise_seed=42):
        seen.update(threshold_hz=threshold_hz, timestep_ms=timestep_ms)
        return np.zeros(10), np.zeros(10)

    def fake_run(thresholds, drive_hz, dd_params, total_ms, timestep_ms=0.1, config_file="alpha_mn_default.yaml"):
        seen.update(n_units=len(thresholds), run_timestep_ms=timestep_ms, config_file=config_file)
        block = neo.Block()
        block.segments.append(neo.Segment())
        return block

    monkeypatch.setattr(ddn, "excitation_to_drive_hz", fake_excitation)
    monkeypatch.setattr(ddn, "run_descending_drive", fake_run)
    monkeypatch.setattr(_builders.CardBuilders, "force_model",
                        lambda self, **kw: (seen.__setitem__("force_n", len(self.thresholds())), _StubForce(0))[1])
    return seen


def test_simulate_applies_drive_and_force_overrides(stub_drive):
    card = load_profile()
    res = card.simulate([(1, 0, 25)], timestep_ms=0.05, threshold_hz=90.0, mvc_model_units=1000.0,
                        recording_frequency__Hz=1024.0, config_file="other.yaml")
    assert stub_drive["timestep_ms"] == stub_drive["run_timestep_ms"] == 0.05
    assert stub_drive["threshold_hz"] == 90.0
    assert stub_drive["config_file"] == "other.yaml"
    assert res.fs_force__Hz == 1024.0
    assert res.force_pct_mvc[0] == pytest.approx(10.0)  # 100 / 1000 * 100
    assert set(res.profile["overrides"]) == {"drive.timestep_ms", "drive.threshold_hz", "drive.mvc_model_units",
                                             "force.recording_frequency__Hz", "motor_neurons.config_file"}


def test_simulate_uses_overridden_excitation_curve(stub_drive):
    card = load_profile()
    card.simulate([(1, 0, 25)], excitation_at_pct_mvc={0: 0.0, 25: 0.5, 100: 1.0})
    assert card.excitation_for_pct_mvc(25) == pytest.approx(0.5)


def test_overrides_carry_to_later_builds(stub_drive):
    card = load_profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        card.thresholds(n_units=60)
        assert len(card.muscle_kwargs()["recruitment_thresholds"]) == 60
        card.simulate([(1, 0, 25)])
    assert stub_drive["n_units"] == stub_drive["force_n"] == 60
    card.thresholds(n_units=120)  # back to the card value: no longer an override
    assert card.overrides() == {}


def test_override_in_another_unit_is_converted():
    reset_notes()
    card = load_profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        kw = card.muscle_kwargs(radius__mm=0.008 * pq.m)
    assert kw["radius__mm"].rescale(pq.mm).magnitude == pytest.approx(8.0)
    assert card.overrides() == {"muscle.radius__mm": pytest.approx(8.0)}


def test_quantity_on_a_unitless_field_raises():
    with pytest.raises(TypeError, match="thresholds.recruitment_range"):
        load_profile().thresholds(recruitment_range=50 * pq.mm)


def test_dispersion_equal_to_card_is_not_an_override():
    from myogen.simulator.core.emg.surface.fiber_dispersion import SurfaceFiberDispersion
    card = load_profile()
    same = card.surface_emg_kwargs()["fiber_dispersion"]
    card.surface_emg_kwargs(fiber_dispersion=copy.deepcopy(same))
    assert card.overrides() == {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        kw = card.surface_emg_kwargs(fiber_dispersion=SurfaceFiberDispersion())
    assert isinstance(kw["fiber_dispersion"], SurfaceFiberDispersion)
    assert list(card.overrides()) == ["surface_emg.fiber_dispersion"]
    assert card.surface_emg_kwargs()["fiber_dispersion"] == SurfaceFiberDispersion()  # carried, rebuilt


def test_unit_cvs_equal_to_card_are_not_an_override():
    card = load_profile()
    cvs = card.muscle_kwargs()[CV]
    card.muscle_kwargs(**{CV: np.array(cvs)})
    assert card.overrides() == {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        kw = card.muscle_kwargs(**{CV: np.full(120, 4.0)})
    np.testing.assert_array_equal(kw[CV], np.full(120, 4.0))
    assert list(card.overrides()) == ["muscle.unit_cv"]
    np.testing.assert_array_equal(card.muscle_kwargs()[CV], np.full(120, 4.0))  # carried


def test_pass_through_keywords_are_recorded_on_the_object():
    card = load_profile()
    fm = card.force_model(longest_duration_rise_time__ms=60.0 * pq.ms)
    assert card.overrides() == {}
    assert fm.profile_record["keywords"] == {"longest_duration_rise_time__ms": "60.0 ms"}


def test_internal_sampling_rate_stays_the_library_default():
    kw = load_profile().surface_emg_kwargs(sampling_frequency__Hz=2048.0 * pq.Hz)
    assert "internal_sampling_frequency__Hz" not in kw


def test_curve_without_ends_extends_to_rest_and_mvc(tmp_path):
    path = tmp_path / "short_curve.yaml"
    path.write_text("name: short_curve\nparent: fdi_semg\nversion: 1\nsections:\n  drive:\n"
                    "    excitation_at_pct_mvc: {value: {10: 0.1, 50: 0.4}, source: assumption}\n", encoding="utf-8")
    card = load_profile(path)
    assert card.excitation_for_pct_mvc(80) == pytest.approx(0.4 + 0.6 * 30 / 50)
    assert card.excitation_for_pct_mvc(5) == pytest.approx(0.05)


def test_card_missing_a_required_field_fails_at_load(tmp_path):
    raw = yaml.safe_load((CARD_DIR / "fdi_semg.yaml").read_text(encoding="utf-8"))
    del raw["sections"]["muscle"]["radius__mm"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ProfileError, match="muscle.radius__mm"):
        load_profile(path)
    del raw["sections"]["muscle"]
    raw["sections"]["thresholds"] = raw["sections"]["thresholds"]
    del raw["seeds"]["cv_scatter"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ProfileError, match="seeds.cv_scatter"):
        load_profile(path)


def _small_muscle():
    from myogen import set_random_seed
    from myogen.simulator import Muscle, RecruitmentThresholds
    set_random_seed(42)
    thresholds, _ = RecruitmentThresholds(N=12, recruitment_range__ratio=30, mode="fuglevand")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ProfileNote)
        muscle = Muscle(thresholds, radius__mm=6 * pq.mm, length__mm=60 * pq.mm,
                        fiber_density__fibers_per_mm2=6 * pq.mm**-2, mean_fiber_length__mm=50 * pq.mm,
                        var_fiber_length__mm=2 * pq.mm, radius_bone__mm=0 * pq.mm, fat_thickness__mm=1 * pq.mm)
    muscle.distribute_innervation_centers()
    muscle.generate_muscle_fiber_centers(verbose=False)
    muscle.assign_mfs2mns(verbose=False, n_jobs=1)
    return muscle


def test_card_surface_emg_muaps_do_not_depend_on_what_ran_in_between():
    from myogen import set_random_seed
    from myogen.simulator import SurfaceElectrodeArray
    from myogen.utils.neo import signal_to_grid

    muscle = _small_muscle()
    card = load_profile()

    def muaps(between):
        array = SurfaceElectrodeArray(1, 1, 10 * pq.mm, 2.5 * pq.mm, center_point__mm_deg=(12 * pq.mm, 0 * pq.deg),
                                      bending_radius__mm=8.3 * pq.mm, differentiation_mode="bipolar_longitudinal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ProfileNote)
            semg = card.surface_emg(muscle, [array], sampling_frequency__Hz=2048 * pq.Hz,
                                    internal_sampling_frequency__Hz=2048 * pq.Hz, MUs_to_simulate=[4, 11],
                                    fiber_dispersion=None, iap_kernel_length__mm=150.0)
        between()
        block = semg.simulate_muaps(n_jobs=1, verbose=False, use_gpu=False)
        return np.array([np.asarray(signal_to_grid(s.analogsignals[0].rescale(pq.mV))) for s in block.groups[0].segments])

    np.testing.assert_array_equal(muaps(lambda: None), muaps(lambda: set_random_seed(999)))


@pytest.mark.parametrize("name", ("fdi_semg", "fdi_semg.published"))
def test_cited_keys_carry_a_locator(name):
    import re
    for path, field in load_profile(name).fields().items():
        src = field.source.strip()
        if src.startswith(("assumption", "derived")):
            continue
        for key, locator in re.findall(r"\[([a-z]+[0-9]{4}[a-z]?)([^\]]*)\]", src):
            assert re.search(r"\b(p\.|Table|Fig\.|eq\.)", locator), f"{path}: [{key}] has no locator"

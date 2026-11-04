"""Running a card in % MVC: excitation mapping, refusal without a mapping, and a short calibrated-drive run."""
import numpy as np
import pytest

from myogen import load_profile
from myogen.profiles import ProfileError


def test_excitation_for_pct_mvc():
    card = load_profile()
    assert card.excitation_for_pct_mvc(25) == pytest.approx(0.18818359375000002)
    assert card.excitation_for_pct_mvc(40) == pytest.approx((0.216552734375 + 0.330078125) / 2)
    assert card.excitation_for_pct_mvc(100) == pytest.approx(1.0)
    assert card.excitation_for_pct_mvc(0) == 0.0
    np.testing.assert_allclose(card.excitation_for_pct_mvc(np.array([5, 10])), [0.048095703125, 0.08164062500000001])
    for bad in (101, -1, float("nan")):
        with pytest.raises(ValueError):
            card.excitation_for_pct_mvc(bad)


def test_card_without_mvc_mapping_refuses_pct_mvc(tmp_path):
    path = tmp_path / "no_mapping.yaml"
    path.write_text("name: no_mapping\nparent: fdi_semg\nversion: 1\nsections:\n  drive:\n"
                    "    excitation_at_pct_mvc: {value: null, source: assumption}\n", encoding="utf-8")
    with pytest.raises(ProfileError, match="drive.excitation_at_pct_mvc"):
        load_profile(path).excitation_for_pct_mvc(25)


def test_published_variant_has_its_own_mvc_mapping():
    card = load_profile("fdi_semg.published")
    assert card.value("drive.mvc_model_units") != load_profile().value("drive.mvc_model_units")
    assert 0.0 < card.excitation_for_pct_mvc(25) < 1.0


def test_simulate_short_run():
    """Slow-ish: 1.5 s of the calibrated 120-unit drive."""
    card = load_profile()
    res = card.simulate([(0.5, 0, 0), (1.0, 0, 25)])
    trains = res.spikes.segments[0].spiketrains
    assert len(trains) == 120
    assert sum(len(st) for st in trains) > 0
    assert res.fs_force__Hz == 2048.0
    assert res.force_pct_mvc.ndim == 1 and res.force_pct_mvc.size == int(1.5 * 2048)
    assert res.excitation.max() == pytest.approx(card.excitation_for_pct_mvc(25), rel=1e-3)
    assert res.spikes.annotations["myogen_profile"] == card.record()
    assert 0.0 <= res.force_pct_mvc.max() < 60.0

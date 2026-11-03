"""Profile cards: loading, variants and sources (docs/superpowers/specs/2026-11-03-named-profiles-design.md)."""
import re
from pathlib import Path

import pytest

from myogen import load_profile
from myogen.profiles import ProfileError

REPO = Path(__file__).resolve().parents[1]
REGISTER = (REPO / "docs" / "literature.md").read_text(encoding="utf-8")
SHIPPED = ("fdi_semg", "fdi_semg.published")


def test_default_is_fdi_semg():
    card = load_profile()
    assert card.name == "fdi_semg"
    assert card.version == 1
    assert card.parent is None


def test_card_values():
    card = load_profile()
    assert card.value("muscle.fiber_density__fibers_per_mm2") == 267
    assert card.value("surface_emg.iap_lambda__per_mm") == 0.63
    assert card.value("thresholds.n_units") == 120
    assert len(card.value("drive.recruitment_pct_mvc")) == 120
    assert card.value("drive.excitation_at_pct_mvc")[25] == pytest.approx(0.18818359375000002)
    assert card.field("motor_neurons.gfluctdv_scale").calibrated is True
    assert card.field("muscle.skin_thickness__mm").unit == "mm"


@pytest.mark.parametrize("name", SHIPPED)
def test_every_field_has_a_registered_source(name):
    for path, field in load_profile(name).fields().items():
        src = field.source.strip()
        assert src, f"{path}: empty source"
        if src.startswith("assumption"):
            continue
        keys = re.findall(r"\[([a-z]+[0-9]{4}[a-z]?)\b", src)
        assert keys or src.startswith("derived"), f"{path}: source {src!r} cites no register key"
        for key in keys:
            assert f'id="{key}"' in REGISTER, f"{path}: {key} is not in docs/literature.md"


def test_variant_inherits_and_changes():
    card = load_profile("fdi_semg.published")
    assert card.parent == "fdi_semg"
    assert card.value("surface_emg.iap_lambda__per_mm") == 1.0
    assert card.value("force.force_frequency_shape") == [2.0, 3.0]
    assert card.value("surface_emg.fiber_dispersion") is None
    assert card.value("muscle.fiber_density__fibers_per_mm2") == 267


def _write(tmp_path, name, text):
    p = tmp_path / f"{name}.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_variant_unknown_field_raises(tmp_path):
    p = _write(tmp_path, "bad", "name: bad\nparent: fdi_semg\nversion: 1\nsections:\n"
                                "  muscle:\n    skin_thicknes__mm: {value: 2.0, source: assumption}\n")
    with pytest.raises(ProfileError, match="muscle.skin_thicknes__mm"):
        load_profile(p)


def test_missing_required_field_raises(tmp_path):
    p = _write(tmp_path, "bad", "name: bad\nversion: 1\nseeds: {}\nsections:\n"
                                "  muscle:\n    radius__mm: {value: 6.91}\n")
    with pytest.raises(ProfileError, match="muscle.radius__mm"):
        load_profile(p)
    p = _write(tmp_path, "bad2", "name: bad2\nversion: 1\nseeds: {}\n")
    with pytest.raises(ProfileError, match="sections"):
        load_profile(p)


def test_unknown_card_name_raises():
    with pytest.raises(ProfileError, match="fdi_semg"):
        load_profile("vl_semg")

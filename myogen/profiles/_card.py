"""Profile cards: YAML files holding every validated setting of one simulation, with sources and notes.

A card is a mapping of sections (``thresholds``, ``motor_neurons``, ``drive``, ``force``, ``muscle``,
``surface_emg``) to fields. Each field has a ``value``, a ``source`` (a literature-register key with a locator, the
word ``assumption``, or ``derived: ...``), and optionally a ``unit``, a ``note`` (printed when the value is
overridden) and ``calibrated: true`` (the descending-drive calibration was fitted with this value). A variant names
its ``parent`` and lists only the fields it changes. See docs/superpowers/specs/2026-11-03-named-profiles-design.md.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from myogen.profiles._builders import CardBuilders

CARD_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = "fdi_semg"
SECTIONS = ("thresholds", "motor_neurons", "drive", "force", "muscle", "surface_emg")
FIELD_KEYS = {"value", "source", "unit", "note", "calibrated"}
# Fields and seeds the builders read: a card (with its parent) must have all of them
REQUIRED_FIELDS = {
    "thresholds": ("n_units", "mode", "recruitment_range"),
    "motor_neurons": ("config_file", "gfluctdv_scale"),
    "drive": ("dd_neurons", "conn_probability", "dd_drive__Hz", "synaptic_weight", "gamma_shape", "network",
              "threshold_hz", "timestep_ms", "mvc_model_units", "excitation_at_pct_mvc", "recruitment_pct_mvc"),
    "force": ("force_frequency_shape", "recording_frequency__Hz"),
    "muscle": ("radius__mm", "fiber_density__fibers_per_mm2", "skin_thickness__mm", "fat_thickness__mm",
               "skin_conductivity__S_per_m", "within_unit_fiber_diameter_log_sd", "unit_cv"),
    "surface_emg": ("fiber_dispersion", "iap_lambda__per_mm", "iap_kernel_length__mm"),
}
REQUIRED_SEEDS = ("thresholds", "muscle", "cv_scatter", "drive", "surface_emg")


class ProfileError(ValueError):
    """A card is missing, malformed, or used in a way it does not support."""


@dataclass(frozen=True)
class CardField:
    value: Any
    source: str
    unit: str | None = None
    note: str | None = None
    calibrated: bool = False


def available_profiles() -> list[str]:
    """Names of the cards shipped with MyoGen."""
    return sorted(p.stem for p in CARD_DIR.glob("*.yaml"))


def _read(name_or_path: str | Path) -> tuple[dict, Path]:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CARD_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise ProfileError(f"no profile {str(name_or_path)!r}; available: {', '.join(available_profiles())}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ProfileError(f"{path.name}: a card must be a YAML mapping")
    return raw, path


def _field(path: str, spec: Any) -> CardField:
    if not isinstance(spec, dict):
        raise ProfileError(f"{path}: a field must be a mapping with 'value' and 'source'")
    unknown = set(spec) - FIELD_KEYS
    if unknown:
        raise ProfileError(f"{path}: unknown keys {sorted(unknown)} (allowed: {sorted(FIELD_KEYS)})")
    for key in ("value", "source"):
        if key not in spec:
            raise ProfileError(f"{path}: missing '{key}'")
    if not isinstance(spec["source"], str) or not spec["source"].strip():
        raise ProfileError(f"{path}: 'source' must be a register key with a locator, 'assumption' or 'derived: ...'")
    return CardField(value=spec["value"], source=spec["source"], unit=spec.get("unit"), note=spec.get("note"),
                     calibrated=bool(spec.get("calibrated", False)))


def _fields(where: str, sections: Any) -> dict[str, CardField]:
    if not isinstance(sections, dict):
        raise ProfileError(f"{where}: 'sections' must be a mapping of {', '.join(SECTIONS)}")
    out = {}
    for section, fields in sections.items():
        if section not in SECTIONS:
            raise ProfileError(f"{where}: unknown section {section!r} (allowed: {', '.join(SECTIONS)})")
        if not isinstance(fields, dict):
            raise ProfileError(f"{where}: section {section!r} must be a mapping of fields")
        for name, spec in fields.items():
            out[f"{section}.{name}"] = _field(f"{section}.{name}", spec)
    return out


class Profile(CardBuilders):
    """A loaded card: its fields (read-only), the builders, and the overrides applied on this instance."""

    def __init__(self, name: str, version: int, description: str, parent: str | None, seeds: dict,
                 fields: dict[str, CardField], calibration: dict, validation: dict | None, path: Path):
        self.name = name
        self.version = version
        self.description = description
        self.parent = parent
        self.seeds = MappingProxyType(dict(seeds))
        self._fields = MappingProxyType(dict(fields))
        self.calibration = MappingProxyType(dict(calibration))
        self.validation = None if validation is None else MappingProxyType(dict(validation))
        self.path = path
        self._overrides: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Profile({self.name!r}, version={self.version})"

    def fields(self) -> dict[str, CardField]:
        return dict(self._fields)

    def field(self, path: str) -> CardField:
        try:
            return self._fields[path]
        except KeyError:
            raise ProfileError(f"{self.name} has no field {path!r}") from None

    def value(self, path: str) -> Any:
        """The card value of ``section.field`` (a copy, so callers cannot change the card)."""
        return copy.deepcopy(self.field(path).value)

    def section(self, section: str) -> dict[str, CardField]:
        prefix = f"{section}."
        return {k[len(prefix):]: v for k, v in self._fields.items() if k.startswith(prefix)}


def load_profile(name: str | Path = DEFAULT_PROFILE) -> Profile:
    """Load a profile card by name (``"fdi_semg"``, ``"fdi_semg.published"``) or from a YAML path."""
    raw, path = _read(name)
    where = path.name
    for key in ("name", "version"):
        if key not in raw:
            raise ProfileError(f"{where}: missing top-level '{key}'")
    if "sections" not in raw:
        raise ProfileError(f"{where}: missing top-level 'sections'")
    own = _fields(where, raw["sections"])
    parent_name = raw.get("parent")
    if parent_name is None:
        if "seeds" not in raw:
            raise ProfileError(f"{where}: missing top-level 'seeds'")
        fields, seeds, calibration = own, dict(raw["seeds"] or {}), dict(raw.get("calibration") or {})
    else:
        parent = load_profile(parent_name)
        fields = parent.fields()
        for key, f in own.items():
            if key not in fields:
                raise ProfileError(f"{where}: {key} is not a field of the parent card {parent_name!r}")
            fields[key] = f
        seeds = {**parent.seeds, **(raw.get("seeds") or {})}
        calibration = {**parent.calibration, **(raw.get("calibration") or {})}
    missing = [f"{s}.{n}" for s, names in REQUIRED_FIELDS.items() for n in names if f"{s}.{n}" not in fields]
    missing += [f"seeds.{k}" for k in REQUIRED_SEEDS if k not in seeds]
    if missing:
        raise ProfileError(f"{where}: missing required {', '.join(missing)}")
    return Profile(name=str(raw["name"]), version=int(raw["version"]), description=str(raw.get("description", "")),
                   parent=parent_name, seeds=seeds, fields=fields, calibration=calibration,
                   validation=raw.get("validation"), path=path)

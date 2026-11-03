"""Card builders: the library objects a card describes, with keyword overrides noted and recorded.

Every builder reseeds the global generator from the card's ``seeds`` before it draws, so a build does not depend on
what ran before it. Keywords equal to the card value (unit-aware) are not overrides. A differing keyword is recorded
(``overrides()``, ``record()``) and, when the field carries a note or the drive calibration depends on it, the note
is printed once per run (``ProfileNote``).
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import quantities as pq

from myogen.profiles._notes import building_from_card, emit_note

# Card field -> constructor keyword, where they differ
_THRESHOLD_KEYWORDS = {"n_units": "N", "recruitment_range": "recruitment_range__ratio", "mode": "mode"}
_UNIT_CV_KEYWORD = "motor_unit_conduction_velocities__m_per_s"


def _magnitude(value: Any, unit: str | None) -> Any:
    if isinstance(value, pq.Quantity):
        return value.rescale(pq.Quantity(1.0, unit)).magnitude if unit else value.magnitude
    return value


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return a == b
    return x.shape == y.shape and bool(np.allclose(x, y, rtol=1e-12, atol=0.0))


def _plain(value: Any, unit: str | None) -> Any:
    """A JSON-friendly copy of an override for the record."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return asdict(value)
    value = _magnitude(value, unit)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class CardBuilders:
    """Builder methods of ``Profile`` (mixed in; uses ``field``, ``value``, ``seeds``, ``name`` and ``version``)."""

    # -- overrides --------------------------------------------------------------------------------------------
    def overrides(self) -> dict[str, Any]:
        """Every override applied so far on this card instance, as ``{"section.field": value}``."""
        return dict(self._overrides)

    def record(self) -> dict[str, Any]:
        """What a result was built with: card name, version, parent and overrides."""
        return {"profile": self.name, "version": self.version, "parent": self.parent, "overrides": self.overrides()}

    def _take(self, section: str, overrides: dict, allowed: set[str], aliases: dict[str, str] | None = None) -> dict:
        """Card values of ``section`` with ``overrides`` applied; returns ``{field: value}`` (card units)."""
        aliases = aliases or {}
        out = {name: self.value(f"{section}.{name}") for name in self.section(section)}
        for key, given in overrides.items():
            name = aliases.get(key, key)
            path = f"{section}.{name}"
            if name in out:
                f = self.field(path)
                if key in aliases:  # a constructor keyword standing in for a card field
                    out[key] = given
                    self._override(path, f, _plain(given, None))
                    continue
                if _same(_magnitude(given, f.unit), f.value) and not _is_object(given):
                    continue
                if _is_object(given) and _same(_plain(given, None), f.value):
                    continue
                out[name] = given
                self._override(path, f, _plain(given, f.unit))
            elif key in allowed:  # a constructor keyword the card does not set (the user's choice): not recorded
                out[key] = given
            else:
                raise TypeError(f"{self.name}: unknown keyword {key!r} for {section} "
                                f"(card fields: {sorted(self.section(section))})")
        return out

    def _override(self, path: str, field, value: Any) -> None:
        others = sorted(p for p in self._overrides if p != path)
        self._overrides[path] = value
        if field.note:
            text = f"{self.name}: {path} overridden ({value!r}). {field.note}"
            if others:
                text += f" You also changed: {', '.join(others)}."
            emit_note(path, text)

    def _q(self, section: str, name: str, value: Any) -> Any:
        """Attach the card unit to a plain value."""
        unit = self.field(f"{section}.{name}").unit
        if unit is None or value is None or isinstance(value, pq.Quantity):
            return value
        return value * pq.Quantity(1.0, unit)

    # -- builders ---------------------------------------------------------------------------------------------
    def thresholds(self, **overrides) -> np.ndarray:
        """Recruitment thresholds, normalised to a largest value of 1."""
        from myogen import set_random_seed
        from myogen.simulator import RecruitmentThresholds

        v = self._take("thresholds", overrides, set())
        set_random_seed(int(self.seeds["thresholds"]))
        thr, _ = RecruitmentThresholds(**{_THRESHOLD_KEYWORDS[k]: v[k] for k in _THRESHOLD_KEYWORDS})
        thr = np.asarray(thr, dtype=float)
        return thr / thr.max()

    def muscle_kwargs(self, **overrides) -> dict[str, Any]:
        """The ``Muscle(...)`` keywords of the card (pure: nothing is built)."""
        from myogen.simulator import Muscle
        from myogen.utils.ramp_analysis import cv_from_recruitment_force

        allowed = set(inspect.signature(Muscle.__init__).parameters) - {"self"}
        v = self._take("muscle", overrides, allowed, aliases={_UNIT_CV_KEYWORD: "unit_cv"})
        kw = {k: val for k, val in v.items() if k in allowed}
        for name in ("radius__mm", "fiber_density__fibers_per_mm2", "skin_thickness__mm", "fat_thickness__mm",
                     "skin_conductivity__S_per_m"):
            kw[name] = self._q("muscle", name, v[name])
        log_sd = v["within_unit_fiber_diameter_log_sd"]
        kw["within_unit_fiber_diameter_log_sd"] = None if log_sd is None else tuple(log_sd)
        if _UNIT_CV_KEYWORD not in v:
            spec = v["unit_cv"]
            if spec is not None:
                if spec.get("from") != "recruitment_force":
                    raise ValueError(f"{self.name}: muscle.unit_cv supports from: recruitment_force, got {spec!r}")
                cv, _ = cv_from_recruitment_force(
                    np.asarray(self.value("drive.recruitment_pct_mvc"), dtype=float),
                    residual_sd__m_s=float(spec["residual_sd__m_s"]),
                    rng=np.random.default_rng(int(self.seeds["cv_scatter"])))
                kw[_UNIT_CV_KEYWORD] = cv
        kw.setdefault("recruitment_thresholds", self.thresholds())
        return kw

    def muscle(self, n_jobs: int = -2, **overrides):
        """The card's ``Muscle``, built (innervation centres, fibres, assignment, fibre properties).

        Fibre properties are generated only when the card's fibre dispersion uses Muscle fibres
        (``fiber_properties: muscle``), with the global generator's state restored afterwards, as in the benchmark
        the card was validated with, so every later draw is the same with or without them.
        """
        from myogen import get_random_generator, set_random_seed
        from myogen.simulator import Muscle

        kw = self.muscle_kwargs(**overrides)
        with building_from_card():
            set_random_seed(int(self.seeds["muscle"]))
            m = Muscle(**kw)
        m.distribute_innervation_centers()
        m.generate_muscle_fiber_centers(verbose=False)
        m.assign_mfs2mns(verbose=False, n_jobs=n_jobs)
        disp = self.value("surface_emg.fiber_dispersion")
        if isinstance(disp, dict) and disp.get("fiber_properties") == "muscle":
            state = get_random_generator().bit_generator.state
            m._generate_fiber_properties()
            get_random_generator().bit_generator.state = state
        m.profile_record = self.record()
        return m

    def force_model(self, **overrides):
        """The card's ``ForceModel`` (twitches and force-frequency curve the drive was calibrated with)."""
        from myogen.simulator.core.force.force_model import ForceModel

        allowed = set(inspect.signature(ForceModel.__init__).parameters) - {"self"}
        v = self._take("force", overrides, allowed)
        kw = {k: val for k, val in v.items() if k in allowed and k not in ("force_frequency_shape",
                                                                             "recording_frequency__Hz")}
        kw.setdefault("recruitment_thresholds", self.thresholds())
        with building_from_card():
            fm = ForceModel(recording_frequency__Hz=self._q("force", "recording_frequency__Hz",
                                                            v["recording_frequency__Hz"]),
                            force_frequency_shape=tuple(float(x) for x in v["force_frequency_shape"]), **kw)
        fm.profile_record = self.record()
        return fm

    def dd_params(self, **overrides) -> dict[str, Any]:
        """Descending-drive parameters in the calibration's format (``run_descending_drive``)."""
        drive = self._take("drive", {k: val for k, val in overrides.items() if k != "gfluctdv_scale"}, set())
        mn = self._take("motor_neurons", {k: val for k, val in overrides.items() if k == "gfluctdv_scale"}, set())
        p = {k: _magnitude(drive[k], self.field(f"drive.{k}").unit)
             for k in ("dd_neurons", "conn_probability", "dd_drive__Hz", "synaptic_weight", "gamma_shape", "network")}
        p["gfluctdv_scale"] = float(mn["gfluctdv_scale"])
        return p

    def motor_neuron_pool(self, **overrides):
        """Motor neuron pool, descending-drive pool and their calibrated wiring (``mn_pool``, ``dd_pool``, ``wiring``)."""
        from myogen import set_random_seed
        from myogen.simulator.neuron.descending_drive_network import build_pools

        mn = self._take("motor_neurons", {k: v for k, v in overrides.items() if k == "config_file"}, set())
        params = self.dd_params(**{k: v for k, v in overrides.items() if k != "config_file"})
        thr = self.thresholds()
        set_random_seed(int(self.seeds["drive"]))
        with building_from_card():
            pools = build_pools(thr, params, float(self.value("drive.timestep_ms")), mn["config_file"])
        pools.profile_record = self.record()
        return pools

    def surface_emg_kwargs(self, **overrides) -> dict[str, Any]:
        """The card's ``SurfaceEMG`` keywords besides the muscle and electrodes (pure: nothing is built)."""
        from myogen.simulator import SurfaceEMG
        from myogen.simulator.core.emg.surface.fiber_dispersion import SurfaceFiberDispersion

        allowed = set(inspect.signature(SurfaceEMG.__init__).parameters) - {"self", "muscle_model", "electrode_arrays"}
        v = self._take("surface_emg", overrides, allowed)
        kw = {k: val for k, val in v.items() if k in allowed}
        disp = v["fiber_dispersion"]
        if isinstance(disp, dict):
            disp = SurfaceFiberDispersion(**{k: tuple(x) if isinstance(x, list) else x for k, x in disp.items()})
        kw["fiber_dispersion"] = disp
        kw["iap_lambda__per_mm"] = float(_magnitude(v["iap_lambda__per_mm"], "1/mm"))
        kw["iap_kernel_length__mm"] = float(_magnitude(v["iap_kernel_length__mm"], "mm"))
        if "sampling_frequency__Hz" in kw and "internal_sampling_frequency__Hz" not in kw:
            kw["internal_sampling_frequency__Hz"] = kw["sampling_frequency__Hz"]  # simulate at the output rate
        return kw

    def surface_emg(self, muscle, electrode_arrays, **overrides):
        """The card's ``SurfaceEMG`` for ``muscle`` (from ``muscle()``) and the user's electrode arrays."""
        from myogen import set_random_seed
        from myogen.simulator import SurfaceEMG

        kw = self.surface_emg_kwargs(**overrides)
        with building_from_card():
            set_random_seed(int(self.seeds["surface_emg"]))
            semg = SurfaceEMG(muscle, electrode_arrays, **kw)
        semg.profile_record = self.record()
        return semg


def _is_object(value: Any) -> bool:
    from dataclasses import is_dataclass
    return is_dataclass(value)

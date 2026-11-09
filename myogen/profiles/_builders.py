"""Card builders: the library objects a card describes, with keyword overrides noted and recorded.

Seeds. Each builder that draws takes a ``seed`` (default: the card's, from ``seeds``; recorded as ``seed`` in the
built object's ``profile_record``) and draws its build from a generator seeded with it, so a build does not depend on
what ran before it, and reproduces the per-stage seeds of the runs the card was validated with. Afterwards the
caller's generator and seed are given back (``myogen._seed_scope``): the user's ``set_random_seed`` stream is as it
was, and nothing is printed. This deviates from base MyoGen, where library code never seeds and the script owns
the seed (one ``set_random_seed`` call, or the import-time default, for the whole script). The cell ID counters are
reset at the start of a seeded build, as ``set_random_seed`` does. On any platform the seeded builds give the
validated thresholds and fibres; ``simulate`` gives the same spikes up to NEURON's floating point, which differs
between platforms (spike times can shift by a time step, and a unit can gain or lose a spike near the end of a run),
so sEMG matches the validation within its limits.
A keyword that names a card field overrides it on this card instance: the override applies to
every later build from the same instance (``thresholds(n_units=60)`` then ``muscle()`` builds 60 units), is recorded
(``overrides()``, ``record()``) and, when the field carries a note, prints the note once per card and field per run
(``ProfileNote``).
A keyword equal to the card value (unit-aware) is not an override, and clears an earlier one. Constructor keywords
the card does not set (electrodes, sampling, units to simulate, ...) are passed through and listed under
``keywords`` in the built object's ``profile_record``.
"""
from __future__ import annotations

import copy
import inspect
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np
import quantities as pq

from myogen.profiles._notes import emit_note

# Card field -> constructor keyword, where they differ
_THRESHOLD_KEYWORDS = {"n_units": "N", "recruitment_range": "recruitment_range__ratio", "mode": "mode"}
_UNIT_CV_KEYWORD = "motor_unit_conduction_velocities__m_per_s"
_PER_UNIT = "per_unit__m_per_s"  # muscle.unit_cv override holding explicit per-unit CVs


def _magnitude(value: Any, unit: str | None) -> Any:
    if isinstance(value, pq.Quantity):
        return value.rescale(pq.Quantity(1.0, unit)).magnitude if unit else value.magnitude
    return value


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict) and isinstance(b, dict):
        return {str(k) for k in a} == {str(k) for k in b} and all(
            _same(a[k], next(b[j] for j in b if str(j) == str(k))) for k in a)
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return a == b
    return x.shape == y.shape and bool(np.allclose(x, y, rtol=1e-12, atol=0.0))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _describe(value: Any) -> Any:
    """A JSON-friendly description of a passed-through keyword for the record."""
    if isinstance(value, pq.Quantity):
        mag = _jsonable(np.asarray(value.magnitude))
        return f"{mag} {value.dimensionality.string}"
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, (str, int, float, bool, type(None), np.ndarray, np.generic, list, tuple, dict)):
        return _jsonable(value)
    return repr(value)


def _dataclass_from(kind: type, spec: dict) -> Any:
    return kind(**{k: tuple(x) if isinstance(x, list) else x for k, x in spec.items()})


class CardBuilders:
    """Builder methods of ``Profile`` (mixed in; uses ``field``, ``value``, ``seeds``, ``name`` and ``version``)."""

    # -- overrides --------------------------------------------------------------------------------------------
    def overrides(self) -> dict[str, Any]:
        """Every override in force on this card instance, as ``{"section.field": value}`` (card units)."""
        return copy.deepcopy(self._overrides)

    def record(self, keywords: dict | None = None, seed: int | None = None) -> dict[str, Any]:
        """What a result was built with: card name, version, parent, overrides (and passed-through keywords, seed)."""
        rec = {"profile": self.name, "version": self.version, "parent": self.parent,
               "parent_version": self.parent_version, "overrides": self.overrides()}
        if keywords is not None:
            rec["keywords"] = dict(keywords)
        if seed is not None:
            rec["seed"] = seed
        return rec

    def _seed(self, stage: str, seed: int | None) -> int:
        """The seed of a build: ``seed`` if given, else the card's seed for ``stage``."""
        if seed is None:
            return int(self.seeds[stage])
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"{self.name}: seed must be an integer, got {seed!r}")
        return int(seed)

    def _card_form(self, path: str, field, key: str, given: Any) -> Any:
        """``given`` in the card's format and units (what the card value would be)."""
        if key == _UNIT_CV_KEYWORD:
            return {_PER_UNIT: np.asarray(_magnitude(given, "m/s"), dtype=float).tolist()}
        if is_dataclass(given):
            return {k: _jsonable(v) for k, v in asdict(given).items()}
        if isinstance(given, pq.Quantity):
            if field.unit is None:
                raise TypeError(f"{self.name}: {path} has no unit; got the quantity {given!r}")
            return _jsonable(_magnitude(given, field.unit))
        return _jsonable(given)

    def _card_value_in_form_of(self, path: str, key: str, given: Any) -> Any:
        """The card value, in the same form as ``_card_form`` gives for ``given``."""
        value = self.value(path)
        if key == _UNIT_CV_KEYWORD:
            cvs = self._unit_cvs(value)
            return None if cvs is None else {_PER_UNIT: cvs.tolist()}
        if is_dataclass(given) and isinstance(value, dict):
            return {k: _jsonable(v) for k, v in asdict(_dataclass_from(type(given), value)).items()}
        return value

    def _take(self, section: str, overrides: dict, allowed: set[str], aliases: dict[str, str] | None = None,
              keywords: dict | None = None) -> dict:
        """Card values of ``section`` with this instance's overrides and ``overrides`` applied (card units).

        Keywords naming a card field (or an alias of one) override it; keywords in ``allowed`` pass through and are
        described in ``keywords``; anything else raises ``TypeError``.
        """
        aliases = aliases or {}
        fields = self.section(section)
        out = {name: self.value(f"{section}.{name}") for name in fields}
        for path, val in self._overrides.items():
            sec, name = path.split(".", 1)
            if sec == section:
                out[name] = copy.deepcopy(val)
        for key, given in overrides.items():
            name = aliases.get(key, key)
            path = f"{section}.{name}"
            if name in fields:
                f = self.field(path)
                plain = self._card_form(path, f, key, given)
                if _same(plain, self._card_value_in_form_of(path, key, given)):
                    self._overrides.pop(path, None)  # back to the card value
                    out[name] = self.value(path)
                    continue
                out[name] = copy.deepcopy(plain)
                self._override(path, f, plain)
            elif key in allowed:  # a constructor keyword the card does not set: the user's choice
                out[key] = given
                if keywords is not None:
                    keywords[key] = _describe(given)
            else:
                raise TypeError(f"{self.name}: unknown keyword {key!r} for {section} "
                                f"(card fields: {sorted(fields)})")
        return out

    def _override(self, path: str, field, value: Any) -> None:
        others = sorted(p for p in self._overrides if p != path)
        self._overrides[path] = copy.deepcopy(value)
        note = field.note
        if not note and field.calibrated:
            note = (f"The descending-drive calibration (trial {self.calibration.get('best_trial', '?')}) was fitted "
                    "with this value, so the drive calibration no longer holds.")
        if note:
            text = f"{self.name}: {path} overridden ({value!r}). {note}"
            if others:
                text += f" You also changed: {', '.join(others)}."
            emit_note(f"{self.name}:{path}", text)

    def _q(self, section: str, name: str, value: Any) -> Any:
        """Attach the card unit to a plain value."""
        unit = self.field(f"{section}.{name}").unit
        if unit is None or value is None or isinstance(value, pq.Quantity):
            return value
        return value * pq.Quantity(1.0, unit)

    def _unit_cvs(self, spec: Any) -> np.ndarray | None:
        """Per-unit fibre CVs (m/s) for a ``muscle.unit_cv`` value, or None for surface CVs."""
        from myogen.utils.ramp_analysis import cv_from_recruitment_force

        if spec is None:
            return None
        if _PER_UNIT in spec:
            return np.asarray(spec[_PER_UNIT], dtype=float)
        if spec.get("from") != "recruitment_force":
            raise ValueError(f"{self.name}: muscle.unit_cv supports from: recruitment_force, got {spec!r}")
        rt = self._take("drive", {}, set())["recruitment_pct_mvc"]
        cv, _ = cv_from_recruitment_force(np.asarray(rt, dtype=float),
                                          residual_sd__m_s=float(spec["residual_sd__m_s"]),
                                          rng=np.random.default_rng(int(self.seeds["cv_scatter"])))
        return np.asarray(cv, dtype=float)

    # -- builders ---------------------------------------------------------------------------------------------
    def thresholds(self, seed: int | None = None, **overrides) -> np.ndarray:
        """Recruitment thresholds, normalised to a largest value of 1."""
        from myogen import _seed_scope
        from myogen.simulator import RecruitmentThresholds

        seed = self._seed("thresholds", seed)
        v = self._take("thresholds", overrides, set())
        with _seed_scope(seed):
            thr, _ = RecruitmentThresholds(**{_THRESHOLD_KEYWORDS[k]: v[k] for k in _THRESHOLD_KEYWORDS})
        thr = np.asarray(thr, dtype=float)
        return thr / thr.max()

    def muscle_kwargs(self, **overrides) -> dict[str, Any]:
        """The ``Muscle(...)`` keywords of the card (nothing is built)."""
        return self._muscle_kwargs(overrides, {})

    def _muscle_kwargs(self, overrides: dict, keywords: dict) -> dict[str, Any]:
        from myogen.simulator import Muscle

        allowed = set(inspect.signature(Muscle.__init__).parameters) - {"self"}
        v = self._take("muscle", overrides, allowed, aliases={_UNIT_CV_KEYWORD: "unit_cv"}, keywords=keywords)
        kw = {k: val for k, val in v.items() if k in allowed}
        for name in ("radius__mm", "fiber_density__fibers_per_mm2", "skin_thickness__mm", "fat_thickness__mm",
                     "skin_conductivity__S_per_m"):
            kw[name] = self._q("muscle", name, v[name])
        log_sd = v["within_unit_fiber_diameter_log_sd"]
        kw["within_unit_fiber_diameter_log_sd"] = None if log_sd is None else tuple(log_sd)
        cvs = self._unit_cvs(v["unit_cv"])
        if cvs is not None:
            kw[_UNIT_CV_KEYWORD] = cvs
        kw.setdefault("recruitment_thresholds", self.thresholds())
        return kw

    def muscle(self, n_jobs: int = -2, seed: int | None = None, **overrides):
        """The card's ``Muscle``, built (innervation centres, fibres, assignment, fibre properties).

        Fibre properties are generated only when the card's fibre dispersion uses Muscle fibres
        (``fiber_properties: muscle``), with the global generator's state restored afterwards, as in the benchmark
        the card was validated with, so every later draw is the same with or without them.
        """
        from myogen import _seed_scope
        from myogen.simulator import Muscle

        seed = self._seed("muscle", seed)
        keywords: dict = {}
        kw = self._muscle_kwargs(overrides, keywords)
        disp = self._take("surface_emg", {}, set())["fiber_dispersion"]
        with _seed_scope(seed) as rng:
            m = Muscle(**kw)
            m.distribute_innervation_centers()
            m.generate_muscle_fiber_centers(verbose=False)
            m.assign_mfs2mns(verbose=False, n_jobs=n_jobs)
            if isinstance(disp, dict) and disp.get("fiber_properties") == "muscle":
                state = rng.bit_generator.state
                m._generate_fiber_properties()
                rng.bit_generator.state = state
        m.profile_record = self.record(keywords, seed=seed)
        return m

    def force_model(self, **overrides):
        """The card's ``ForceModel`` (twitches and force-frequency curve the drive was calibrated with)."""
        from myogen.simulator.core.force.force_model import ForceModel

        allowed = set(inspect.signature(ForceModel.__init__).parameters) - {"self"}
        keywords: dict = {}
        v = self._take("force", overrides, allowed, keywords=keywords)
        kw = {k: val for k, val in v.items() if k in allowed and k not in ("force_frequency_shape",
                                                                             "recording_frequency__Hz")}
        kw.setdefault("recruitment_thresholds", self.thresholds())
        fm = ForceModel(recording_frequency__Hz=self._q("force", "recording_frequency__Hz", v["recording_frequency__Hz"]),
                        force_frequency_shape=tuple(float(x) for x in v["force_frequency_shape"]), **kw)
        fm.profile_record = self.record(keywords)
        return fm

    def dd_params(self, **overrides) -> dict[str, Any]:
        """Descending-drive parameters in the calibration's format (``run_descending_drive``).

        Keywords may name fields of the drive or the motor_neurons section.
        """
        mn_fields = set(self.section("motor_neurons"))
        drive = self._take("drive", {k: v for k, v in overrides.items() if k not in mn_fields}, set())
        mn = self._take("motor_neurons", {k: v for k, v in overrides.items() if k in mn_fields}, set())
        p = {k: drive[k] for k in ("dd_neurons", "conn_probability", "dd_drive__Hz", "synaptic_weight", "gamma_shape",
                                   "network")}
        p["gfluctdv_scale"] = float(mn["gfluctdv_scale"])
        return p

    def motor_neuron_pool(self, seed: int | None = None, **overrides):
        """Motor neuron pool, descending-drive pool and their calibrated wiring (``mn_pool``, ``dd_pool``, ``wiring``)."""
        from myogen import _seed_scope
        from myogen.simulator.neuron.descending_drive_network import build_pools

        seed = self._seed("drive", seed)
        params = self.dd_params(**overrides)
        mn = self._take("motor_neurons", {}, set())
        drive = self._take("drive", {}, set())
        thr = self.thresholds()
        with _seed_scope(seed):
            pools = build_pools(thr, params, float(drive["timestep_ms"]), mn["config_file"])
        pools.profile_record = self.record(seed=seed)
        return pools

    def surface_emg_kwargs(self, **overrides) -> dict[str, Any]:
        """The card's ``SurfaceEMG`` keywords besides the muscle and electrodes (nothing is built)."""
        return self._surface_emg_kwargs(overrides, {})

    def _surface_emg_kwargs(self, overrides: dict, keywords: dict) -> dict[str, Any]:
        from myogen.simulator import SurfaceEMG
        from myogen.simulator.core.emg.surface.fiber_dispersion import SurfaceFiberDispersion

        allowed = set(inspect.signature(SurfaceEMG.__init__).parameters) - {"self", "muscle_model", "electrode_arrays"}
        v = self._take("surface_emg", overrides, allowed, keywords=keywords)
        kw = {k: val for k, val in v.items() if k in allowed}
        disp = v["fiber_dispersion"]
        kw["fiber_dispersion"] = _dataclass_from(SurfaceFiberDispersion, disp) if isinstance(disp, dict) else disp
        kw["iap_lambda__per_mm"] = float(v["iap_lambda__per_mm"])
        kw["iap_kernel_length__mm"] = float(v["iap_kernel_length__mm"])
        return kw

    def surface_emg(self, muscle, electrode_arrays, seed: int | None = None, **overrides):
        """The card's ``SurfaceEMG`` for ``muscle`` (from ``muscle()``) and the user's electrode arrays.

        The seed and the generator state after construction are kept on the object (``profile_seed``,
        ``profile_rng_state``); ``simulate_muaps`` draws from them and gives the caller's generator back, so the MUAPs
        do not depend on what ran between building and simulating.
        """
        from myogen import _seed_scope
        from myogen.simulator import SurfaceEMG

        seed = self._seed("surface_emg", seed)
        keywords: dict = {}
        kw = self._surface_emg_kwargs(overrides, keywords)
        with _seed_scope(seed) as rng:
            semg = SurfaceEMG(muscle, electrode_arrays, **kw)
            semg.profile_rng_state = copy.deepcopy(rng.bit_generator.state)
        semg.profile_seed = seed
        semg.profile_record = self.record(keywords, seed=seed)
        return semg

    # -- running in % MVC -------------------------------------------------------------------------------------
    def excitation_for_pct_mvc(self, pct):
        """Excitation (0-1) for a force level in % MVC, interpolated on the card's MVC-ramp curve.

        The curve is extended to rest (0 % MVC, excitation 0) and MVC (100 %, excitation 1) when it stops short.
        """
        from myogen.profiles._card import ProfileError

        curve = self._take("drive", {}, set())["excitation_at_pct_mvc"]
        if curve is None:
            raise ProfileError(f"{self.name}: drive.excitation_at_pct_mvc is not filled in, so % MVC levels cannot be "
                               "mapped to excitation (run benchmarks/profiles/validate_card.py --mvc-mapping)")
        pct_arr = np.asarray(pct, dtype=float)
        if not np.all(np.isfinite(pct_arr)) or np.any(pct_arr < 0) or np.any(pct_arr > 100):
            raise ValueError(f"force levels must be within 0-100 % MVC, got {pct!r}")
        points = {float(k): float(v) for k, v in curve.items()}
        points.setdefault(0.0, 0.0)
        points.setdefault(100.0, 1.0)
        levels = sorted(points)
        e = np.interp(pct_arr, levels, [points[k] for k in levels])
        return float(e) if np.ndim(e) == 0 else e

    def simulate(self, protocol, seed: int | None = None, **overrides) -> "SimulationResult":
        """Run the calibrated drive for ``protocol``, segments ``(duration_s, start_pct_mvc, end_pct_mvc)``.

        Each segment's ends are mapped to excitation with ``excitation_for_pct_mvc`` and the excitation is linear in
        between, as in the calibration. Returns the motor-unit spike trains, force in % MVC and the excitation.
        Keywords override card fields of the thresholds, motor_neurons, drive and force sections; ``seed`` (default:
        the card's drive seed) seeds the drive noise and the network.
        """
        from myogen import _seed_scope
        from myogen.simulator.neuron import descending_drive_network as ddn

        seed = self._seed("drive", seed)

        by_section = {s: {} for s in ("thresholds", "motor_neurons", "drive", "force")}
        for key, val in overrides.items():
            owner = next((s for s in by_section if key in self.section(s)), None)
            if owner is None:
                raise TypeError(f"{self.name}: unknown keyword {key!r} for simulate "
                                f"(fields of {', '.join(by_section)})")
            by_section[owner][key] = val
        thr = self.thresholds(**by_section["thresholds"])
        params = self.dd_params(**by_section["drive"], **by_section["motor_neurons"])
        self._take("force", by_section["force"], set())
        drive = self._take("drive", {}, set())
        mn = self._take("motor_neurons", {}, set())
        segments = [(float(d), self.excitation_for_pct_mvc(a), self.excitation_for_pct_mvc(b)) for d, a, b in protocol]
        dt = float(drive["timestep_ms"])
        e, hz = ddn.excitation_to_drive_hz(segments, float(drive["threshold_hz"]), params["dd_drive__Hz"],
                                           timestep_ms=dt, noise_seed=seed)
        with _seed_scope(seed):
            block = ddn.run_descending_drive(thr, hz, params, len(e) * dt, timestep_ms=dt,
                                             config_file=mn["config_file"])
        fm = self.force_model()
        force = np.asarray(fm.generate_force(block, verbose=False).magnitude, dtype=float)[:, 0]
        record = self.record(seed=seed)
        block.annotations["myogen_profile"] = record
        fs = float(self._take("force", {}, set())["recording_frequency__Hz"])
        return SimulationResult(spikes=block, force_pct_mvc=100.0 * force / float(drive["mvc_model_units"]),
                                excitation=e, fs_force__Hz=fs, profile=record)


@dataclass
class SimulationResult:
    """Spike trains (Neo block, annotated with the card record), force in % MVC at ``fs_force__Hz``, excitation per
    drive time step, and the card record."""
    spikes: Any
    force_pct_mvc: np.ndarray
    excitation: np.ndarray
    fs_force__Hz: float
    profile: dict

"""Per-fibre dispersion for surface EMG: end-plates, axon-branch delays, fibre CV.

Every option defaults to off, which reproduces the undispersed model (one
end-plate per motor unit, no nerve delays, every fibre at the mean conduction
velocity). The options are independent so that each effect can be measured
on its own.

End-plates and nerve delays come from MyoGen's two-layer nerve model
(:mod:`myogen.simulator.core.emg.nerve_branches`), the same one intramuscular
EMG uses. Draws use generators seeded from ``seed`` only, never the global
MyoGen generator, so switching an option on leaves every other draw intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal, Optional, Sequence

import numpy as np

from ...muscle.muscle import CV_INTERCEPT__M_S, CV_PER_DIAMETER__M_S_PER_MM
from ..nerve_branches import NMJ_SCHEDULE_COEFFICIENTS, nerve_delays__ms, nmj_branch_schedule, two_layer_nmj

NMJ_MODELS = ("uniform_mu", "two_layer", "two_layer_intra")
FIBER_PROPERTY_SOURCES = ("config", "muscle")


@dataclass(frozen=True)
class SurfaceFiberDispersion:
    """Configuration of per-fibre dispersion (all off by default).

    Parameters
    ----------
    nmj_model : {"uniform_mu", "two_layer", "two_layer_intra"}
        ``"uniform_mu"``: every fibre of a motor unit has the unit's end-plate
        (the undispersed model). ``"two_layer"``: the two-layer nerve model as
        intramuscular EMG uses it (branch points scattered about the end-plate
        band centre, fibres scattered about their branch point).
        ``"two_layer_intra"``: the two-layer scatter re-centred on each unit's
        undispersed end-plate, isolating the spread within a motor unit.
    nmj_scatter_scale : float
        Multiplies both two-layer z spreads (1 = the coefficients as given).
    nmj_coefficients : {"published", "reference_code"}
        Two-layer z-spread coefficients: the values printed in Akhmadeev et al.
        (2020), eq. 5-6 [akhmadeev2020 eq. 5-6] (default, as in intramuscular EMG), or those of
        Akhmadeev's reference simulator code, 1.5-2 times wider.
    nerve_delays : bool
        Delay each fibre by its axon-branch conduction time (two-layer nerve
        paths over ``nerve_cvs__m_per_s``). Independent of ``nmj_model``: with
        ``"uniform_mu"`` only the timing changes, not the end-plate positions.
    nerve_cvs__m_per_s : (float, float)
        Conduction velocity of the first (axon to branch point) and second
        (branch point to end-plate) layer. Default (5, 2) m/s as in
        intramuscular EMG (Akhmadeev et al. 2020) [akhmadeev2020].
    fiber_cv_inter_mu_spread : float
        Spread of motor-unit mean fibre CV linear in recruitment order: the
        first-recruited unit propagates at ``mean * (1 - spread / 2)``, the
        last at ``mean * (1 + spread / 2)``.
    fiber_cv_threshold_slope : float
        Motor-unit mean fibre CV linear in recruitment threshold:
        ``mean * (1 + slope * (rt - mean(rt)))`` with ``rt`` the threshold
        relative to the largest. Del Vecchio et al. (2018) measured CV rising
        linearly with recruitment threshold in tibialis anterior (406 units from
        10 subjects; 3.89 m/s at 0-30 % MVC, 5.12 m/s at 50-70 % MVC, overall mean
        4.32) [delvecchio2018 p. 2], about 0.63 of the mean per unit of threshold
        (derived from those values). Cannot be combined with
        ``fiber_cv_inter_mu_spread``.
    fiber_cv_per_unit__m_s : tuple of float, optional
        Mean fibre CV of each motor unit (m/s), in pool order, replacing
        ``mean * threshold factor``; e.g. CV mapped from each unit's measured
        recruitment force. Cannot be combined with ``fiber_cv_threshold_slope``
        or ``fiber_cv_inter_mu_spread``.
    fiber_cv_within_mu_cov : float
        Coefficient of variation of fibre CV within a motor unit (normal,
        clipped at three standard deviations). No measurement of this spread
        was found; Del Vecchio et al. (2018) call it unknown [delvecchio2018 p. 5].
    fiber_properties : {"config", "muscle"}
        Source of each fibre's conduction velocity and diameter. ``"config"`` (default): built
        here from the options below (muscle mean CV, threshold factor or per-unit CVs, within-unit
        spread; diameter read from the CV line). ``"muscle"``: taken fibre by fibre from ``Muscle``
        (``muscle_fiber_conduction_velocities__mm_per_s``, ``muscle_fiber_diameters__mm``: typed by
        motor unit, the fibres intramuscular EMG uses), so both signals share one set of fibres.
        Cannot be combined with ``fiber_cv_threshold_slope``, ``fiber_cv_inter_mu_spread``,
        ``fiber_cv_within_mu_cov`` or ``fiber_cv_per_unit__m_s``.
    diameter_scales_source : bool
        Scale each fibre's source by its cross-section. With ``fiber_properties="muscle"`` the
        factor is ``(d / SurfaceEMG.fiber_diameter__um)**2`` with ``d`` the ``Muscle`` diameter, not
        normalised, so each source scales with its own absolute ``d**2`` and
        ``fiber_diameter__um`` has no effect. Otherwise the diameter is read
        from the CV line ``cv = cv_intercept + cv_per_diameter * d``
        (0.95 m/s + 0.05 m/s per µm; Nandedkar & Stålberg 1983 [nandedkar1983], used in the
        same form by Del Vecchio et al. 2018 [delvecchio2018 p. 7 eq. 1]; the measured fit of
        Blijham et al. 2006, 0.83 m/s + 0.043 m/s per µm [blijham2006 p. 1840], gives the same
        relative spread within 2 %). The factor ``d^2`` is normalised to average 1 over every
        fibre of the pool, so ``SurfaceEMG.fiber_diameter__um`` stays the population
        root-mean-square diameter and the line only sets the spread between fibres; its
        absolute anchor (3.7 m/s at 55 µm, biceps [nandedkar1983 p. 163]) does not fit FDI.
    cv_per_diameter__m_s_per_mm, cv_intercept__m_s : float
        Slope and intercept of that line.
    tendon_end_band__mm : float
        Width of the band over which each fibre end is placed, independently
        at either end, uniform about ``±mean_fiber_length / 2``. Replaces the
        fibre-length variation (which moves both ends together). 0 keeps it.
    tendon_end_correlation__mm : float
        Correlation length of the fibre-end position across the cross-section.
        0 draws every end independently over the band. Above 0, each end
        follows a smooth Gaussian random field (squared-exponential covariance,
        one field per tendon) plus ``tendon_end_jitter__mm`` per fibre, with the
        total SD of the uniform band, ``tendon_end_band__mm / sqrt(12)``.
    tendon_end_jitter__mm : float
        SD of each fibre's own offset from the smooth field.
    seed : int
        Seed of the dispersion draws.
    """

    nmj_model: Literal["uniform_mu", "two_layer", "two_layer_intra"] = "uniform_mu"
    nmj_scatter_scale: float = 1.0
    nmj_coefficients: str = "published"
    nerve_delays: bool = False
    nerve_cvs__m_per_s: tuple[float, float] = (5.0, 2.0)
    fiber_cv_inter_mu_spread: float = 0.0
    fiber_cv_threshold_slope: float = 0.0
    fiber_cv_within_mu_cov: float = 0.0
    fiber_cv_per_unit__m_s: Optional[tuple[float, ...]] = None
    diameter_scales_source: bool = False
    cv_per_diameter__m_s_per_mm: float = CV_PER_DIAMETER__M_S_PER_MM
    cv_intercept__m_s: float = CV_INTERCEPT__M_S
    fiber_properties: Literal["config", "muscle"] = "config"
    tendon_end_band__mm: float = 0.0
    tendon_end_correlation__mm: float = 0.0
    tendon_end_jitter__mm: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.nmj_model not in NMJ_MODELS:
            raise ValueError(f"nmj_model must be one of {NMJ_MODELS}, got {self.nmj_model!r}")
        if self.nmj_coefficients not in NMJ_SCHEDULE_COEFFICIENTS:
            raise ValueError(f"nmj_coefficients must be one of {tuple(NMJ_SCHEDULE_COEFFICIENTS)}")
        if self.nmj_scatter_scale < 0:
            raise ValueError("nmj_scatter_scale must be non-negative")
        if len(self.nerve_cvs__m_per_s) != 2 or min(self.nerve_cvs__m_per_s) <= 0:
            raise ValueError("nerve_cvs__m_per_s must be two positive velocities")
        if not 0 <= self.fiber_cv_inter_mu_spread < 2:
            raise ValueError("fiber_cv_inter_mu_spread must be in [0, 2)")
        if min(self.tendon_end_band__mm, self.tendon_end_correlation__mm, self.tendon_end_jitter__mm) < 0:
            raise ValueError("tendon_end_band__mm, tendon_end_correlation__mm and tendon_end_jitter__mm "
                             "must be non-negative")
        if self.tendon_end_correlation__mm > 0:
            if self.tendon_end_band__mm == 0:
                raise ValueError("tendon_end_correlation__mm needs tendon_end_band__mm")
            if self.tendon_end_jitter__mm >= self.tendon_end_band__mm / np.sqrt(12):
                raise ValueError("tendon_end_jitter__mm must be below the band SD, tendon_end_band__mm / sqrt(12)")
        if self.fiber_cv_threshold_slope < 0:
            raise ValueError("fiber_cv_threshold_slope must be non-negative")
        if self.fiber_cv_threshold_slope and self.fiber_cv_inter_mu_spread:
            raise ValueError("use either fiber_cv_threshold_slope or fiber_cv_inter_mu_spread, not both")
        if not 0 <= self.fiber_cv_within_mu_cov < 1 / 3:
            raise ValueError("fiber_cv_within_mu_cov must be in [0, 1/3)")
        if self.fiber_properties not in FIBER_PROPERTY_SOURCES:
            raise ValueError(f"fiber_properties must be one of {FIBER_PROPERTY_SOURCES}, got {self.fiber_properties!r}")
        if self.fiber_properties == "muscle":
            for name in ("fiber_cv_threshold_slope", "fiber_cv_inter_mu_spread", "fiber_cv_within_mu_cov"):
                if getattr(self, name):
                    raise ValueError(f"fiber_properties='muscle' takes each fibre's CV from Muscle; set {name} to 0")
            if self.fiber_cv_per_unit__m_s is not None:
                raise ValueError("fiber_properties='muscle' takes each fibre's CV from Muscle; "
                                 "leave fiber_cv_per_unit__m_s unset")
        if self.fiber_cv_per_unit__m_s is not None:
            if self.fiber_cv_threshold_slope or self.fiber_cv_inter_mu_spread:
                raise ValueError("fiber_cv_per_unit__m_s replaces fiber_cv_threshold_slope and "
                                 "fiber_cv_inter_mu_spread; set those to 0")
            if min(self.fiber_cv_per_unit__m_s) <= 0:
                raise ValueError("fiber_cv_per_unit__m_s must be positive")

    @classmethod
    def literature_candidate(cls, seed: int = 0) -> "SurfaceFiberDispersion":
        """Dispersion with each value taken from the literature or marked as an assumption.

        - End-plates: two-layer within each unit, coefficients of Akhmadeev et al.
          (2020), eq. 5-6 [akhmadeev2020 eq. 5-6].
        - Nerve branches: 5 / 2 m/s, the ``IntramuscularEMG`` default (Akhmadeev
          et al. 2020) [akhmadeev2020]. Within-unit delay SD is about 0.1-0.2 ms in the sweep
          scenarios of ``benchmarks/semg_dispersion``.
        - Unit conduction velocity: linear in recruitment threshold, slope 0.63
          (Del Vecchio et al. 2018) [delvecchio2018 p. 2].
        - Fibre velocity within a unit: CoV 5 %. A modelling assumption; no
          measurement was found.
        - Tendon ends: 5 mm band (Keenan et al. 2005) [keenan2005], correlated over 5 mm with
          0.3 mm per-fibre jitter (assumptions; no measurement of fibre-end stagger
          was found).
        - Diameter-scaled source: off, pending an amplitude check against published
          ranges.
        """
        return cls(nmj_model="two_layer_intra", nmj_scatter_scale=1.0, nmj_coefficients="published",
                   nerve_delays=True, nerve_cvs__m_per_s=(5.0, 2.0), fiber_cv_threshold_slope=0.63,
                   fiber_cv_within_mu_cov=0.05, tendon_end_band__mm=5.0, tendon_end_correlation__mm=5.0,
                   tendon_end_jitter__mm=0.3, seed=seed)

    @classmethod
    def muscle_candidate(cls, seed: int = 0) -> "SurfaceFiberDispersion":
        """``literature_candidate`` with every fibre's CV and diameter taken from ``Muscle``.

        End-plates, nerve delays and tendon ends as in ``literature_candidate``; the threshold
        slope and within-unit CV spread give way to the ``Muscle`` fibres (typed by motor unit,
        CV 4.32 +/- 0.71 m/s over fibres [delvecchio2018 p. 2]); each source scales with its own
        cross-section.
        """
        return replace(cls.literature_candidate(seed=seed), fiber_properties="muscle",
                       fiber_cv_threshold_slope=0.0, fiber_cv_within_mu_cov=0.0, diameter_scales_source=True)

    @property
    def active(self) -> bool:
        return (self.nmj_model != "uniform_mu" or self.nerve_delays
                or self.fiber_cv_inter_mu_spread > 0 or self.fiber_cv_threshold_slope > 0
                or self.fiber_cv_within_mu_cov > 0 or self.fiber_cv_per_unit__m_s is not None
                or self.diameter_scales_source or self.tendon_end_band__mm > 0
                or self.fiber_properties == "muscle")


@dataclass
class FiberDispersionRealization:
    """Per-motor-unit lists of per-fibre values, and a per-unit summary."""

    zi__mm: list[np.ndarray]
    cv__m_s: list[np.ndarray]
    delay__ms: list[np.ndarray]
    amplitude_scale: list[np.ndarray]
    end_plus__mm: Optional[list[np.ndarray]] = None
    end_minus__mm: Optional[list[np.ndarray]] = None
    report: dict = field(default_factory=dict)


def _smooth_field(rng: np.random.Generator, correlation__mm: float, sd__mm: float, n_features: int = 1000):
    """Gaussian random field over the cross-section with covariance
    ``sd² exp(-r² / (2 correlation²))``, by random Fourier features."""
    k = rng.normal(0.0, 1.0 / correlation__mm, size=(n_features, 2))
    phase = rng.uniform(0.0, 2 * np.pi, size=n_features)
    weight = sd__mm * np.sqrt(2.0 / n_features)
    return lambda xy: weight * np.cos(xy @ k.T + phase).sum(axis=1)


def realize_fiber_dispersion(
    config: SurfaceFiberDispersion,
    fiber_positions__mm: Sequence[np.ndarray],
    recruitment_thresholds: np.ndarray,
    mu_zones__mm: np.ndarray,
    mean_cv__m_s: float,
    fiber_lengths__mm: Optional[Sequence[np.ndarray]] = None,
    end_margin__mm: float = 1.0,
    mean_fiber_length__mm: Optional[float] = None,
    fiber_diameter__um: Optional[float] = None,
) -> FiberDispersionRealization:
    """Draw per-fibre end-plate, nerve delay, CV and source scale for every motor unit.

    ``mu_zones__mm`` holds each unit's undispersed end-plate (relative to the
    fibre centre). With ``fiber_lengths__mm`` given, each fibre's ends sit at
    ``±length / 2`` (or are drawn in the tendon band, which needs
    ``mean_fiber_length__mm``), and end-plates are clipped to
    ``end_margin__mm`` inside the ends with the clipped count reported.
    ``fiber_diameter__um`` (the ``SurfaceEMG`` source diameter) gives the
    effective diameters in the report (``diameter_mean__um``; NaN without it);
    ``line_diameter_mean__um`` is the diameter read from the CV line.
    """
    n_mu = len(fiber_positions__mm)
    counts = np.array([len(p) for p in fiber_positions__mm])
    nerve_rng = np.random.default_rng([config.seed, 0])
    cv_rng = np.random.default_rng([config.seed, 1])
    end_rng = np.random.default_rng([config.seed, 2])
    band = config.tendon_end_band__mm
    if band > 0 and (fiber_lengths__mm is None or mean_fiber_length__mm is None):
        raise ValueError("tendon_end_band__mm needs fiber_lengths__mm and mean_fiber_length__mm")
    end_fields = None
    if band > 0 and config.tendon_end_correlation__mm > 0:
        field_sd = np.sqrt(band**2 / 12 - config.tendon_end_jitter__mm**2)
        end_fields = [_smooth_field(end_rng, config.tendon_end_correlation__mm, field_sd) for _ in range(2)]
    needs_nerve = config.nmj_model != "uniform_mu" or config.nerve_delays
    if needs_nerve:
        n_branches, branch_sd, arbor_sd = nmj_branch_schedule(recruitment_thresholds, config.nmj_coefficients)

    report = {key: np.full(n_mu, np.nan) for key in (
        "zi_mean__mm", "zi_sd__mm", "cv_mean__m_s", "cv_cov", "delay_mean__ms", "delay_sd__ms",
        "amplitude_scale_mean", "diameter_mean__um", "line_diameter_mean__um", "end_sd__mm")}
    report["n_fibers"] = counts
    report["n_branches"] = np.zeros(n_mu, dtype=int)
    report["zi_clipped"] = np.zeros(n_mu, dtype=int)
    zi_all, cv_all, delay_all, scale_all, plus_all, minus_all = [], [], [], [], [], []
    rank = np.arange(n_mu) / max(n_mu - 1, 1)
    rt = np.asarray(recruitment_thresholds, dtype=float)
    rt = rt / rt.max()
    threshold_factor = 1 + config.fiber_cv_threshold_slope * (rt - rt.mean())
    if config.fiber_cv_per_unit__m_s is not None and len(config.fiber_cv_per_unit__m_s) != n_mu:
        raise ValueError(f"fiber_cv_per_unit__m_s has {len(config.fiber_cv_per_unit__m_s)} values for {n_mu} motor units")
    if np.any(threshold_factor <= 0):
        raise ValueError("fiber_cv_threshold_slope gives a non-positive CV for the first-recruited units")
    for mu in range(n_mu):
        n = int(counts[mu])
        zi = np.full(n, float(mu_zones__mm[mu]))
        delay = np.zeros(n)
        if needs_nerve and n:
            branches = int(min(n_branches[mu], n))
            report["n_branches"][mu] = branches
            nmj_z, paths = two_layer_nmj(
                fiber_positions__mm[mu], branches, 0.0,
                config.nmj_scatter_scale * branch_sd[mu], config.nmj_scatter_scale * arbor_sd[mu],
                rng=nerve_rng, kmeans_seed=int(nerve_rng.integers(2**31)))
            if config.nmj_model == "two_layer":
                zi = nmj_z
            elif config.nmj_model == "two_layer_intra":
                zi = zi + (nmj_z - np.mean(nmj_z))
            if config.nerve_delays:
                delay = nerve_delays__ms(paths, config.nerve_cvs__m_per_s)
        if fiber_lengths__mm is not None:
            if end_fields is not None:
                xy = np.asarray(fiber_positions__mm[mu], dtype=float)[:, :2]
                jitter = config.tendon_end_jitter__mm
                end_plus = mean_fiber_length__mm / 2 + end_fields[0](xy) + end_rng.normal(0, jitter, size=n)
                end_minus = -mean_fiber_length__mm / 2 + end_fields[1](xy) + end_rng.normal(0, jitter, size=n)
            elif band > 0:
                end_plus = mean_fiber_length__mm / 2 + end_rng.uniform(-band / 2, band / 2, size=n)
                end_minus = -mean_fiber_length__mm / 2 + end_rng.uniform(-band / 2, band / 2, size=n)
            else:
                end_plus = np.asarray(fiber_lengths__mm[mu]) / 2
                end_minus = -np.asarray(fiber_lengths__mm[mu]) / 2
            clipped = np.clip(zi, end_minus + end_margin__mm, end_plus - end_margin__mm)
            report["zi_clipped"][mu] = int(np.count_nonzero(clipped != zi))
            zi = clipped
            plus_all.append(end_plus)
            minus_all.append(end_minus)
            if n:
                report["end_sd__mm"][mu] = np.std(end_plus)

        cv_mu = mean_cv__m_s * threshold_factor[mu] * (1 - config.fiber_cv_inter_mu_spread / 2
                                                       + config.fiber_cv_inter_mu_spread * rank[mu])
        if config.fiber_cv_per_unit__m_s is not None:
            cv_mu = float(config.fiber_cv_per_unit__m_s[mu])
        cov = config.fiber_cv_within_mu_cov
        if cov > 0:
            cv = np.clip(cv_rng.normal(cv_mu, cov * cv_mu, size=n), cv_mu * (1 - 3 * cov), cv_mu * (1 + 3 * cov))
        else:
            cv = np.full(n, cv_mu)
        diameter__um = (cv - config.cv_intercept__m_s) / config.cv_per_diameter__m_s_per_mm * 1e3
        if config.diameter_scales_source:
            if np.any(diameter__um <= 0):
                raise ValueError("a fibre CV lies at or below the CV-diameter intercept")
            scale = diameter__um**2
        else:
            scale = np.ones(n)

        zi_all.append(zi)
        cv_all.append(cv)
        delay_all.append(delay)
        scale_all.append(scale)
        if n:
            report["zi_mean__mm"][mu], report["zi_sd__mm"][mu] = np.mean(zi), np.std(zi)
            report["cv_mean__m_s"][mu], report["cv_cov"][mu] = np.mean(cv), np.std(cv) / np.mean(cv)
            report["delay_mean__ms"][mu], report["delay_sd__ms"][mu] = np.mean(delay), np.std(delay)
            report["line_diameter_mean__um"][mu] = np.mean(diameter__um)
    if config.diameter_scales_source and counts.sum():
        mean_square = np.mean(np.concatenate(scale_all))
        scale_all = [scale / mean_square for scale in scale_all]
    for mu, scale in enumerate(scale_all):
        if len(scale):
            report["amplitude_scale_mean"][mu] = np.mean(scale)
            if fiber_diameter__um is not None:
                report["diameter_mean__um"][mu] = np.mean(fiber_diameter__um * np.sqrt(scale))
    report["config"] = asdict(config)
    ends = (plus_all, minus_all) if fiber_lengths__mm is not None else (None, None)
    return FiberDispersionRealization(zi_all, cv_all, delay_all, scale_all, *ends, report=report)

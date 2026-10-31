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

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from ..nerve_branches import nerve_delays__ms, nmj_branch_schedule, two_layer_nmj

NMJ_MODELS = ("uniform_mu", "two_layer", "two_layer_intra")


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
        Multiplies both two-layer z spreads (1 = the intramuscular default).
    nerve_delays : bool
        Delay each fibre by its axon-branch conduction time (two-layer nerve
        paths over ``nerve_cvs__m_per_s``). Independent of ``nmj_model``: with
        ``"uniform_mu"`` only the timing changes, not the end-plate positions.
    nerve_cvs__m_per_s : (float, float)
        Conduction velocity of the first (axon to branch point) and second
        (branch point to end-plate) layer. Default (5, 2) m/s as in
        intramuscular EMG (Konstantin et al. 2020).
    fiber_cv_inter_mu_spread : float
        Size-principle spread of motor-unit mean fibre CV: the first-recruited
        unit propagates at ``mean * (1 - spread / 2)``, the last at
        ``mean * (1 + spread / 2)``, linear in recruitment order.
    fiber_cv_within_mu_cov : float
        Coefficient of variation of fibre CV within a motor unit (normal,
        clipped at three standard deviations). Stålberg & Karlsson (1973)
        report 5-10 % in healthy muscle.
    diameter_scales_source : bool
        Scale each fibre's source by its cross-section, with diameter linear in
        CV: ``d / d_ref = 1 + (cv - mean_cv) / (cv_per_diameter * d_ref)``.
    cv_per_diameter__m_s_per_mm, reference_diameter__mm : float
        Slope of CV against fibre diameter and the diameter at the mean CV.
    seed : int
        Seed of the dispersion draws.
    """

    nmj_model: Literal["uniform_mu", "two_layer", "two_layer_intra"] = "uniform_mu"
    nmj_scatter_scale: float = 1.0
    nerve_delays: bool = False
    nerve_cvs__m_per_s: tuple[float, float] = (5.0, 2.0)
    fiber_cv_inter_mu_spread: float = 0.0
    fiber_cv_within_mu_cov: float = 0.0
    diameter_scales_source: bool = False
    cv_per_diameter__m_s_per_mm: float = 77.0
    reference_diameter__mm: float = 0.05
    seed: int = 0

    def __post_init__(self):
        if self.nmj_model not in NMJ_MODELS:
            raise ValueError(f"nmj_model must be one of {NMJ_MODELS}, got {self.nmj_model!r}")
        if self.nmj_scatter_scale < 0:
            raise ValueError("nmj_scatter_scale must be non-negative")
        if len(self.nerve_cvs__m_per_s) != 2 or min(self.nerve_cvs__m_per_s) <= 0:
            raise ValueError("nerve_cvs__m_per_s must be two positive velocities")
        if not 0 <= self.fiber_cv_inter_mu_spread < 2:
            raise ValueError("fiber_cv_inter_mu_spread must be in [0, 2)")
        if not 0 <= self.fiber_cv_within_mu_cov < 1 / 3:
            raise ValueError("fiber_cv_within_mu_cov must be in [0, 1/3)")

    @property
    def active(self) -> bool:
        return (self.nmj_model != "uniform_mu" or self.nerve_delays
                or self.fiber_cv_inter_mu_spread > 0 or self.fiber_cv_within_mu_cov > 0
                or self.diameter_scales_source)


@dataclass
class FiberDispersionRealization:
    """Per-motor-unit lists of per-fibre values, and a per-unit summary."""

    zi__mm: list[np.ndarray]
    cv__m_s: list[np.ndarray]
    delay__ms: list[np.ndarray]
    amplitude_scale: list[np.ndarray]
    report: dict = field(default_factory=dict)


def realize_fiber_dispersion(
    config: SurfaceFiberDispersion,
    fiber_positions__mm: Sequence[np.ndarray],
    recruitment_thresholds: np.ndarray,
    mu_zones__mm: np.ndarray,
    mean_cv__m_s: float,
    fiber_lengths__mm: Optional[Sequence[np.ndarray]] = None,
    end_margin__mm: float = 1.0,
) -> FiberDispersionRealization:
    """Draw per-fibre end-plate, nerve delay, CV and source scale for every motor unit.

    ``mu_zones__mm`` holds each unit's undispersed end-plate (relative to the
    fibre centre). With ``fiber_lengths__mm`` given, end-plates are clipped to
    ``end_margin__mm`` inside each fibre and the clipped count is reported.
    """
    n_mu = len(fiber_positions__mm)
    counts = np.array([len(p) for p in fiber_positions__mm])
    nerve_rng = np.random.default_rng([config.seed, 0])
    cv_rng = np.random.default_rng([config.seed, 1])
    needs_nerve = config.nmj_model != "uniform_mu" or config.nerve_delays
    if needs_nerve:
        n_branches, branch_sd, arbor_sd = nmj_branch_schedule(recruitment_thresholds)

    report = {key: np.full(n_mu, np.nan) for key in (
        "zi_mean__mm", "zi_sd__mm", "cv_mean__m_s", "cv_cov", "delay_mean__ms", "delay_sd__ms",
        "amplitude_scale_mean")}
    report["n_fibers"] = counts
    report["n_branches"] = np.zeros(n_mu, dtype=int)
    report["zi_clipped"] = np.zeros(n_mu, dtype=int)
    zi_all, cv_all, delay_all, scale_all = [], [], [], []
    rank = np.arange(n_mu) / max(n_mu - 1, 1)
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
        if fiber_lengths__mm is not None and n:
            limit = np.asarray(fiber_lengths__mm[mu]) / 2 - end_margin__mm
            clipped = np.clip(zi, -limit, limit)
            report["zi_clipped"][mu] = int(np.count_nonzero(clipped != zi))
            zi = clipped

        cv_mu = mean_cv__m_s * (1 - config.fiber_cv_inter_mu_spread / 2
                                + config.fiber_cv_inter_mu_spread * rank[mu])
        cov = config.fiber_cv_within_mu_cov
        if cov > 0:
            cv = np.clip(cv_rng.normal(cv_mu, cov * cv_mu, size=n), cv_mu * (1 - 3 * cov), cv_mu * (1 + 3 * cov))
        else:
            cv = np.full(n, cv_mu)
        if config.diameter_scales_source:
            ratio = 1 + (cv - mean_cv__m_s) / (config.cv_per_diameter__m_s_per_mm * config.reference_diameter__mm)
            scale = ratio**2
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
            report["amplitude_scale_mean"][mu] = np.mean(scale)
    report["config"] = asdict(config)
    return FiberDispersionRealization(zi_all, cv_all, delay_all, scale_all, report)

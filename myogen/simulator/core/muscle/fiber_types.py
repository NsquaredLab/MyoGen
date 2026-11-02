"""Fibre types and diameters by motor unit.

Every motor unit has one fibre type, and its mean fibre diameter rises with its recruitment
threshold along one ladder across the whole pool (design: docs/superpowers/specs/
2026-11-02-fdi-fibre-typing-design.md, option A).

- Types: units in threshold order are type I until they hold the type I fibre fraction
  (FDI 57.4 % [polgar1973 p. 312 Table 4]). Strict order is a modelling simplification: in
  human medial gastrocnemius slow and fast fatigue-resistant units interleave at low force and
  only the fast fatigable (type 2b) units come strictly last [garnett1979 p. 42].
- Ladder: a unit's position is the midpoint of its share of the pool's cumulative fibre count;
  its log-mean diameter is ``sigma_between * clip(Phi^-1(q), -3, 3)``. ``sigma_between`` is solved
  so that the type I : type II fibre-weighted mean ratio is 50.0 : 67.0 um [polgar1973 p. 310
  Table 2]; ``sigma_within`` so that the pool's fibre log-SD equals that of Polgar's two-type
  mixture with ``type_log_sd`` per type (0.2, a modelling assumption).
- Fibres scatter log-normally about their unit's mean; the combined log-deviation is truncated at
  3 SD (a modelling assumption), and the diameters are scaled to the mixture mean (57.2 um).

A per-type ladder would make CV drop at the type boundary, against the linear CV-threshold
relation measured with HD-sEMG [delvecchio2018 p. 2].
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, truncnorm


@dataclass(frozen=True)
class FiberTypeLadder:
    """Per-unit fibre types and ladder positions, with the derived spreads.

    Attributes
    ----------
    unit_types : np.ndarray
        Fibre type of each unit (1 or 2), in pool order.
    unit_z : np.ndarray
        Ladder position of each unit, ``clip(Phi^-1(q), -3, 3)``.
    sigma_between, sigma_within : float
        Log-SD of the unit means along the ladder, and of fibres within a unit.
    type_i_fiber_fraction : float
        Realised fraction of fibres in type I units.
    target_mean__um : float
        Mean fibre diameter of the pool (um).
    """

    unit_types: np.ndarray
    unit_z: np.ndarray
    sigma_between: float
    sigma_within: float
    type_i_fiber_fraction: float
    target_mean__um: float


def fiber_type_ladder(
    fibers_per_unit: np.ndarray,
    recruitment_thresholds: np.ndarray,
    type_i_fiber_fraction: float = 0.574,
    type_mean_diameters__um: tuple[float, float] = (50.0, 67.0),
    type_log_sd: float = 0.2,
) -> FiberTypeLadder:
    """Type the units and place them on the diameter ladder.

    Parameters
    ----------
    fibers_per_unit : np.ndarray
        Number of fibres of each unit, in pool order.
    recruitment_thresholds : np.ndarray
        Recruitment threshold of each unit; ties keep pool order.
    type_i_fiber_fraction : float
        Fraction of fibres in type I units [polgar1973 p. 312 Table 4].
    type_mean_diameters__um : tuple of float
        Mean fibre diameter of type I and type II fibres [polgar1973 p. 310 Table 2].
    type_log_sd : float
        Log-SD of fibre diameter within each type (a modelling assumption).

    Raises
    ------
    ValueError
        If either type gets no fibres, or the type spread is too small for the ladder.
    """
    n = np.asarray(fibers_per_unit, dtype=float)
    f = float(type_i_fiber_fraction)
    m1, m2 = (float(v) for v in type_mean_diameters__um)

    order = np.argsort(np.asarray(recruitment_thresholds), kind="stable")
    cum = np.cumsum(n[order])
    q = np.empty(len(n))
    q[order] = (cum - n[order] / 2) / cum[-1]
    k = int(np.argmin(np.abs(cum / cum[-1] - f)))
    unit_types = np.full(len(n), 2)
    unit_types[order[: k + 1]] = 1
    type_i = unit_types == 1
    if n[~type_i].sum() == 0:
        raise ValueError("fiber_type_ladder: no type II fibres; check the unit count and type_i_fiber_fraction")
    if n[type_i].sum() == 0:
        raise ValueError("fiber_type_ladder: no type I fibres; check type_i_fiber_fraction")

    unit_z = np.clip(norm.ppf(q), -3.0, 3.0)

    def ratio(s):
        w = n * np.exp(s * unit_z)
        return (w[type_i].sum() / n[type_i].sum()) / (w[~type_i].sum() / n[~type_i].sum()) - m1 / m2

    sigma_between = float(brentq(ratio, 1e-6, 3.0))
    mixture_var = type_log_sd**2 + f * (1 - f) * np.log(m2 / m1) ** 2
    within_var = mixture_var - sigma_between**2 * np.average(unit_z**2, weights=n)
    if within_var <= 0:
        raise ValueError(
            f"fiber_type_ladder: sigma_within^2 = {within_var:.4f} <= 0; type_log_sd {type_log_sd} is too "
            "small for the spread the type means already need"
        )
    return FiberTypeLadder(
        unit_types=unit_types,
        unit_z=unit_z,
        sigma_between=sigma_between,
        sigma_within=float(np.sqrt(within_var)),
        type_i_fiber_fraction=float(n[type_i].sum() / n.sum()),
        target_mean__um=f * m1 + (1 - f) * m2,
    )


def draw_fiber_diameters__um(
    ladder: FiberTypeLadder, assignment: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw each fibre's diameter (um) about its unit's ladder mean.

    The combined log-deviation ``sigma_between * unit_z + sigma_within * z`` is truncated at
    3 SD of the pool (a modelling assumption); diameters are then scaled to
    ``ladder.target_mean__um``.
    """
    unit = np.asarray(assignment).astype(int)
    sb, sw = ladder.sigma_between, ladder.sigma_within
    bound = 3.0 * np.hypot(sb, sw)
    centre = sb * ladder.unit_z[unit]
    z = truncnorm.rvs((-bound - centre) / sw, (bound - centre) / sw, random_state=rng)
    d = np.exp(centre + sw * np.atleast_1d(z))
    return d * (ladder.target_mean__um / d.mean())

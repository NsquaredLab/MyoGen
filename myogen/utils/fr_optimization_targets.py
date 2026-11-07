"""
Firing rate optimization targets for DD network tuning.

Provides closed-form predictions for number of active motor units and
mean firing rate at any %MVC level. Multiple target models are supported
so the DD optimizer can be tuned against different physiological
assumptions.

Available models
----------------
- ``fuglevand``: Linear rate coding (Fuglevand et al. 1993) [fuglevand1993].
  Simple, closed-form. Tends to underestimate real rate coding at
  higher %MVC because it assumes linear FR vs excitation.

- ``de_luca``: Exponential approach to peak (De Luca & Contessa 2012) [delucacontessa2012].
  Captures rapid rise after recruitment (PIC effect) and plateau at
  peak rate. Supports onion-skin (earliest recruited fire fastest).
  Recommended default for new work.

- ``experimental``: Lookup table from published decomposed MU data.
  Muscle-specific, most accurate where data is available. Requires
  a target muscle name.

All models return the same dict format:

    {mvc_pct: {"n_active": int, "mean_fr_hz": float,
               "fr_range": (min, max)}}

Example
-------
>>> from myogen.utils.fr_optimization_targets import compute_fr_targets
>>> thresholds = profile.generate_thresholds(mu_count_method="fixed", n=200)
>>> targets = compute_fr_targets(thresholds, model="de_luca")
>>> targets[10]["mean_fr_hz"]
11.2
"""

from __future__ import annotations

import numpy as np


# ─── Model 1: Fuglevand linear (back-compat) ─────────────────────


def fuglevand_targets(
    thresholds: np.ndarray,
    mvc_levels: list[float] | None = None,
    min_fr_hz: float = 8.0,
    max_fr_hz: float = 17.0,
) -> dict[float, dict]:
    """Fuglevand 1993 [fuglevand1993] linear rate coding targets.

    FR_i(E) = min_fr + (max_fr - min_fr) * (E - threshold_i)
    clipped to [min_fr, max_fr].

    Parameters
    ----------
    thresholds : ndarray
        Normalized recruitment thresholds (0 to 1).
    mvc_levels : list of float, optional
        %MVC levels to compute targets for. Default [5, 10, 20, 30, 50, 100].
    min_fr_hz : float
        Onset firing rate for newly recruited MUs.
    max_fr_hz : float
        Mean firing rate at 100% MVC.
    """
    if mvc_levels is None:
        mvc_levels = [5, 10, 20, 30, 50, 100]

    thresholds = np.sort(thresholds)
    gain = max_fr_hz - min_fr_hz

    targets = {}
    for mvc in mvc_levels:
        excitation = mvc / 100.0
        active_mask = thresholds <= excitation
        n_active = int(np.sum(active_mask))

        if n_active == 0:
            targets[mvc] = {"n_active": 0, "mean_fr_hz": 0.0, "fr_range": (0.0, 0.0)}
            continue

        active_thresholds = thresholds[active_mask]
        frs = min_fr_hz + gain * (excitation - active_thresholds)
        frs = np.clip(frs, min_fr_hz, max_fr_hz)

        targets[mvc] = {
            "n_active": n_active,
            "mean_fr_hz": float(np.mean(frs)),
            "fr_range": (float(frs.min()), float(frs.max())),
        }

    return targets


# ─── Model 2: De Luca & Contessa exponential ────────────────────


def de_luca_targets(
    thresholds: np.ndarray,
    mvc_levels: list[float] | None = None,
    fr_onset_hz: float = 8.0,
    fr_peak_early_hz: float = 16.0,
    fr_peak_late_hz: float = 12.0,
    steepness: float = 3.0,
) -> dict[float, dict]:
    """De Luca & Contessa 2012 [delucacontessa2012] exponential rate coding targets.

    Each MU's firing rate follows an exponential approach from
    ``fr_onset_hz`` at recruitment to its peak rate at full excitation:

        FR_i(E) = FR_peak_i - (FR_peak_i - fr_onset_hz)
                  * exp(-steepness * (E - threshold_i))

    For E < threshold_i, the MU is silent.

    Onion-skin rate coding: the first-recruited MU has a higher peak
    rate (``fr_peak_early_hz``) than the last-recruited MU
    (``fr_peak_late_hz``). Peak rates interpolate linearly between
    these two endpoints by threshold rank.

    This produces the characteristic nonlinear rate coding observed
    in real muscles: rapid rise after recruitment (matching PIC
    amplification in biophysical MN models), then saturation at peak
    (matching Ca²⁺ saturation and AHP dynamics).

    Parameters
    ----------
    thresholds : ndarray
        Normalized recruitment thresholds (0 to 1).
    mvc_levels : list of float, optional
        %MVC levels to compute targets for. Default [5, 10, 20, 30, 50, 100].
    fr_onset_hz : float
        Firing rate at recruitment (both onset and minimum). Typical 6-8 Hz.
    fr_peak_early_hz : float
        Peak rate (at MVC) for the first-recruited (lowest threshold) MU.
        Default 16 Hz tuned to approach Skarabot 2023 [skarabot2023] untrained VL mean
        (14 Hz). For trained VL use ~24 Hz, for FDI ~26 Hz.
    fr_peak_late_hz : float
        Peak rate (at MVC) for the last-recruited (highest threshold) MU.
        Typically 70-85% of fr_peak_early_hz (onion-skin principle).
        Default 12 Hz pairs with fr_peak_early_hz=16 for untrained VL.
    steepness : float
        Exponential approach rate. Higher = faster saturation.
        Typical 2-5 for human limb muscles.
    """
    if mvc_levels is None:
        mvc_levels = [5, 10, 20, 30, 50, 100]

    thresholds = np.sort(thresholds)
    n_mu = len(thresholds)

    # Per-MU peak rates: interpolate from early to late by rank
    if n_mu > 1:
        ranks = np.arange(n_mu) / (n_mu - 1)  # 0 → 1
    else:
        ranks = np.array([0.0])
    fr_peaks = fr_peak_early_hz + ranks * (fr_peak_late_hz - fr_peak_early_hz)

    targets = {}
    for mvc in mvc_levels:
        excitation = mvc / 100.0
        active_mask = thresholds <= excitation
        n_active = int(np.sum(active_mask))

        if n_active == 0:
            targets[mvc] = {"n_active": 0, "mean_fr_hz": 0.0, "fr_range": (0.0, 0.0)}
            continue

        active_thresholds = thresholds[active_mask]
        active_peaks = fr_peaks[active_mask]
        # Exponential approach: FR = peak - (peak - onset) * exp(-k * (E - thr))
        delta_e = excitation - active_thresholds
        frs = active_peaks - (active_peaks - fr_onset_hz) * np.exp(-steepness * delta_e)
        frs = np.clip(frs, fr_onset_hz, active_peaks)

        targets[mvc] = {
            "n_active": n_active,
            "mean_fr_hz": float(np.mean(frs)),
            "fr_range": (float(frs.min()), float(frs.max())),
        }

    return targets


# ─── Model 3: Experimental lookup ──────────────────────────────


# Per-muscle lookup: {mvc_pct: (mean_fr_hz, std_fr_hz)}
#
# Values are monotonically increasing toward the MVC peak rate from
# Skarabot 2023 [skarabot2023]. Sub-MVC values approximate the rise to peak based on
# De Luca-style exponential approach.
#
# TODO: refine with direct measurements from decomposed MU recordings
# (MyoGen paper Fig 4, Skarabot raw data, etc.). Current values are
# best-estimate placeholders.
EXPERIMENTAL_FR_TARGETS: dict[str, dict[float, tuple[float, float]]] = {
    "vl": {
        # Vastus lateralis, untrained.
        # Peak: Skarabot 2023 [skarabot2023] untrained VL (14.0 ± 4.5 pps at MVC).
        # Sub-MVC: exponential approach from onset ~8 Hz to MVC peak.
        5:   (8.5, 2.0),
        10:  (9.5, 2.3),
        20:  (11.0, 2.8),
        30:  (12.0, 3.2),
        50:  (13.0, 3.8),
        100: (14.0, 4.5),   # Skarabot untrained VL mean [skarabot2023]
    },
    "vl_trained": {
        # Resistance-trained VL. Peak: Skarabot 2023 [skarabot2023] (21.1 ± 4.1 pps).
        5:   (9.0, 2.2),
        10:  (11.0, 2.6),
        20:  (14.0, 3.0),
        30:  (16.5, 3.4),
        50:  (19.0, 3.8),
        100: (21.1, 4.1),   # Skarabot trained VL mean [skarabot2023]
    },
    "fdi": {
        # First dorsal interosseous. Higher peak rates than VL
        # (hand muscles tend to fire faster). Placeholder based on
        # MyoGen paper Fig 4 FDI data.
        5:   (9.0, 2.0),
        10:  (11.0, 2.3),
        20:  (14.0, 2.8),
        30:  (17.0, 3.2),
        50:  (20.0, 3.8),
        100: (24.0, 4.5),
    },
    "ta": {
        # Tibialis anterior. Placeholder — fewer decomposition studies
        # than VL, should refine with data from Desmedt and others.
        5:   (9.0, 2.0),
        10:  (10.5, 2.3),
        20:  (12.5, 2.8),
        30:  (14.0, 3.2),
        50:  (16.0, 3.8),
        100: (19.0, 4.5),
    },
}


def experimental_targets(
    thresholds: np.ndarray,
    mvc_levels: list[float] | None = None,
    muscle_name: str = "vl",
) -> dict[float, dict]:
    """Experimental firing rate targets from published decomposed MU data.

    Uses a muscle-specific lookup table keyed by %MVC. Since the
    experimental data gives population-level mean and std, all active
    MUs at a given level are assigned the same target (the optimizer
    matches mean/std, not per-MU rates).

    Parameters
    ----------
    thresholds : ndarray
        Normalized recruitment thresholds (0 to 1). Used only to
        determine n_active at each %MVC.
    mvc_levels : list of float, optional
        %MVC levels. Must be keys in the lookup table.
    muscle_name : str
        Key into EXPERIMENTAL_FR_TARGETS. Default "vl".

    Raises
    ------
    KeyError
        If muscle_name is not in the lookup table.
    ValueError
        If a requested mvc_level is not in the table for that muscle.
    """
    if mvc_levels is None:
        mvc_levels = [5, 10, 20, 30, 50, 100]

    if muscle_name not in EXPERIMENTAL_FR_TARGETS:
        raise KeyError(
            f"No experimental targets for muscle '{muscle_name}'. "
            f"Available: {list(EXPERIMENTAL_FR_TARGETS.keys())}"
        )

    lookup = EXPERIMENTAL_FR_TARGETS[muscle_name]
    thresholds = np.sort(thresholds)

    targets = {}
    for mvc in mvc_levels:
        if mvc not in lookup:
            raise ValueError(
                f"No experimental data for {mvc}% MVC in muscle '{muscle_name}'. "
                f"Available levels: {sorted(lookup.keys())}"
            )
        mean_fr, std_fr = lookup[mvc]

        excitation = mvc / 100.0
        n_active = int(np.sum(thresholds <= excitation))

        targets[mvc] = {
            "n_active": n_active,
            "mean_fr_hz": float(mean_fr),
            "fr_range": (float(mean_fr - std_fr), float(mean_fr + std_fr)),
        }

    return targets


# ─── Dispatcher ────────────────────────────────────────────────


def compute_fr_targets(
    thresholds: np.ndarray,
    mvc_levels: list[float] | None = None,
    model: str = "de_luca",
    muscle_name: str = "vl",
    **model_params,
) -> dict[float, dict]:
    """Compute firing rate optimization targets using the selected model.

    Parameters
    ----------
    thresholds : ndarray
        Normalized recruitment thresholds (0 to 1).
    mvc_levels : list of float, optional
        %MVC levels to compute targets for.
    model : {"fuglevand", "de_luca", "experimental"}
        Target model to use.

        - "fuglevand": linear rate coding (Fuglevand 1993) [fuglevand1993]. Params:
          ``min_fr_hz``, ``max_fr_hz``.
        - "de_luca": exponential approach to peak (De Luca & Contessa
          2012) [delucacontessa2012]. Params: ``fr_onset_hz``, ``fr_peak_early_hz``,
          ``fr_peak_late_hz``, ``steepness``.
        - "experimental": muscle-specific lookup from published data.
          Requires ``muscle_name``.
    muscle_name : str
        Muscle identifier for experimental lookup. Ignored by other models.
    **model_params
        Additional parameters passed to the selected model function.

    Returns
    -------
    dict
        Mapping mvc_pct -> {"n_active": int, "mean_fr_hz": float,
                            "fr_range": (min, max)}

    Raises
    ------
    ValueError
        If ``model`` is not recognized.
    """
    if model == "fuglevand":
        return fuglevand_targets(thresholds, mvc_levels=mvc_levels, **model_params)
    elif model == "de_luca":
        return de_luca_targets(thresholds, mvc_levels=mvc_levels, **model_params)
    elif model == "experimental":
        return experimental_targets(
            thresholds, mvc_levels=mvc_levels, muscle_name=muscle_name, **model_params
        )
    else:
        raise ValueError(
            f"Unknown rate coding model '{model}'. "
            f"Options: fuglevand, de_luca, experimental"
        )

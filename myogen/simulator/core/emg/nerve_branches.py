"""Two-layer motor-nerve branching model (Konstantin et al. 2020).

Single authority for where a motor unit's end-plates sit along its fibres and
how far the action potential travels along the axon branches to reach them.
Used by intramuscular EMG (``MotorUnitSim``) and surface EMG.

Each motor unit's fibres are grouped into ``n_branches`` primary branches
(k-means on the fibre cross-section positions). A branch point sits at
``endplate_center + N(0, branches_z_std)``; each fibre's end-plate sits at its
branch point's z plus ``N(0, arborization_z_std)``. The nerve path to a fibre
is the distance from the motor unit's centre to its branch point (first layer)
plus the distance from the branch point to the end-plate (second layer).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def nmj_branch_schedule(recruitment_thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Branch count and z spreads per motor unit, growing with motor-unit size.

    Returns ``(n_branches, branches_z_std__mm, arborization_z_std__mm)``. The
    spread factor is the share of total threshold below each unit.
    """
    thresholds = np.asarray(recruitment_thresholds, dtype=float)
    n_branches = (1 + np.round(np.log(thresholds / thresholds[0]))).astype(int)
    spread = np.concatenate([[0.0], np.cumsum(thresholds)[:-1]]) / np.sum(thresholds)
    return n_branches, 1.5 + spread * 4.0, 0.5 + spread * 1.5


def two_layer_nmj(
    fiber_xy__mm: np.ndarray,
    n_branches: int,
    endplate_center__mm: float,
    branches_z_std__mm: float,
    arborization_z_std__mm: float,
    *,
    rng: np.random.Generator,
    kmeans_seed: int,
    return_branches: bool = False,
):
    """End-plate z per fibre and the two nerve-path lengths (mm) per fibre.

    Returns ``(nmj_z, nerve_paths)`` with ``nerve_paths[:, 0]`` the motor-unit
    centre to branch point distance and ``nerve_paths[:, 1]`` the branch point
    to end-plate distance. With ``return_branches`` the branch points
    ``(xy, z)`` and each fibre's branch index are appended.
    """
    fiber_xy__mm = np.asarray(fiber_xy__mm, dtype=float)
    n_fibers = len(fiber_xy__mm)
    kmeans = KMeans(n_clusters=n_branches, init="k-means++", max_iter=100, random_state=kmeans_seed)
    idx = kmeans.fit_predict(fiber_xy__mm)
    branch_xy = kmeans.cluster_centers_
    branch_z = endplate_center__mm + branches_z_std__mm * rng.standard_normal(size=(n_branches, 1))
    # One scalar draw per fibre, in fibre order (the published draw sequence).
    nmj_z = np.array([branch_z[idx[i], 0] + arborization_z_std__mm * rng.standard_normal()
                      for i in range(n_fibers)])

    centre = np.concatenate([np.mean(fiber_xy__mm, axis=0), [np.mean(nmj_z)]])
    branch = np.column_stack([branch_xy[idx], branch_z[idx, 0]])
    nmj = np.column_stack([fiber_xy__mm, nmj_z])
    paths = np.column_stack([np.linalg.norm(centre - branch, axis=1),
                             np.linalg.norm(nmj - branch, axis=1)])
    if return_branches:
        return nmj_z, paths, branch_xy, branch_z, idx
    return nmj_z, paths


def nerve_delays__ms(nerve_paths__mm: np.ndarray, branch_cvs__m_per_s) -> np.ndarray:
    """Axon-branch conduction time per fibre (ms): sum of path / velocity per layer."""
    cvs = np.asarray(branch_cvs__m_per_s, dtype=float)  # m/s == mm/ms
    return np.sum(np.asarray(nerve_paths__mm, dtype=float) / cvs[None, :], axis=1)

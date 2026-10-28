import numpy as np


def fuglevand_gain(r: np.ndarray) -> np.ndarray:
    """
    Compute the rate-dependent twitch gain from Fuglevand et al. (1993), eq. 17.

    Parameters
    ----------
    r : np.ndarray
        Dimensionless contraction time divided by the preceding inter-spike interval.
        Both times must be expressed in the same units.

    Returns
    -------
    np.ndarray
        Gain of 1 for r <= 0.4, otherwise (S(r) / r) / (S(0.4) / 0.4),
        where S(r) = 1 - exp(-2 * r**3).
    """
    r = np.asarray(r, dtype=float)
    gain = np.ones_like(r)
    mask = r > 0.4
    gain[mask] = ((1 - np.exp(-2 * r[mask] ** 3)) / r[mask]) / (
        (1 - np.exp(-2 * 0.4**3)) / 0.4
    )
    return gain

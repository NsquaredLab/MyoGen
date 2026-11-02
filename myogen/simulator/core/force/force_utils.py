import numpy as np

# Force-frequency saturation S(r) = 1 - exp(-a r^c) fitted to human single motor units stimulated through their axons
# (mean plateau force vs contraction time x rate): Macefield et al. 1996 (J Neurophysiol 75:2509), Fuglevand et al.
# 1999 (J Neurophysiol 81:1718), McNulty et al. 2000 (J Physiol 526:445). rms error 0.05 (Fuglevand 1993: 0.13);
# predicts twitch:tetanus 0.23 (measured 0.22-0.36; Fuglevand 1993: 0.11).
FUGLEVAND_FORCE_FREQUENCY = (2.0, 3.0)
HUMAN_UNIT_FORCE_FREQUENCY = (0.93, 1.29)


def fuglevand_gain(r: np.ndarray, a: float = 2.0, c: float = 3.0) -> np.ndarray:
    """
    Compute the rate-dependent twitch gain from Fuglevand et al. (1993), eq. 17.

    Parameters
    ----------
    r : np.ndarray
        Dimensionless contraction time divided by the preceding inter-spike interval.
        Both times must be expressed in the same units.
    a, c : float, default 2.0, 3.0
        Shape of the saturation S(r) = 1 - exp(-a * r**c); Fuglevand et al. (1993) use a = 2, c = 3.
        ``HUMAN_UNIT_FORCE_FREQUENCY`` is a fit to human single motor units.

    Returns
    -------
    np.ndarray
        Gain of 1 for r <= 0.4, otherwise (S(r) / r) / (S(0.4) / 0.4).
    """
    r = np.asarray(r, dtype=float)
    gain = np.ones_like(r)
    mask = r > 0.4
    gain[mask] = ((1 - np.exp(-a * r[mask] ** c)) / r[mask]) / (
        (1 - np.exp(-a * 0.4**c)) / 0.4
    )
    return gain

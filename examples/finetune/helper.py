import numpy as np
import quantities as pq
from elephant.statistics import isi


def get_gamma_shape_for_mvc(
    mvc_percent,
    mvc_shape_value: float = 11.1,
) -> float | np.ndarray:
    """
    Get gamma shape for given MVC percentage using linear interpolation.

    Higher MVC → higher shape → lower CV → more regular firing.

    Parameters
    ----------
    mvc_percent : float or array-like
        Maximum voluntary contraction percentage (0-100).

    -------
    float or np.ndarray
        Gamma shape parameter for the given MVC level.
    """
    mvc_values = np.array([0, 100])
    shape_values = np.array([0.0, mvc_shape_value])

    return np.interp(mvc_percent, mvc_values, shape_values)


def calculate_SD_FR(isi):
    """
    Calculate SD_FR from inter-spike intervals (ISI).

    Parameters:
    isi : numpy 1D array of inter-spike intervals in ms

    Returns:
    SD_FR : float, the standard deviation of firing rate
    """
    if len(isi) < 2:
        return 0.0

    # Calculate moments
    mu = np.mean(isi)
    SD_isi = np.std(isi, ddof=1)  # Sample standard deviation
    mu_3 = np.mean((isi - mu) ** 3)  # Third central moment

    # Calculate SD_FR using the formula
    variance_term = (
        (SD_isi**2 / mu**3) + (1 / 6) + (SD_isi**4 / (2 * mu**4)) - (mu_3 / (3 * mu**3))
    )

    # Handle numerical issues: if variance is negative, clip to zero
    if variance_term < 0:
        return 0.0

    return np.sqrt(variance_term)


def calculate_firing_rate_statistics(spiketrains):
    """Calculate firing rate statistics from spike trains using per-neuron then ensemble approach."""
    firing_rates = []
    sd_frs = []

    for spiketrain in spiketrains:
        if len(spiketrain) > 1:
            # Extract ISIs for this neuron
            isis_values = isi(spiketrain.rescale(pq.s))

            if len(isis_values) > 0:
                neuron_fr = 1.0 / np.mean(isis_values.magnitude)  # type: ignore

                if neuron_fr >= 0.5:
                    firing_rates.append(neuron_fr)
                    # Compute SD_FR for this neuron
                    sd_frs.append(calculate_SD_FR(np.array(isis_values.magnitude)))  # type: ignore

    if not firing_rates:
        return {
            "FR_mean": 0.0,
            "FR_std": 0.0,
            "n_active": 0,
            "firing_rates": np.array([]),
        }

    # Ensemble statistics: mean across neurons for both FR and SD_FR
    return {
        "FR_mean": np.mean(firing_rates),
        "FR_std": np.std(firing_rates, ddof=1),
        "n_active": len(firing_rates),
        "firing_rates": np.array(firing_rates),
    }

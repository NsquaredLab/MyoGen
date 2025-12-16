import numpy as np
import pandas as pd
import quantities as pq

try:
    from elephant.statistics import isi

    HAS_ELEPHANT = True
except ImportError:
    HAS_ELEPHANT = False
    isi = None  # type: ignore


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
    variance_term = (SD_isi**2 / mu**3) + (1 / 6) + (SD_isi**4 / (2 * mu**4)) - (mu_3 / (3 * mu**3))

    # Handle numerical issues: if variance is negative, clip to zero
    if variance_term < 0:
        return 0.0

    return np.sqrt(variance_term)


def calculate_firing_rate_statistics(
    spiketrains,
    plateau_start_ms=None,
    plateau_end_ms=None,
    return_per_neuron=False,
    min_spikes_for_cv=3,
    min_firing_rate=None,
):
    """
    Calculate firing rate statistics from spike trains using per-neuron then ensemble approach.

    IMPORTANT: ISI and FR should only be computed during the plateau phase
    where firing is stable, not during ramp-up/down where rates change by design.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        List of spike train objects to analyze.
    plateau_start_ms : float, optional
        Start time of plateau phase in milliseconds.
        If provided along with plateau_end_ms, only spikes within
        the plateau phase will be used for FR/ISI calculation.
    plateau_end_ms : float, optional
        End time of plateau phase in milliseconds.
    return_per_neuron : bool, optional
        If True, returns DataFrame with per-neuron statistics including CV_ISI.
        If False (default), returns ensemble statistics as dict.
    min_spikes_for_cv : int, optional
        Minimum number of spikes required to compute CV (default: 3).
        Only used when return_per_neuron=True.
    min_firing_rate : float, optional
        Minimum firing rate threshold in Hz. Neurons below this are excluded.
        If None, uses 0.5 Hz for ensemble stats (backward compatibility) or
        no filtering for per-neuron stats.

    Returns
    -------
    dict or pd.DataFrame
        If return_per_neuron=False:
            Dictionary with keys: FR_mean, FR_std, n_active, firing_rates
        If return_per_neuron=True:
            DataFrame with columns: MU_ID, mean_firing_rate_Hz, CV_ISI
    """
    if not HAS_ELEPHANT:
        raise ImportError(
            "Elephant is required for firing rate statistics (ISI calculation). "
            "Install with: pip install myogen[elephant]"
        )

    # Set default min_firing_rate based on mode
    if min_firing_rate is None:
        min_firing_rate = 0.5 if not return_per_neuron else 0.0

    firing_rates = []
    sd_frs = []
    per_neuron_results = []

    for mu_id, spiketrain in enumerate(spiketrains):
        # Filter to plateau phase if boundaries are provided
        if plateau_start_ms is not None and plateau_end_ms is not None:
            # Use time_slice to extract only plateau spikes
            plateau_spiketrain = spiketrain.time_slice(
                plateau_start_ms * pq.ms, plateau_end_ms * pq.ms
            )
        else:
            plateau_spiketrain = spiketrain

        # For per-neuron CV calculation, need at least min_spikes_for_cv spikes
        min_spikes = min_spikes_for_cv if return_per_neuron else 2

        if len(plateau_spiketrain) > min_spikes - 1:
            # Extract ISIs for this neuron (only from plateau phase)
            isis_values = isi(plateau_spiketrain.rescale(pq.s))

            if len(isis_values) > 0:
                isis_array = np.array(isis_values.magnitude)  # type: ignore
                neuron_fr = 1.0 / np.mean(isis_array)

                if neuron_fr >= min_firing_rate:
                    firing_rates.append(neuron_fr)

                    if return_per_neuron:
                        # Compute CV of inter-spike intervals
                        cv = np.std(isis_array, ddof=1) / np.mean(isis_array)
                        per_neuron_results.append(
                            {
                                "MU_ID": mu_id,
                                "mean_firing_rate_Hz": neuron_fr,
                                "CV_ISI": cv,
                            }
                        )
                    else:
                        # Compute SD_FR for ensemble statistics
                        sd_frs.append(calculate_SD_FR(isis_array))

    # Return per-neuron DataFrame or ensemble statistics
    if return_per_neuron:
        return pd.DataFrame(per_neuron_results)
    else:
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

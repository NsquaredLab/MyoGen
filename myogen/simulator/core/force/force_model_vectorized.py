from .force_model import ForceModel


class ForceModelVectorized(ForceModel):
    """
    Force model kept for backward compatibility; identical to ForceModel.

    Parameters
    ----------
    recruitment_thresholds : RECRUITMENT_THRESHOLDS__ARRAY
        Recruitment thresholds for each motor unit.
    recording_frequency__Hz : Quantity__Hz
        Recording frequency in Hz. Determines temporal resolution of the force output.
    longest_duration_rise_time__ms : Quantity__ms, default=90.0 * pq.ms
        Longest duration of the rise time in milliseconds.
    contraction_time_range_factor : float, default=3.0
        Contraction time range factor. Determines the spread of contraction times.

    Notes
    -----
    All parameters, attributes and methods are inherited from ForceModel.
    contraction_times__samples are in samples of recording_frequency__Hz.
    Force is in units of peak twitch force P, independent of recording_frequency__Hz.
    Each unit saturates at approximately 9.05 P; the pool maximum is approximately
    9.05 * sum(P). Twitches are summed exactly on the spike-train grid without truncation.

    References
    ----------
    [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993). Models of recruitment
    and rate coding organization in motor-unit pools. J Neurophysiol, 70(6), 2470–2488.
    https://doi.org/10.1152/jn.1993.70.6.2470.
    """

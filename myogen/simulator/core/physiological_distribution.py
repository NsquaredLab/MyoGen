from typing import Literal

import numpy as np

from myogen.utils.types import beartowertype, RECRUITMENT_THRESHOLDS__ARRAY


@beartowertype
def generate_mu_recruitment_thresholds(
    N: int,
    recruitment_range__ratio: float,
    deluca__slope: float | None = None,
    konstantin__max_threshold__ratio: float = 1.0,
    mode: Literal["fuglevand", "deluca", "konstantin", "combined"] = "konstantin",
) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
    r"""
    Generate recruitment thresholds for a pool of motor units using different models.

    This function computes the recruitment thresholds (and zero-based thresholds) for a pool of N motor units
    according to one of several models from the literature. The distribution of thresholds is controlled by the
    recruitment range (RR) and, for some models, additional parameters.

    Following models are available:  
        - Fuglevand et al. (1993) [1]_
        - De Luca & Contessa (2012) [2]_
        - Konstantin et al. (2020) [3]_
        - Combined model

    Parameters
    ----------
    N : int
        Number of motor units in the pool.
    recruitment_range__ratio : float
        Recruitment range (dimensionless ratio), defined as the ratio of the largest to smallest threshold
        :math:`(rt(N)/rt(1))`.
    deluca__slope : float, optional
        Dimensionless slope parameter for the ``'deluca'`` mode. Required if ``mode='deluca'``.
        Controls the curvature of the threshold distribution. Typical values range from 0.001-100.
    konstantin__max_threshold__ratio : float, optional
        Maximum recruitment threshold (dimensionless ratio) for the ``'konstantin'`` mode. Required if ``mode='konstantin'``.
        Sets the absolute scale of all thresholds. Default is 1.0.
    mode : RecruitmentMode, optional
        Model to use for threshold generation. One of ``'fuglevand'``, ``'deluca'``, ``'konstantin'``, or ``'combined'``.
        Default is ``'konstantin'``.

    Returns
    -------
    rt : RECRUITMENT_THRESHOLDS__ARRAY
        Recruitment thresholds for each motor unit (shape: (N,)).
        Values are monotonically increasing from ``rt[0]`` to ``rt[N-1]``.
    rtz : RECRUITMENT_THRESHOLDS__ARRAY
        Zero-based recruitment thresholds where :math:`rtz[0] = 0` (shape: (N,)).
        Computed as :math:`rtz = rt - rt[0]`, convenient for simulation.

    Raises
    ------
    ValueError
        If a required mode-specific parameter is not provided or if an unknown mode is specified.

    References
    ----------
    .. [1] Fuglevand, A.J., Winter, D.A., Patla, A.E., 1993. 
           Models of recruitment and rate coding organization in motor-unit pools. 
           Journal of Neurophysiology 70, 2470–2488. https://doi.org/10.1152/jn.1993.70.6.2470
    .. [2] De Luca, C.J., Contessa, P., 2012. 
           Hierarchical control of motor units in voluntary contractions. 
           Journal of Neurophysiology 107, 178–195. https://doi.org/10.1152/jn.00961.2010
    .. [3] Konstantin, A., Yu, T., Le Carpentier, E., Aoustin, Y., Farina, D., 2020. 
           Simulation of Motor Unit Action Potential Recordings From Intramuscular Multichannel Scanning Electrodes. 
           IEEE Transactions on Biomedical Engineering 67, 2005–2014. https://doi.org/10.1109/TBME.2019.2953680




    Notes
    -----
    **fuglevand** : Fuglevand et al. (1993) [1]_ exponential model
        .. math:: rt(i) = \exp( \frac{i \cdot \ln(RR)}{N} ) / 100

        where :math:`i = 1, 2, \ldots, N`

    **deluca** : De Luca & Contessa (2012) [2]_ model with slope correction
        .. math::
            rt(i) = \frac{b \cdot i}{N} \cdot \exp\left(\frac{i \cdot \ln(RR / b)}{N}\right) / 100

        where :math:`b` = ``deluca__slope``, :math:`i = 1, 2, \ldots, N`

    **konstantin** : Konstantin et al. (2020) [3]_ model allowing explicit maximum threshold control
        .. math::
            rt(i) &= \frac{RT_{max}}{RR} \cdot \exp\left(\frac{(i - 1) \cdot \ln(RR)}{N - 1}\right) \\
            rtz(i) &= \frac{RT_{max}}{RR} \cdot \left(\exp\left(\frac{(i - 1) \cdot \ln(RR + 1)}{N}\right) - 1\right)

        where :math:`RT_{max}` = ``konstantin__max_threshold__ratio``, :math:`i = 1, 2, \ldots, N`

    **combined** : A corrected De Luca model that uses the slope parameter for shape control but properly respects the RR constraint and maximum threshold like the Konstantin model
        .. math::
            rt(i) = \frac{RT_{max}}{RR} + \left(\frac{b \cdot i}{N} \cdot \exp\left(\frac{i \cdot \ln(RR / b)}{N}\right) - \frac{RT_{max}}{RR}\right) \cdot \left(\frac{RT_{max} - RT_{max}/RR}{b \cdot N \cdot \exp\left(\frac{i \cdot \ln(RR / b)}{N}\right) - \frac{RT_{max}}{RR}}\right)

        where :math:`b` = ``deluca__slope``, :math:`RT_{max}` = ``konstantin__max_threshold__ratio``, :math:`i = 1, 2, \ldots, N`

    .. note::
        - All models ensure :math:`rt(1) < rt(2) < \ldots < rt(N)` (monotonically increasing)
        - The recruitment range :math:`RR = rt(N) / rt(1)` is preserved across all models
        - ``rtz`` is a zero-based version where :math:`rtz(1) = 0`, useful for simulation
        - Motor units are recruited when excitation :math:`> rt(i)` for unit :math:`i`

    Examples
    --------
    >>> # Generate thresholds using Fuglevand model
    >>> rt, rtz = generate_mu_recruitment_thresholds(
    ...     N=100, recruitment_range__ratio=50.0, mode='fuglevand'
    ... )
    >>>
    >>> # Generate thresholds using Konstantin model with explicit max threshold
    >>> rt, rtz = generate_mu_recruitment_thresholds(
    ...     N=100, recruitment_range__ratio=50.0, konstantin__max_threshold__ratio=1.0, mode='konstantin'
    ... )
    """
    i = np.arange(N)

    match mode:
        case "fuglevand":
            rt = np.exp((np.log(recruitment_range__ratio) / N) * i) / 100
            rtz = rt - rt[0]
        case "deluca":
            if deluca__slope is None:
                raise ValueError("deluca__slope must be provided for 'deluca' mode.")
            rt = (
                (deluca__slope * i / N)
                * np.exp((np.log(recruitment_range__ratio / deluca__slope) / N) * i)
                / 100
            )
            rtz = rt - rt[0]
        case "konstantin":
            if konstantin__max_threshold__ratio is None:
                raise ValueError(
                    "konstantin__max_threshold__ratio must be provided for 'konstantin' mode."
                )
            rt = (konstantin__max_threshold__ratio / recruitment_range__ratio) * np.exp(
                (i - 1) * np.log(recruitment_range__ratio) / (N - 1)
            )
            rtz = (konstantin__max_threshold__ratio / recruitment_range__ratio) * (
                np.exp((i - 1) * np.log(recruitment_range__ratio + 1) / N) - 1
            )
        case "combined":
            if deluca__slope is None or konstantin__max_threshold__ratio is None:
                raise ValueError(
                    "Both deluca__slope and konstantin__max_threshold__ratio must be provided for 'combined' mode."
                )
            # Create a De Luca-style curve with slope parameter controlling curvature
            # but properly scaled to respect RR and max threshold

            # Generate base De Luca shape (without the /100 scaling)
            base_shape = (deluca__slope * i / N) * np.exp(
                (np.log(recruitment_range__ratio / deluca__slope) / N) * i
            )

            # Scale to ensure exact RR and max threshold
            # We want: rt[0] = konstantin__max_threshold__ratio / RR and rt[-1] = konstantin__max_threshold__ratio
            min_val = base_shape[0]
            max_val = base_shape[-1]

            # Scale the shape to achieve the desired RR while respecting max threshold
            rt = (konstantin__max_threshold__ratio / recruitment_range__ratio) + (
                base_shape - min_val
            ) * (
                konstantin__max_threshold__ratio
                - konstantin__max_threshold__ratio / recruitment_range__ratio
            ) / (max_val - min_val)

            rtz = rt - rt[0]

        case _:
            raise ValueError(f"Unknown mode: {mode}")

    # Normalize the thresholds to the maximum threshold
    rtz = rtz * np.max(rt) / np.max(rtz)

    return rt, rtz

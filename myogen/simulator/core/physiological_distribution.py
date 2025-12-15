from typing import Literal

import numpy as np
from beartype.typing import Iterator

from myogen.utils.decorators import beartowertype
from myogen.utils.types import RECRUITMENT_THRESHOLDS__ARRAY


@beartowertype
class RecruitmentThresholds:
    """
    Motor unit recruitment threshold generator using physiological models.

    This class computes recruitment thresholds for motor unit pools using
    different established models from the literature.
    """

    def __init__(
        self,
        N: int,
        recruitment_range__ratio: float,
        deluca__slope: float | None = None,
        konstantin__max_threshold__ratio: float = 1.0,
        mode: Literal["fuglevand", "deluca", "konstantin", "combined"] = "konstantin",
    ) -> None:
        r"""
        Generate recruitment thresholds for a pool of motor units using different models.

        This class computes the recruitment thresholds (and zero-based thresholds) for a pool of N motor units
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

        Attributes
        ----------
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
               Journal of Neurophysiology 70, 2470-2488. https://doi.org/10.1152/jn.1993.70.6.2470
        .. [2] De Luca, C.J., Contessa, P., 2012. 
               Hierarchical control of motor units in voluntary contractions. 
               Journal of Neurophysiology 107, 178-195. https://doi.org/10.1152/jn.00961.2010
        .. [3] Konstantin, A., Yu, T., Le Carpentier, E., Aoustin, Y., Farina, D., 2020. 
               Simulation of Motor Unit Action Potential Recordings From Intramuscular Multichannel Scanning Electrodes. 
               IEEE Transactions on Biomedical Engineering 67, 2005-2014. https://doi.org/10.1109/TBME.2019.2953680

        Notes
        -----
        **fuglevand** : Fuglevand et al. (1993) [1]_ exponential model
            .. math:: rt(i) = \\exp( \\frac{i \\cdot \\ln(RR)}{N} ) / 100

            where :math:`i = 1, 2, \\ldots, N`

        **deluca** : De Luca & Contessa (2012) [2]_ model with slope correction
            .. math::
                rt(i) = \\frac{b \\cdot i}{N} \\cdot \\exp\\left(\\frac{i \\cdot \\ln(RR / b)}{N}\\right) / 100

            where :math:`b` = ``deluca__slope``, :math:`i = 1, 2, \\ldots, N`

        **konstantin** : Konstantin et al. (2020) [3]_ model allowing explicit maximum threshold control
            .. math::
                rt(i) &= \\frac{RT_{max}}{RR} \\cdot \\exp\\left(\\frac{(i - 1) \\cdot \\ln(RR)}{N - 1}\\right) \\\\
                rtz(i) &= \\frac{RT_{max}}{RR} \\cdot \\left(\\exp\\left(\\frac{(i - 1) \\cdot \\ln(RR + 1)}{N}\\right) - 1\\right)

            where :math:`RT_{max}` = ``konstantin__max_threshold__ratio``, :math:`i = 1, 2, \\ldots, N`

        **combined** : A corrected De Luca model that uses the slope parameter for shape control but properly respects the RR constraint and maximum threshold like the Konstantin model
            .. math::
                rt(i) = \\frac{RT_{max}}{RR} + \\left(\\frac{b \\cdot i}{N} \\cdot \\exp\\left(\\frac{i \\cdot \\ln(RR / b)}{N}\\right) - \\frac{RT_{max}}{RR}\\right) \\cdot \\left(\\frac{RT_{max} - RT_{max}/RR}{b \\cdot N \\cdot \\exp\\left(\\frac{i \\cdot \\ln(RR / b)}{N}\\right) - \\frac{RT_{max}}{RR}}\\right)

            where :math:`b` = ``deluca__slope``, :math:`RT_{max}` = ``konstantin__max_threshold__ratio``, :math:`i = 1, 2, \\ldots, N`

        Examples
        --------
        >>> # Generate thresholds using Fuglevand model
        >>> thresholds = RecruitmentThresholds(
        ...     N=100, recruitment_range__ratio=50.0, mode='fuglevand'
        ... )
        >>> rt, rtz = thresholds  # Tuple unpacking works
        >>> # Or access directly
        >>> rt = thresholds.rt
        >>> rtz = thresholds.rtz
        >>>
        >>> # Generate thresholds using Konstantin model with explicit max threshold
        >>> thresholds = RecruitmentThresholds(
        ...     N=100, recruitment_range__ratio=50.0, konstantin__max_threshold__ratio=1.0, mode='konstantin'
        ... )
        >>> rt, rtz = thresholds
        """
        # Store immutable public parameters (for joblib serialization)
        self.N = N
        self.recruitment_range__ratio = recruitment_range__ratio
        self.deluca__slope = deluca__slope
        self.konstantin__max_threshold__ratio = konstantin__max_threshold__ratio
        self.mode = mode

        # Generate and store thresholds
        self.rt, self.rtz = self._generate_thresholds()

    def __iter__(self) -> Iterator[RECRUITMENT_THRESHOLDS__ARRAY]:
        """Allow tuple unpacking: rt, rtz = thresholds"""
        return iter((self.rt, self.rtz))

    def __getitem__(self, index: int) -> RECRUITMENT_THRESHOLDS__ARRAY:
        """Allow indexing: thresholds[0] -> rt, thresholds[1] -> rtz"""
        if index == 0:
            return self.rt
        elif index == 1:
            return self.rtz
        else:
            raise IndexError("Index out of range. Use 0 for rt and 1 for rtz.")

    def _generate_thresholds(
        self,
    ) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
        """Generate recruitment thresholds based on the specified model."""
        match self.mode:
            case "fuglevand":
                rt, rtz = self._fuglevand_model()
            case "deluca":
                rt, rtz = self._deluca_model()
            case "konstantin":
                rt, rtz = self._konstantin_model()
            case "combined":
                rt, rtz = self._combined_model()
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")

        # Normalize the thresholds to the maximum threshold
        rtz = rtz * np.max(rt) / np.max(rtz)

        return rt, rtz

    def _fuglevand_model(
        self,
    ) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
        i = np.arange(self.N) + 1
        rt = np.exp((np.log(self.recruitment_range__ratio) / self.N) * i) / 100
        rtz = rt - rt[0]
        return rt, rtz

    def _deluca_model(
        self,
    ) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
        if self.deluca__slope is None:
            raise ValueError("deluca__slope must be provided for 'deluca' mode.")

        i = np.arange(self.N) + 1
        rt = (
            (self.deluca__slope * i / self.N)
            * np.exp((np.log(self.recruitment_range__ratio / self.deluca__slope) / self.N) * i)
            / 100
        )
        rtz = rt - rt[0]
        return rt, rtz

    def _konstantin_model(
        self,
    ) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
        if self.konstantin__max_threshold__ratio is None:
            raise ValueError(
                "konstantin__max_threshold__ratio must be provided for 'konstantin' mode."
            )

        i = np.arange(self.N) + 1
        rt = (self.konstantin__max_threshold__ratio / self.recruitment_range__ratio) * np.exp(
            (i - 1) * np.log(self.recruitment_range__ratio) / (self.N - 1)
        )
        rtz = (self.konstantin__max_threshold__ratio / self.recruitment_range__ratio) * (
            np.exp((i - 1) * np.log(self.recruitment_range__ratio + 1) / self.N) - 1
        )
        return rt, rtz

    def _combined_model(
        self,
    ) -> tuple[RECRUITMENT_THRESHOLDS__ARRAY, RECRUITMENT_THRESHOLDS__ARRAY]:
        if self.deluca__slope is None or self.konstantin__max_threshold__ratio is None:
            raise ValueError(
                "Both deluca__slope and konstantin__max_threshold__ratio must be provided for 'combined' mode."
            )

        i = np.arange(self.N) + 1

        # Create a De Luca-style curve with slope parameter controlling curvature
        # but properly scaled to respect RR and max threshold

        # Generate base De Luca shape (without the /100 scaling)
        base_shape = (self.deluca__slope * i / self.N) * np.exp(
            (np.log(self.recruitment_range__ratio / self.deluca__slope) / self.N) * i
        )

        # Scale to ensure exact RR and max threshold
        # We want: rt[0] = konstantin__max_threshold__ratio / RR and rt[-1] = konstantin__max_threshold__ratio
        min_val = base_shape[0]
        max_val = base_shape[-1]

        # Scale the shape to achieve the desired RR while respecting max threshold
        rt = (self.konstantin__max_threshold__ratio / self.recruitment_range__ratio) + (
            base_shape - min_val
        ) * (
            self.konstantin__max_threshold__ratio
            - self.konstantin__max_threshold__ratio / self.recruitment_range__ratio
        ) / (max_val - min_val)

        rtz = rt - rt[0]
        return rt, rtz

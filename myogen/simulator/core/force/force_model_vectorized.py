import logging
from typing import Optional

try:
    import elephant
    import elephant.utils

    HAS_ELEPHANT = True
except ImportError:
    HAS_ELEPHANT = False
    elephant = None  # type: ignore
import numpy as np
import quantities as pq
from neo import AnalogSignal
from tqdm import tqdm

from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    RECRUITMENT_THRESHOLDS__ARRAY,
    FORCE__AnalogSignal,
    SPIKE_TRAIN__Block,
)

from .force_utils_vectorized import (
    get_gain,
    sawtooth2ipi,
    spikes2sawtooth,
    generate_force_vectorized,
)


@beartowertype
class ForceModelVectorized:
    """
    Vectorized force model based on Fuglevand et al. (1993) [1]_.

    This is an optimized version of ForceModel that uses numpy vectorization
    for significantly better performance, especially for long simulations.

    Parameters
    ----------
    recruitment_thresholds : RECRUITMENT_THRESHOLDS__ARRAY
        Recruitment thresholds for each motor unit.
    recording_frequency__Hz : float
        Recording frequency in Hz.
    longest_duration_rise_time__ms : float, default=90.0
        Longest duration of the rise time in milliseconds.
    contraction_time_range__unitless : float, default=3.0
        Contraction time range factor.

    References
    ----------
    .. [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993).
        Models of recruitment and rate coding in motor-unit pools.
        Journal of Neurophysiology, 70(2), 782-797.
    """

    def __init__(
        self,
        recruitment_thresholds: RECRUITMENT_THRESHOLDS__ARRAY,
        recording_frequency__Hz: float,
        longest_duration_rise_time__ms: float = 90.0,
        contraction_time_range__unitless: float = 3.0,
    ) -> None:
        # Input validation
        if len(recruitment_thresholds) == 0:
            raise ValueError(
                "recruitment_thresholds cannot be empty. "
                "Please provide at least one recruitment threshold value."
            )

        if not np.all(recruitment_thresholds > 0):
            raise ValueError(
                "All recruitment thresholds must be positive. "
                "Found values: min={:.3f}, max={:.3f}. "
                "Recruitment thresholds typically range from 0.01 to 1.0.".format(
                    np.min(recruitment_thresholds), np.max(recruitment_thresholds)
                )
            )

        if recording_frequency__Hz <= 0:
            raise ValueError(
                f"recording_frequency__Hz must be positive, got {recording_frequency__Hz}. "
                "Typical values for EMG/force recordings are between 1000-10000 Hz."
            )

        if longest_duration_rise_time__ms <= 0:
            raise ValueError(
                f"longest_duration_rise_time__ms must be positive, got {longest_duration_rise_time__ms}. "
                "Typical values range from 50-150 ms for human motor units."
            )

        if contraction_time_range__unitless <= 1.0:
            raise ValueError(
                f"contraction_time_range__unitless must be greater than 1.0, got {contraction_time_range__unitless}. "
                "This parameter determines the spread of contraction times. Typical values are 2.0-5.0."
            )

        # Immutable public access
        self.recruitment_thresholds = recruitment_thresholds
        self.recording_frequency__Hz = recording_frequency__Hz
        self.longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self.contraction_time_range__unitless = contraction_time_range__unitless

        # Private copies for internal modifications
        self._recruitment_thresholds = recruitment_thresholds.copy()
        self._recording_frequency__Hz = recording_frequency__Hz
        self._longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self._contraction_time_range__unitless = contraction_time_range__unitless

        # Derived properties
        self._number_of_neurons = len(self._recruitment_thresholds)
        self._recruitment_ratio = (
            self._recruitment_thresholds[-1] / self._recruitment_thresholds[0]
        )

        self._longest_duration_rise_time__samples = (
            self._longest_duration_rise_time__ms / 1000 * self._recording_frequency__Hz
        )

        # Simulation results
        self._peak_twitch_forces__unitless: Optional[np.ndarray] = None
        self._contraction_times__samples: Optional[np.ndarray] = None
        self._twitch_mat: Optional[np.ndarray] = None
        self._twitch_list: Optional[list[np.ndarray]] = None

        # Initialize model parameters
        self._compute_twitch_parameters()

    def _compute_twitch_parameters(self) -> None:
        """Compute peak twitch forces and contraction times based on Fuglevand model."""
        self._peak_twitch_forces__unitless = np.exp(
            (np.log(self._recruitment_ratio) / self._number_of_neurons)
            * np.arange(1, self._number_of_neurons + 1)
        )

        self._contraction_times__samples = (
            self._longest_duration_rise_time__samples
            * np.power(
                1 / self._peak_twitch_forces__unitless,
                1
                / np.emath.logn(
                    self._contraction_time_range__unitless, self._recruitment_ratio
                ),
            )
        )

        self._initialize_twitches()

    def _initialize_twitches(self) -> None:
        """Initialize the twitches matrix and the twitch list."""
        if (
            self._peak_twitch_forces__unitless is None
            or self._contraction_times__samples is None
        ):
            raise ValueError(
                "Twitch parameters not computed. "
                "Call _compute_twitch_parameters() first."
            )

        max_twitch_length = int(np.ceil(5 * np.max(self._contraction_times__samples)))
        twitch_timelines_reshaped = np.arange(max_twitch_length)[:, np.newaxis]

        self._twitch_mat = (
            self._peak_twitch_forces__unitless
            / self._contraction_times__samples
            * twitch_timelines_reshaped
            * np.exp(1 - twitch_timelines_reshaped / self._contraction_times__samples)
        )

        self._twitch_list = [
            self._twitch_mat[:L, i]
            for i, L in enumerate(
                np.minimum(
                    max_twitch_length,
                    np.ceil(5 * self._contraction_times__samples).astype(int),
                )
            )
        ]

    def generate_force(
        self, spike_train__Block: SPIKE_TRAIN__Block
    ) -> FORCE__AnalogSignal:
        """
        Generate force output from motor unit spike trains using the Fuglevand model.

        This vectorized version provides significantly better performance for long simulations.

        Parameters
        ----------
        spike_train__Block : SPIKE_TRAIN__Block
            Spike train block containing spike train data.

        Returns
        -------
        FORCE__AnalogSignal
            Force output as neo.AnalogSignal.
        """
        if self._twitch_list is None:
            raise ValueError(
                "Twitch parameters not available. "
                "This should not occur if the model was properly initialized."
            )

        if not HAS_ELEPHANT:
            raise ImportError(
                "Elephant is required for force simulation. "
                "Install with: pip install myogen[elephant]"
            )

        # Extract timing information
        spiketrain_timestep__ms = float(
            spike_train__Block.segments[0]
            .spiketrains[0]
            .sampling_period.rescale("ms")
            .magnitude
        )

        forces = []
        for i, segment in enumerate(spike_train__Block.segments):
            if len(segment.spiketrains) != self._number_of_neurons:
                raise ValueError(
                    f"MU pool {i} has {len(segment.spiketrains)} neurons, "
                    f"but force model was initialized with {self._number_of_neurons} motor units."
                )

            elephant_utils_logger = logging.getLogger(elephant.utils.__file__)
            original_level = elephant_utils_logger.level
            elephant_utils_logger.setLevel(logging.ERROR)

            try:
                spike_array = (
                    elephant.conversion.BinnedSpikeTrain(
                        segment.spiketrains,
                        bin_size=segment.spiketrains[0].sampling_period,
                        t_start=segment.t_start,
                        t_stop=segment.t_stop,
                    )
                    .to_array()
                    .astype(bool)
                    .T
                )

                # Generate force with vectorized implementation
                force_output = self._generate_force_vectorized(
                    spike_array, spiketrain_timestep__ms, prefix=f"Pool {i + 1}"
                )
                forces.append(force_output)

            finally:
                elephant_utils_logger.setLevel(original_level)

        return AnalogSignal(
            np.stack(forces, axis=-1) * pq.dimensionless,
            t_start=spike_train__Block.segments[0].t_start,
            sampling_rate=self._recording_frequency__Hz * pq.Hz,
        )

    def _generate_force_vectorized(
        self, spikes: np.ndarray, spiketrain_timestep__ms: float, prefix: str = ""
    ) -> np.ndarray:
        """Generate force using vectorized operations for better performance."""
        L = spikes.shape[0]

        # Calculate timing parameters
        spiketrain_timestep__s = spiketrain_timestep__ms / 1000.0
        force_timestep__s = 1.0 / self._recording_frequency__Hz

        # IPI signal generation
        _, ipi = sawtooth2ipi(
            spikes2sawtooth(
                np.vstack([spikes[1:], np.zeros((1, self._number_of_neurons))])
            )
        )

        # Calculate gain for all motor units
        gain = np.full_like(spikes, np.nan, dtype=float)
        for n in range(self._number_of_neurons):
            gain[:, n] = get_gain(ipi[:, n], self._contraction_times__samples[n])

        # Resample twitches to spike train sampling rate
        resampled_twitches = []
        for force_twitch in self._twitch_list:
            resampled_twitches.append(
                np.interp(
                    x=np.arange(
                        0,
                        force_twitch.shape[0] * force_timestep__s,
                        spiketrain_timestep__s,
                    ),
                    xp=np.arange(
                        0, force_twitch.shape[0] * force_timestep__s, force_timestep__s
                    ),
                    fp=force_twitch,
                )
            )

        # Use vectorized force generation
        print(f"{prefix} Generating force with vectorized implementation...")
        force = generate_force_vectorized(spikes, gain, resampled_twitches)

        # Resample to recording frequency
        return np.interp(
            x=np.arange(0, force.shape[0] * spiketrain_timestep__s, force_timestep__s),
            xp=np.arange(
                0, force.shape[0] * spiketrain_timestep__s, spiketrain_timestep__s
            ),
            fp=force,
        )

    # Property accessors
    @property
    def peak_twitch_forces__unitless(self) -> np.ndarray:
        if self._peak_twitch_forces__unitless is None:
            raise ValueError("Peak twitch forces not computed.")
        return self._peak_twitch_forces__unitless

    @property
    def contraction_times__samples(self) -> np.ndarray:
        if self._contraction_times__samples is None:
            raise ValueError("Contraction times not computed.")
        return self._contraction_times__samples

    @property
    def twitch_mat(self) -> np.ndarray:
        if self._twitch_mat is None:
            raise ValueError("Twitch matrix not computed.")
        return self._twitch_mat

    @property
    def twitch_list(self) -> list[np.ndarray]:
        if self._twitch_list is None:
            raise ValueError("Twitch list not computed.")
        return self._twitch_list

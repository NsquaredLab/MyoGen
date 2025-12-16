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
import scipy.sparse as sp
from neo import AnalogSignal
from tqdm import tqdm

from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    RECRUITMENT_THRESHOLDS__ARRAY,
    FORCE__AnalogSignal,
    Quantity__Hz,
    Quantity__ms,
    SPIKE_TRAIN__Block,
)

from .force_utils import get_gain_vectorized, sawtooth2ipi, spikes2sawtooth


@beartowertype
class ForceModel:
    """
    Force model based on Fuglevand et al. (1993) [1]_.

    This class implements the Fuglevand force generation model for motor unit pools,
    computing individual motor unit twitch responses and their nonlinear gain modulation
    based on discharge rates. The model generates realistic force outputs from spike trains
    using physiologically-based parameters.

    Parameters
    ----------
    recruitment_thresholds : RECRUITMENT_THRESHOLDS__ARRAY
        Recruitment thresholds for each motor unit. Array of values typically ranging
        from 0 to 1 where larger motor units have higher thresholds.
    recording_frequency__Hz : Quantity__Hz
        Recording frequency in Hz. Determines temporal resolution of force calculations.
        Typical values: 100-1000 Hz.
    longest_duration_rise_time__ms : Quantity__ms, default=90.0 * pq.ms
        Longest duration of the rise time in milliseconds. This parameter (T_L in _[1])
        determines the contraction time of the slowest motor unit. Typical range: 50-150 ms.
    contraction_time_range_factor : float, default=3.0
        Contraction time range factor (RT in _[1]). Determines the spread of contraction
        times across motor units. Generally between 2 and 5. Higher values create
        larger differences between fast and slow motor units.

    Attributes
    ----------
    peak_twitch_forces__unitless : np.ndarray
        Peak twitch forces for each motor unit (unitless). Available after initialization.
        Computed according to Fuglevand equation 13.
    contraction_times__samples : np.ndarray
        Contraction times for each motor unit in samples. Available after initialization.
        Computed according to Fuglevand equation 14.
    twitch_mat : np.ndarray
        Complete twitch matrix for all motor units. Available after initialization.
        Shape: (max_twitch_length, n_motor_units).
    twitch_list : list[np.ndarray]
        List of individual twitch responses for each motor unit. Available after initialization.
        Each element contains the twitch response for one motor unit.

    Raises
    ------
    ValueError
        If recruitment_thresholds is empty or contains invalid values.
        If recording_frequency__Hz is not positive.
        If longest_duration_rise_time__ms is not positive.
        If contraction_time_range__unitless is not greater than 1.

    References
    ----------
    .. [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993).
        Models of recruitment and rate coding in motor-unit pools.
        Journal of Neurophysiology, 70(2), 782-797.

    Examples
    --------
    >>> import numpy as np
    >>> from myogen.simulator.core.force import ForceModel
    >>> thresholds = np.linspace(0.1, 1.0, 10)
    >>> force_model = ForceModel(
    ...     recruitment_thresholds=thresholds,
    ...     recording_frequency__Hz=2000.0
    ... )
    """

    def __init__(
        self,
        recruitment_thresholds: RECRUITMENT_THRESHOLDS__ARRAY,
        recording_frequency__Hz: Quantity__Hz,
        longest_duration_rise_time__ms: Quantity__ms = 90.0 * pq.ms,
        contraction_time_range_factor: float = 3.0,
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

        if contraction_time_range_factor <= 1.0:
            raise ValueError(
                f"contraction_time_range__unitless must be greater than 1.0, got {contraction_time_range_factor}. "
                "This parameter determines the spread of contraction times. Typical values are 2.0-5.0."
            )

        # Immutable public access
        self.recruitment_thresholds = recruitment_thresholds
        self.recording_frequency__Hz = recording_frequency__Hz
        self.longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self.contraction_time_range__unitless = contraction_time_range_factor

        # Private copies for internal modifications
        self._recruitment_thresholds = recruitment_thresholds.copy()
        self._recording_frequency__Hz = recording_frequency__Hz
        self._longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self._contraction_time_range__unitless = contraction_time_range_factor

        # Derived properties
        self._number_of_neurons = len(self._recruitment_thresholds)
        self._recruitment_ratio = (
            self._recruitment_thresholds[-1] / self._recruitment_thresholds[0]
        )  # referred in [1] as RP

        self._longest_duration_rise_time__samples = float(
            (
                self._longest_duration_rise_time__ms.rescale("s") * self._recording_frequency__Hz
            ).magnitude
        )  # referred in [1] as T_L (see eq. 14)

        # Simulation results - stored privately, accessed via properties
        self._peak_twitch_forces__unitless: Optional[np.ndarray] = None
        self._contraction_times__samples: Optional[np.ndarray] = None
        self._twitch_mat: Optional[np.ndarray] = None
        self._twitch_list: Optional[list[np.ndarray]] = None

        # Initialize model parameters
        self._compute_twitch_parameters()

    def _compute_twitch_parameters(self) -> None:
        """
        Compute peak twitch forces and contraction times based on Fuglevand model.

        This method calculates the physiological parameters for each motor unit
        according to the Fuglevand et al. (1993) model equations.

        Results are stored in the `peak_twitch_forces` and `contraction_times`
        properties after execution.
        """
        self._peak_twitch_forces__unitless = np.exp(
            (np.log(self._recruitment_ratio) / self._number_of_neurons)
            * np.arange(1, self._number_of_neurons + 1)
        )  # referred in [1] as P(i) (see eq. 13)

        self._contraction_times__samples = self._longest_duration_rise_time__samples * np.power(
            1 / self._peak_twitch_forces__unitless,
            1 / np.emath.logn(self._contraction_time_range__unitless, self._recruitment_ratio),
        )  # referred in [1] as T(i) (see eq. 14)

        self._initialize_twitches()

    def _initialize_twitches(self) -> None:
        """
        Initialize the twitches matrix and the twitch list.

        This method computes the individual motor unit twitch responses
        based on the calculated peak forces and contraction times.

        Results are stored in the `twitch_mat` and `twitch_list`
        properties after execution.

        Raises
        ------
        ValueError
            If twitch parameters have not been computed first.
        """
        if self._peak_twitch_forces__unitless is None or self._contraction_times__samples is None:
            raise ValueError(
                "Twitch parameters not computed. Call _compute_twitch_parameters() first."
            )

        # 5 is a rule of thumb number so we capture the entire twitch.
        max_twitch_length = int(np.ceil(5 * np.max(self._contraction_times__samples)))

        twitch_timelines_reshaped = np.arange(max_twitch_length)[:, np.newaxis]

        self._twitch_mat = (
            self._peak_twitch_forces__unitless
            / self._contraction_times__samples
            * twitch_timelines_reshaped
            * np.exp(1 - twitch_timelines_reshaped / self._contraction_times__samples)
        )  # referred in [1] as f_i(t) (see eq. 11 and 12)

        # Truncate the twitch to the effective length
        self._twitch_list = [
            self._twitch_mat[:L, i]
            for i, L in enumerate(
                np.minimum(
                    max_twitch_length,
                    np.ceil(5 * self._contraction_times__samples).astype(int),
                )
            )
        ]

    def generate_force(self, spike_train__Block: SPIKE_TRAIN__Block) -> FORCE__AnalogSignal:
        """
        Generate force output from motor unit spike trains using the Fuglevand model.

        This method simulates muscle force by converting spike trains into force output
        through individual motor unit twitches with nonlinear gain modulation based on
        discharge rate. Each motor unit contributes to the total force according to its
        twitch properties and firing pattern. The output is resampled to match the
        recording_frequency__Hz parameter.

        Parameters
        ----------
        spike_train__Block : SPIKE_TRAIN__Block
            Spike train block containing spike train data for multiple motor neuron pools.

        Returns
        -------
        FORCE__AnalogSignal
            Force output neo.AnalogSignal representing muscle force over time.
            Each channel corresponds to one motor neuron pool's force response.
            Sampling rate matches the recording_frequency__Hz parameter.

        Raises
        ------
        ValueError
            If spike train matrix dimensions don't match the number of motor units.
            If twitch parameters have not been computed.

        Notes
        -----
        The force generation follows these steps:
        1. Convert spike trains to inter-pulse intervals (IPIs)
        2. Calculate nonlinear gain based on discharge rates
        3. Sum weighted twitch responses for each spike
        4. Apply gain modulation to final force output
        5. Resample output to match recording_frequency__Hz
        """
        if self._twitch_list is None:
            raise ValueError(
                "Twitch parameters not available. "
                "This should not occur if the model was properly initialized. "
                "Please reinitialize the ForceModel."
            )

        if not HAS_ELEPHANT:
            raise ImportError(
                "Elephant is required for force simulation. "
                "Install with: pip install myogen[elephant]"
            )

        # Extract timing information from spike trains
        spiketrain_timestep__ms = float(
            spike_train__Block.segments[0].spiketrains[0].sampling_period.rescale("ms").magnitude
        )

        forces = []
        for i, segment in enumerate(spike_train__Block.segments):
            if len(segment.spiketrains) != self._number_of_neurons:
                raise ValueError(
                    f"MU pool {i} has {len(segment.spiketrains)} neurons, "
                    f"but force model was initialized with {self._number_of_neurons} motor units. "
                    "The number of neurons in the spike train neo.Block must match the number of recruitment thresholds."
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
                    .to_sparse_bool_array()
                    .T
                )

                # Generate force with resampling handled internally
                force_output = self._generate_force(
                    spike_array, spiketrain_timestep__ms, prefix=f"Pool {i + 1}"
                )
                forces.append(force_output)

            finally:
                elephant_utils_logger.setLevel(original_level)

        return AnalogSignal(
            np.stack(forces, axis=-1) * pq.dimensionless,
            t_start=spike_train__Block.segments[0].t_start.rescale("s"),
            sampling_rate=self._recording_frequency__Hz,
        )

    def _generate_force(
        self, spikes, spiketrain_timestep__ms: float, prefix: str = ""
    ) -> np.ndarray:
        """Generate force offline from spike trains with resampling to recording frequency."""
        # Convert sparse to dense once at the start
        if sp.issparse(spikes):
            spikes_dense = spikes.toarray()
        else:
            spikes_dense = spikes

        L = spikes_dense.shape[0]

        # Calculate target length for resampling to recording frequency
        spiketrain_timestep__s = spiketrain_timestep__ms / 1000.0
        force_timestep__s = float((1.0 / self._recording_frequency__Hz).rescale("s").magnitude)

        # IPI signal generation out of spikes signal (for gain nonlinearity)
        _, ipi = sawtooth2ipi(
            spikes2sawtooth(np.vstack([spikes_dense[1:], np.zeros((1, self._number_of_neurons))])),
            spikes_dense,
        )

        gain = get_gain_vectorized(ipi, self._contraction_times__samples)

        # Optimize twitch resampling - pre-compute interpolation grids
        resampled_twitches = []
        for force_twitch in self._twitch_list:
            # Pre-compute grid arrays - ensure exact length match
            twitch_length = force_twitch.shape[0]
            xp_orig = np.arange(twitch_length) * force_timestep__s
            twitch_duration_s = (twitch_length - 1) * force_timestep__s
            x_new = np.arange(0, twitch_duration_s + spiketrain_timestep__s, spiketrain_timestep__s)

            resampled_twitches.append(np.interp(x_new, xp_orig, force_twitch))

        # Generate force at spike train sampling rate - optimized to only iterate over spikes
        force = np.zeros(L)

        # Try to use Numba if available for additional speedup
        try:
            force = self._generate_force_numba(
                force, spikes_dense, gain, resampled_twitches, L, prefix
            )
        except (ImportError, AttributeError):
            # Fallback to optimized NumPy version
            for n in tqdm(
                range(self._number_of_neurons),
                desc=f"{prefix} Twitch trains are generated",
                unit="MU",
                disable=False,
            ):
                # Only iterate over spike times, not all time points
                spike_indices = np.where(spikes_dense[:, n])[0]
                twitch = resampled_twitches[n]

                for spike_t in spike_indices:
                    to_take = min(len(twitch), L - spike_t)
                    force[spike_t : spike_t + to_take] += gain[spike_t, n] * twitch[:to_take]

        # Final resampling to target frequency
        output_times = np.arange(0, L * spiketrain_timestep__s, force_timestep__s)
        input_times = np.arange(0, L * spiketrain_timestep__s, spiketrain_timestep__s)

        return np.interp(output_times, input_times, force)

    def _generate_force_numba(
        self,
        force: np.ndarray,
        spikes: np.ndarray,
        gain: np.ndarray,
        resampled_twitches: list,
        L: int,
        prefix: str,
    ) -> np.ndarray:
        """Generate force using Numba JIT compilation for maximum speed."""
        try:
            import numba
        except ImportError:
            raise ImportError("Numba not available, falling back to NumPy")

        @numba.jit(nopython=True, parallel=False)
        def add_twitches_jit(force, spikes, gain, n, twitch, L):
            """Inner loop compiled with Numba for speed."""
            for t in range(L):
                if spikes[t, n]:
                    to_take = min(len(twitch), L - t)
                    for i in range(to_take):
                        force[t + i] += gain[t, n] * twitch[i]
            return force

        # Process each neuron
        for n in tqdm(
            range(self._number_of_neurons),
            desc=f"{prefix} Twitch trains (Numba)",
            unit="MU",
        ):
            force = add_twitches_jit(force, spikes, gain, n, resampled_twitches[n], L)

        return force

    # Property accessors for computed results
    @property
    def peak_twitch_forces__unitless(self) -> np.ndarray:
        """
        Peak twitch forces for each motor unit (unitless).

        Returns
        -------
        np.ndarray
            Array of peak twitch forces according to Fuglevand model equation 13.

        Raises
        ------
        ValueError
            If twitch parameters have not been computed yet.
        """
        if self._peak_twitch_forces__unitless is None:
            raise ValueError(
                "Peak twitch forces not computed. "
                "Twitch parameters are computed automatically during initialization."
            )
        return self._peak_twitch_forces__unitless

    @property
    def contraction_times__samples(self) -> np.ndarray:
        """
        Contraction times for each motor unit in samples.

        Returns
        -------
        np.ndarray
            Array of contraction times according to Fuglevand model equation 14.

        Raises
        ------
        ValueError
            If twitch parameters have not been computed yet.
        """
        if self._contraction_times__samples is None:
            raise ValueError(
                "Contraction times not computed. "
                "Twitch parameters are computed automatically during initialization."
            )
        return self._contraction_times__samples

    @property
    def twitch_mat(self) -> np.ndarray:
        """
        Complete twitch matrix for all motor units.

        Returns
        -------
        np.ndarray
            Matrix of shape (max_twitch_length, n_motor_units) containing
            the twitch responses for each motor unit.

        Raises
        ------
        ValueError
            If twitches have not been initialized yet.
        """
        if self._twitch_mat is None:
            raise ValueError(
                "Twitch matrix not computed. "
                "Twitches are initialized automatically during model setup."
            )
        return self._twitch_mat

    @property
    def twitch_list(self) -> list[np.ndarray]:
        """
        List of individual twitch responses for each motor unit.

        Returns
        -------
        list[np.ndarray]
            List where each element is the twitch response array for one motor unit.
            Each array may have different lengths based on the motor unit's contraction time.

        Raises
        ------
        ValueError
            If twitches have not been initialized yet.
        """
        if self._twitch_list is None:
            raise ValueError(
                "Twitch list not computed. "
                "Twitches are initialized automatically during model setup."
            )
        return self._twitch_list

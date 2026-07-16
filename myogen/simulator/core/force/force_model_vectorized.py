from typing import Optional

import numpy as np
import quantities as pq
import scipy.sparse as sp
from neo import AnalogSignal

from myogen.utils.binning import bin_spike_trains
from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    RECRUITMENT_THRESHOLDS__ARRAY,
    FORCE__AnalogSignal,
    Quantity__Hz,
    Quantity__ms,
    SPIKE_TRAIN__Block,
)

# Share helpers with ``ForceModel`` so the two implementations cannot drift.
from .force_utils import get_gain_vectorized, sawtooth2ipi, spikes2sawtooth
from .force_utils_vectorized import generate_force_vectorized


@beartowertype
class ForceModelVectorized:
    """
    Vectorized force model based on Fuglevand et al. (1993) [1].

    This is an optimized version of `ForceModel` that uses numpy
    vectorization for significantly better performance, especially for long
    simulations. It shares the IPI/gain/twitch pipeline with the reference
    implementation via ``force_utils`` so the output is guaranteed to match
    the reference model bit-for-bit (modulo per-spike accumulation order).

    Parameters
    ----------
    recruitment_thresholds : RECRUITMENT_THRESHOLDS__ARRAY
        Recruitment thresholds for each motor unit.
    recording_frequency__Hz : Quantity__Hz
        Recording frequency in Hz. Determines temporal resolution of force
        calculations. Typical values: 100-1000 Hz.
    longest_duration_rise_time__ms : Quantity__ms, default=90.0 * pq.ms
        Longest duration of the rise time in milliseconds.
    contraction_time_range_factor : float, default=3.0
        Contraction time range factor. Determines the spread of contraction
        times across motor units. Generally between 2 and 5.

    References
    ----------
    [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993). Models of recruitment and rate coding in motor-unit pools. Journal of Neurophysiology, 70(2), 782-797.
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
                f"contraction_time_range_factor must be greater than 1.0, got {contraction_time_range_factor}. "
                "This parameter determines the spread of contraction times. Typical values are 2.0-5.0."
            )

        # Immutable public access
        self.recruitment_thresholds = recruitment_thresholds
        self.recording_frequency__Hz = recording_frequency__Hz
        self.longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self.contraction_time_range_factor = contraction_time_range_factor

        # Private copies for internal modifications
        self._recruitment_thresholds = recruitment_thresholds.copy()
        self._recording_frequency__Hz = recording_frequency__Hz
        self._longest_duration_rise_time__ms = longest_duration_rise_time__ms
        self._contraction_time_range_factor = contraction_time_range_factor

        # Derived properties
        self._number_of_neurons = len(self._recruitment_thresholds)
        self._recruitment_ratio = (
            self._recruitment_thresholds[-1] / self._recruitment_thresholds[0]
        )

        # Match ForceModel's quantity-aware sample conversion exactly.
        self._longest_duration_rise_time__samples = float(
            (
                self._longest_duration_rise_time__ms.rescale("s")
                * self._recording_frequency__Hz
            ).magnitude
        )

        # Simulation results
        self._peak_twitch_forces__unitless: Optional[np.ndarray] = None
        self._contraction_times__samples: Optional[np.ndarray] = None
        self._twitch_mat: Optional[np.ndarray] = None
        self._twitch_list: Optional[list[np.ndarray]] = None

        # Initialize model parameters
        self._compute_twitch_parameters()

    def _compute_twitch_parameters(self) -> None:
        """Compute peak twitch forces and contraction times (Fuglevand)."""
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
                    self._contraction_time_range_factor, self._recruitment_ratio
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
        self, spike_train__Block: SPIKE_TRAIN__Block, verbose: bool = True
    ) -> FORCE__AnalogSignal:
        """
        Generate force output from motor unit spike trains.

        The body mirrors `ForceModel.generate_force` so that the two
        implementations cannot drift apart silently. Only the per-spike
        accumulation differs (vectorized vs. per-spike loop).
        """
        if self._twitch_list is None:
            raise ValueError(
                "Twitch parameters not available. "
                "This should not occur if the model was properly initialized."
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

            spike_array = bin_spike_trains(
                segment.spiketrains,
                bin_size=segment.spiketrains[0].sampling_period,
                t_start=segment.t_start,
                t_stop=segment.t_stop,
                sparse=True,
            ).T

            # Generate force with vectorized implementation
            force_output = self._generate_force_vectorized(
                spike_array,
                spiketrain_timestep__ms,
                prefix=f"Pool {i + 1}",
                verbose=verbose,
            )
            forces.append(force_output)

        return AnalogSignal(
            np.stack(forces, axis=-1) * pq.dimensionless,
            t_start=spike_train__Block.segments[0].t_start.rescale("s"),
            sampling_rate=self._recording_frequency__Hz,
        )

    def _generate_force_vectorized(
        self,
        spikes,
        spiketrain_timestep__ms: float,
        prefix: str = "",
        verbose: bool = True,
    ) -> np.ndarray:
        """Generate force using vectorized per-time-step accumulation.

        Mirrors `ForceModel._generate_force` byte-for-byte except for
        the inner per-spike accumulation, which is delegated to
        `generate_force_vectorized`.
        """
        # Convert sparse to dense once at the start (mirrors ForceModel)
        if sp.issparse(spikes):
            spikes_dense = spikes.toarray()
        else:
            spikes_dense = spikes

        L = spikes_dense.shape[0]

        # Calculate timing parameters (mirrors ForceModel)
        spiketrain_timestep__s = spiketrain_timestep__ms / 1000.0
        force_timestep__s = float(
            (1.0 / self._recording_frequency__Hz).rescale("s").magnitude
        )

        # IPI signal generation out of spikes signal (for gain nonlinearity)
        _, ipi = sawtooth2ipi(
            spikes2sawtooth(
                np.vstack([spikes_dense[1:], np.zeros((1, self._number_of_neurons))])
            ),
            spikes_dense,
        )

        gain = get_gain_vectorized(ipi, self._contraction_times__samples)

        # Optimize twitch resampling - pre-compute interpolation grids
        resampled_twitches = []
        for force_twitch in self._twitch_list:
            twitch_length = force_twitch.shape[0]
            xp_orig = np.arange(twitch_length) * force_timestep__s
            twitch_duration_s = (twitch_length - 1) * force_timestep__s
            x_new = np.arange(
                0, twitch_duration_s + spiketrain_timestep__s, spiketrain_timestep__s
            )

            resampled_twitches.append(np.interp(x_new, xp_orig, force_twitch))

        # Use vectorized force generation
        if verbose:
            print(f"{prefix} Generating force with vectorized implementation...")
        force = generate_force_vectorized(spikes_dense, gain, resampled_twitches)

        # Final resampling to target frequency (mirrors ForceModel)
        output_times = np.arange(0, L * spiketrain_timestep__s, force_timestep__s)
        input_times = np.arange(0, L * spiketrain_timestep__s, spiketrain_timestep__s)

        return np.interp(output_times, input_times, force)

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

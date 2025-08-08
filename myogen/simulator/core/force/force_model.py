from typing import Optional
from beartype import beartype
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from myogen.utils.types import SPIKE_TRAIN__MATRIX, beartowertype


@beartowertype
class ForceModel:
    """
    Force model based on Fuglevand et al. (1993) [1]_.

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        Recruitment thresholds for each motor unit.
    recording_frequency__Hz : float
        Recording frequency in Hz.
    longest_duration_rise_time__ms : float, optional
        Longest duration of the rise time in milliseconds.
    twitch_force_range : float, optional
        Twitch force range.
    contraction_time_range : float, optional
        Contraction time range. Generally between 2 and 5.

    References
    ----------
    .. [1] Fuglevand, A. J., Winter, D. A., & Patla, A. E. (1993).
        Models of recruitment and rate coding in motor-unit pools.
        Journal of Neurophysiology, 70(2), 782-797.
    """

    def __init__(
        self,
        recruitment_thresholds: np.ndarray,
        recording_frequency__Hz: float,
        longest_duration_rise_time__ms: float = 90.0,
        contraction_time_range: float = 3.0,
    ):
        self.recruitment_thresholds = recruitment_thresholds
        self._number_of_neurons = len(recruitment_thresholds)
        self._recruitment_ratio = (
            self.recruitment_thresholds[-1] / self.recruitment_thresholds[0]
        )  # referred in [1] as RP

        self.recording_frequency__Hz = recording_frequency__Hz

        self.longest_duration_rise_time__ms = (
            longest_duration_rise_time__ms  # referred in [1] as T_L (see eq. 14)
        )
        self._longest_duration_rise_time__samples = (
            self.longest_duration_rise_time__ms / 1000 * self.recording_frequency__Hz
        )  # referred in [1] as T_L (see eq. 14)

        self.contraction_time_range = (
            contraction_time_range  # referred in [1] as RT (see eq. 14)
        )

        self.peak_twitch_forces = np.exp(
            (np.log(self._recruitment_ratio) / self._number_of_neurons)
            * np.arange(1, self._number_of_neurons + 1)
        )  # referred in [1] as P(i) (see eq. 13)

        self.contraction_times = self._longest_duration_rise_time__samples * np.power(
            1 / self.peak_twitch_forces,
            1 / np.emath.logn(self.contraction_time_range, self._recruitment_ratio),
        )  # referred in [1] as T(i) (see eq. 14)#

        self._initialize_twitches()

    def _initialize_twitches(self):
        """
        Initialize the twitches matrix and the twitch list.
        """

        # 5 is a rule of thumb number so we capute the entire twitch.
        max_twitch_length = int(np.ceil(5 * np.max(self.contraction_times)))

        twitch_timelines_reshaped = np.arange(max_twitch_length)[:, np.newaxis]

        self.twitch_mat = (
            self.peak_twitch_forces
            / self.contraction_times
            * twitch_timelines_reshaped
            * np.exp(1 - twitch_timelines_reshaped / self.contraction_times)
        )  # referred in [1] as f_i(t) (see eq. 11 and 12)

        # Truncate the twitch to the effective length
        self.twitch_list = [
            self.twitch_mat[:L, i]
            for i, L in enumerate(
                np.minimum(
                    max_twitch_length, np.ceil(5 * self.contraction_times).astype(int)
                )
            )
        ]

    def generate_force(self, spike_train__matrix: SPIKE_TRAIN__MATRIX) -> np.ndarray:
        """
        Generate force output from motor unit spike trains using the Fuglevand model.

        This method simulates muscle force by converting spike trains into force output
        through individual motor unit twitches with nonlinear gain modulation based on
        discharge rate.

        Parameters
        ----------
        spike_train__matrix : SPIKE_TRAIN__MATRIX
            Spike train matrix with shape (n_pools, n_neurons, n_time_points).
            Binary array where 1 indicates a spike occurrence.

        Returns
        -------
        np.ndarray
            Force output array with shape (n_pools, n_time_points).
            Force values in arbitrary units representing muscle force over time.
        """
        return np.array(
            [self._generate_force(spike_train.T) for spike_train in spike_train__matrix]
        )

    def _generate_force(self, spikes: np.ndarray, prefix: str = "") -> np.ndarray:
        """Generate force offline from spike trains."""
        L = spikes.shape[0]

        # IPI signal generation out of spikes signal (for gain nonlinearity)
        _, ipi = self.sawtooth2ipi(
            self.spikes2sawtooth(
                np.vstack([spikes[1:], np.zeros((1, self._number_of_neurons))])
            )
        )

        gain = np.full_like(spikes, np.nan)
        for n in range(self._number_of_neurons):
            gain[:, n] = self.get_gain(ipi[:, n], self.contraction_times[n])

        # Generate force
        force = np.zeros(L)
        for n in tqdm(
            range(self._number_of_neurons), desc=f"{prefix} Twitch trains are generated"
        ):
            for t in range(L):
                if spikes[t, n]:
                    twitch_to_add = self.twitch_list[n]
                    to_take = min(len(twitch_to_add), L - t)
                    force[t : t + to_take] += gain[t, n] * twitch_to_add[:to_take]

        return force

    def get_gain(self, ipi: np.ndarray, T: float) -> np.ndarray:
        """
        Returns the gain value for the force output for a motor unit with current
        inter-pulse-interval ipi and T-parameter of the twitch T. This function
        corresponds to Fuglevand's nonlinear gain model for the force output,
        see Fuglevand - Models of Rate Coding ..., eq. 17.

        Parameters:
        -----------
        ipi: np.ndarray
            Inter-pulse interval vector.
        T: float
            T-parameter of the twitch.

        Returns:
        --------
        np.ndarray
            Gain vector.
        """
        Sf = lambda x: 1 - np.exp(-2 * x**3)

        inst_dr = T / ipi  # Instantaneous discharge rate
        gain = np.ones_like(inst_dr)  # Gain

        mask = inst_dr > 0.4
        gain[mask] = (Sf(inst_dr[mask]) / inst_dr[mask]) / (Sf(0.4) / 0.4)

        return gain

    def spikes2sawtooth(
        self, spikes: np.ndarray, initial_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert spikes to sawtooth signal.

        The sawtooth signal represents inter-pulse intervals (IPIs) where each sawtooth
        tooth height equals the number of time steps between consecutive spikes. This is
        used to calculate instantaneous discharge rates for force generation.

        **Sawtooth Sizing:**
            - Each time step without a spike: sawtooth value increments by 1
            - When a spike occurs: sawtooth resets to 0
            - Sawtooth height = IPI in time samples (at recording_frequency__Hz)
            - Example: height of 50 at 10kHz = 5ms IPI

        Parameters:
        -----------
        spikes: np.ndarray
            Spike train matrix (time x neurons). Binary values where 1 indicates a spike.
        initial_values: np.ndarray, optional
            Initial values for each neuron. Default is ones. Will be reset to 0 if
            there's a spike at t=0.

        Returns:
        --------
        np.ndarray
            Sawtooth sequence where each value represents the time since the last spike
            in samples. Used for calculating inter-pulse intervals and gain modulation.
        """
        if initial_values is None:
            initial_values = np.ones((1, spikes.shape[1]))

        l, w = spikes.shape
        seq = np.zeros((l, w))

        # Set initial values, but reset to 0 if there's a spike at t=0
        initial_values = initial_values * (spikes[0] != 1)
        seq[0] = initial_values

        for j in range(w):
            for i in range(1, l):
                if spikes[i, j] == 1:
                    seq[i, j] = 0
                else:
                    seq[i, j] = seq[i - 1, j] + 1

        return seq

    def sawtooth2spikes(self, sawtooth: np.ndarray) -> np.ndarray:
        """
        Convert sawtooth signal to spikes.

        A spike occurs when the sawtooth resets to 0, indicating the end of an
        inter-pulse interval. This is the inverse operation of spikes2sawtooth.

        **Spike Detection:**
            - Spike at t=0 if sawtooth starts at 0
            - Spike when sawtooth decreases (resets from peak back to 0)
            - Each spike marks the end of one IPI and start of the next

        Parameters:
        -----------
        sawtooth: np.ndarray
            Sawtooth signal matrix where values represent time since last spike.

        Returns:
        --------
        np.ndarray
            Spike train matrix with binary values (1 = spike, 0 = no spike).
        """
        spikes = np.zeros_like(sawtooth, dtype=bool)

        spikes[sawtooth == 0] = 1

        return spikes.astype(int)

    def sawtooth2ipi(
        self, sawtooth: np.ndarray, ipi_saturation: float = np.inf
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert sawtooth signal to inter-pulse intervals.

        This method converts sawtooth heights (representing time since last spike) into
        inter-pulse intervals (IPIs) by combining forward and backward spike information.
        The IPI values are used to calculate instantaneous discharge rates for the
        Fuglevand gain model.

        **IPI Calculation:**
        - Forward sawtooth: time since last spike
        - Backward sawtooth: time until next spike
        - IPI = forward + backward sawtooth values
        - IPI represents the full inter-pulse interval in time samples
        - Convert to physiological units: IPI_ms = IPI_samples / (recording_frequency__Hz / 1000)

        Parameters:
        -----------
        sawtooth: np.ndarray
            Sawtooth signal matrix where values represent time since last spike.
        ipi_saturation: float, optional
            Maximum IPI value (saturation) to prevent infinite values when spikes
            are far apart. Default is infinity.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            ipi_filled: IPI signal with zeros where no full IPI can be calculated.
            ipi_filled_seamless: IPI signal with zeros filled by preceding values
            for smooth gain calculations.
        """
        spikes = self.sawtooth2spikes(sawtooth)

        i_sawtooth = self.spikes2sawtooth(spikes[::-1], sawtooth[0])
        i_sawtooth = i_sawtooth[::-1]
        ipi_filled = sawtooth + i_sawtooth
        ipi_filled = np.minimum(ipi_filled, ipi_saturation)

        ipi_filled_seamless = ipi_filled.copy()
        ipi_filled_seamless[ipi_filled == 0] = ipi_filled_seamless[ipi_filled == 0] - 1
        ipi_filled_seamless = np.minimum(ipi_filled_seamless, ipi_saturation)

        return ipi_filled, ipi_filled_seamless


if __name__ == "__main__":
    from myogen import simulator
    from myogen.utils.plotting.force import (
        plot_twitch_parameter_assignment,
        plot_twitches,
    )
    from myogen.utils.plotting import plot_spike_trains
    from myogen.utils.currents import create_trapezoid_current

    recruitment_thresholds, _ = simulator.generate_mu_recruitment_thresholds(
        N=100, recruitment_range=25
    )

    force_model = ForceModel(
        recruitment_thresholds=recruitment_thresholds, recording_frequency__Hz=10000.0
    )

    _, ax = plt.subplots(figsize=(10, 6))

    plot_twitch_parameter_assignment(
        force_model, ax, [1, 10], flip_x=True, apply_default_formatting=True
    )
    plt.tight_layout()
    plt.show()

    plot_twitches(force_model, plt.gca(), apply_default_formatting=True)
    plt.tight_layout()
    plt.show()

    simulation_duration__ms = 10000.0  # 1 seconds
    timestep__ms = 0.05  # 0.1 ms time step
    t_points = int(simulation_duration__ms / timestep__ms)

    trap_amplitudes = [100.0]  # Peak amplitudes
    trap_rise_times = [1000.0]  # Rise durations (ms)
    trap_plateau_times = [3000.0]  # Plateau durations (ms)
    trap_fall_times = [1000.0]  # Fall durations (ms)
    trap_offsets = [0.0]  # Baseline currents
    trap_delays = [2500.0]  # Initial delays (ms)

    trapezoid_currents = create_trapezoid_current(
        n_pools=1,
        t_points=t_points,
        timestep_ms=timestep__ms,
        amplitudes__muV=trap_amplitudes,
        rise_times__ms=trap_rise_times,
        plateau_times__ms=trap_plateau_times,
        fall_times__ms=trap_fall_times,
        offsets__muV=trap_offsets,
        delays__ms=trap_delays,
    )

    plt.plot(trapezoid_currents[0])
    plt.show()

    motor_neuron_pool = simulator.MotorNeuronPool(recruitment_thresholds)

    spike_trains_matrix, active_neuron_indices, data = (
        motor_neuron_pool.generate_spike_trains(
            input_current__matrix=trapezoid_currents,
            timestep__ms=timestep__ms,
            noise_mean__nA=0.0,
            noise_stdev__nA=0.0,
        )
    )

    plot_spike_trains(spike_trains_matrix, timestep__ms, [plt.gca()])
    plt.show()

    force_hfs = force_model.generate_force(spike_trains_matrix)
    plt.plot(
        force_hfs[0]
        + np.random.randn(len(force_hfs[0])) * 0.0001 * np.mean(force_hfs[0])
    )
    plt.show()

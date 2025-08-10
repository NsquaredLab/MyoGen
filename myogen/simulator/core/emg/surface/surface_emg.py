import warnings
from typing import Optional

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import numpy as np
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.muscle import Muscle
from myogen.simulator.core.spike_train import MotorNeuronPool
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.utils.types import (
    SURFACE_MUAP_SHAPE__TENSOR,
    SURFACE_EMG__TENSOR,
    beartowertype,
)

# Suppress warnings for cleaner output during batch simulations
warnings.filterwarnings("ignore")


@beartowertype
class SurfaceEMG:
    """
    Surface Electromyography (sEMG) Simulation.

    This class provides a simulation framework for generating
    surface electromyography signals from the muscle. It implements the
    multi-layered cylindrical volume conductor model from Farina et al. 2004 [1]_.

    Parameters
    ----------
    muscle_model : Muscle
        Pre-computed muscle model (see :class:`myogen.simulator.Muscle`).
    electrode_arrays : list[SurfaceElectrodeArray]
        List of electrode arrays to use for simulation (see :class:`myogen.simulator.SurfaceElectrodeArray`).
    sampling_frequency__Hz : float, default=2048.0
        Sampling frequency in Hz. Default is set to 2048 Hz as used by the Quattrocento (OT Bioelettronica, Turin, Italy) system.
    sampling_points_in_t_and_z_domains : int, default=256
        Spatial and temporal discretization resolution for numerical integration.
        Controls the accuracy of the volume conductor calculations but significantly
        impacts computational cost (scales quadratically).
        Higher values provide better numerical accuracy at the expense of simulation time.
        Default is set to 256 samples.
    sampling_points_in_theta_domain : int, default=180
        Angular discretization for cylindrical coordinate system in degrees.
        Higher values provide better spatial resolution for circumferential electrode placement studies.
        Default is set to 180 points, which provides 2° angular resolution.
        This is suitable for most EMG studies.
    MUs_to_simulate : list[int], optional
        Indices of motor units to simulate. If None, all motor units are simulated.
        Default is None. For computational efficiency, consider
        simulating subsets for initial analysis.
        Indices correspond to the recruitment order (0 is recruited first).

    Attributes
    ----------
    muaps : list[SURFACE_MUAP_SHAPE__TENSOR]
        Motor Unit Action Potential (MUAP) templates for each electrode array. Available after simulate_muaps().
    surface_emg__tensors : list[SURFACE_EMG__TENSOR]
        Surface EMG signals for each electrode array. Available after simulate_surface_emg().
    noisy_surface_emg__tensors : list[SURFACE_EMG__TENSOR]
        Noisy surface EMG signals for each electrode array. Available after add_noise().
    motor_neuron_pool : MotorNeuronPool
        Motor neuron pool used for EMG generation. Available after simulate_surface_emg().

    References
    ----------
    .. [1] Farina, D., Mesin, L., Martina, S., Merletti, R., 2004. A surface EMG generation model with multilayer cylindrical description of the volume conductor. IEEE Transactions on Biomedical Engineering 51, 415–426. https://doi.org/10.1109/TBME.2003.820998
    """

    def __init__(
        self,
        muscle_model: Muscle,
        electrode_arrays: list[SurfaceElectrodeArray],
        sampling_frequency__Hz: float = 2048.0,
        sampling_points_in_t_and_z_domains: int = 256,
        sampling_points_in_theta_domain: int = 180,
        MUs_to_simulate: list[int] | None = None,
    ):
        # Immutable public arguments - never modify these
        self.muscle_model = muscle_model
        self.electrode_arrays = electrode_arrays
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self.sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self.MUs_to_simulate = MUs_to_simulate

        # Private copies for internal modifications
        self._muscle_model = muscle_model
        self._electrode_arrays = electrode_arrays
        self._sampling_frequency__Hz = sampling_frequency__Hz
        self._sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self._sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self._MUs_to_simulate = MUs_to_simulate

        # Derived properties from muscle model - immutable public access
        self.mean_conduction_velocity__m_s = (
            self._muscle_model.mean_conduction_velocity__m_s
        )
        self.mean_fiber_length__mm = self._muscle_model.mean_fiber_length__mm
        self.var_fiber_length__mm = self._muscle_model.var_fiber_length__mm
        self.radius_bone__mm = self._muscle_model.radius_bone__mm
        self.fat_thickness__mm = self._muscle_model.fat_thickness__mm
        self.skin_thickness__mm = self._muscle_model.skin_thickness__mm
        self.muscle_conductivity_radial__S_m = (
            self._muscle_model.muscle_conductivity_radial__S_m
        )
        self.muscle_conductivity_longitudinal__S_m = (
            self._muscle_model.muscle_conductivity_longitudinal__S_m
        )
        self.fat_conductivity__S_m = self._muscle_model.fat_conductivity__S_m
        self.skin_conductivity__S_m = self._muscle_model.skin_conductivity__S_m

        # Private copies for internal modifications
        self._mean_conduction_velocity__m_s = (
            self._muscle_model.mean_conduction_velocity__m_s
        )
        self._mean_fiber_length__mm = self._muscle_model.mean_fiber_length__mm
        self._var_fiber_length__mm = self._muscle_model.var_fiber_length__mm
        self._radius_bone__mm = self._muscle_model.radius_bone__mm
        self._fat_thickness__mm = self._muscle_model.fat_thickness__mm
        self._skin_thickness__mm = self._muscle_model.skin_thickness__mm
        self._muscle_conductivity_radial__S_m = (
            self._muscle_model.muscle_conductivity_radial__S_m
        )
        self._muscle_conductivity_longitudinal__S_m = (
            self._muscle_model.muscle_conductivity_longitudinal__S_m
        )
        self._fat_conductivity__S_m = self._muscle_model.fat_conductivity__S_m
        self._skin_conductivity__S_m = self._muscle_model.skin_conductivity__S_m

        # Calculate total radius - immutable and private
        self.radius_total = (
            self._muscle_model.radius__mm
            + self._fat_thickness__mm
            + self._skin_thickness__mm
        )
        self._radius_total = self.radius_total

        # Simulation results - stored privately, accessed via properties
        self._muaps: Optional[list[SURFACE_MUAP_SHAPE__TENSOR]] = None
        self._surface_emg__tensors: Optional[list[SURFACE_EMG__TENSOR]] = None
        self._noisy_surface_emg__tensors: Optional[list[SURFACE_EMG__TENSOR]] = None
        self._motor_neuron_pool: Optional[MotorNeuronPool] = None

    def simulate_muaps(self) -> list[SURFACE_MUAP_SHAPE__TENSOR]:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        This method generates Motor Unit Action Potential (MUAP) templates that represent
        the electrical signature of each motor unit as recorded by the surface electrodes.
        The simulation uses the multi-layered cylindrical volume conductor model.

        Returns
        -------
        list[SURFACE_MUAP_SHAPE__TENSOR]
            List of generated MUAP templates for each electrode array.
            Results are stored in the `muaps` property after execution.

        Notes
        -----
        This method must be called before simulate_surface_emg(). The generated MUAP
        templates are used as basis functions for EMG signal synthesis.
        """
        # Set default MUs to simulate
        if self._MUs_to_simulate is None:
            self._MUs_to_simulate = list(
                range(len(self._muscle_model.resulting_number_of_innervated_fibers))
            )

        # Calculate innervation zone variance
        innervation_zone_variance = (
            self._mean_fiber_length__mm * 0.1
        )  # 10% of the mean fiber length (see Botelho et al. 2019 [6]_)

        # Extract fiber counts
        number_of_fibers_per_MUs = (
            self._muscle_model.resulting_number_of_innervated_fibers
        )

        # Create time array
        t = np.linspace(
            0,
            (self._sampling_points_in_t_and_z_domains - 1)
            / self._sampling_frequency__Hz
            * 1e-3,
            self._sampling_points_in_t_and_z_domains,
        )

        # Pre-calculate innervation zones
        innervation_zones = RANDOM_GENERATOR.uniform(
            low=-innervation_zone_variance / 2,
            high=innervation_zone_variance / 2,
            size=len(self._MUs_to_simulate),
        )

        # Initialize output arrays for each electrode array
        # Each electrode array can have different dimensions
        electrode_array_results = []

        for array_idx, electrode_array in enumerate(self._electrode_arrays):
            # Initialize array for this electrode configuration
            array_result = np.zeros(
                (
                    len(self._MUs_to_simulate),
                    electrode_array.num_rows,
                    electrode_array.num_cols,
                    len(t),
                )
            )

            # Matrix optimization variables
            A_matrix = None
            B_incomplete = None

            # Process each motor unit
            for MU_number, MU_index in enumerate(self._MUs_to_simulate):
                number_of_fibers = number_of_fibers_per_MUs[MU_index]

                if number_of_fibers == 0:
                    continue

                # Get fiber positions
                position_of_fibers = self._muscle_model.resulting_fiber_assignment(
                    MU_index
                )
                innervation_zone = innervation_zones[MU_number]

                # Process each fiber
                for fiber_number in tqdm(
                    range(number_of_fibers),
                    desc=f"Electrode Array {array_idx + 1}/{len(self._electrode_arrays)} MU {MU_number + 1}/{len(self._MUs_to_simulate)}",
                    unit="fiber(s)",
                ):
                    fiber_position = position_of_fibers[fiber_number]

                    # Calculate fiber distance from center
                    R = np.sqrt(fiber_position[0] ** 2 + fiber_position[1] ** 2)

                    # Generate fiber length
                    fiber_length__mm = (
                        self._mean_fiber_length__mm
                        + RANDOM_GENERATOR.uniform(
                            low=-self._var_fiber_length__mm,
                            high=self._var_fiber_length__mm,
                        )
                    )

                    # Calculate fiber end positions
                    L2 = abs(innervation_zone - fiber_length__mm / 2)
                    L1 = abs(innervation_zone + fiber_length__mm / 2)

                    # Use the new simulate_fiber_v2 function
                    if fiber_number == 0 or A_matrix is None:
                        phi_temp, A_matrix, B_incomplete = simulate_fiber_v2(
                            Fs=self._sampling_frequency__Hz * 1e-3,
                            v=self._mean_conduction_velocity__m_s,
                            N=self._sampling_points_in_t_and_z_domains,
                            M=self._sampling_points_in_theta_domain,
                            r=self._radius_total,
                            r_bone=self._radius_bone__mm,
                            th_fat=self._fat_thickness__mm,
                            th_skin=self._skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            electrode_array=electrode_array,
                            sig_muscle_rho=self._muscle_conductivity_radial__S_m,
                            sig_muscle_z=self._muscle_conductivity_longitudinal__S_m,
                            sig_skin=self._skin_conductivity__S_m,
                            sig_fat=self._fat_conductivity__S_m,
                        )
                    else:
                        phi_temp, _, _ = simulate_fiber_v2(
                            Fs=self._sampling_frequency__Hz * 1e-3,
                            v=self._mean_conduction_velocity__m_s,
                            N=self._sampling_points_in_t_and_z_domains,
                            M=self._sampling_points_in_theta_domain,
                            r=self._radius_total,
                            r_bone=self._radius_bone__mm,
                            th_fat=self._fat_thickness__mm,
                            th_skin=self._skin_thickness__mm,
                            R=R,
                            L1=L1,
                            L2=L2,
                            zi=innervation_zone,
                            electrode_array=electrode_array,
                            sig_muscle_rho=self._muscle_conductivity_radial__S_m,
                            sig_muscle_z=self._muscle_conductivity_longitudinal__S_m,
                            sig_skin=self._skin_conductivity__S_m,
                            sig_fat=self._fat_conductivity__S_m,
                            A_matrix=A_matrix,
                            B_incomplete=B_incomplete,
                        )

                    array_result[MU_number] += phi_temp

            electrode_array_results.append(array_result)

        # Store results privately
        self._muaps = electrode_array_results

        return electrode_array_results

    def simulate_surface_emg(
        self, motor_neuron_pool: MotorNeuronPool
    ) -> list[SURFACE_EMG__TENSOR]:
        """
        Generate surface EMG signals for all electrode arrays using the provided motor neuron pool.

        This method convolves the pre-computed MUAP templates with the motor neuron spike trains
        to synthesize realistic surface EMG signals. The process includes temporal resampling
        to match the motor neuron pool timestep and supports both CPU and GPU acceleration.

        Parameters
        ----------
        motor_neuron_pool : MotorNeuronPool
            Motor neuron pool with spike trains computed (see :class:`myogen.simulator.MotorNeuronPool`).

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Surface EMG signals for each electrode array.
            Results are stored in the `surface_emg__tensors` property after execution.

        Raises
        ------
        ValueError
            If MUAP templates have not been generated. Call simulate_muaps() first.
        """
        if self._muaps is None:
            raise ValueError(
                "MUAP templates have not been generated. Call simulate_muaps() first."
            )

        # Store motor neuron pool privately
        self._motor_neuron_pool = motor_neuron_pool

        # Handle MUs to simulate
        if self._MUs_to_simulate is None:
            MUs_to_simulate = set(
                range(len(self._muscle_model.resulting_number_of_innervated_fibers))
            )
        else:
            MUs_to_simulate = set(self._MUs_to_simulate)

        emg_results = []

        for array_idx, muap_array in enumerate(self._muaps):
            # Temporal resampling
            muap_shapes = np.zeros(
                (
                    muap_array.shape[0],
                    muap_array.shape[1],
                    muap_array.shape[2],
                    int(
                        muap_array.shape[3]
                        / self._sampling_frequency__Hz
                        * 1
                        / self._motor_neuron_pool.timestep__ms
                        * 1000
                    ),
                )
            )

            for muap_nr in range(muap_shapes.shape[0]):
                for row in range(muap_shapes.shape[1]):
                    for col in range(muap_shapes.shape[2]):
                        muap_shapes[muap_nr, row, col] = np.interp(
                            np.arange(
                                0,
                                muap_array.shape[-1] / self._sampling_frequency__Hz,
                                self._motor_neuron_pool.timestep__ms / 1000,
                            ),
                            np.arange(
                                0,
                                muap_array.shape[-1] / self._sampling_frequency__Hz,
                                1 / self._sampling_frequency__Hz,
                            ),
                            muap_array[muap_nr, row, col],
                        )

            n_pools = self._motor_neuron_pool.spike_trains.shape[0]
            n_rows = muap_shapes.shape[1]
            n_cols = muap_shapes.shape[2]

            # Initialize result array
            sample_conv = np.convolve(
                self._motor_neuron_pool.spike_trains[0, 0],
                muap_shapes[0, 0, 0],
                mode="same",
            )

            surface_emg = np.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

            muap_shapes /= np.max(np.abs(muap_shapes))  # Normalize MUAP shapes

            # Perform convolution for each pool using GPU acceleration if available
            if HAS_CUPY:
                # Use GPU acceleration with CuPy
                spike_gpu = cp.asarray(self._motor_neuron_pool.spike_trains)
                muap_gpu = cp.asarray(muap_shapes)
                surface_emg_gpu = cp.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

                for pool_idx in tqdm(
                    range(n_pools),
                    desc=f"Electrode Array {array_idx + 1}/{len(self._muaps)} Surface EMG (GPU)",
                    unit="pools",
                ):
                    active_neuron_indices = set(
                        self._motor_neuron_pool.active_neuron_indices[pool_idx]
                    )

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs on GPU
                            convolutions = cp.array(
                                [
                                    cp.correlate(
                                        spike_gpu[pool_idx, mu_idx],
                                        muap_gpu[i, row_idx, col_idx],
                                        mode="same",
                                    )
                                    for i, mu_idx in enumerate(
                                        MUs_to_simulate.intersection(
                                            active_neuron_indices
                                        )
                                    )
                                ]
                            )
                            # Sum across MUAPs on GPU
                            if len(convolutions) > 0:
                                surface_emg_gpu[pool_idx, row_idx, col_idx] = cp.sum(
                                    convolutions, axis=0
                                )

                # Transfer results back to CPU
                surface_emg = cp.asnumpy(surface_emg_gpu)
            else:
                # Fallback to CPU computation with NumPy
                for pool_idx in tqdm(
                    range(n_pools),
                    desc=f"Electrode Array {array_idx + 1}/{len(self._muaps)} Surface EMG (CPU)",
                    unit="pools",
                ):
                    active_neuron_indices = set(
                        self._motor_neuron_pool.active_neuron_indices[pool_idx]
                    )

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs
                            convolutions = []
                            for i, mu_idx in enumerate(
                                MUs_to_simulate.intersection(active_neuron_indices)
                            ):
                                conv = np.correlate(
                                    self._motor_neuron_pool.spike_trains[
                                        pool_idx, mu_idx
                                    ],
                                    muap_shapes[i, row_idx, col_idx],
                                    mode="same",
                                )
                                convolutions.append(conv)

                            if convolutions:
                                surface_emg[pool_idx, row_idx, col_idx] = np.sum(
                                    convolutions, axis=0
                                )

            surface_emg_resampled = np.zeros(
                (
                    n_pools,
                    n_rows,
                    n_cols,
                    int(
                        surface_emg.shape[-1]
                        / (1 / self._motor_neuron_pool.timestep__ms * 1000)
                        * self._sampling_frequency__Hz
                    ),
                )
            )
            for pool_idx in range(n_pools):
                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        surface_emg_resampled[pool_idx, row_idx, col_idx] = np.interp(
                            np.arange(
                                0,
                                surface_emg.shape[-1]
                                * (self._motor_neuron_pool.timestep__ms / 1000),
                                1 / self._sampling_frequency__Hz,
                            ),
                            np.arange(
                                0,
                                surface_emg.shape[-1]
                                * (self._motor_neuron_pool.timestep__ms / 1000),
                                self._motor_neuron_pool.timestep__ms / 1000,
                            ),
                            surface_emg[pool_idx, row_idx, col_idx],
                        )

            emg_results.append(surface_emg_resampled)

        # Store results privately
        self._surface_emg__tensors = emg_results
        return emg_results

    def add_noise(
        self, snr_db: float, noise_type: str = "gaussian"
    ) -> list[SURFACE_EMG__TENSOR]:
        """
        Add noise to all electrode arrays.

        This method adds realistic noise to the simulated surface EMG signals
        based on a specified signal-to-noise ratio. The noise characteristics
        can be customized to match different recording conditions.

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB. Higher values result in cleaner signals.
            Typical physiological EMG has SNR ranging from 10-40 dB.
        noise_type : str, default="gaussian"
            Type of noise to add. Currently supports "gaussian" for white noise.

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Noisy EMG signals for each electrode array.
            Results are stored in the `noisy_surface_emg__tensors` property after execution.

        Raises
        ------
        ValueError
            If surface EMG has not been simulated. Call simulate_surface_emg() first.
        """
        if self._surface_emg__tensors is None:
            raise ValueError(
                "Surface EMG has not been simulated. Call simulate_surface_emg() first."
            )

        noisy_emg_results = []

        for _, emg_array in enumerate(self._surface_emg__tensors):
            # Calculate signal power
            signal_power = np.mean(emg_array**2)

            # Calculate noise power
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear

            # Generate noise
            if noise_type.lower() == "gaussian":
                noise_std = np.sqrt(noise_power)
                noise = RANDOM_GENERATOR.normal(
                    loc=0.0, scale=noise_std, size=emg_array.shape
                )
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            # Add noise
            noisy_emg = emg_array + noise
            noisy_emg_results.append(noisy_emg)

        # Store results privately
        self._noisy_surface_emg__tensors = noisy_emg_results
        return noisy_emg_results

    # Property accessors for computed results
    @property
    def muaps(self) -> list[SURFACE_MUAP_SHAPE__TENSOR]:
        """
        Motor Unit Action Potential (MUAP) templates for each electrode array.

        Returns
        -------
        list[SURFACE_MUAP_SHAPE__TENSOR]
            List of MUAP templates for each electrode array.

        Raises
        ------
        ValueError
            If MUAP templates have not been computed yet.
        """
        if self._muaps is None:
            raise ValueError(
                "MUAP templates not computed. Call simulate_muaps() first."
            )
        return self._muaps

    @property
    def surface_emg__tensors(self) -> list[SURFACE_EMG__TENSOR]:
        """
        Surface EMG signals for each electrode array.

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Surface EMG signals for each electrode array.

        Raises
        ------
        ValueError
            If surface EMG has not been computed yet.
        """
        if self._surface_emg__tensors is None:
            raise ValueError(
                "Surface EMG signals not computed. Call simulate_surface_emg() first."
            )
        return self._surface_emg__tensors

    @property
    def noisy_surface_emg__tensors(self) -> list[SURFACE_EMG__TENSOR]:
        """
        Noisy surface EMG signals for each electrode array.

        Returns
        -------
        list[SURFACE_EMG__TENSOR]
            Noisy surface EMG signals for each electrode array.

        Raises
        ------
        ValueError
            If noisy surface EMG has not been computed yet.
        """
        if self._noisy_surface_emg__tensors is None:
            raise ValueError(
                "Noisy surface EMG signals not computed. Call add_noise() first."
            )
        return self._noisy_surface_emg__tensors

    @property
    def motor_neuron_pool(self) -> MotorNeuronPool:
        """
        Motor neuron pool used for EMG generation.

        Returns
        -------
        MotorNeuronPool
            The motor neuron pool used in the simulation.

        Raises
        ------
        ValueError
            If motor neuron pool has not been set yet.
        """
        if self._motor_neuron_pool is None:
            raise ValueError(
                "Motor neuron pool not set. Call simulate_surface_emg() first."
            )
        return self._motor_neuron_pool

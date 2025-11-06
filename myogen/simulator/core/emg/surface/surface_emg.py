from typing import Optional

from myogen.utils.neo import GridAnalogSignal

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import logging

import elephant
import elephant.utils
import numpy as np
import quantities as pq
from neo import Block, Group, Segment
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.muscle import Muscle
from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    SPIKE_TRAIN__Block,
    SURFACE_EMG__Block,
    SURFACE_MUAP__Block,
)


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
    muaps__Block : SURFACE_MUAP__Block
        Motor Unit Action Potential (MUAP) templates for each electrode array as a neo.Block. Available after simulate_muaps().
    surface_emg__Block : SURFACE_EMG__Block
        Surface EMG signals for each electrode array as a neo.Block. Available after simulate_surface_emg().
    noisy_surface_emg__Block : SURFACE_EMG__Block
        Noisy surface EMG signals for each electrode array as a neo.Block. Available after add_noise().
    spike_train__Block : SPIKE_TRAIN__Block
        Spike train block used for EMG generation signals. Available after simulate_surface_emg().

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
        self._muaps__Block: Optional[SURFACE_MUAP__Block] = None
        self._surface_emg__Block: Optional[SURFACE_EMG__Block] = None
        self._noisy_surface_emg__Block: Optional[SURFACE_EMG__Block] = None
        self._spike_train__Block: Optional[SPIKE_TRAIN__Block] = None

    def simulate_muaps(self) -> SURFACE_MUAP__Block:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        This method generates Motor Unit Action Potential (MUAP) templates that represent
        the electrical signature of each motor unit as recorded by the surface electrodes.
        The simulation uses the multi-layered cylindrical volume conductor model.

        Returns
        -------
        SURFACE_MUAP__Block
            neo.Block of generated MUAP templates for each electrode array.
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

        # Get total number of motor units
        n_motor_units = len(number_of_fibers_per_MUs)

        # Pre-calculate innervation zones for all MUs
        innervation_zones = RANDOM_GENERATOR.uniform(
            low=-innervation_zone_variance / 2,
            high=innervation_zone_variance / 2,
            size=n_motor_units,
        )

        block = Block()
        for array_idx, electrode_array in enumerate(self._electrode_arrays):
            group = Group(name=f"ElectrodeArray_{array_idx}")
            block.groups.append(group)

            # Matrix optimization variables
            A_matrix = None
            B_incomplete = None

            # Process all motor units (not just selected ones for consistent normalization)
            for MU_index in range(n_motor_units):
                segment = Segment(name=f"MUAP_{MU_index}")
                group.segments.append(segment)

                array_result = np.zeros(
                    (
                        electrode_array.num_rows,
                        electrode_array.num_cols,
                        len(t),
                    )
                )

                number_of_fibers = number_of_fibers_per_MUs[MU_index]

                if number_of_fibers == 0:
                    # Add empty signal for MUs with no fibers
                    segment.analogsignals.append(
                        GridAnalogSignal(
                            signal=np.zeros((len(t), electrode_array.num_rows, electrode_array.num_cols)) * pq.dimensionless,
                            t_start=0 * pq.ms,
                            sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                        )
                    )
                    continue

                # Get fiber positions
                position_of_fibers = self._muscle_model.resulting_fiber_assignment(
                    MU_index
                )
                innervation_zone = innervation_zones[MU_index]

                # Process each fiber
                for fiber_number in tqdm(
                    range(number_of_fibers),
                    desc=f"Electrode Array {array_idx + 1}/{len(self._electrode_arrays)} MU {MU_index + 1}/{n_motor_units}",
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

                    theta = np.arctan2(fiber_position[1], fiber_position[0])

                    electrode_array._center_point__mm_deg = (
                        electrode_array._center_point__mm_deg[0],
                        electrode_array._center_point__mm_deg[1] - np.rad2deg(theta),
                    )
                    electrode_array._create_electrode_grid()

                    # Calculate fiber end positions
                    L1 = abs(innervation_zone + fiber_length__mm / 2)
                    L2 = abs(innervation_zone - fiber_length__mm / 2)

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

                    array_result += phi_temp

                segment.analogsignals.append(
                    GridAnalogSignal(
                        signal=np.transpose(array_result, (2, 0, 1)) * pq.dimensionless,
                        t_start=0 * pq.ms,
                        sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                    )
                )

        # Store results privately
        self._muaps__Block = block

        return block

    def simulate_surface_emg(
        self, spike_train__Block: SPIKE_TRAIN__Block
    ) -> SURFACE_EMG__Block:
        """
        Generate surface EMG signals for all electrode arrays using the provided spike train block.

        This method convolves the pre-computed MUAP templates with the spike trains
        to synthesize realistic surface EMG signals. The process includes temporal resampling
        to match the spike train timestep and supports both CPU and GPU acceleration.

        Parameters
        ----------
        spike_train__Block : SPIKE_TRAIN__Block
            Block containing spike trains organized as segments (pools) with spiketrains.

        Returns
        -------
        SURFACE_EMG__Block
            Surface EMG signals for each electrode array stored in a neo.Block.
            Results are stored in the `surface_emg__tensors` property after execution.

        Raises
        ------
        ValueError
            If MUAP templates have not been generated. Call simulate_muaps() first.
        """
        if self._muaps__Block is None:
            raise ValueError(
                "MUAP templates have not been generated. Call simulate_muaps() first."
            )

        # Store spike train data privately
        self._spike_train__Block = spike_train__Block

        # Extract timestep from the first spike train
        muap_timestep__ms = float((1 / self._sampling_frequency__Hz) * 1000) * pq.ms

        # Convert spike train block to numpy arrays
        n_pools = len(spike_train__Block.segments)
        n_neurons = len(spike_train__Block.segments[0].spiketrains)

        # Extract spike train durations to determine time length
        first_spiketrain = spike_train__Block.segments[0].spiketrains[0]
        spiketrain_timestep__ms = first_spiketrain.sampling_period.rescale("ms")

        # Convert spike trains to binary arrays using Elephant, suppressing rounding error logging
        elephant_utils_logger = logging.getLogger(elephant.utils.__file__)
        original_level = elephant_utils_logger.level
        elephant_utils_logger.setLevel(logging.ERROR)

        try:
            spike_trains = np.array(
                [
                    elephant.conversion.BinnedSpikeTrain(
                        segment.spiketrains, bin_size=spiketrain_timestep__ms
                    )
                    .to_array()
                    .astype(bool)
                    for segment in spike_train__Block.segments
                ]
            )
        finally:
            elephant_utils_logger.setLevel(original_level)

        # Handle MUs to simulate
        if self._MUs_to_simulate is None:
            MUs_to_simulate = set(range(n_neurons))
        else:
            MUs_to_simulate = set(self._MUs_to_simulate)

        # Create active neuron indices (all neurons are active in each pool for spike train block)
        active_neuron_indices = [list(range(n_neurons)) for _ in range(n_pools)]

        block = Block()

        muap_data_list = [
            np.array([seg.analogsignals[0].magnitude for seg in group.segments])
            for group in self._muaps__Block.groups
        ]

        for array_idx, muap_array in enumerate(muap_data_list):
            emg_group = Group(name=f"ElectrodeArray_{array_idx}")
            block.groups.append(emg_group)

            muap_array = np.transpose(muap_array, (0, 2, 3, 1))

            # Temporal resampling
            new_muap_time_length = max(
                1,
                np.round(
                    muap_array.shape[3]
                    / self._sampling_frequency__Hz
                    * (1 / spiketrain_timestep__ms.rescale("s"))
                ).astype(int),
            )

            muap_shapes = np.zeros(
                (
                    muap_array.shape[0],
                    muap_array.shape[1],
                    muap_array.shape[2],
                    new_muap_time_length,
                )
            )

            for muap_nr in range(muap_shapes.shape[0]):
                for row in range(muap_shapes.shape[1]):
                    for col in range(muap_shapes.shape[2]):
                        muap_shapes[muap_nr, row, col] = np.interp(
                            x=np.arange(
                                start=0,
                                stop=muap_array.shape[-1]
                                / self._sampling_frequency__Hz,
                                step=spiketrain_timestep__ms.rescale(pq.s).magnitude,
                            ),
                            xp=np.arange(
                                start=0,
                                stop=muap_array.shape[-1]
                                / self._sampling_frequency__Hz,
                                step=muap_timestep__ms.rescale(pq.s).magnitude,
                            ),
                            fp=muap_array[muap_nr, row, col],
                        )

            # n_pools already defined above from spike_train_block
            n_rows = muap_shapes.shape[1]
            n_cols = muap_shapes.shape[2]

            # Initialize result array
            sample_conv = np.convolve(
                spike_trains[0, 0], muap_shapes[0, 0, 0], mode="same"
            )

            surface_emg = np.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

            muap_shapes /= np.max(np.abs(muap_shapes))  # Normalize MUAP shapes

            # Perform convolution for each pool using GPU acceleration if available
            if HAS_CUPY:
                # Use GPU acceleration with CuPy
                spike_gpu = cp.asarray(spike_trains)
                muap_gpu = cp.asarray(muap_shapes)
                surface_emg_gpu = cp.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

                for pool_idx in tqdm(
                    range(n_pools),
                    desc=f"Electrode Array {array_idx + 1}/{len(self._muaps__Block.groups)} Surface EMG (GPU)",
                    unit="pools",
                ):
                    pool_active_neurons = set(active_neuron_indices[pool_idx])

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs on GPU
                            convolutions = cp.array(
                                [
                                    cp.correlate(
                                        spike_gpu[pool_idx, mu_idx],
                                        muap_gpu[mu_idx, row_idx, col_idx],
                                        mode="same",
                                    )
                                    for mu_idx in MUs_to_simulate.intersection(
                                        pool_active_neurons
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
                    desc=f"Electrode Array {array_idx + 1}/{len(self._muaps__Block.groups)} Surface EMG (CPU)",
                    unit="pools",
                ):
                    pool_active_neurons = set(active_neuron_indices[pool_idx])

                    for row_idx in range(n_rows):
                        for col_idx in range(n_cols):
                            # Process all active MUs
                            convolutions = []
                            for mu_idx in MUs_to_simulate.intersection(
                                pool_active_neurons
                            ):
                                conv = np.correlate(
                                    spike_trains[pool_idx, mu_idx],
                                    muap_shapes[mu_idx, row_idx, col_idx],
                                    mode="same",
                                )
                                convolutions.append(conv)

                            if convolutions:
                                surface_emg[pool_idx, row_idx, col_idx] = np.sum(
                                    convolutions, axis=0
                                )

            # Temporal resampling
            surface_emg_resampled = np.zeros(
                (
                    n_pools,
                    n_rows,
                    n_cols,
                    int(
                        surface_emg.shape[-1]
                        * spiketrain_timestep__ms.rescale(pq.s).magnitude
                        * self._sampling_frequency__Hz
                    ),
                )
            )
            for pool_idx in range(n_pools):
                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        surface_emg_resampled[pool_idx, row_idx, col_idx] = np.interp(
                            x=np.arange(
                                start=0,
                                stop=surface_emg.shape[-1]
                                * spiketrain_timestep__ms.rescale(pq.s).magnitude,
                                step=1 / self._sampling_frequency__Hz,
                            ),
                            xp=np.arange(
                                start=0,
                                stop=surface_emg.shape[-1]
                                * spiketrain_timestep__ms.rescale(pq.s).magnitude,
                                step=spiketrain_timestep__ms.rescale(pq.s).magnitude,
                            ),
                            fp=surface_emg[pool_idx, row_idx, col_idx],
                        )

            # Create segments for each motor unit pool within this electrode array group
            for pool_idx in range(n_pools):
                segment = Segment(name=f"Pool_{pool_idx}")
                emg_group.segments.append(segment)

                # Create GridAnalogSignal for this pool's EMG data
                segment.analogsignals.append(
                    GridAnalogSignal(
                        signal=np.transpose(surface_emg_resampled[pool_idx], (2, 0, 1))
                        * pq.dimensionless,
                        sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                    )
                )

        # Store results privately
        self._surface_emg__Block = block
        return block

    def add_noise(
        self, snr__dB: float, noise_type: str = "gaussian"
    ) -> SURFACE_EMG__Block:
        """
        Add noise to all electrode arrays.

        This method adds realistic noise to the simulated surface EMG signals
        based on a specified signal-to-noise ratio. The noise is calculated
        and applied independently for each electrode channel to ensure that
        channels with different signal amplitudes maintain the specified SNR.

        Parameters
        ----------
        snr__dB : float
            Signal-to-noise ratio in dB. Higher values result in cleaner signals.
            Typical physiological EMG has SNR ranging from 10-40 dB.
            The SNR is applied independently to each electrode channel.
        noise_type : str, default="gaussian"
            Type of noise to add. Currently supports "gaussian" for white noise.

        Returns
        -------
        SURFACE_EMG__Block
            Noisy EMG signals for each electrode array as a neo.Block.
            Results are stored in the `noisy_surface_emg__Block` property after execution.

        Raises
        ------
        ValueError
            If surface EMG has not been simulated. Call simulate_surface_emg() first.

        Notes
        -----
        The noise is computed per-channel (per electrode) to maintain the specified
        SNR independently across all channels. This ensures that electrodes with
        different signal amplitudes receive appropriately scaled noise.
        """
        if self._surface_emg__Block is None:
            raise ValueError(
                "Surface EMG has not been simulated. Call simulate_surface_emg() first."
            )

        noisy_block = Block()

        for array_idx, emg_group in enumerate(self._surface_emg__Block.groups):
            noisy_group = Group(name=f"ElectrodeArray_{array_idx}")
            noisy_block.groups.append(noisy_group)

            for pool_idx, segment in enumerate(emg_group.segments):
                noisy_segment = Segment(name=f"Pool_{pool_idx}")
                noisy_group.segments.append(noisy_segment)

                # Get the EMG signal data
                emg_signal = segment.analogsignals[0]
                emg_array = emg_signal.magnitude  # Shape: (time, rows, cols)

                # Calculate signal power PER CHANNEL (per electrode)
                # Mean along time axis (axis=0) gives power per spatial location
                signal_power_per_channel = np.mean(emg_array**2, axis=0)  # Shape: (rows, cols)

                # Calculate noise power per channel
                snr_linear = 10 ** (snr__dB / 10)
                noise_power_per_channel = signal_power_per_channel / snr_linear
                noise_std_per_channel = np.sqrt(noise_power_per_channel)  # Shape: (rows, cols)

                # Generate noise
                if noise_type.lower() == "gaussian":
                    # Generate standard normal noise, then scale per channel
                    noise = RANDOM_GENERATOR.normal(
                        loc=0.0, scale=1.0, size=emg_array.shape
                    )
                    # Broadcast noise_std_per_channel along time axis
                    # noise shape: (time, rows, cols)
                    # noise_std_per_channel shape: (rows, cols)
                    # Broadcasting: (time, rows, cols) * (1, rows, cols)
                    noise = noise * noise_std_per_channel[np.newaxis, :, :]
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")

                # Add noise
                noisy_emg = emg_array + noise

                # Create new GridAnalogSignal with noise
                noisy_segment.analogsignals.append(
                    GridAnalogSignal(
                        signal=noisy_emg * emg_signal.units,
                        t_start=emg_signal.t_start,
                        sampling_rate=emg_signal.sampling_rate,
                    )
                )

        # Store results privately
        self._noisy_surface_emg__Block = noisy_block
        return noisy_block

    # Property accessors for computed results
    @property
    def muaps__Block(self) -> SURFACE_MUAP__Block:
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
        if self._muaps__Block is None:
            raise ValueError(
                "MUAP templates not computed. Call simulate_muaps() first."
            )
        return self._muaps__Block

    @property
    def surface_emg__Block(self) -> SURFACE_EMG__Block:
        """
        Surface EMG signals for each electrode array stored in a neo.Block.

        Returns
        -------
        SURFACE_EMG__Block
            Surface EMG signals for each electrode array stored in a neo.Block.

        Raises
        ------
        ValueError
            If surface EMG has not been computed yet.
        """
        if self._surface_emg__Block is None:
            raise ValueError(
                "Surface EMG signals not computed. Call simulate_surface_emg() first."
            )
        return self._surface_emg__Block

    @property
    def noisy_surface_emg__Block(self) -> SURFACE_EMG__Block:
        """
        Noisy surface EMG signals for each electrode array.

        Returns
        -------
        SURFACE_EMG__Block
            Noisy surface EMG signals for each electrode array.

        Raises
        ------
        ValueError
            If noisy surface EMG has not been computed yet.
        """
        if self._noisy_surface_emg__Block is None:
            raise ValueError(
                "Noisy surface EMG signals not computed. Call add_noise() first."
            )
        return self._noisy_surface_emg__Block

    @property
    def spike_train__Block(self) -> SPIKE_TRAIN__Block:
        """
        Spike train block used for EMG generation.

        Returns
        -------
        SPIKE_TRAIN__Block
            The spike train block used in the simulation.

        Raises
        ------
        ValueError
            If spike train block has not been set yet.
        """
        if self._spike_train__Block is None:
            raise ValueError(
                "Spike train block not set. Call simulate_surface_emg() first."
            )
        return self._spike_train__Block

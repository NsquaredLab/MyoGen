from typing import Optional

from myogen.utils.neo import GridAnalogSignal

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import elephant
    import elephant.utils

    HAS_ELEPHANT = True
except ImportError:
    HAS_ELEPHANT = False
    elephant = None  # type: ignore

import logging
from copy import deepcopy

import numpy as np
import quantities as pq
from joblib import Parallel, delayed
from neo import Block, Group, Segment
from scipy.signal import resample
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.emg.electrodes import SurfaceElectrodeArray
from myogen.simulator.core.emg.surface.simulate_fiber import simulate_fiber_v2
from myogen.simulator.core.muscle import Muscle
from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    Quantity__Hz,
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
    sampling_points_in_theta_domain : int, default=32
        Angular discretization for cylindrical coordinate system in degrees.
        Higher values provide better spatial resolution but cause numerical overflow in Bessel functions.
        Default is set to 32 points to avoid numerical instability.
        WARNING: Values >64 cause extreme Bessel function overflow leading to incorrect results.
        This is suitable for most EMG studies.
    MUs_to_simulate : list[int], optional
        Indices of motor units to simulate. If None, all motor units are simulated.
        Default is None. For computational efficiency, consider
        simulating subsets for initial analysis.
        Indices correspond to the recruitment order (0 is recruited first).
    internal_sampling_frequency__Hz : Quantity__Hz, optional
        Internal sampling frequency for MUAP computation before downsampling.
        If None, defaults to 10 kHz. Higher values provide better temporal resolution
        but increase computation time. Default is 10 kHz.
    iap_kernel_length__mm : float, optional
        Physical spatial extent for intracellular action potential (IAP) kernel evaluation in mm.
        If None (default), uses individual fiber lengths from muscle model, ensuring
        MUAP duration is physiologically accurate and independent of sampling resolution.

        **Recommended**: Leave as None to use fiber-specific lengths for most realistic MUAPs.

        Alternatively, set to a fixed value (e.g., 80-100 mm) to use the same kernel
        extent for all fibers, which can simplify analysis but may be less physiologically
        accurate for muscles with variable fiber lengths.

        This parameter controls the spatial extent over which the IAP waveform is computed,
        directly affecting MUAP duration: duration ≈ iap_kernel_length__mm / (2 * v) ms.

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
        sampling_frequency__Hz: Quantity__Hz = 2048.0 * pq.Hz,
        sampling_points_in_t_and_z_domains: int = 256,
        sampling_points_in_theta_domain: int = 32,
        MUs_to_simulate: list[int] | None = None,
        internal_sampling_frequency__Hz: Quantity__Hz | None = None,
        iap_kernel_length__mm: float | None = None,
    ):
        # Immutable public arguments - never modify these
        self.muscle_model = muscle_model
        self.electrode_arrays = electrode_arrays
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self.sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self.MUs_to_simulate = MUs_to_simulate
        self.iap_kernel_length__mm = iap_kernel_length__mm

        # Internal sampling frequency for higher resolution MUAP computation
        # If not specified, defaults to 10 kHz for better MUAP resolution
        if internal_sampling_frequency__Hz is None:
            internal_sampling_frequency__Hz = 10000.0 * pq.Hz
        self.internal_sampling_frequency__Hz = internal_sampling_frequency__Hz

        # Private copies for internal modifications (extract magnitudes)
        self._muscle_model = muscle_model
        self._electrode_arrays = electrode_arrays
        self._sampling_frequency__Hz = float(sampling_frequency__Hz.rescale(pq.Hz).magnitude)
        self._internal_sampling_frequency__Hz = float(
            internal_sampling_frequency__Hz.rescale(pq.Hz).magnitude
        )

        # Calculate upsampling factor and internal sample count
        self._upsampling_factor = (
            self._internal_sampling_frequency__Hz / self._sampling_frequency__Hz
        )
        self._internal_sampling_points = int(
            np.round(sampling_points_in_t_and_z_domains * self._upsampling_factor)
        )

        self._sampling_points_in_t_and_z_domains = sampling_points_in_t_and_z_domains
        self._sampling_points_in_theta_domain = sampling_points_in_theta_domain
        self._MUs_to_simulate = MUs_to_simulate
        self._iap_kernel_length__mm = iap_kernel_length__mm

        # Derived properties from muscle model - immutable public access
        self.mean_conduction_velocity__m_s = self._muscle_model.mean_conduction_velocity__m_s
        self.mean_fiber_length__mm = self._muscle_model.mean_fiber_length__mm
        self.var_fiber_length__mm = self._muscle_model.var_fiber_length__mm
        self.radius_bone__mm = self._muscle_model.radius_bone__mm
        self.fat_thickness__mm = self._muscle_model.fat_thickness__mm
        self.skin_thickness__mm = self._muscle_model.skin_thickness__mm
        self.muscle_conductivity_radial__S_m = self._muscle_model.muscle_conductivity_radial__S_m
        self.muscle_conductivity_longitudinal__S_m = (
            self._muscle_model.muscle_conductivity_longitudinal__S_m
        )
        self.fat_conductivity__S_m = self._muscle_model.fat_conductivity__S_m
        self.skin_conductivity__S_m = self._muscle_model.skin_conductivity__S_m

        # Private copies for internal modifications (extract magnitudes if Quantity objects)
        def _extract_value(val):
            """Helper to extract magnitude from Quantity or return float directly."""
            if hasattr(val, "magnitude"):
                return float(val.magnitude)
            return float(val)

        self._mean_conduction_velocity__m_s = _extract_value(
            self._muscle_model.mean_conduction_velocity__m_s
        )
        self._mean_fiber_length__mm = _extract_value(self._muscle_model.mean_fiber_length__mm)
        self._var_fiber_length__mm = _extract_value(self._muscle_model.var_fiber_length__mm)
        self._radius_bone__mm = _extract_value(self._muscle_model.radius_bone__mm)
        self._fat_thickness__mm = _extract_value(self._muscle_model.fat_thickness__mm)
        self._skin_thickness__mm = _extract_value(self._muscle_model.skin_thickness__mm)
        self._muscle_conductivity_radial__S_m = _extract_value(
            self._muscle_model.muscle_conductivity_radial__S_m
        )
        self._muscle_conductivity_longitudinal__S_m = _extract_value(
            self._muscle_model.muscle_conductivity_longitudinal__S_m
        )
        self._fat_conductivity__S_m = _extract_value(self._muscle_model.fat_conductivity__S_m)
        self._skin_conductivity__S_m = _extract_value(self._muscle_model.skin_conductivity__S_m)

        # Calculate total radius - immutable and private
        self._radius_muscle__mm = _extract_value(self._muscle_model.radius__mm)
        self.radius_total = (
            self._radius_muscle__mm + self._fat_thickness__mm + self._skin_thickness__mm
        )
        self._radius_total = self.radius_total

        # Simulation results - stored privately, accessed via properties
        self._muaps__Block: Optional[SURFACE_MUAP__Block] = None
        self._surface_emg__Block: Optional[SURFACE_EMG__Block] = None
        self._noisy_surface_emg__Block: Optional[SURFACE_EMG__Block] = None
        self._spike_train__Block: Optional[SPIKE_TRAIN__Block] = None

    def simulate_muaps(self, n_jobs: int = -2) -> SURFACE_MUAP__Block:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        This method generates Motor Unit Action Potential (MUAP) templates that represent
        the electrical signature of each motor unit as recorded by the surface electrodes.
        The simulation uses the multi-layered cylindrical volume conductor model with
        parallel processing for improved performance.

        Parameters
        ----------
        n_jobs : int, optional
            Number of parallel workers for motor unit processing. Default is -2.
            - n_jobs=-1: Use all CPU cores
            - n_jobs=-2: Use all cores except one (recommended, keeps system responsive)
            - n_jobs=-3: Use all cores except two
            - n_jobs=1: No parallelization
            - n_jobs=N: Use exactly N cores

        Returns
        -------
        SURFACE_MUAP__Block
            neo.Block of generated MUAP templates for each electrode array.
            Results are stored in the `muaps` property after execution.

        Notes
        -----
        This method must be called before simulate_surface_emg(). The generated MUAP
        templates are used as basis functions for EMG signal synthesis.

        The motor units are processed in parallel using joblib, with each motor unit's
        fibers processed sequentially to maintain optimization efficiency.
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
        number_of_fibers_per_MUs = self._muscle_model.resulting_number_of_innervated_fibers

        # Create time array at INTERNAL sampling frequency for higher resolution
        t_internal = np.linspace(
            0,
            (self._internal_sampling_points - 1) / self._internal_sampling_frequency__Hz * 1e-3,
            self._internal_sampling_points,
        )

        # Get total number of motor units
        n_motor_units = len(number_of_fibers_per_MUs)

        # Pre-calculate innervation zones for all MUs
        innervation_zones = RANDOM_GENERATOR.uniform(
            low=-innervation_zone_variance / 2,
            high=innervation_zone_variance / 2,
            size=n_motor_units,
        )

        # Pre-allocate result shape at INTERNAL resolution (optimization: avoid repeated shape calculations)
        # Will be downsampled to output resolution after simulation
        internal_result_shape = (
            self._electrode_arrays[0].num_rows,
            self._electrode_arrays[0].num_cols,
            len(t_internal),
        )

        # Final output shape after downsampling
        output_result_shape = (
            self._electrode_arrays[0].num_rows,
            self._electrode_arrays[0].num_cols,
            self._sampling_points_in_t_and_z_domains,
        )

        # Helper function to process a single motor unit
        def _process_single_mu(
            MU_index: int,
            electrode_array_original: SurfaceElectrodeArray,
        ) -> tuple[np.ndarray, str]:
            """
            Process a single motor unit (all its fibers) in parallel.

            Parameters
            ----------
            MU_index : int
                Index of the motor unit to process.
            electrode_array_original : SurfaceElectrodeArray
                Original electrode array (will be deep-copied to avoid threading issues).

            Returns
            -------
            tuple[np.ndarray, str]
                Tuple of (array_result, segment_name) where array_result is the accumulated
                MUAP signal for this MU and segment_name is the name for the segment.
            """
            try:
                # Deep copy electrode array to avoid threading issues
                electrode_array = deepcopy(electrode_array_original)

                # Pre-allocated result array at INTERNAL resolution (optimization: use pre-computed shape)
                array_result_internal = np.zeros(internal_result_shape, dtype=np.float64)

                number_of_fibers = number_of_fibers_per_MUs[MU_index]

                if number_of_fibers == 0:
                    # Return empty signal (downsampled to output resolution)
                    array_result_downsampled = np.zeros(output_result_shape, dtype=np.float64)
                    return array_result_downsampled, f"MUAP_{MU_index}"

                # Get fiber positions
                position_of_fibers_raw = self._muscle_model.resulting_fiber_assignment(MU_index)
                # Extract magnitude if Quantity, otherwise use as-is
                if hasattr(position_of_fibers_raw, "magnitude"):
                    position_of_fibers = position_of_fibers_raw.magnitude
                else:
                    position_of_fibers = position_of_fibers_raw

                innervation_zone = innervation_zones[MU_index]

                # Batch generate random fiber lengths (optimization: single RNG call)
                fiber_length_variations = RANDOM_GENERATOR.uniform(
                    low=-self._var_fiber_length__mm,
                    high=self._var_fiber_length__mm,
                    size=number_of_fibers,
                )

                # Pre-compute geometric values for all fibers (optimization: vectorized)
                R_values = np.sqrt(position_of_fibers[:, 0] ** 2 + position_of_fibers[:, 1] ** 2)
                theta_values = np.arctan2(position_of_fibers[:, 1], position_of_fibers[:, 0])
                fiber_lengths = self._mean_fiber_length__mm + fiber_length_variations

                # Matrix optimization variables (local to this MU)
                A_matrix = None
                B_incomplete = None

                # Process each fiber (inner loop - must remain sequential)
                for fiber_number in range(number_of_fibers):
                    # Use pre-computed values (optimization: vectorized calculations)
                    R = R_values[fiber_number]
                    theta = theta_values[fiber_number]
                    fiber_length__mm = fiber_lengths[fiber_number]

                    electrode_array._center_point__mm_deg = (
                        electrode_array._center_point__mm_deg[0],
                        electrode_array._center_point__mm_deg[1] - np.rad2deg(theta),
                    )
                    electrode_array._create_electrode_grid()

                    # Calculate fiber end positions
                    L1 = abs(innervation_zone + fiber_length__mm / 2)
                    L2 = abs(innervation_zone - fiber_length__mm / 2)

                    # Determine IAP kernel length: use fixed value or scaled mean fiber length
                    # IMPORTANT: Use scaled mean, not individual fiber_length__mm, to avoid boundary artifacts
                    if self._iap_kernel_length__mm is not None:
                        kernel_length = self._iap_kernel_length__mm
                    else:
                        # Use scaled mean fiber length (2.5×) for all fibers to prevent boundary truncation
                        IAP_SCALE_FACTOR = 2.5
                        kernel_length = self._mean_fiber_length__mm * IAP_SCALE_FACTOR

                    # Use the new simulate_fiber_v2 function with INTERNAL sampling frequency
                    phi_temp, A_matrix, B_incomplete = simulate_fiber_v2(
                        Fs=self._internal_sampling_frequency__Hz * 1e-3,
                        v=self._mean_conduction_velocity__m_s,
                        N=self._internal_sampling_points,
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
                        A_matrix=None if fiber_number == 0 else A_matrix,
                        B_incomplete=None if fiber_number == 0 else B_incomplete,
                        fiber_length__mm=kernel_length,  # NEW: Use fiber-specific or fixed IAP kernel length
                    )

                    array_result_internal += phi_temp

                # Downsample from internal resolution to output resolution
                # resample operates on the last axis (time), which is axis=2
                array_result_downsampled = resample(
                    array_result_internal, self._sampling_points_in_t_and_z_domains, axis=2
                )

                return array_result_downsampled, f"MUAP_{MU_index}"

            except Exception as e:
                # Log error and return empty result to avoid crashing entire parallel job
                logging.error(
                    f"Failed to process MU {MU_index} for electrode array {array_idx}: {e}"
                )
                # Return empty signal with error marker at output resolution
                empty_result = np.zeros(output_result_shape, dtype=np.float64)
                return empty_result, f"MUAP_{MU_index}_FAILED"

        block = Block()
        for array_idx, electrode_array in enumerate(self._electrode_arrays):
            group = Group(name=f"ElectrodeArray_{array_idx}")
            block.groups.append(group)

            # Process only specified motor units in parallel
            n_mus_to_compute = len(self._MUs_to_simulate)
            logging.info(
                f"Processing {n_mus_to_compute}/{n_motor_units} motor units for electrode array {array_idx + 1}/{len(self._electrode_arrays)} using parallel processing..."
            )

            # Parallel execution of motor units with tqdm progress bar
            # batch_size="auto" optimizes task distribution across workers
            # Only compute MUs in the subset
            results = {}  # Use dict to map MU_index -> result
            with tqdm(
                total=n_mus_to_compute,
                desc=f"Electrode Array {array_idx + 1}/{len(self._electrode_arrays)}",
            ) as pbar:
                for array_result, segment_name in Parallel(
                    n_jobs=n_jobs,
                    return_as="generator",
                    verbose=0,
                    batch_size="auto",
                )(
                    delayed(_process_single_mu)(MU_index, electrode_array)
                    for MU_index in self._MUs_to_simulate
                ):
                    # Extract MU index from segment name "MUAP_{MU_index}"
                    mu_idx = int(segment_name.split("_")[1].split("_")[0])
                    results[mu_idx] = (array_result, segment_name)
                    pbar.update(1)

            # Calculate actual MUAP duration based on fiber lengths
            # Use iap_kernel_length__mm if specified, otherwise use scaled fiber length from muscle model
            if self._iap_kernel_length__mm is not None:
                kernel_length_mm = self._iap_kernel_length__mm
            else:
                # Scale fiber length to avoid boundary truncation of IAP kernel
                # The IAP kernel needs ~2.5x the fiber length to fully develop and decay
                # This prevents edge artifacts while maintaining proportionality to fiber length
                IAP_SCALE_FACTOR = 2.5
                kernel_length_mm = self._mean_fiber_length__mm * IAP_SCALE_FACTOR

            # Physical duration based on IAP kernel length (after /=2 scaling in simulate_fiber)
            # Duration = (kernel_length_mm / 2) / velocity
            muap_duration__s = (
                kernel_length_mm / 2.0
            ) / self._mean_conduction_velocity__m_s / 1000.0  # Convert ms to s

            # Create custom time array for this duration
            times__s = np.linspace(0, muap_duration__s, self._sampling_points_in_t_and_z_domains)

            # Calculate effective sampling rate for these times
            effective_sampling_rate__Hz = (
                self._sampling_points_in_t_and_z_domains - 1
            ) / muap_duration__s

            use_custom_times = True

            # Create segments for ALL MUs (maintaining index order)
            # Non-computed MUs get empty signals at output resolution
            for MU_index in range(n_motor_units):
                if MU_index in results:
                    array_result, segment_name = results[MU_index]
                else:
                    # Create empty MUAP for non-computed MUs at output resolution
                    array_result = np.zeros(output_result_shape, dtype=np.float64)
                    segment_name = f"MUAP_{MU_index}"

                segment = Segment(name=segment_name)
                group.segments.append(segment)

                # Use actual physical duration based on fiber lengths
                segment.analogsignals.append(
                    GridAnalogSignal(
                        signal=np.transpose(array_result, (2, 0, 1)) * pq.mV,
                        t_start=0 * pq.s,
                        times=times__s * pq.s,
                        sampling_rate=effective_sampling_rate__Hz * pq.Hz,
                    )
                )

        # Store results privately
        self._muaps__Block = block

        return block

    def simulate_surface_emg(self, spike_train__Block: SPIKE_TRAIN__Block) -> SURFACE_EMG__Block:
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
            raise ValueError("MUAP templates have not been generated. Call simulate_muaps() first.")

        if not HAS_ELEPHANT:
            raise ImportError(
                "Elephant is required for surface EMG simulation. "
                "Install with: pip install myogen[elephant]"
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
                                stop=muap_array.shape[-1] / self._sampling_frequency__Hz,
                                step=spiketrain_timestep__ms.rescale(pq.s).magnitude,
                            ),
                            xp=np.arange(
                                start=0,
                                stop=muap_array.shape[-1] / self._sampling_frequency__Hz,
                                step=muap_timestep__ms.rescale(pq.s).magnitude,
                            ),
                            fp=muap_array[muap_nr, row, col],
                        )

            # n_pools already defined above from spike_train_block
            n_rows = muap_shapes.shape[1]
            n_cols = muap_shapes.shape[2]

            # Initialize result array
            sample_conv = np.convolve(spike_trains[0, 0], muap_shapes[0, 0, 0], mode="same")

            surface_emg = np.zeros((n_pools, n_rows, n_cols, len(sample_conv)))

            # No normalization needed - MUAPs are in absolute units (mV) from biophysical model

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
                                    for mu_idx in MUs_to_simulate.intersection(pool_active_neurons)
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
                            for mu_idx in MUs_to_simulate.intersection(pool_active_neurons):
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
                        signal=np.transpose(surface_emg_resampled[pool_idx], (2, 0, 1)) * pq.mV,
                        sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                    )
                )

        # Store results privately
        self._surface_emg__Block = block
        return block

    def add_noise(self, snr__dB: float, noise_type: str = "gaussian") -> SURFACE_EMG__Block:
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
                    noise = RANDOM_GENERATOR.normal(loc=0.0, scale=1.0, size=emg_array.shape)
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
            raise ValueError("MUAP templates not computed. Call simulate_muaps() first.")
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
            raise ValueError("Surface EMG signals not computed. Call simulate_surface_emg() first.")
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
            raise ValueError("Noisy surface EMG signals not computed. Call add_noise() first.")
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
            raise ValueError("Spike train block not set. Call simulate_surface_emg() first.")
        return self._spike_train__Block

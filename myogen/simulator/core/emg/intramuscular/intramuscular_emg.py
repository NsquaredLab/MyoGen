"""
Intramuscular Electromyography (iEMG) Simulation.

This module provides the main simulation framework for generating intramuscular
electromyography signals using needle electrodes. It integrates motor unit
simulation, electrode modeling, and signal generation with realistic noise.
"""

import logging
import warnings
from copy import deepcopy
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
from joblib import Parallel, delayed
from neo import AnalogSignal, Block, Segment
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.emg.electrodes import IntramuscularElectrodeArray
from myogen.simulator.core.muscle import Muscle
from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    INTRAMUSCULAR_EMG__Block,
    INTRAMUSCULAR_MUAP__Block,
    Quantity__Hz,
    Quantity__m_per_s,
    Quantity__mm,
    Quantity__s,
    SPIKE_TRAIN__Block,
)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from .motor_unit_sim import MotorUnitSim

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@beartowertype
class IntramuscularEMG:
    """
    Intramuscular Electromyography (iEMG) Simulation.

    This class provides a comprehensive simulation framework for generating
    intramuscular EMG signals detected by needle electrodes.

    Parameters
    ----------
    muscle_model : Muscle
        Pre-computed muscle model (see :class:`myogen.simulator.Muscle`).
    electrode_array : IntramuscularElectrodeArray
        Intramuscular electrode array configuration to use for simulation (see :class:`myogen.simulator.IntramuscularElectrodeArray`).
    sampling_frequency__Hz : Quantity__Hz, default=10240.0 * pq.Hz
        Sampling frequency in Hz for EMG simulation.
        Default is set to 10240 Hz as used by the Quattrocento (OT Bioelettronica, Turin, Italy) system.
    spatial_resolution__mm : Quantity__mm, default=0.01 * pq.mm
        Spatial resolution for fiber action potential calculation in mm.
        Default is set to 0.01 mm.
    endplate_center__percent : float, default=50
        Percentage of muscle length where the endplate is located.
        By default, the endplate is located at the center of the muscle (50% of the muscle length).
    nmj_jitter__s : Quantity__s, default=35e-6 * pq.s
        Standard deviation of neuromuscular junction jitter in seconds.
        Default is set to 35e-6 s as determined by Konstantin et al. 2020 [1]_.
    branch_cvs__m_per_s : tuple[Quantity__m_per_s, Quantity__m_per_s], default=(5.0 * pq.m / pq.s, 2.0 * pq.m / pq.s)
        Conduction velocities for the two-layer model of the neuromuscular junction in m/s.
        Default is set to (5.0, 2.0) m/s as determined by Konstantin et al. 2020 [1]_.

        .. note::
            The two-layer model is a simplification of the actual arborization pattern, but it is a good approximation for the purposes of this simulation.
            Follows the implementation of Kontos et al. 2020 [1]_.
    MUs_to_simulate : list[int], optional
        Indices of motor units to simulate. If None, all motor units are simulated.
        Default is None. For computational efficiency, consider
        simulating subsets for initial analysis.
        Indices correspond to the recruitment order (0 is recruited first).

    Attributes
    ----------
    muaps__Block : INTRAMUSCULAR_MUAP__Block
        Intramuscular MUAP shapes for the electrode array as a neo.Block. Available after simulate_muaps().
    intramuscular_emg__Block : INTRAMUSCULAR_EMG__Block
        Intramuscular EMG signals for the electrode array as a neo.Block. Available after simulate_intramuscular_emg().
    noisy_intramuscular_emg__Block : INTRAMUSCULAR_EMG__Block
        Noisy intramuscular EMG signals for the electrode array as a neo.Block. Available after add_noise().
    spike_train__Block : SPIKE_TRAIN__Block
        Spike train block used for EMG generation. Available after simulate_intramuscular_emg().

    References
    ----------
    .. [1] Konstantin, A., Yu, T., Le Carpentier, E., Aoustin, Y., Farina, D., 2020. Simulation of Motor Unit Action Potential Recordings From Intramuscular Multichannel Scanning Electrodes. IEEE Transactions on Biomedical Engineering 67, 2005–2014. https://doi.org/10.1109/TBME.2019.2953680
    """

    def __init__(
        self,
        muscle_model: Muscle,
        electrode_array: IntramuscularElectrodeArray,
        sampling_frequency__Hz: Quantity__Hz = 10240.0 * pq.Hz,
        spatial_resolution__mm: Quantity__mm = 0.01 * pq.mm,
        endplate_center__percent: float = 50,
        nmj_jitter__s: Quantity__s = 35e-6 * pq.s,
        branch_cvs__m_per_s: tuple[Quantity__m_per_s, Quantity__m_per_s] = (
            5.0 * pq.m / pq.s,
            2.0 * pq.m / pq.s,
        ),
        MUs_to_simulate: list[int] | None = None,
    ):
        # Immutable public arguments - never modify these
        self.muscle_model = muscle_model
        self.electrode_array = electrode_array
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.spatial_resolution__mm = spatial_resolution__mm
        self.endplate_center__percent = endplate_center__percent
        self.nmj_jitter__s = nmj_jitter__s
        self.branch_cvs__m_per_s = branch_cvs__m_per_s
        self.MUs_to_simulate = MUs_to_simulate

        # Private copies for internal modifications (extract magnitudes)
        self._muscle_model = muscle_model
        self._electrode_array = electrode_array
        self._sampling_frequency__Hz = float(sampling_frequency__Hz.rescale(pq.Hz).magnitude)
        self._spatial_resolution__mm = float(spatial_resolution__mm.rescale(pq.mm).magnitude)
        self._endplate_center__percent = endplate_center__percent
        self._nmj_jitter__s = float(nmj_jitter__s.rescale(pq.s).magnitude)
        self._branch_cvs__m_per_s = (
            float(branch_cvs__m_per_s[0].rescale(pq.m / pq.s).magnitude),
            float(branch_cvs__m_per_s[1].rescale(pq.m / pq.s).magnitude),
        )
        self._MUs_to_simulate = MUs_to_simulate

        # Derived parameters - immutable public access
        self.branch_cvs__mm_per_s = (
            float(branch_cvs__m_per_s[0].rescale(pq.mm / pq.s).magnitude),
            float(branch_cvs__m_per_s[1].rescale(pq.mm / pq.s).magnitude),
        )
        self.endplate_center__mm = self._muscle_model.length__mm * (
            self._endplate_center__percent / 100.0
        )

        # Private copies for internal modifications
        self._branch_cvs__mm_per_s: tuple[float, float] = (
            self._branch_cvs__m_per_s[0] * 1000.0,
            self._branch_cvs__m_per_s[1] * 1000.0,
        )
        # Extract magnitude for internal use (calculations expect floats)
        length_mm = (
            float(self._muscle_model.length__mm.rescale(pq.mm).magnitude)
            if hasattr(self._muscle_model.length__mm, "magnitude")
            else float(self._muscle_model.length__mm)
        )
        self._endplate_center__mm = length_mm * (self._endplate_center__percent / 100.0)

        # Derived parameters - private for internal use
        self._dt = 1.0 / self._sampling_frequency__Hz
        self._dz = self._spatial_resolution__mm
        self._n_motor_units = len(self._muscle_model.recruitment_thresholds)

        # Motor unit selection
        if self._MUs_to_simulate is None:
            self._MUs_to_simulate = list(range(self._n_motor_units))
        else:
            self._MUs_to_simulate = self._MUs_to_simulate

        # Motor unit simulations - private storage
        self._motor_units: list[MotorUnitSim] = []  # List of motor unit simulators
        self._muaps__Block: Optional[INTRAMUSCULAR_MUAP__Block] = None
        self._max_muap_length: int = 0

        # Simulation results - stored privately, accessed via properties
        self._intramuscular_emg__Block: Optional[INTRAMUSCULAR_EMG__Block] = None
        self._noisy_intramuscular_emg__Block: Optional[INTRAMUSCULAR_EMG__Block] = None
        self._spike_train__Block: Optional[SPIKE_TRAIN__Block] = None

    def simulate_muaps(self, n_jobs: int = -2) -> INTRAMUSCULAR_MUAP__Block:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        This method generates intramuscular Motor Unit Action Potential (MUAP) templates
        by simulating individual motor units with realistic neuromuscular junction
        distributions and fiber action potential propagation.

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
        INTRAMUSCULAR_MUAP_SHAPE__TENSOR
            Intramuscular MUAP shapes for all electrode arrays.
            Results are stored in the `muaps` property after execution.

        Notes
        -----
        This method must be called before simulate_intramuscular_emg(). The process
        includes: (1) motor unit initialization, (2) neuromuscular junction simulation,
        and (3) MUAP calculation with spatial filtering.
        """
        self._initialize_motor_units()
        self._simulate_neuromuscular_junctions()
        return self._calculate_muaps(n_jobs=n_jobs)

    def _initialize_motor_units(self) -> None:
        """
        Initialize individual motor unit simulators.

        This method creates MotorUnitSim objects for each motor unit based on
        the muscle model fiber assignments and properties.
        """
        if not hasattr(self._muscle_model, "assignment") or self._muscle_model.assignment is None:
            raise ValueError(
                "Muscle model must have fiber assignments. Call muscle.assign_mfs2mns() first."
            )

        # Initialize list with None for all MUs (will be filled with simulators)
        self._motor_units = [None] * self._n_motor_units

        for mu_idx in tqdm(
            range(self._n_motor_units),
            desc="Creating motor unit simulators",
            unit="Simulator",
        ):
            # Get fibers assigned to this motor unit
            fiber_mask = self._muscle_model.assignment == mu_idx
            if not np.any(fiber_mask):
                continue

            # Create motor unit simulator at the correct index
            # Extract magnitudes from quantities for MotorUnitSim (expects floats/arrays)
            def _extract_magnitude(val):
                """Helper to extract magnitude from Quantity or return value as-is."""
                if hasattr(val, "magnitude"):
                    return val.magnitude
                return val

            self._motor_units[mu_idx] = MotorUnitSim(
                muscle_fiber_centers__mm=_extract_magnitude(
                    self._muscle_model.muscle_fiber_centers__mm[fiber_mask]
                ),
                muscle_length__mm=float(_extract_magnitude(self._muscle_model.length__mm)),
                muscle_fiber_diameters__mm=_extract_magnitude(
                    self._muscle_model.muscle_fiber_diameters__mm[fiber_mask]
                ),
                muscle_fiber_conduction_velocity__mm_per_s=_extract_magnitude(
                    self._muscle_model.muscle_fiber_conduction_velocities__mm_per_s[fiber_mask]
                ),
                neuromuscular_junction_conduction_velocities__mm_per_s=list(
                    self._branch_cvs__mm_per_s
                ),
                nominal_center__mm=_extract_magnitude(
                    self._muscle_model.innervation_center_positions__mm[mu_idx]
                ),
            )

    def _simulate_neuromuscular_junctions(self) -> None:
        """
        Simulate neuromuscular junction distributions for all motor units.

        This implements the logic from s08_cl_init_muaps.m for generating
        realistic NMJ branch patterns with size-dependent complexity.
        """
        if not self._motor_units:
            raise ValueError("Must call _initialize_motor_units() first")

        n_branches = 1 + np.round(
            np.log(
                self._muscle_model.recruitment_thresholds
                / self._muscle_model.recruitment_thresholds[0]
            )
        )

        for mu_idx, mu_sim in enumerate(
            tqdm(self._motor_units, desc="Setting up NMJ distributions", unit="Simulator")
        ):
            spread_factor = np.sum(self._muscle_model.recruitment_thresholds[:mu_idx]) / np.sum(
                self._muscle_model.recruitment_thresholds
            )

            # Create NMJ distribution
            # Branch spread increases with motor unit size
            mu_sim.sim_nmj_branches_two_layers(
                n_branches=int(n_branches[mu_idx]),
                endplate_center=self._endplate_center__mm,
                branches_z_std=1.5 + spread_factor * 4.0,
                arborization_z_std=0.5 + spread_factor * 1.5,
            )

    def _calculate_muaps(self, n_jobs: int = -2) -> INTRAMUSCULAR_MUAP__Block:
        """
        Pre-calculate motor unit action potentials (MUAPs) using parallel processing.

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
        INTRAMUSCULAR_MUAP_SHAPE__TENSOR
            Intramuscular MUAP shapes for all electrode arrays.
        """
        if not self._motor_units:
            raise ValueError("Must call _initialize_motor_units() first")

        # Set default MUs to simulate
        if self._MUs_to_simulate is None:
            self._MUs_to_simulate = list(range(len(self._motor_units)))

        # Helper function to process a single motor unit
        def _process_single_mu(
            mu_idx: int,
            mu_sim: Optional[MotorUnitSim],
            dt: float,
            dz: float,
            electrode_positions: np.ndarray,
        ) -> tuple[Optional[np.ndarray], int]:
            """
            Process a single motor unit (calculate SFAP and MUAP).

            Parameters
            ----------
            mu_idx : int
                Index of the motor unit to process.
            mu_sim : Optional[MotorUnitSim]
                Motor unit simulator object (None if MU has no fibers).
            dt : float
                Temporal resolution.
            dz : float
                Spatial resolution.
            electrode_positions : np.ndarray
                Electrode positions array.

            Returns
            -------
            tuple[Optional[np.ndarray], int]
                Tuple of (muap_array, muap_length) where muap_array is the MUAP signal
                for this MU (or None if MU has no fibers) and muap_length is the length.
            """
            try:
                if mu_sim is None:
                    return None, 0

                # Deep copy to avoid threading issues
                mu_sim_copy = deepcopy(mu_sim)

                # Calculate SFAPs
                mu_sim_copy.calc_sfaps(
                    index=mu_idx,
                    dt=dt,
                    dz=dz,
                    electrode_positions=electrode_positions,
                )

                # Calculate MUAP (no jitter for templates)
                muap = mu_sim_copy.calc_muap(jitter_std=0.0)

                return muap, muap.shape[0]

            except Exception as e:
                # Log error and return None to avoid crashing entire parallel job
                logging.error(f"Failed to process MU {mu_idx}: {e}")
                return None, 0

        # Process only specified motor units in parallel
        n_motor_units = len(self._motor_units)
        n_mus_to_compute = len(self._MUs_to_simulate)

        logging.info(
            f"Processing {n_mus_to_compute}/{n_motor_units} motor units using parallel processing..."
        )

        # Parallel execution with progress bar
        # Only compute MUs in the subset
        results = {}  # Use dict to map MU_index -> result
        max_length = 0

        with tqdm(total=n_mus_to_compute, desc="Computing MUAPs", unit="MU") as pbar:
            for muap, muap_length in Parallel(
                n_jobs=n_jobs,
                return_as="generator",
                verbose=0,
                batch_size="auto",
            )(
                delayed(_process_single_mu)(
                    mu_idx,
                    self._motor_units[mu_idx],
                    self._dt,
                    self._dz,
                    self._electrode_array.pts,
                )
                for mu_idx in self._MUs_to_simulate
            ):
                # Results come in order, map back to original MU index
                mu_idx = self._MUs_to_simulate[len(results)]
                results[mu_idx] = muap
                max_length = max(max_length, muap_length)
                pbar.update(1)

        # Create neo Block structure
        # Create segments for ALL MUs (maintaining index order)
        # Non-computed MUs get empty signals
        block = Block()

        for mu_idx in range(n_motor_units):
            muap = results.get(mu_idx)

            if muap is None:
                # Create empty segment for MUs with no fibers or not selected
                block.segments.append(segment := Segment(name="MUAP_None"))
                segment.analogsignals.append(
                    AnalogSignal(
                        np.zeros((1, len(self._electrode_array.pts))) * pq.dimensionless,
                        sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                    )
                )
            else:
                block.segments.append(segment := Segment(name=f"MUAP_{muap.shape[0]}"))
                segment.analogsignals.append(
                    AnalogSignal(
                        muap * pq.dimensionless,
                        sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                    )
                )

        self._muaps__Block = block

        return self._muaps__Block

    def _analyze_detectable_motor_units(self) -> tuple[np.ndarray, list[int]]:
        """
        Analyze which motor units are detectable by the electrode.

        This implements s11_cl_get_detectable_mus.m logic for determining
        motor unit visibility based on signal-to-noise ratio and contribution.

        Returns
        -------
        tuple[np.ndarray, List[int]]
            Boolean array of detectable motor units and their indices
        """
        if self._muaps__Block is None:
            raise ValueError("Must call simulate_muaps() first")

        print("Analyzing motor unit detectability...")

        detectable = np.zeros(len(self._motor_units), dtype=bool)

        # Prominence criterion: MUAP amplitude vs noise
        over_noise_threshold = 2.0  # 2x noise level

        for i, mu_sim in enumerate(self._motor_units):
            # Get peak MUAP amplitude across all channels
            muap_data = self._muaps__Block.segments[i].analogsignals[0].magnitude
            muap_amplitudes = np.max(np.abs(muap_data), axis=0)
            max_amplitude = np.max(muap_amplitudes)

            # Check if MUAP is prominent enough above noise
            # Use a simple noise threshold estimate
            all_muaps = np.array(
                [seg.analogsignals[0].magnitude for seg in self._muaps__Block.segments]
            )
            noise_estimate = np.std(all_muaps) * 0.1  # Simple noise estimate
            is_prominent = max_amplitude > over_noise_threshold * noise_estimate

            # Additional criterion: contribution to total signal variance
            relative_size = (i + 1) / len(self._motor_units)
            min_contribution = 0.05  # Minimum 5% contribution
            contributes_enough = relative_size > min_contribution

            detectable[i] = is_prominent and contributes_enough

        detectable_indices = [i for i, det in enumerate(detectable) if det]

        print(f"Found {np.sum(detectable)} detectable motor units out of {len(self._motor_units)}")

        return detectable, detectable_indices

    def simulate_intramuscular_emg(
        self,
        spike_train__Block: SPIKE_TRAIN__Block,
    ) -> INTRAMUSCULAR_EMG__Block:
        """
        Generate intramuscular EMG signals using the provided spike train block.

        This method convolves the pre-computed MUAP templates with spike trains
        to synthesize realistic intramuscular EMG signals. The process includes temporal
        resampling and supports both CPU and GPU acceleration for efficient computation.

        Parameters
        ----------
        spike_train__Block : SPIKE_TRAIN__Block
            Block containing spike trains organized as segments (pools) with spiketrains.

        Returns
        -------
        INTRAMUSCULAR_EMG__Block
            Intramuscular EMG signals for the electrode array stored in a neo.Block.
            Results are stored in the `intramuscular_emg__Block` property after execution.

        Raises
        ------
        ValueError
            If MUAP templates have not been generated. Call simulate_muaps() first.
        """
        if self._muaps__Block is None:
            raise ValueError("MUAP templates have not been generated. Call simulate_muaps() first.")

        if not HAS_ELEPHANT:
            raise ImportError(
                "Elephant is required for intramuscular EMG simulation. "
                "Install with: pip install myogen[elephant]"
            )

        # Store spike train data privately
        self._spike_train__Block = spike_train__Block

        # Handle MUs to simulate
        if self._MUs_to_simulate is None:
            MUs_to_simulate = set(
                range(len(self._muscle_model.resulting_number_of_innervated_fibers))
            )
        else:
            MUs_to_simulate = set(self._MUs_to_simulate)

        # Extract MUAP data from Block and pad to same length
        muap_data_list = [seg.analogsignals[0].magnitude for seg in self._muaps__Block.segments]

        # Find the maximum length among all MUAPs
        max_length = max(muap.shape[0] for muap in muap_data_list)
        n_electrodes = muap_data_list[0].shape[1]

        # Pad all MUAPs to the same length (centered, pad both sides)
        padded_muaps = []
        for muap in muap_data_list:
            pad_total = max_length - muap.shape[0]
            if pad_total > 0:
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                pad_width = ((pad_left, pad_right), (0, 0))
                padded_muap = np.pad(muap, pad_width, mode="constant", constant_values=0)
            else:
                padded_muap = muap
            padded_muaps.append(padded_muap)

        muap_array = np.array(padded_muaps)

        # Extract timestep from the first spike train
        first_spiketrain = spike_train__Block.segments[0].spiketrains[0]
        spiketrain_timestep__ms = first_spiketrain.sampling_period.rescale("ms")

        target_length = int(
            np.round(
                muap_array.shape[1]
                / self._sampling_frequency__Hz
                * 1
                / (spiketrain_timestep__ms.rescale("s").magnitude)
            )
        )
        muap_shapes = np.zeros((muap_array.shape[0], muap_array.shape[2], target_length))
        for muap_nr in range(muap_shapes.shape[0]):
            for electrode_nr in range(muap_shapes.shape[1]):
                muap_shapes[muap_nr, electrode_nr] = np.interp(
                    np.linspace(
                        0,
                        muap_array.shape[1] / self._sampling_frequency__Hz,
                        target_length,
                        endpoint=False,
                    ),
                    np.arange(
                        0,
                        muap_array.shape[1] / self._sampling_frequency__Hz,
                        1 / self._sampling_frequency__Hz,
                    ),
                    muap_array[muap_nr, :, electrode_nr],
                )

        # Convert spike train block to numpy arrays
        n_pools = len(spike_train__Block.segments)
        n_neurons = len(spike_train__Block.segments[0].spiketrains)
        n_electrodes = muap_shapes.shape[1]

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

        # Create active neuron indices (all neurons are active in each pool for spike train block)
        active_neuron_indices = [list(range(n_neurons)) for _ in range(n_pools)]

        # Initialize result array
        sample_conv = np.convolve(
            spike_trains[0, 0],
            muap_shapes[0, 0],
            mode="same",
        )
        intramuscular_emg = np.zeros((n_pools, n_electrodes, len(sample_conv)))

        # Normalize MUAP shapes before convolution
        # Note: Unlike surface EMG, intramuscular MUAP amplitudes from biophysical
        # calculations are in arbitrary units and must be normalized to prevent
        # numerical overflow during convolution. Final EMG amplitudes are determined
        # by the spike train convolution, not the raw MUAP amplitudes.
        muap_shapes /= np.max(np.abs(muap_shapes))

        # Perform convolution for each pool using GPU acceleration if available
        if HAS_CUPY:
            # Use GPU acceleration with CuPy
            spike_gpu = cp.asarray(spike_trains)
            muap_gpu = cp.asarray(muap_shapes)
            intramuscular_emg_gpu = cp.zeros((n_pools, n_electrodes, len(sample_conv)))

            for pool_idx in tqdm(
                range(n_pools),
                desc="Intramuscular EMG (GPU)",
                unit="pool",
            ):
                pool_active_neurons = set(active_neuron_indices[pool_idx])

                for e_idx in range(n_electrodes):
                    # Process all active MUs on GPU
                    convolutions = []
                    for mu_idx in MUs_to_simulate.intersection(pool_active_neurons):
                        # Use mu_idx directly since muap_shapes now contains all MUs
                        if mu_idx < muap_gpu.shape[0]:
                            conv = cp.correlate(
                                spike_gpu[pool_idx, mu_idx],
                                muap_gpu[mu_idx, e_idx],
                                mode="same",
                            )
                            convolutions.append(conv)

                    convolutions = cp.array(convolutions) if convolutions else cp.array([])
                    # Sum across MUAPs on GPU
                    if len(convolutions) > 0:
                        intramuscular_emg_gpu[pool_idx, e_idx] = cp.sum(convolutions, axis=0)

            # Transfer results back to CPU
            intramuscular_emg = cp.asnumpy(intramuscular_emg_gpu)
        else:
            # Fallback to CPU computation with NumPy
            for pool_idx in tqdm(
                range(n_pools),
                desc="Intramuscular EMG (CPU)",
                unit="pool",
            ):
                pool_active_neurons = set(active_neuron_indices[pool_idx])

                for e_idx in range(n_electrodes):
                    # Process all active MUs
                    convolutions = []
                    for mu_idx in MUs_to_simulate.intersection(pool_active_neurons):
                        # Use mu_idx directly since muap_shapes now contains all MUs
                        if mu_idx < muap_shapes.shape[0]:
                            conv = np.correlate(
                                spike_trains[pool_idx, mu_idx],
                                muap_shapes[mu_idx, e_idx],
                                mode="same",
                            )
                            convolutions.append(conv)

                    if convolutions:
                        intramuscular_emg[pool_idx, e_idx] = np.sum(convolutions, axis=0)

        # Temporal resampling
        intramuscular_emg_resampled = np.zeros(
            (
                n_pools,
                n_electrodes,
                int(
                    intramuscular_emg.shape[-1]
                    * spiketrain_timestep__ms.rescale("s").magnitude
                    * self._sampling_frequency__Hz
                ),
            )
        )
        for pool_idx in range(n_pools):
            for e_idx in range(n_electrodes):
                intramuscular_emg_resampled[pool_idx, e_idx] = np.interp(
                    x=np.arange(
                        start=0,
                        stop=intramuscular_emg.shape[-1]
                        * spiketrain_timestep__ms.rescale("s").magnitude,
                        step=1 / self._sampling_frequency__Hz,
                    ),
                    xp=np.arange(
                        start=0,
                        stop=intramuscular_emg.shape[-1]
                        * spiketrain_timestep__ms.rescale("s").magnitude,
                        step=spiketrain_timestep__ms.rescale("s").magnitude,
                    ),
                    fp=intramuscular_emg[pool_idx, e_idx],
                )

        # Create neo Block structure
        block = Block()

        # Create segments for each motor unit pool
        for pool_idx in range(n_pools):
            segment = Segment(name=f"Pool_{pool_idx}")
            block.segments.append(segment)

            # Create AnalogSignal for this pool's EMG data
            segment.analogsignals.append(
                AnalogSignal(
                    intramuscular_emg_resampled[pool_idx].T * pq.dimensionless,
                    t_start=0 * pq.ms,
                    sampling_rate=self._sampling_frequency__Hz * pq.Hz,
                )
            )

        # Store results privately
        self._intramuscular_emg__Block = block
        return block

    def add_noise(self, snr__dB: float, noise_type: str = "gaussian") -> INTRAMUSCULAR_EMG__Block:
        """
        Add noise to the electrode array.

        This method adds realistic noise to the simulated intramuscular EMG signals
        based on a specified signal-to-noise ratio. The noise is calculated
        and applied independently for each electrode channel to ensure that
        channels with different signal amplitudes maintain the specified SNR.

        Parameters
        ----------
        snr__dB : float
            Signal-to-noise ratio in dB. Higher values result in cleaner signals.
            Typical intramuscular EMG has SNR ranging from 15-50 dB.
            The SNR is applied independently to each electrode channel.
        noise_type : str, default="gaussian"
            Type of noise to add. Currently supports "gaussian" for white noise.

        Returns
        -------
        INTRAMUSCULAR_EMG__Block
            Noisy intramuscular EMG signals for the electrode array as a neo.Block.
            Results are stored in the `noisy_intramuscular_emg__Block` property after execution.

        Raises
        ------
        ValueError
            If intramuscular EMG has not been simulated. Call simulate_intramuscular_emg() first.

        Notes
        -----
        The noise is computed per-channel (per electrode) to maintain the specified
        SNR independently across all channels. This ensures that electrodes with
        different signal amplitudes receive appropriately scaled noise.
        """
        if self._intramuscular_emg__Block is None:
            raise ValueError(
                "Intramuscular EMG has not been simulated. Call simulate_intramuscular_emg() first."
            )

        noisy_block = Block()

        for pool_idx, segment in enumerate(self._intramuscular_emg__Block.segments):
            noisy_segment = Segment(name=f"Pool_{pool_idx}")
            noisy_block.segments.append(noisy_segment)

            # Get the EMG signal data
            emg_signal = segment.analogsignals[0]
            emg_array = emg_signal.magnitude  # Shape: (time, n_electrodes)

            # Calculate signal power PER CHANNEL (per electrode)
            # Mean along time axis (axis=0) gives power per electrode
            signal_power_per_channel = np.mean(emg_array**2, axis=0)  # Shape: (n_electrodes,)

            # Calculate noise power per channel
            snr_linear = 10 ** (snr__dB / 10)
            noise_power_per_channel = signal_power_per_channel / snr_linear
            noise_std_per_channel = np.sqrt(noise_power_per_channel)  # Shape: (n_electrodes,)

            # Generate noise
            if noise_type.lower() == "gaussian":
                # Generate standard normal noise, then scale per channel
                noise = RANDOM_GENERATOR.normal(loc=0.0, scale=1.0, size=emg_array.shape)
                # Broadcast noise_std_per_channel along time axis
                # noise shape: (time, n_electrodes)
                # noise_std_per_channel shape: (n_electrodes,)
                # Broadcasting: (time, n_electrodes) * (1, n_electrodes)
                noise = noise * noise_std_per_channel[np.newaxis, :]
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")

            # Add noise
            noisy_emg = emg_array + noise

            # Create new AnalogSignal with noise
            noisy_segment.analogsignals.append(
                AnalogSignal(
                    noisy_emg * emg_signal.units,
                    t_start=emg_signal.t_start,
                    sampling_rate=emg_signal.sampling_rate,
                )
            )

        # Store results privately
        self._noisy_intramuscular_emg__Block = noisy_block
        return noisy_block

    # Property accessors for computed results
    @property
    def muaps__Block(self) -> INTRAMUSCULAR_MUAP__Block:
        """
        Intramuscular MUAP shapes for the electrode array.

        Returns
        -------
        INTRAMUSCULAR_MUAP__Block
            Intramuscular MUAP templates for the electrode array as a neo.Block.

        Raises
        ------
        ValueError
            If MUAP templates have not been computed yet.
        """
        if self._muaps__Block is None:
            raise ValueError("MUAP templates not computed. Call simulate_muaps() first.")
        return self._muaps__Block

    @property
    def intramuscular_emg__Block(self) -> INTRAMUSCULAR_EMG__Block:
        """
        Intramuscular EMG signals for the electrode array.

        Returns
        -------
        INTRAMUSCULAR_EMG__Block
            Intramuscular EMG signals for the electrode array as a neo.Block.

        Raises
        ------
        ValueError
            If intramuscular EMG has not been computed yet.
        """
        if self._intramuscular_emg__Block is None:
            raise ValueError(
                "Intramuscular EMG signals not computed. Call simulate_intramuscular_emg() first."
            )
        return self._intramuscular_emg__Block

    @property
    def noisy_intramuscular_emg__Block(self) -> INTRAMUSCULAR_EMG__Block:
        """
        Noisy intramuscular EMG signals for the electrode array.

        Returns
        -------
        INTRAMUSCULAR_EMG__Block
            Noisy intramuscular EMG signals for the electrode array as a neo.Block.

        Raises
        ------
        ValueError
            If noisy intramuscular EMG has not been computed yet.
        """
        if self._noisy_intramuscular_emg__Block is None:
            raise ValueError(
                "Noisy intramuscular EMG signals not computed. Call add_noise() first."
            )
        return self._noisy_intramuscular_emg__Block

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
            raise ValueError("Spike train block not set. Call simulate_intramuscular_emg() first.")
        return self._spike_train__Block

"""
Intramuscular Electromyography (iEMG) Simulation.

This module provides the main simulation framework for generating intramuscular
electromyography signals using needle electrodes. It integrates motor unit
simulation, electrode modeling, and signal generation with realistic noise.
"""

import logging
import warnings
from typing import Optional

import elephant
import elephant.utils
import numpy as np
import quantities as pq
from neo import AnalogSignal, Block, Group, Segment
from tqdm import tqdm

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.emg.electrodes import IntramuscularElectrodeArray
from myogen.simulator.core.muscle import Muscle
from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    INTRAMUSCULAR_EMG__Block,
    INTRAMUSCULAR_MUAP__Block,
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
    sampling_frequency__Hz : float, default=10240.0
        Sampling frequency in Hz for EMG simulation.
        Default is set to 10240 Hz as used by the Quattrocento (OT Bioelettronica, Turin, Italy) system.
    spatial_resolution__mm : float, default=0.01
        Spatial resolution for fiber action potential calculation in mm.
        Default is set to 0.01 mm.
    endplate_center__percent : float, default=50
        Percentage of muscle length where the endplate is located.
        By default, the endplate is located at the center of the muscle (50% of the muscle length).
    nmj_jitter__s : float, default=35e-6
        Standard deviation of neuromuscular junction jitter in seconds.
        Default is set to 35e-6 s as determined by Konstantin et al. 2020 [1]_.
    branch_cvs__m_per_s : tuple[float, float], default=(5.0, 2.0)
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
        sampling_frequency__Hz: float = 10240.0,
        spatial_resolution__mm: float = 0.01,
        endplate_center__percent: float = 50,
        nmj_jitter__s: float = 35e-6,
        branch_cvs__m_per_s: tuple[float, float] = (5.0, 2.0),
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

        # Private copies for internal modifications
        self._muscle_model = muscle_model
        self._electrode_array = electrode_array
        self._sampling_frequency__Hz = sampling_frequency__Hz
        self._spatial_resolution__mm = spatial_resolution__mm
        self._endplate_center__percent = endplate_center__percent
        self._nmj_jitter__s = nmj_jitter__s
        self._branch_cvs__m_per_s = branch_cvs__m_per_s
        self._MUs_to_simulate = MUs_to_simulate

        # Derived parameters - immutable public access
        self.branch_cvs__mm_per_s = list(
            (
                self._branch_cvs__m_per_s[0] * 1000.0,
                self._branch_cvs__m_per_s[1] * 1000.0,
            )
        )
        self.endplate_center__mm = self._muscle_model.length__mm * (
            self._endplate_center__percent / 100.0
        )

        # Private copies for internal modifications
        self._branch_cvs__mm_per_s: list[float] = list(
            (
                self._branch_cvs__m_per_s[0] * 1000.0,
                self._branch_cvs__m_per_s[1] * 1000.0,
            )
        )
        self._endplate_center__mm = self._muscle_model.length__mm * (
            self._endplate_center__percent / 100.0
        )

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

    def simulate_muaps(self) -> INTRAMUSCULAR_MUAP__Block:
        """
        Simulate MUAPs for all electrode arrays using the provided muscle model.

        This method generates intramuscular Motor Unit Action Potential (MUAP) templates
        by simulating individual motor units with realistic neuromuscular junction
        distributions and fiber action potential propagation.

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
        return self._calculate_muaps()

    def _initialize_motor_units(self) -> None:
        """
        Initialize individual motor unit simulators.

        This method creates MotorUnitSim objects for each motor unit based on
        the muscle model fiber assignments and properties.
        """
        if (
            not hasattr(self._muscle_model, "assignment")
            or self._muscle_model.assignment is None
        ):
            raise ValueError(
                "Muscle model must have fiber assignments. Call muscle.assign_mfs2mns() first."
            )

        self._motor_units = []

        for mu_idx in tqdm(
            self._MUs_to_simulate,
            desc="Creating motor unit simulators",
            unit="Simulator",
        ):
            # Get fibers assigned to this motor unit
            fiber_mask = self._muscle_model.assignment == mu_idx
            if not np.any(fiber_mask):
                continue

            # Create motor unit simulator
            self._motor_units.append(
                MotorUnitSim(
                    muscle_fiber_centers__mm=self._muscle_model.muscle_fiber_centers__mm[
                        fiber_mask
                    ],
                    muscle_length__mm=self._muscle_model.length__mm,
                    muscle_fiber_diameters__mm=self._muscle_model.muscle_fiber_diameters__mm[
                        fiber_mask
                    ],
                    muscle_fiber_conduction_velocity__mm_per_s=self._muscle_model.muscle_fiber_conduction_velocities__mm_per_s[
                        fiber_mask
                    ],
                    neuromuscular_junction_conduction_velocities__mm_per_s=self._branch_cvs__mm_per_s,
                    nominal_center__mm=self._muscle_model.innervation_center_positions__mm[
                        mu_idx
                    ],
                )
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
            tqdm(
                self._motor_units, desc="Setting up NMJ distributions", unit="Simulator"
            )
        ):
            spread_factor = np.sum(
                self._muscle_model.recruitment_thresholds[:mu_idx]
            ) / np.sum(self._muscle_model.recruitment_thresholds)

            # Create NMJ distribution
            # Branch spread increases with motor unit size
            mu_sim.sim_nmj_branches_two_layers(
                n_branches=int(n_branches[mu_idx]),
                endplate_center=self._endplate_center__mm,
                branches_z_std=1.5 + spread_factor * 4.0,
                arborization_z_std=0.5 + spread_factor * 1.5,
            )

    def _calculate_muaps(self) -> INTRAMUSCULAR_MUAP__Block:
        """
        Pre-calculate motor unit action potentials (MUAPs).

        Returns
        -------
        INTRAMUSCULAR_MUAP_SHAPE__TENSOR
            Intramuscular MUAP shapes for all electrode arrays.
        """
        if not self._motor_units:
            raise ValueError("Must call _initialize_motor_units() first")

        # Calculate SFAPs for each motor unit
        for i, mu_sim in enumerate(self._motor_units):
            mu_sim.calc_sfaps(
                index=i,
                dt=self._dt,
                dz=self._dz,
                electrode_positions=self._electrode_array.pts,
            )

        # Calculate MUAPs (no jitter for templates)
        muaps_list: list[np.ndarray] = []
        max_length = 0

        for mu_sim in tqdm(self._motor_units, desc="Computing MUAPs", unit="MU"):
            muap = mu_sim.calc_muap(jitter_std=0.0)  # No jitter for templates
            muaps_list.append(muap)
            max_length = max(max_length, muap.shape[0])

        block = Block()
        for muap in muaps_list:
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

        print(
            f"Found {np.sum(detectable)} detectable motor units out of {len(self._motor_units)}"
        )

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
            raise ValueError(
                "MUAP templates have not been generated. Call simulate_muaps() first."
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
        muap_data_list = [
            seg.analogsignals[0].magnitude for seg in self._muaps__Block.segments
        ]

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
                padded_muap = np.pad(
                    muap, pad_width, mode="constant", constant_values=0
                )
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
        muap_shapes = np.zeros(
            (muap_array.shape[0], muap_array.shape[2], target_length)
        )
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

        muap_shapes /= np.max(np.abs(muap_shapes))  # Normalize MUAP shapes

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
                        # Find the MUAP index in our list of simulated MUs
                        muap_idx = (
                            self._MUs_to_simulate.index(mu_idx)
                            if mu_idx in self._MUs_to_simulate
                            else None
                        )
                        if muap_idx is not None and muap_idx < muap_gpu.shape[0]:
                            conv = cp.correlate(
                                spike_gpu[pool_idx, mu_idx],
                                muap_gpu[muap_idx, e_idx],
                                mode="same",
                            )
                            convolutions.append(conv)

                    convolutions = (
                        cp.array(convolutions) if convolutions else cp.array([])
                    )
                    # Sum across MUAPs on GPU
                    if len(convolutions) > 0:
                        intramuscular_emg_gpu[pool_idx, e_idx] = cp.sum(
                            convolutions, axis=0
                        )

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
                        # Find the MUAP index in our list of simulated MUs
                        muap_idx = (
                            self._MUs_to_simulate.index(mu_idx)
                            if mu_idx in self._MUs_to_simulate
                            else None
                        )
                        if muap_idx is not None and muap_idx < muap_shapes.shape[0]:
                            conv = np.correlate(
                                spike_trains[pool_idx, mu_idx],
                                muap_shapes[muap_idx, e_idx],
                                mode="same",
                            )
                            convolutions.append(conv)

                    if convolutions:
                        intramuscular_emg[pool_idx, e_idx] = np.sum(
                            convolutions, axis=0
                        )

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

    def add_noise(
        self, snr__dB: float, noise_type: str = "gaussian"
    ) -> INTRAMUSCULAR_EMG__Block:
        """
        Add noise to the electrode array.

        This method adds realistic noise to the simulated intramuscular EMG signals
        based on a specified signal-to-noise ratio. The noise characteristics
        can be customized to match different recording conditions.

        Parameters
        ----------
        snr__dB : float
            Signal-to-noise ratio in dB. Higher values result in cleaner signals.
            Typical intramuscular EMG has SNR ranging from 15-50 dB.
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
            emg_array = emg_signal.magnitude

            # Calculate signal power
            signal_power = np.mean(emg_array**2)

            # Calculate noise power
            snr_linear = 10 ** (snr__dB / 10)
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
            raise ValueError(
                "MUAP templates not computed. Call simulate_muaps() first."
            )
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
            raise ValueError(
                "Spike train block not set. Call simulate_intramuscular_emg() first."
            )
        return self._spike_train__Block

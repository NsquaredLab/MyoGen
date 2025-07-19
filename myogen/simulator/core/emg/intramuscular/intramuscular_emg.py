"""
Intramuscular Electromyography (iEMG) Simulation.

This module provides the main simulation framework for generating intramuscular
electromyography signals using needle electrodes. It integrates motor unit
simulation, electrode modeling, and signal generation with realistic noise.

Based on the MATLAB iemg_simulator scripts, particularly s10_cl_generate_emg.m.
"""

import warnings
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt

from myogen import RANDOM_GENERATOR
from myogen.simulator.core.muscle import Muscle
from myogen.simulator.core.spike_train import MotorNeuronPool
from myogen.simulator.core.emg.electrodes import IntramuscularElectrodeArray
from myogen.utils.types import (
    SPIKE_TRAIN__MATRIX,
    INTRAMUSCULAR_EMG__TENSOR,
    beartowertype,
)
from .motor_unit_sim import MotorUnitSim

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@beartowertype
class IntramuscularEMG:
    """
    Intramuscular Electromyography (iEMG) Simulation.

    This class provides a comprehensive simulation framework for generating
    intramuscular EMG signals detected by needle electrodes. It includes:
    - Motor unit action potential (MUAP) pre-calculation
    - Signal generation from spike trains
    - Realistic noise modeling
    - Electrode trajectory simulation
    - Motor unit detectability analysis

    Parameters
    ----------
    muscle_model : Muscle
        Pre-computed muscle model containing fiber positions and motor unit assignments
    electrode_array : IntramuscularElectrodeArray
        Intramuscular electrode array configuration
    sampling_frequency__Hz : float, default=10000.0
        Sampling frequency in Hz for EMG simulation
    spatial_resolution__mm : float, default=0.5
        Spatial resolution for fiber action potential calculation in mm
    muscle_length__mm : float, default=30.0
        Total muscle length in mm
    endplate_center__mm : float, default=15.0
        Center position of the endplate zone in mm (typically muscle length / 2)
    nmj_jitter__s : float, default=35e-6
        Standard deviation of neuromuscular junction jitter in seconds
    branch_cv__mm_s : List[float], default=[5000.0, 2000.0]
        Conduction velocities for neuromuscular junction branches in mm/s
    snr__dB : float, default=20.0
        Signal-to-noise ratio in dB for noise modeling
    MUs_to_simulate : Optional[List[int]], default=None
        Indices of motor units to simulate. If None, all motor units are simulated.
    """

    def __init__(
        self,
        muscle_model: Muscle,
        electrode_array: IntramuscularElectrodeArray,
        sampling_frequency__Hz: float = 10000.0,
        spatial_resolution__mm: float = 0.5,
        muscle_length__mm: float = 30.0,
        endplate_center__mm: float = 15.0,
        nmj_jitter__s: float = 35e-6,
        branch_cv__mm_s: List[float] = [5000.0, 2000.0],
        snr__dB: float = 20.0,
        MUs_to_simulate: Optional[List[int]] = None,
    ):
        self.muscle_model = muscle_model
        self.electrode_array = electrode_array
        self.sampling_frequency__Hz = sampling_frequency__Hz
        self.spatial_resolution__mm = spatial_resolution__mm
        self.muscle_length__mm = muscle_length__mm
        self.endplate_center__mm = endplate_center__mm
        self.nmj_jitter__s = nmj_jitter__s
        self.branch_cv__mm_s = branch_cv__mm_s
        self.snr__dB = snr__dB

        # Derived parameters
        self.dt = 1.0 / sampling_frequency__Hz
        self.dz = spatial_resolution__mm
        self.n_motor_units = len(muscle_model.recruitment_thresholds)

        # Motor unit selection
        if MUs_to_simulate is None:
            self.MUs_to_simulate = list(range(self.n_motor_units))
        else:
            self.MUs_to_simulate = MUs_to_simulate

        # Motor unit simulators
        self.motor_units: List[MotorUnitSim] = []
        self.muaps: Optional[np.ndarray] = None  # Pre-calculated MUAPs
        self.max_muap_length: int = 0

        # Noise parameters
        self.mvc_emg_std: Optional[np.ndarray] = None
        self.emg_noise_std: Optional[np.ndarray] = None

        # Detectability analysis
        self.detectable_mus: Optional[np.ndarray] = None
        self.detectable_indices: Optional[List[int]] = None

        # Simulation results
        self.emg_signals: Optional[INTRAMUSCULAR_EMG__TENSOR] = None
        self.clean_emg_signals: Optional[INTRAMUSCULAR_EMG__TENSOR] = None

    def initialize_motor_units(self):
        """
        Initialize individual motor unit simulators.

        This method creates MotorUnitSim objects for each motor unit based on
        the muscle model fiber assignments and properties.
        """
        if (
            not hasattr(self.muscle_model, "assignment")
            or self.muscle_model.assignment is None
        ):
            raise ValueError(
                "Muscle model must have fiber assignments. Call muscle.assign_mfs2mns() first."
            )

        print("Initializing motor units...")
        self.motor_units = []

        for mu_idx in tqdm(self.MUs_to_simulate, desc="Creating motor unit simulators"):
            # Get fibers assigned to this motor unit
            fiber_mask = self.muscle_model.assignment == mu_idx
            if not np.any(fiber_mask):
                continue

            # Extract fiber properties for this motor unit
            mf_centers = self.muscle_model.mf_centers[fiber_mask]
            mf_diameters = self.muscle_model.mf_diameters[fiber_mask]
            mf_cv = self.muscle_model.mf_cv[fiber_mask]

            # Create motor unit simulator
            mu_sim = MotorUnitSim(
                mf_centers=mf_centers,
                muscle_length=self.muscle_length__mm,
                mf_diameters=mf_diameters,
                mf_cv=mf_cv,
                nmj_cv=self.branch_cv__mm_s,
            )

            # Set nominal center (innervation center)
            mu_sim.nominal_center = self.muscle_model.innervation_center_positions[
                mu_idx
            ]

            self.motor_units.append(mu_sim)

    def simulate_neuromuscular_junctions(self):
        """
        Simulate neuromuscular junction distributions for all motor units.

        This implements the logic from s08_cl_init_muaps.m for generating
        realistic NMJ branch patterns with size-dependent complexity.
        """
        if not self.motor_units:
            raise ValueError("Must call initialize_motor_units() first")

        print("Simulating neuromuscular junctions...")

        for mu_idx, mu_sim in enumerate(
            tqdm(self.motor_units, desc="Setting up NMJ distributions")
        ):
            # Number of branches increases with motor unit size (log relationship)
            relative_size = (mu_idx + 1) / len(self.motor_units)
            n_branches = 1 + int(
                np.log(relative_size + 0.1) * 2
            )  # Simplified relationship
            n_branches = max(1, min(n_branches, 5))  # Limit to reasonable range

            # Branch spread increases with motor unit size
            arborization_z_std = 0.5 + relative_size * 1.5
            branches_z_std = 1.5 + relative_size * 4.0

            # Create NMJ distribution
            mu_sim.sim_nmj_branches_two_layers(
                n_branches=n_branches,
                endplate_center=self.endplate_center__mm,
                branches_z_std=branches_z_std,
                arborization_z_std=arborization_z_std,
            )

    def calculate_muaps(self):
        """
        Pre-calculate motor unit action potentials (MUAPs).

        This implements the core SFAP calculation from s08_cl_init_muaps.m,
        computing MUAPs for all motor units at all electrode positions.
        """
        if not self.motor_units:
            raise ValueError("Must call initialize_motor_units() first")

        print("Calculating MUAPs...")

        # Get electrode positions for all trajectory nodes
        electrode_positions = self._get_electrode_positions()

        # Calculate SFAPs for each motor unit
        for mu_sim in tqdm(self.motor_units, desc="Calculating SFAPs"):
            mu_sim.calc_sfaps(
                dt=self.dt, dz=self.dz, electrode_positions=electrode_positions
            )

        # Calculate MUAPs (no jitter for templates)
        print("Generating MUAP templates...")
        muaps_list = []
        max_length = 0

        for mu_sim in tqdm(self.motor_units, desc="Computing MUAPs"):
            muap = mu_sim.calc_muap(jitter_std=0.0)  # No jitter for templates
            muaps_list.append(muap)
            max_length = max(max_length, muap.shape[0])

        # Pad MUAPs to same length and store
        self.max_muap_length = max_length
        n_electrodes = electrode_positions.shape[0]
        self.muaps = np.zeros((len(self.motor_units), max_length, n_electrodes))

        for i, muap in enumerate(muaps_list):
            self.muaps[i, : muap.shape[0], :] = muap

    def generate_mvc_emg(self, duration__s: float = 5.0) -> np.ndarray:
        """
        Generate maximum voluntary contraction (MVC) EMG for noise reference.

        This implements s09_cl_generate_mvc_emg.m logic to establish noise levels.

        Parameters
        ----------
        duration__s : float, default=5.0
            Duration of MVC simulation in seconds

        Returns
        -------
        np.ndarray
            MVC EMG signal for noise level estimation
        """
        if self.muaps is None:
            raise ValueError("Must call calculate_muaps() first")

        print("Generating MVC EMG for noise reference...")

        # Create MVC spike trains (all units active at high rates)
        n_samples = int(duration__s * self.sampling_frequency__Hz)
        mvc_spikes = np.ones((len(self.motor_units), n_samples), dtype=bool)

        # Generate MVC EMG by convolving spikes with MUAPs
        electrode_positions = self._get_electrode_positions()
        mvc_emg = self._generate_emg_from_spikes(mvc_spikes, use_jitter=False)

        # Calculate noise standard deviation
        self.mvc_emg_std = np.std(mvc_emg[int(self.sampling_frequency__Hz) :], axis=0)
        self.emg_noise_std = self.mvc_emg_std * 10 ** (-self.snr__dB / 20)

        # Make noise level consistent across channels
        self.emg_noise_std = np.mean(self.emg_noise_std) * np.ones_like(
            self.emg_noise_std
        )

        return mvc_emg

    def analyze_detectable_motor_units(self) -> Tuple[np.ndarray, List[int]]:
        """
        Analyze which motor units are detectable by the electrode.

        This implements s11_cl_get_detectable_mus.m logic for determining
        motor unit visibility based on signal-to-noise ratio and contribution.

        Returns
        -------
        Tuple[np.ndarray, List[int]]
            Boolean array of detectable motor units and their indices
        """
        if self.muaps is None or self.emg_noise_std is None:
            raise ValueError("Must call calculate_muaps() and generate_mvc_emg() first")

        print("Analyzing motor unit detectability...")

        detectable = np.zeros(len(self.motor_units), dtype=bool)

        # Prominence criterion: MUAP amplitude vs noise
        over_noise_threshold = 6.0  # 6x noise level

        for i, mu_sim in enumerate(self.motor_units):
            # Get peak MUAP amplitude across all channels
            muap_amplitudes = np.max(np.abs(self.muaps[i]), axis=0)
            max_amplitude = np.max(muap_amplitudes)

            # Check if MUAP is prominent enough above noise
            is_prominent = max_amplitude > over_noise_threshold * np.max(
                self.emg_noise_std
            )

            # Additional criterion: contribution to total signal variance
            # (simplified version of the explained variance criterion)
            relative_size = (i + 1) / len(self.motor_units)
            min_contribution = 0.05  # Minimum 5% contribution
            contributes_enough = relative_size > min_contribution

            detectable[i] = is_prominent and contributes_enough

        self.detectable_mus = detectable
        self.detectable_indices = [i for i, det in enumerate(detectable) if det]

        print(
            f"Found {np.sum(detectable)} detectable motor units out of {len(self.motor_units)}"
        )

        return detectable, self.detectable_indices

    def simulate_emg(
        self,
        spike_trains: SPIKE_TRAIN__MATRIX,
        use_jitter: bool = True,
        add_noise: bool = True,
        electrode_trajectory_parameter: Optional[np.ndarray] = None,
    ) -> INTRAMUSCULAR_EMG__TENSOR:
        """
        Generate intramuscular EMG signals from spike trains.

        This implements the core logic from s10_cl_generate_emg.m.

        Parameters
        ----------
        spike_trains : SPIKE_TRAIN__MATRIX
            Spike trains matrix (n_pools, n_motor_units, n_time_points)
        use_jitter : bool, default=True
            Whether to apply neuromuscular junction jitter
        add_noise : bool, default=True
            Whether to add realistic noise to the signal
        electrode_trajectory_parameter : Optional[np.ndarray], default=None
            Parameter controlling electrode position along trajectory (0 to 1)

        Returns
        -------
        INTRAMUSCULAR_EMG__TENSOR
            Generated EMG signals (n_pools, n_electrodes, n_time_points)
        """
        if self.muaps is None:
            raise ValueError("Must call calculate_muaps() first")

        if add_noise and self.emg_noise_std is None:
            raise ValueError(
                "Must call generate_mvc_emg() first to establish noise levels"
            )

        n_pools, n_motor_units, n_time_points = spike_trains.shape

        # Ensure we don't simulate more units than available
        n_units_to_sim = min(n_motor_units, len(self.motor_units))

        print(f"Generating EMG for {n_pools} pools, {n_units_to_sim} motor units...")

        # Get electrode positions
        electrode_positions = self._get_electrode_positions()
        n_electrodes = electrode_positions.shape[0]

        # Initialize output
        self.emg_signals = np.zeros((n_pools, n_electrodes, n_time_points))
        self.clean_emg_signals = np.zeros((n_pools, n_electrodes, n_time_points))

        # Process each pool
        for pool_idx in tqdm(range(n_pools), desc="Generating EMG signals"):
            # Extract spike trains for this pool
            pool_spikes = spike_trains[pool_idx, :n_units_to_sim, :]

            # Generate clean EMG
            clean_emg = self._generate_emg_from_spikes(
                pool_spikes,
                use_jitter=use_jitter,
                trajectory_parameter=electrode_trajectory_parameter,
            )

            self.clean_emg_signals[pool_idx] = clean_emg.T

            # Add noise if requested
            if add_noise:
                noise = RANDOM_GENERATOR.normal(
                    0, self.emg_noise_std, (n_time_points, n_electrodes)
                )
                noisy_emg = clean_emg + noise
                self.emg_signals[pool_idx] = noisy_emg.T
            else:
                self.emg_signals[pool_idx] = clean_emg.T

        return self.emg_signals

    def _get_electrode_positions(self) -> np.ndarray:
        """Get electrode positions for all trajectory nodes."""
        # For now, return initial positions
        # TODO: Implement full trajectory support
        positions = np.array(self.electrode_array.position__mm)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        # Replicate for each electrode in array
        n_electrodes = self.electrode_array.num_electrodes
        electrode_positions = np.zeros((n_electrodes, 3))

        for i in range(n_electrodes):
            electrode_positions[i] = positions[0]
            # Add offset for each electrode in array
            electrode_positions[i, 2] += (
                i * self.electrode_array.inter_electrode_distance__mm
            )

        return electrode_positions

    def _generate_emg_from_spikes(
        self,
        spike_trains: np.ndarray,
        use_jitter: bool = True,
        trajectory_parameter: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate EMG signal by convolving spike trains with MUAPs.

        Parameters
        ----------
        spike_trains : np.ndarray
            Spike trains (n_motor_units, n_time_points)
        use_jitter : bool
            Whether to apply jitter
        trajectory_parameter : Optional[np.ndarray]
            Trajectory parameter for electrode movement

        Returns
        -------
        np.ndarray
            EMG signal (n_time_points, n_electrodes)
        """
        n_motor_units, n_time_points = spike_trains.shape
        n_electrodes = self.muaps.shape[2]

        # Initialize EMG signal
        emg = np.zeros((n_time_points, n_electrodes))

        # Process each motor unit
        for mu_idx in range(min(n_motor_units, len(self.motor_units))):
            mu_spikes = spike_trains[mu_idx]

            # Find spike times
            spike_times = np.where(mu_spikes)[0]

            if len(spike_times) == 0:
                continue

            # Get MUAP for this motor unit
            muap = self.muaps[mu_idx]
            jitter_std = self.nmj_jitter__s if use_jitter else 0.0

            # Add each spike's contribution
            for spike_time in spike_times:
                # Apply jitter if requested
                if jitter_std > 0:
                    jitter_samples = int(
                        RANDOM_GENERATOR.normal(0, jitter_std) / self.dt
                    )
                    actual_spike_time = spike_time + jitter_samples
                else:
                    actual_spike_time = spike_time

                # Add MUAP to signal
                start_idx = max(0, actual_spike_time)
                end_idx = min(n_time_points, actual_spike_time + muap.shape[0])
                muap_start = max(0, -actual_spike_time)
                muap_end = muap_start + (end_idx - start_idx)

                if start_idx < end_idx and muap_start < muap.shape[0]:
                    emg[start_idx:end_idx] += muap[muap_start:muap_end]

        return emg

    def plot_muaps(
        self, mu_indices: Optional[List[int]] = None, electrode_idx: int = 0
    ):
        """
        Plot motor unit action potentials.

        Parameters
        ----------
        mu_indices : Optional[List[int]]
            Motor unit indices to plot. If None, plot first 10.
        electrode_idx : int, default=0
            Electrode channel to plot
        """
        if self.muaps is None:
            raise ValueError("Must call calculate_muaps() first")

        if mu_indices is None:
            mu_indices = list(range(min(10, len(self.motor_units))))

        plt.figure(figsize=(12, 8))
        time_axis = np.arange(self.muaps.shape[1]) * self.dt * 1000  # Convert to ms

        for i, mu_idx in enumerate(mu_indices):
            plt.subplot(len(mu_indices), 1, i + 1)
            plt.plot(time_axis, self.muaps[mu_idx, :, electrode_idx])
            plt.title(f"Motor Unit {mu_idx + 1} MUAP")
            plt.ylabel("Amplitude")
            if i == len(mu_indices) - 1:
                plt.xlabel("Time (ms)")

        plt.tight_layout()
        plt.show()

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary of simulation parameters and results.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary
        """
        summary = {
            "sampling_frequency_Hz": self.sampling_frequency__Hz,
            "spatial_resolution_mm": self.spatial_resolution__mm,
            "muscle_length_mm": self.muscle_length__mm,
            "n_motor_units_total": self.n_motor_units,
            "n_motor_units_simulated": len(self.motor_units),
            "n_electrodes": self.electrode_array.num_electrodes,
            "snr_dB": self.snr__dB,
            "nmj_jitter_s": self.nmj_jitter__s,
        }

        if self.detectable_indices:
            summary["n_detectable_motor_units"] = len(self.detectable_indices)
            summary["detectable_motor_units"] = self.detectable_indices

        if self.muaps is not None:
            summary["max_muap_length_samples"] = self.max_muap_length
            summary["max_muap_duration_ms"] = self.max_muap_length * self.dt * 1000

        return summary

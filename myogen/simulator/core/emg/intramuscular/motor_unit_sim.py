"""
Motor Unit Simulation for Intramuscular EMG.

This module implements individual motor unit simulation including neuromuscular
junction modeling, single fiber action potential (SFAP) calculation, and
motor unit action potential (MUAP) generation with realistic jitter.

Based on the MU_Sim class from the MATLAB iemg_simulator.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial.distance import cdist
from myogen import RANDOM_GENERATOR
from .bioelectric import (
    get_current_density,
    get_elementary_current_response,
    calculate_sfap,
)


class MotorUnitSim:
    """
    Simulation of individual motor unit for intramuscular EMG.

    This class handles the simulation of a single motor unit including:
    - Muscle fiber spatial distribution
    - Neuromuscular junction positioning and timing
    - Single fiber action potential (SFAP) calculation
    - Motor unit action potential (MUAP) generation with jitter

    Parameters
    ----------
    mf_centers : np.ndarray
        Muscle fiber center positions (N × 3) in mm [x, y, z]
    muscle_length : float
        Total muscle length in mm
    mf_diameters : np.ndarray
        Muscle fiber diameters in mm (N,)
    mf_cv : np.ndarray
        Muscle fiber conduction velocities in mm/s (N,)
    nmj_cv : List[float]
        Neuromuscular junction branch conduction velocities in mm/s
    """

    def __init__(
        self,
        mf_centers: np.ndarray,
        muscle_length: float,
        mf_diameters: np.ndarray,
        mf_cv: np.ndarray,
        nmj_cv: List[float] = [5000.0, 2000.0],
    ):
        self.mf_centers = np.asarray(mf_centers)
        self.Nmf = len(mf_centers)
        self.muscle_length = muscle_length
        self.mf_diameters = np.asarray(mf_diameters)
        self.mf_cv = np.asarray(mf_cv)
        self.nmj_cv = nmj_cv

        # Initialize fiber end positions
        self.mf_left_end = np.zeros(self.Nmf)
        self.mf_right_end = np.full(self.Nmf, muscle_length)

        # Neuromuscular junction properties
        self.nmj_z: Optional[np.ndarray] = None  # Will be set by sim_nmj_branches
        self.nmj_delays: Optional[np.ndarray] = None
        self.branch_points_xy: Optional[List] = None
        self.branch_points_z: Optional[List] = None
        self.nerve_paths: Optional[np.ndarray] = None

        # Simulation results
        self.sfaps: Optional[np.ndarray] = None  # Single fiber action potentials
        self.muap: Optional[np.ndarray] = None  # Motor unit action potential

        # Simulation parameters
        self.dt: Optional[float] = None
        self.dz: Optional[float] = None
        self.Npt: Optional[int] = None  # Number of electrode points

        # Centers
        self.nominal_center: Optional[np.ndarray] = None
        self.actual_center = np.mean(mf_centers, axis=0)

    def sim_nmj_branches_two_layers(
        self,
        n_branches: int,
        endplate_center: float,
        branches_z_std: float,
        arborization_z_std: float,
    ):
        """
        Simulate neuromuscular junction branches using two-layer model.

        This creates a realistic distribution of neuromuscular junctions
        with primary branches and secondary arborizations.

        Parameters
        ----------
        n_branches : int
            Number of primary branches
        endplate_center : float
            Center position of endplate zone in mm
        branches_z_std : float
            Standard deviation of primary branch distribution in mm
        arborization_z_std : float
            Standard deviation of secondary arborization in mm
        """
        rng = RANDOM_GENERATOR

        # Primary branch positions
        primary_branches_z = rng.normal(endplate_center, branches_z_std, n_branches)

        # Assign fibers to branches
        fibers_per_branch = self.Nmf // n_branches
        remaining_fibers = self.Nmf % n_branches

        self.nmj_z = np.zeros(self.Nmf)
        self.branch_points_z = []
        self.branch_points_xy = []
        self.nerve_paths = np.zeros((self.Nmf, 2))  # Two segments: axon + branch

        fiber_idx = 0
        for branch_idx in range(n_branches):
            # Number of fibers for this branch
            n_fibers_this_branch = fibers_per_branch + (
                1 if branch_idx < remaining_fibers else 0
            )

            if n_fibers_this_branch == 0:
                continue

            # Primary branch position
            branch_z = primary_branches_z[branch_idx]
            self.branch_points_z.append(branch_z)

            # Secondary arborization positions for fibers
            fiber_nmj_positions = rng.normal(
                branch_z, arborization_z_std, n_fibers_this_branch
            )

            # Assign to fibers
            for i in range(n_fibers_this_branch):
                if fiber_idx < self.Nmf:
                    self.nmj_z[fiber_idx] = fiber_nmj_positions[i]

                    # Calculate nerve path lengths (simplified)
                    # Path 1: From spinal cord to branch point
                    branch_distance = np.sqrt(
                        (self.mf_centers[fiber_idx, 0] - self.actual_center[0]) ** 2
                        + (self.mf_centers[fiber_idx, 1] - self.actual_center[1]) ** 2
                        + (branch_z - endplate_center) ** 2
                    )

                    # Path 2: From branch point to NMJ
                    nmj_distance = np.sqrt(
                        (self.mf_centers[fiber_idx, 0] - self.mf_centers[fiber_idx, 0])
                        ** 2
                        + (
                            self.mf_centers[fiber_idx, 1]
                            - self.mf_centers[fiber_idx, 1]
                        )
                        ** 2
                        + (fiber_nmj_positions[i] - branch_z) ** 2
                    )

                    self.nerve_paths[fiber_idx, 0] = branch_distance
                    self.nerve_paths[fiber_idx, 1] = nmj_distance

                    fiber_idx += 1

        # Calculate delays
        self._calculate_nmj_delays()

    def sim_nmj_branches_gaussian(self, endplate_center: float, branches_z_std: float):
        """
        Simulate neuromuscular junctions with simple Gaussian distribution.

        Parameters
        ----------
        endplate_center : float
            Center of endplate zone in mm
        branches_z_std : float
            Standard deviation of NMJ distribution in mm
        """
        rng = RANDOM_GENERATOR
        self.nmj_z = rng.normal(endplate_center, branches_z_std, self.Nmf)

        # Simplified nerve paths (single segment)
        self.nerve_paths = np.zeros((self.Nmf, 1))
        for i in range(self.Nmf):
            distance = np.sqrt(
                (self.mf_centers[i, 0] - self.actual_center[0]) ** 2
                + (self.mf_centers[i, 1] - self.actual_center[1]) ** 2
                + (self.nmj_z[i] - endplate_center) ** 2
            )
            self.nerve_paths[i, 0] = distance

        self._calculate_nmj_delays()

    def _calculate_nmj_delays(self):
        """Calculate neuromuscular junction propagation delays."""
        if self.nerve_paths is None:
            return

        self.nmj_delays = np.zeros(self.Nmf)

        for i in range(self.Nmf):
            total_delay = 0.0
            for segment_idx in range(self.nerve_paths.shape[1]):
                path_length = self.nerve_paths[i, segment_idx]
                if segment_idx < len(self.nmj_cv):
                    cv = self.nmj_cv[segment_idx]
                else:
                    cv = self.nmj_cv[-1]  # Use last velocity for additional segments
                total_delay += path_length / cv

            self.nmj_delays[i] = total_delay

    def calc_sfaps(
        self,
        dt: float,
        dz: float,
        electrode_positions: np.ndarray,
        electrode_normals: Optional[np.ndarray] = None,
        min_radial_dist: Optional[float] = None,
    ):
        """
        Calculate single fiber action potentials (SFAPs) for all fibers.

        Parameters
        ----------
        dt : float
            Time step in seconds
        dz : float
            Spatial step in mm
        electrode_positions : np.ndarray
            Electrode positions (N_electrodes × 3) in mm
        electrode_normals : np.ndarray, optional
            Electrode normal vectors (not used for point electrodes)
        min_radial_dist : float, optional
            Minimum radial distance for stability (default: mean diameter * 1000)
        """
        self.dt = dt
        self.dz = dz
        self.Npt = len(electrode_positions)

        if min_radial_dist is None:
            min_radial_dist = float(
                np.mean(self.mf_diameters) * 1000
            )  # Convert to micrometers

        # Check that nmj_z is set
        if self.nmj_z is None:
            raise ValueError(
                "Must call sim_nmj_branches_* method first to set neuromuscular junction positions"
            )

        # Calculate maximum simulation time needed
        max_time_1 = float(np.max((self.nmj_z - self.mf_left_end) / self.mf_cv))
        max_time_2 = float(np.max((self.mf_right_end - self.nmj_z) / self.mf_cv))
        max_propagation_time = 2 * max(max_time_1, max_time_2)

        max_delay = (
            float(np.max(self.nmj_delays)) if self.nmj_delays is not None else 0.0
        )
        t_max = max_propagation_time + max_delay
        t = np.arange(0, t_max + dt, dt)

        # Initialize SFAP storage: (time, electrodes, fibers)
        self.sfaps = np.zeros((len(t), self.Npt, self.Nmf))

        # Calculate SFAPs for each fiber at each electrode
        for fiber_idx in range(self.Nmf):
            for electrode_idx in range(self.Npt):
                # Calculate radial distance from fiber to electrode
                fiber_pos = self.mf_centers[fiber_idx]
                electrode_pos = electrode_positions[electrode_idx]

                r_distance = np.sqrt(
                    (fiber_pos[0] - electrode_pos[0]) ** 2
                    + (fiber_pos[1] - electrode_pos[1]) ** 2
                )
                r_distance = max(
                    r_distance, min_radial_dist * 1e-3
                )  # Convert back to mm

                # Fiber parameters
                fiber_length_L1 = self.mf_right_end[fiber_idx] - self.nmj_z[fiber_idx]
                fiber_length_L2 = self.nmj_z[fiber_idx] - self.mf_left_end[fiber_idx]

                # Create spatial grid along fiber
                z_min = self.mf_left_end[fiber_idx]
                z_max = self.mf_right_end[fiber_idx]
                z_fiber = np.arange(z_min, z_max + dz, dz)

                # Calculate current density
                current_density = get_current_density(
                    t,
                    z_fiber,
                    self.nmj_z[fiber_idx],
                    fiber_length_L1,
                    fiber_length_L2,
                    self.mf_cv[fiber_idx],
                    self.mf_diameters[fiber_idx],
                )

                # Calculate volume conductor response
                h_response = get_elementary_current_response(
                    z_fiber, electrode_pos[2], np.full_like(z_fiber, r_distance)
                )

                # Convolve to get SFAP
                sfap = np.zeros(len(t))
                for z_idx, z_pos in enumerate(z_fiber):
                    sfap += current_density[z_idx, :] * h_response[z_idx] * dz

                # Apply neuromuscular junction delay if present
                if self.nmj_delays is not None:
                    delay_samples = int(self.nmj_delays[fiber_idx] / dt)
                    if delay_samples > 0 and delay_samples < len(sfap):
                        sfap_delayed = np.zeros_like(sfap)
                        sfap_delayed[delay_samples:] = sfap[:-delay_samples]
                        sfap = sfap_delayed

                self.sfaps[:, electrode_idx, fiber_idx] = sfap

    def calc_muap(self, jitter_std: float = 0.0) -> np.ndarray:
        """
        Calculate motor unit action potential (MUAP) with optional jitter.

        Parameters
        ----------
        jitter_std : float, default=0.0
            Standard deviation of neuromuscular junction jitter in seconds

        Returns
        -------
        np.ndarray
            MUAP signal (time × electrodes)
        """
        if self.sfaps is None:
            raise ValueError("Must call calc_sfaps() first")

        if self.dt is None:
            raise ValueError("dt not set - call calc_sfaps() first")

        rng = RANDOM_GENERATOR
        n_time, n_electrodes, n_fibers = self.sfaps.shape

        # Initialize MUAP
        muap = np.zeros((n_time, n_electrodes))

        # Add each fiber's contribution with jitter
        for fiber_idx in range(n_fibers):
            # Apply jitter if specified
            if jitter_std > 0:
                jitter_delay = rng.normal(0, jitter_std)
                jitter_samples = int(jitter_delay / self.dt)
            else:
                jitter_samples = 0

            # Add fiber SFAP to MUAP with jitter
            for electrode_idx in range(n_electrodes):
                sfap = self.sfaps[:, electrode_idx, fiber_idx]

                if jitter_samples != 0:
                    # Apply jitter by shifting
                    sfap_jittered = np.zeros_like(sfap)
                    if jitter_samples > 0 and jitter_samples < len(sfap):
                        sfap_jittered[jitter_samples:] = sfap[:-jitter_samples]
                    elif jitter_samples < 0 and abs(jitter_samples) < len(sfap):
                        sfap_jittered[:jitter_samples] = sfap[-jitter_samples:]
                    else:
                        sfap_jittered = sfap
                    sfap = sfap_jittered

                muap[:, electrode_idx] += sfap

        self.muap = muap
        return muap

    def get_muap_duration(self, threshold_fraction: float = 0.1) -> float:
        """
        Get MUAP duration based on threshold crossing.

        Parameters
        ----------
        threshold_fraction : float, default=0.1
            Fraction of peak amplitude to use as threshold

        Returns
        -------
        float
            MUAP duration in seconds
        """
        if self.muap is None or self.dt is None:
            return 0.0

        # Use first electrode channel
        signal = self.muap[:, 0]
        peak_amplitude = np.max(np.abs(signal))
        threshold = threshold_fraction * peak_amplitude

        # Find first and last threshold crossings
        above_threshold = np.abs(signal) > threshold
        if not np.any(above_threshold):
            return 0.0

        start_idx = np.where(above_threshold)[0][0]
        end_idx = np.where(above_threshold)[0][-1]

        return (end_idx - start_idx) * self.dt

    def get_muap_amplitude(self, electrode_idx: int = 0) -> float:
        """
        Get peak-to-peak MUAP amplitude.

        Parameters
        ----------
        electrode_idx : int, default=0
            Electrode index to analyze

        Returns
        -------
        float
            Peak-to-peak amplitude
        """
        if self.muap is None:
            return 0.0

        signal = self.muap[:, electrode_idx]
        return float(np.max(signal) - np.min(signal))

    @property
    def fiber_count(self) -> int:
        """Number of muscle fibers in this motor unit."""
        return self.Nmf

    @property
    def territory_center(self) -> np.ndarray:
        """Center of motor unit territory."""
        return self.actual_center

    @property
    def territory_radius(self) -> float:
        """Approximate radius of motor unit territory."""
        distances = cdist([self.actual_center[:2]], self.mf_centers[:, :2])
        return float(np.mean(distances))

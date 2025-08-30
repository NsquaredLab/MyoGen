"""
Motor Unit Simulation for Intramuscular EMG.

This module implements individual motor unit simulation including neuromuscular
junction modeling, single fiber action potential (SFAP) calculation, and
motor unit action potential (MUAP) generation with realistic jitter.

Based on the MU_Sim class from the MATLAB iemg_simulator.
"""

from typing import Optional, List

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from tqdm import tqdm

from myogen import RANDOM_GENERATOR, SEED
from myogen.utils.decorators import beartowertype
from .bioelectric import (
    get_current_density,
    get_elementary_current_response,
    shift_padding,
    hr_shift_template,
)


@beartowertype
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
    muscle_fiber_centers__mm : np.ndarray
        Muscle fiber center positions (N × 3) in mm [x, y, z].
    muscle_length__mm : float
        Total muscle length in mm.
    muscle_fiber_diameters__mm : np.ndarray
        Muscle fiber diameters in mm (N,).
    muscle_fiber_conduction_velocity__mm_per_s : np.ndarray
        Muscle fiber conduction velocities in mm/s (N,).
    neuromuscular_junction_conduction_velocities__mm_per_s : List[float]
        Neuromuscular junction branch conduction velocities in mm/s.
    nominal_center__mm : np.ndarray, optional
        Nominal center of the motor unit in mm [x, y]. This is the target center of the motor unit,
        which is used to calculate the nerve paths.
    """

    def __init__(
        self,
        muscle_fiber_centers__mm: np.ndarray,
        muscle_length__mm: float,
        muscle_fiber_diameters__mm: np.ndarray,
        muscle_fiber_conduction_velocity__mm_per_s: np.ndarray,
        nominal_center__mm: np.ndarray,
        neuromuscular_junction_conduction_velocities__mm_per_s: list[float] = [
            5000.0,
            2000.0,
        ],
    ):
        self.muscle_fiber_centers__mm = muscle_fiber_centers__mm
        self.muscle_length__mm = muscle_length__mm
        self.muscle_fiber_diameters__mm = muscle_fiber_diameters__mm[..., None]
        self.muscle_fiber_conduction_velocity__mm_per_s = (
            muscle_fiber_conduction_velocity__mm_per_s[..., None]
        )
        self.neuromuscular_junction_conduction_velocities__mm_per_s = (
            neuromuscular_junction_conduction_velocities__mm_per_s
        )
        self.nominal_center__mm = nominal_center__mm

        self._number_of_muscle_fibers = len(muscle_fiber_centers__mm)

        # Initialize fiber end positions
        self._muscle_fiber_left_ends__mm = np.zeros(
            shape=(self._number_of_muscle_fibers, 1)
        )  # coordinates of muscle fibers left ends
        self._muscle_fiber_right_ends__mm = np.full(
            fill_value=muscle_length__mm, shape=(self._number_of_muscle_fibers, 1)
        )  # coordinates of muscle fibers right ends

        # Neuromuscular junction properties
        self._neuromuscular_z_coordinates__mm: Optional[np.ndarray] = None
        self._neuromuscular_delays: Optional[np.ndarray] = None
        self._branch_points_xy__mm: Optional[List] = None
        self._branch_points_z__mm: Optional[List] = None
        self._nerve_paths: Optional[np.ndarray] = None

        # Simulation results
        self._sfaps: Optional[np.ndarray] = None  # Single fiber action potentials
        self._muap: Optional[np.ndarray] = None  # Motor unit action potential

        # Simulation parameters
        self._dt: Optional[float] = None
        self._dz: Optional[float] = None
        self._number_of_electrode_points: Optional[int] = (
            None  # Number of electrode points
        )

        # Centers
        self._actual_center = np.mean(muscle_fiber_centers__mm, axis=0)

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

        self.nerve_paths = np.zeros(
            (self._number_of_muscle_fibers, 2)
        )  # Point coordinates

        kmeans = KMeans(
            n_clusters=n_branches, init="k-means++", max_iter=100, random_state=SEED
        )
        idx = kmeans.fit_predict(self.muscle_fiber_centers__mm)
        c = kmeans.cluster_centers_

        self.branch_points_xy = c
        self.branch_points_z = endplate_center + branches_z_std * rng.standard_normal(
            size=(n_branches, 1)
        )

        self._neuromuscular_z_coordinates__mm = np.array(
            [
                self.branch_points_z[idx[i]]
                + arborization_z_std * rng.standard_normal()
                for i in range(self._number_of_muscle_fibers)
            ]
        )

        self._actual_center = np.concatenate(
            [
                np.mean(self.muscle_fiber_centers__mm, axis=0),
                np.mean(self._neuromuscular_z_coordinates__mm, axis=0),
            ]
        )[None]
        for i in range(self._number_of_muscle_fibers):
            cluster_center = np.concatenate(
                [
                    self.branch_points_xy[idx[i]],
                    self._neuromuscular_z_coordinates__mm[idx[i]],
                ]
            )[None]
            nmj_coordinates = np.concatenate(
                [
                    self.muscle_fiber_centers__mm[i],
                    self._neuromuscular_z_coordinates__mm[i],
                ]
            )[None]

            self.nerve_paths[i, 0] = np.linalg.norm(
                self._actual_center - cluster_center, axis=-1
            )
            self.nerve_paths[i, 1] = np.linalg.norm(
                nmj_coordinates - cluster_center, axis=-1
            )

        # Calculate delays
        # self._calculate_nmj_delays()

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
                (self.muscle_fiber_centers__mm[i, 0] - self.actual_center[0]) ** 2
                + (self.muscle_fiber_centers__mm[i, 1] - self.actual_center[1]) ** 2
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
        index: int,
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
        self.Npt = electrode_positions.shape[0]

        if min_radial_dist is None:
            min_radial_dist = float(
                np.mean(self.muscle_fiber_diameters__mm) * 1000
            )  # Convert to micrometers

        if self._neuromuscular_z_coordinates__mm is None:
            raise ValueError(
                "Must call sim_nmj_branches_* method first to set neuromuscular junction positions"
            )

        t = np.arange(
            start=0,
            stop=2
            * np.max(
                [
                    np.divide(
                        self._neuromuscular_z_coordinates__mm
                        - self._muscle_fiber_left_ends__mm,
                        self.muscle_fiber_conduction_velocity__mm_per_s,
                    ),
                    np.divide(
                        self._muscle_fiber_right_ends__mm
                        - self._neuromuscular_z_coordinates__mm,
                        self.muscle_fiber_conduction_velocity__mm_per_s,
                    ),
                ]
            )
            + dt,
            step=dt,
        )[..., None]
        self.sfaps = np.zeros((len(t), self.Npt, self._number_of_muscle_fibers))

        for fiber_idx in tqdm(
            range(self._number_of_muscle_fibers),
            desc=f"MU {index}: Calculating SFAPs",
            unit="fiber",
        ):
            z_left = np.arange(
                start=self._neuromuscular_z_coordinates__mm[fiber_idx],
                step=-dz,
                stop=self._muscle_fiber_left_ends__mm[fiber_idx] - dz,
            )
            z_right = np.arange(
                start=self._neuromuscular_z_coordinates__mm[fiber_idx],
                step=dz,
                stop=self._muscle_fiber_right_ends__mm[fiber_idx] + dz,
            )
            z = np.concatenate((z_left[::-1], z_right[1:]))[:, None]
            mf_coord_3d = np.concatenate(
                [
                    np.matlib.repmat(
                        a=self.muscle_fiber_centers__mm[fiber_idx], m=len(z), n=1
                    ),
                    z,
                ],
                axis=1,
            )

            current_density = get_current_density(
                t,
                z,
                self._neuromuscular_z_coordinates__mm[fiber_idx],
                self._muscle_fiber_right_ends__mm[fiber_idx]
                - self._neuromuscular_z_coordinates__mm[fiber_idx],
                self._neuromuscular_z_coordinates__mm[fiber_idx]
                - self._muscle_fiber_left_ends__mm[fiber_idx],
                self.muscle_fiber_conduction_velocity__mm_per_s[fiber_idx],
                self.muscle_fiber_diameters__mm[fiber_idx],
            )

            for electrode_idx in range(self.Npt):
                # Calculate radial distance from fiber to electrode
                radial_distance = np.sqrt(
                    np.sum(
                        (
                            electrode_positions[electrode_idx, :2]
                            - self.muscle_fiber_centers__mm[fiber_idx]
                        )
                        ** 2,
                        keepdims=True,
                    )
                )
                if radial_distance < min_radial_dist:
                    radial_distance = min_radial_dist

                response_to_elem_current = get_elementary_current_response(
                    z,
                    electrode_positions[electrode_idx, 2],
                    radial_distance,
                )

                self.sfaps[:, electrode_idx, fiber_idx] = (
                    current_density.T @ response_to_elem_current
                )[:, 0]

        self.shift_sfaps(dt)

    def calc_mnap_delays(self):
        self.mnap_delays = np.divide(
            self.nerve_paths,
            np.matlib.repmat(
                np.array(self.neuromuscular_junction_conduction_velocities__mm_per_s)[
                    None
                ],
                self._number_of_muscle_fibers,
                1,
            ),
        ).sum(axis=1, keepdims=True)

    def shift_sfaps(self, dt):
        self.calc_mnap_delays()

        for fb in range(self._number_of_muscle_fibers):
            for pt in range(self.Npt):
                self.sfaps[:, pt, fb] = shift_padding(
                    self.sfaps[:, pt, fb],
                    int(np.floor(self.mnap_delays[fb] / dt)),
                    axis=0,
                )
                self.sfaps[:, pt, fb] = hr_shift_template(
                    self.sfaps[:, pt, fb], int(np.mod(self.mnap_delays[fb], dt))
                )

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
            raise ValueError("_dt not set - call calc_sfaps() first")

        if jitter_std != 0:
            delays = jitter_std * RANDOM_GENERATOR.standard_normal(
                size=(self._number_of_muscle_fibers, 1)
            )
            jittered_sfaps = np.zeros_like(self.sfaps)
            for fiber_idx in range(self._number_of_muscle_fibers):
                for electrode_idx in range(self.Npt):
                    jittered_sfaps[:, electrode_idx, fiber_idx] = hr_shift_template(
                        self.sfaps[:, electrode_idx, fiber_idx],
                        delays[fiber_idx] / self.dt,
                    )

                self.muap = np.sum(jittered_sfaps, axis=2)
        else:
            self.muap = np.sum(self.sfaps, axis=2)

        return self.muap

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
        distances = cdist(
            [self.actual_center[:2]], self.muscle_fiber_centers__mm[:, :2]
        )
        return float(np.mean(distances))

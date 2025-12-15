import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import skfmm
from joblib import Parallel, delayed
from scipy.integrate import dblquad
from scipy.stats import chi2, multivariate_normal
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import quantities as pq

from myogen import RANDOM_GENERATOR
from myogen.utils.types import (
    RECRUITMENT_THRESHOLDS__ARRAY,
    Quantity__S_per_m,
    Quantity__m_per_s,
    Quantity__mm,
    Quantity__mm2,
    Quantity__mm_per_s,
    Quantity__per_mm2,
)
from myogen.utils.decorators import beartowertype


def _perform_fast_marching(speed_map: np.ndarray, seed_points: np.ndarray) -> np.ndarray:
    """
    Perform fast marching using scikit-fmm to compute distance maps.

    This function implements the Fast Marching Method to solve the Eikonal equation,
    which is used to distribute innervation centers optimally within the muscle cross-section.
    The method ensures that innervation centers are spaced as far apart as possible from
    each other, mimicking the natural distribution of motor unit territories.

    Parameters
    ----------
    speed_map : np.ndarray
        2D speed map (inverse of a density map) defining the propagation speed
        at each grid point. Higher values indicate faster propagation.
        Should be > 1e-10 for valid regions and ≤ 1e-10 for invalid regions.
    seed_points : np.ndarray
        Seed points as 2×N array where each column is a point [x, y].
        Uses 1-based indexing like MATLAB. These are the starting points
        for the distance computation.

    Returns
    -------
    np.ndarray
        Distance map from seed points. Each element represents the minimum
        distance to any of the seed points. Invalid regions (outside the
        circular muscle boundary) are set to -1e10.

    Notes
    -----
    This function is used internally by the muscle distribution algorithm
    to implement a greedy approach for placing innervation centers such that
    each new electrode_grid_center is placed at the location farthest from all previously
    placed centers.
    """
    # Create a mask for valid regions (inside the circular domain)
    valid_mask = speed_map > 1e-10

    # Create a signed distance function for the domain
    # Initialize with large positive values (far from boundary)
    phi = np.ones_like(speed_map) * 1000.0

    # Set seed points to 0 (starting points for Fast Marching)
    for i in range(seed_points.shape[1]):
        x, y = int(seed_points[0, i] - 1), int(seed_points[1, i] - 1)
        # Ensure indices are within bounds
        if 0 <= x < speed_map.shape[0] and 0 <= y < speed_map.shape[1]:
            phi[x, y] = 0.0  # Starting points

    # Set invalid regions (outside circle) to negative values
    phi[~valid_mask] = -1000.0

    # Use scikit-fmm to solve the Eikonal equation
    distance = skfmm.distance(phi, dx=1.0)

    # Set invalid regions to very small values so they won't be selected
    distance[~valid_mask] = -1e10

    return distance


@beartowertype
class Muscle:
    """
    A muscle model based on the cylindrical description of the volume conductor by Farina et al. 2004 [1]_ and the motor unit distribution by Konstantin et al. 2020 [2]_.

    .. note::
        All default values are set to simulate the First Dorsal Interosseous (FDI) muscle. Values are pulled from the literature.

    Parameters
    ----------
    recruitment_thresholds : RECRUITMENT_THRESHOLDS__ARRAY
        Array of recruitment thresholds for each motor unit (see `myogen.simulator.generate_mu_recruitment_thresholds`).
        Values range from 0 to 1 with the largest motor units having thresholds near 1.
    radius__mm : float, default=6.91
        Radius of the muscle cross-section in millimeters. Default is set to 6.91 mm as determined by Jacobson et al. 1992 [3]_.
    length__mm : float, default=30.0
        Length of the muscle in millimeters. Default is set to 30.0 mm as determined by no one.
    fiber_density__fibers_per_mm2 : float, default=350
        Density of muscle fibers per square millimeter. Default is set to 350 fibers/mm² as determined by Bettelho et al. 2019 [7]_.
    max_innervation_area_to_total_muscle_area__ratio : float, default=0.25
        Ratio defining the maximum territory size relative to total muscle area.
        Default is set to 0.25 as determined by no one but it is a good starting point.
        A value of 0.25 means the largest motor unit can innervate up to 25%
        of the total muscle cross-sectional area.
        Must be in range (0, 1].
    mean_conduction_velocity__m_per_s : float, default=4.2
        Mean conduction velocity in m/s. Default is set to 4.2 m/s as determined by Nishizono et al. 1990 [4]_.
        Experimental range determined by Nishizono et al. 1990 [4]_ is between 3.2 and 5.0 m/s.
    mean_fiber_length__mm : float, default=31.7
        Mean fiber length in mm. Default is set to 31.7 mm as determined by Jacobson et al. 1992 [3]_ (Table 1).
    var_fiber_length__mm : float, default=2.8
        Fiber length variance in mm. Default is set to 2.8 mm as determined by Jacobson et al. 1992 [3]_ (Table 1).
    radius_bone__mm : float, default=1
        Bone radius in mm. Default is set to 1 mm.
    fat_thickness__mm : float, default=0.3
        Fat thickness in mm. Default is set to 0.3 mm as determined by Störchle et al. 2018 [5]_.
    skin_thickness__mm : float, default=1.29
        Skin thickness in mm. Default is set to the male skin thickness average of 1.29 mm as determined by Brodar 1960 [6]_.
    muscle_conductivity_radial__S_m : float, default=0.09
        Muscle conductivity in radial direction. Default is set to 0.09 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    muscle_conductivity_longitudinal__S_m : float, default=0.4
        Muscle conductivity in longitudinal direction. Default is set to 0.4 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    fat_conductivity__S_per_m : float, default=4.07E-2
        Fat conductivity. Default is set to 4.07E-2 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    skin_conductivity__S_per_m : float, default=4.88E-4
        Skin conductivity. Default is set to 4.88E-4 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    grid_resolution : int, default=256
        Resolution of the computational grid used for innervation the muscle.
        Higher values provide more accurate spatial distribution but increase computational cost.
        Default is set to 256.
    autorun : bool, default=False
        If True, automatically executes the complete muscle simulation pipeline: innervation distribution, muscle fiber generation, and fiber-to-motor unit assignment.
        If False, these steps must be called manually.

    Attributes
    ----------
    innervation_center_positions__mm : np.ndarray
        Motor unit innervation center positions [x, y] in mm. Available after distribute_innervation_centers().
    muscle_fiber_centers__mm : np.ndarray
        Muscle fiber center positions [x, y] in mm. Available after generate_muscle_fiber_centers().
    muscle_fiber_diameters__mm : np.ndarray
        Muscle fiber diameters in mm. Available after _generate_fiber_properties().
    muscle_fiber_conduction_velocities__mm_per_s : np.ndarray
        Muscle fiber conduction velocities in mm/s. Available after _generate_fiber_properties().
    assignment : np.ndarray
        Motor unit assignment for each muscle fiber. Available after assign_mfs2mns().
    number_of_muscle_fibers : int
        Total number of muscle fibers. Available after generate_muscle_fiber_centers().
    muscle_border__mm : np.ndarray
        Muscle boundary points for visualization. Available after generate_muscle_fiber_centers().
    resulting_number_of_innervated_fibers : np.ndarray
        Actual number of fibers per motor unit. Available after assign_mfs2mns().
    resulting_innervation_areas__mm2 : np.ndarray
        Actual innervation areas per motor unit in mm². Available after assign_mfs2mns().

    Raises
    ------
    ValueError
        If max_innervation_area_to_total_muscle_area__ratio is not in (0, 1].

    References
    ----------
    .. [1] Farina, D., Mesin, L., Martina, S., Merletti, R., 2004. A surface EMG generation model with multilayer cylindrical description of the volume conductor. IEEE Transactions on Biomedical Engineering 51, 415–426. https://doi.org/10.1109/TBME.2003.820998

    .. [2] Konstantin, A., Yu, T., Le Carpentier, E., Aoustin, Y., Farina, D., 2020. Simulation of Motor Unit Action Potential Recordings From Intramuscular Multichannel Scanning Electrodes. IEEE Transactions on Biomedical Engineering 67, 2005–2014. https://doi.org/10.1109/TBME.2019.2953680

    .. [3] Jacobson, M.D., Raab, R., Fazeli, B.M., Abrams, R.A., Botte, M.J., Lieber, R.L., 1992. Architectural design of the human intrinsic hand muscles. The Journal of Hand Surgery 17, 804–809. https://doi.org/10.1016/0363-5023(92)90446-V

    .. [4] Nishizono, H., Fujimoto, T., Ohtake, H., Miyashita, M., 1990. Muscle fiber conduction velocity and contractile properties estimated from surface electrode arrays. Electroencephalography and Clinical Neurophysiology 75, 75–81. https://doi.org/10.1016/0013-4694(90)90154-C

    .. [5] Störchle, P., Müller, W., Sengeis, M., Lackner, S., Holasek, S., Fürhapter-Rieger, A., 2018. Measurement of mean subcutaneous fat thickness: eight standardised ultrasound sites compared to 216 randomly selected sites. Sci Rep 8, 16268. https://doi.org/10.1038/s41598-018-34213-0

    .. [6] Brodar, V., 1960. Observations on skin thickness and subcutaneous tissue in man. Zeitschrift für Morphologie und Anthropologie 50, 386–395.

    .. [7] Botelho, D.P., Curran, K., Lowery, M.M., 2019. Anatomically accurate model of EMG during index finger flexion and abduction derived from diffusion tensor imaging. PLOS Computational Biology 15, e1007267. https://doi.org/10.1371/journal.pcbi.1007267
    """

    def __init__(
        self,
        recruitment_thresholds: RECRUITMENT_THRESHOLDS__ARRAY,
        radius__mm: Quantity__mm = 6.91 * pq.mm,
        length__mm: Quantity__mm = 30.0 * pq.mm,
        fiber_density__fibers_per_mm2: Quantity__per_mm2 = 350 * pq.mm**-2,
        max_innervation_area_to_total_muscle_area__ratio: float = 1 / 4,
        mean_conduction_velocity__m_per_s: Quantity__m_per_s = 4.2 * pq.m / pq.s,
        mean_fiber_length__mm: Quantity__mm = 31.7 * pq.mm,
        var_fiber_length__mm: Quantity__mm = 2.8 * pq.mm,
        radius_bone__mm: Quantity__mm = 0 * pq.mm,
        fat_thickness__mm: Quantity__mm = 0.3 * pq.mm,
        skin_thickness__mm: Quantity__mm = 1.29 * pq.mm,
        muscle_conductivity_radial__S_per_m: Quantity__S_per_m = 0.09 * pq.S / pq.m,
        muscle_conductivity_longitudinal__S_per_m: Quantity__S_per_m = 0.4 * pq.S / pq.m,
        fat_conductivity__S_per_m: Quantity__S_per_m = 4.07e-2 * pq.S / pq.m,
        skin_conductivity__S_per_m: Quantity__S_per_m = 4.88e-4 * pq.S / pq.m,
        grid_resolution: int = 256,
        autorun: bool = False,
    ) -> None:
        # Muscle properties - immutable public access
        self.radius__mm = radius__mm
        self.length__mm = length__mm
        self.fiber_density__fibers_per_mm2 = fiber_density__fibers_per_mm2
        self.max_innervation_area_to_total_muscle_area__ratio = (
            max_innervation_area_to_total_muscle_area__ratio
        )
        self.mean_conduction_velocity__m_s = mean_conduction_velocity__m_per_s
        self.mean_fiber_length__mm = mean_fiber_length__mm
        self.var_fiber_length__mm = var_fiber_length__mm
        self.radius_bone__mm = radius_bone__mm
        self.fat_thickness__mm = fat_thickness__mm
        self.skin_thickness__mm = skin_thickness__mm
        self.muscle_conductivity_radial__S_m = muscle_conductivity_radial__S_per_m
        self.muscle_conductivity_longitudinal__S_m = muscle_conductivity_longitudinal__S_per_m
        self.fat_conductivity__S_m = fat_conductivity__S_per_m
        self.skin_conductivity__S_m = skin_conductivity__S_per_m
        self.grid_resolution = grid_resolution
        self.autorun = autorun
        # Private copies for internal modifications
        self._radius__mm = radius__mm
        self._length__mm = length__mm
        self._fiber_density__fibers_per_mm2 = fiber_density__fibers_per_mm2
        self._max_innervation_area_to_total_muscle_area__ratio = (
            max_innervation_area_to_total_muscle_area__ratio
        )
        self._mean_conduction_velocity__m_s = mean_conduction_velocity__m_per_s
        self._mean_fiber_length__mm = mean_fiber_length__mm
        self._var_fiber_length__mm = var_fiber_length__mm
        self._radius_bone__mm = radius_bone__mm
        self._fat_thickness__mm = fat_thickness__mm
        self._skin_thickness__mm = skin_thickness__mm
        self._muscle_conductivity_radial__S_m = muscle_conductivity_radial__S_per_m
        self._muscle_conductivity_longitudinal__S_m = muscle_conductivity_longitudinal__S_per_m
        self._fat_conductivity__S_m = fat_conductivity__S_per_m
        self._skin_conductivity__S_m = skin_conductivity__S_per_m
        self._grid_resolution = grid_resolution
        self._autorun = autorun
        self._recruitment_thresholds = recruitment_thresholds.copy()

        # Derived properties
        self.muscle_area__mm2 = np.pi * (self._radius__mm**2)
        self.max_innervation_area_scaling_factor = (
            1 / self._max_innervation_area_to_total_muscle_area__ratio
        )
        self._number_of_neurons = len(self._recruitment_thresholds)

        # Simulation results - stored privately, accessed via properties
        self._innervation_center_positions__mm: Optional[Quantity__mm] = None
        self._muscle_fiber_centers__mm: Optional[Quantity__mm] = None
        self._assignment: Optional[np.ndarray] = None
        self._muscle_fiber_diameters__mm: Optional[Quantity__mm] = None
        self._muscle_fiber_conduction_velocities__mm_per_s: Optional[Quantity__mm_per_s] = None
        self._number_of_muscle_fibers: Optional[int] = None
        self._muscle_border__mm: Optional[Quantity__mm] = None

        # Validate the ratio
        if not (0 < max_innervation_area_to_total_muscle_area__ratio <= 1):
            raise ValueError(
                '"max_innervation_area_to_total_muscle_area__ratio" must be in (0, 1]. '
                "This ratio defines how much of the muscle area the largest motor unit can occupy. "
                "For realistic simulations, try values between 0.1 and 0.5."
            )

        self.desired_innervation_areas__mm2 = (
            self._recruitment_thresholds
            / np.max(self._recruitment_thresholds)
            * self.muscle_area__mm2
            / self.max_innervation_area_scaling_factor
        )

        self.desired_number_of_innervated_fibers = np.round(
            (
                self.desired_innervation_areas__mm2
                / np.sum(self.desired_innervation_areas__mm2)
                * self.muscle_area__mm2
                * self._fiber_density__fibers_per_mm2
            ).magnitude
        ).astype(int)

        if autorun:
            self.distribute_innervation_centers()
            self.generate_muscle_fiber_centers()
            self.assign_mfs2mns()
            self._generate_fiber_properties()

    def _generate_fiber_properties(self) -> None:
        """
        Generate muscle fiber diameters and conduction velocities based on physiological models.

        This method should be called after generate_muscle_fiber_centers() to generate
        realistic fiber properties based on the number and positions of muscle fibers.

        Results are stored in the `mf_diameters` and `mf_cv` properties after execution.

        Raises
        ------
        ValueError
            If muscle fiber centers have not been generated first. Call generate_muscle_fiber_centers() first.
        """
        if self._muscle_fiber_centers__mm is None:
            raise ValueError(
                "Muscle fiber centers must be generated first. "
                "Call generate_muscle_fiber_centers() before generating fiber properties."
            )

        n_fibers = len(self._muscle_fiber_centers__mm)

        # Generate muscle fiber diameters using log-normal distribution
        # Based on physiological measurements (Brooke & Kaiser, 1970)
        # Mean diameter ~50um, range 20-80um
        mean_diameter__mm = 50e-3  # mm (50 um)
        std_diameter__mm = 15e-3  # mm (15 um)

        self._muscle_fiber_diameters__mm = (
            RANDOM_GENERATOR.lognormal(mean=np.log(mean_diameter__mm), sigma=0.3, size=n_fibers)
            * pq.mm
        )

        # Ensure diameters are within physiological range (20-80 um)
        self._muscle_fiber_diameters__mm = np.clip(
            self._muscle_fiber_diameters__mm, 20e-3 * pq.mm, 80e-3 * pq.mm
        )

        # Generate conduction velocities based on fiber diameter
        # CV = k * diameter + c, where k ≈ 4.5-6.0 (m/s)/mm, c ≈ 0.5-1.0 m/s
        # Based on Hakansson (1956) and later studies
        k = 5.5 * (pq.m / pq.s) / pq.mm  # (m/s)/mm
        c = 0.8 * pq.m / pq.s  # m/s

        # Add some biological variability
        cv_base = k * self._muscle_fiber_diameters__mm + c
        cv_noise = RANDOM_GENERATOR.normal(0, 0.2, n_fibers) * pq.m / pq.s  # 20% CV variation

        self._muscle_fiber_conduction_velocities__mm_per_s = cv_base + cv_noise

        # Ensure velocities are within physiological range (2-6 m/s)
        self._muscle_fiber_conduction_velocities__mm_per_s = np.clip(
            self._muscle_fiber_conduction_velocities__mm_per_s, 2.0 * pq.m / pq.s, 6.0 * pq.m / pq.s
        )

        # Convert to mm/s for consistency with the rest of the code
        self._muscle_fiber_conduction_velocities__mm_per_s = (
            self._muscle_fiber_conduction_velocities__mm_per_s.rescale(pq.mm / pq.s)
        )

    def distribute_innervation_centers(self) -> None:
        """
        Distribute innervation center positions using the fast marching method.

        This method implements an optimal packing algorithm to distribute motor unit
        innervation centers within the circular muscle cross-section. The algorithm
        uses the Fast Marching Method to ensure that each new innervation center is
        placed at the location that maximizes the minimum distance to all previously
        placed centers.

        Results are stored in the `innervation_center_positions` property after execution.

        Notes
        -----
        This method must be called before generate_muscle_fiber_centers() and
        assign_mfs2mns(). The resulting distribution approximates the optimal
        packing problem for circles, leading to realistic motor unit territory
        arrangements.
        """
        density_map = np.ones((self._grid_resolution, self._grid_resolution))
        X, Y = np.meshgrid(
            np.arange(self._grid_resolution),
            np.arange(self._grid_resolution),
        )
        density_map[
            np.sqrt((X - self._grid_resolution / 2) ** 2 + (Y - self._grid_resolution / 2) ** 2)
            > self._grid_resolution / 2 - 1
        ] = 1e-10

        vertices = np.zeros((2, self._number_of_neurons + 1))
        vertices[:, 0] = [1, 1]

        # MATLAB: for i = 2:(obj.N+1)
        for i in range(1, self._number_of_neurons + 1):
            # Use scikit-fmm for fast marching
            # Create speed map, avoiding division by zero
            ind = np.argmax(_perform_fast_marching(density_map.copy(), vertices[:, :i]))
            x, y = np.unravel_index(ind, (self._grid_resolution, self._grid_resolution))
            vertices[:, i] = [x, y]

        vertices = vertices * pq.mm

        # MATLAB: obj.innervation_center_positions = vertices(:,end:-1:2)';
        # This takes columns from end down to 2 (1-indexed), then transposes
        # In Python: vertices[:, -1:0:-1] gives us columns from end down to 1 (0-indexed)
        self._innervation_center_positions__mm = vertices[:, -1:0:-1].T

        # Only proceed if we have valid innervation_center_positions
        if (
            self._innervation_center_positions__mm.shape[0] > 0
            and self._innervation_center_positions__mm.shape[1] == 2
        ):
            center_offset = (
                self._innervation_center_positions__mm - (self._grid_resolution / 2) * pq.mm
            )
            max_dist = np.max(np.sqrt(center_offset[:, 0] ** 2 + center_offset[:, 1] ** 2))
            if max_dist > 0:  # Avoid division by zero
                self._innervation_center_positions__mm = (
                    center_offset / max_dist
                ) * self._radius__mm
            else:
                self._innervation_center_positions__mm = (
                    center_offset  # Keep original if max_dist is 0
                )

    def generate_muscle_fiber_centers(self) -> None:
        """
        Generate muscle fiber center positions using a pre-computed Voronoi distribution.

        This method creates the spatial distribution of muscle fiber centers
        within the circular muscle cross-section. The distribution is based on a
        Voronoi tessellation pattern that mimics the natural packing of muscle fibers
        observed in histological studies.

        Results are stored in the following properties after execution:

            - `mf_centers`: Array of shape (n_fibers, 2) with fiber positions [x, y] in mm
            - `number_of_muscle_fibers`: Total number of muscle fibers
            - `muscle_border`: Array of border points for visualization

        Notes
        -----
        This method should be called after distribute_innervation_centers() and
        before assign_mfs2mns(). The Voronoi-based distribution provides more
        realistic fiber spacing compared to regular grids or purely random distributions.

        The reference dataset ('voronoi_pi1e5.csv') contains 100,000 pre-computed
        Voronoi cell centers optimized for circular domains, ensuring efficient
        and consistent fiber distributions across simulations.
        """

        # Expected number of muscle fibers in the muscle
        self._number_of_muscle_fibers = int(
            np.rint(((self._radius__mm**2) * np.pi * self._fiber_density__fibers_per_mm2).magnitude)
        )

        self._muscle_fiber_centers__mm = (
            pd.read_csv(
                Path(inspect.getfile(self.__class__)).parent / "voronoi_pi1e5.csv",
                header=None,
            ).values
            * pq.mm
        )

        # Adjust the loaded innervation_center_positions to the expected number of fibers and muscle radius
        self._muscle_fiber_centers__mm = (self._muscle_fiber_centers__mm - (5 * pq.mm)) / 4
        dists = np.sqrt(
            self._muscle_fiber_centers__mm[:, 0] ** 2 + self._muscle_fiber_centers__mm[:, 1] ** 2
        )
        sorted_indices = np.argsort(dists)

        if len(sorted_indices) >= self._number_of_muscle_fibers + 1:
            self._muscle_fiber_centers__mm = (
                self._muscle_fiber_centers__mm[sorted_indices[: self._number_of_muscle_fibers], :]
                / dists[sorted_indices[self._number_of_muscle_fibers]]
                * self._radius__mm
            )
        else:
            self._muscle_fiber_centers__mm = (
                self._muscle_fiber_centers__mm[sorted_indices, :]
                / dists[sorted_indices[-1]]
                * self._radius__mm
            )
            self._number_of_muscle_fibers = len(self._muscle_fiber_centers__mm)

        # Remove fibers inside the bone boundary
        # Fibers should only exist in the muscle tissue, not in the bone core
        if self._radius_bone__mm.magnitude > 0:
            fiber_radial_dists = np.sqrt(
                self._muscle_fiber_centers__mm[:, 0] ** 2
                + self._muscle_fiber_centers__mm[:, 1] ** 2
            )
            valid_fiber_mask = fiber_radial_dists > self._radius_bone__mm
            n_fibers_removed = np.sum(~valid_fiber_mask)

            if n_fibers_removed > 0:
                self._muscle_fiber_centers__mm = self._muscle_fiber_centers__mm[valid_fiber_mask]
                self._number_of_muscle_fibers = len(self._muscle_fiber_centers__mm)
                print(
                    f"Removed {n_fibers_removed} fibers inside bone radius "
                    f"(r < {self._radius_bone__mm.magnitude:.3f} mm)"
                )

        # Create muscle border for plotting
        phi_circle = np.linspace(0, 2 * np.pi, 1000)
        phi_circle = phi_circle[:-1]
        self._muscle_border__mm = (
            np.column_stack(
                [
                    self._radius__mm.magnitude * np.cos(phi_circle),
                    self._radius__mm.magnitude * np.sin(phi_circle),
                ]
            )
            * pq.mm
        )

    def assign_mfs2mns(self, n_neighbours: int = 3, conf: float = 0.999, n_jobs: int = -2) -> None:
        """
        Assign muscle fibers to motor neurons using biologically realistic principles.

        This method implements an assignment algorithm that balances
        multiple biological constraints:

        1. Proximity: Fibers closer to innervation centers are more likely to be assigned
        2. Territory size: Each motor unit has a target number of fibers based on its size
        3. Self-avoidance: Neighboring fibers avoid belonging to the same motor unit
        4. Gaussian territories: Fiber territories follow roughly Gaussian distributions

        The assignment uses a probabilistic approach where each fiber is assigned
        based on the posterior probability computed from prior probabilities (target
        fiber numbers) and likelihoods (spatial clustering with Gaussian territories).

        Parameters
        ----------
        n_neighbours : int, default 3
            Number of neighboring fibers to consider for self-avoiding phenomena.
            Higher values increase intermingling between motor units but may slow
            computation. Typical range: 2-5.
        conf : float, default 0.999
            Confidence interval that defines the relationship between innervation
            area and Gaussian distribution variance. Higher values create tighter,
            more compact territories. Should be between 0.9 and 0.999.
        n_jobs : int, default -2
            Number of parallel workers for out-of-circle coefficient computation.

            - n_jobs=-1: Use all CPU cores
            - n_jobs=-2: Use all cores except one (recommended, keeps system responsive)
            - n_jobs=-3: Use all cores except two
            - n_jobs=1: No parallelization
            - n_jobs=N: Use exactly N cores

        Results are stored in the `assignment` property after execution.

        Raises
        ------
        ValueError
            If innervation_center_positions is None. Call distribute_innervation_centers()
            first, or if muscle fiber centers are not available.

        Notes
        -----
        The algorithm compensates for out-of-muscle effects by calculating how much
        of each motor unit's Gaussian distribution falls outside the circular muscle
        boundary and adjusting the in-muscle probabilities accordingly.

        The self-avoidance mechanism promotes realistic intermingling by reducing
        the probability of assigning a fiber to a motor unit if its neighbors are
        already assigned to that unit.
        """
        # Ensure innervation_center_positions is available
        if self._innervation_center_positions__mm is None:
            raise ValueError(
                "Innervation center positions not computed. "
                "Call distribute_innervation_centers() first."
            )

        if self._muscle_fiber_centers__mm is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )

        # Out-of-muscle area compensation
        # Calculates how much of the MU's gaussian distribution is outside of the
        # muscle border and inflates the rest of the distribution according to it
        # Work with magnitude to avoid quantity issues in integration
        radius_magnitude = self._radius__mm.magnitude

        c = chi2.ppf(conf, 2)

        def sigma(ia):
            # ia should be in mm^2, extract magnitude if it's a quantity
            ia_magnitude = ia.magnitude if hasattr(ia, "magnitude") else ia
            return np.eye(2) * ia_magnitude / np.pi / c

        # Helper function for parallel computation of out-of-circle coefficients
        def _compute_out_circle_coeff_single_mu(
            mu_index: int,
            radius_mag: float,
            innervation_center_mean: np.ndarray,
            desired_area,
            c_value: float,
        ) -> float:
            """
            Compute out-of-circle coefficient for a single motor unit.

            This function is designed to be called in parallel for each motor unit.

            Parameters
            ----------
            mu_index : int
                Motor unit index (for reference only, not used in computation).
            radius_mag : float
                Muscle radius magnitude in mm.
            innervation_center_mean : np.ndarray
                Innervation center position [x, y] in mm (magnitude only).
            desired_area : float
                Desired innervation area in mm² (magnitude only).
            c_value : float
                Chi-squared value for confidence interval.

            Returns
            -------
            float
                Out-of-circle coefficient for this motor unit.
            """

            def borderfun_pos(x):
                return np.real(np.sqrt(radius_mag**2 - x**2))

            def borderfun_neg(x):
                return np.real(-np.sqrt(radius_mag**2 - x**2))

            # Compute covariance matrix
            ia_magnitude = (
                desired_area.magnitude if hasattr(desired_area, "magnitude") else desired_area
            )
            cov = np.eye(2) * ia_magnitude / np.pi / c_value

            def probfun(y, x):
                points = (
                    np.column_stack([x.ravel(), y.ravel()])
                    if hasattr(x, "ravel")
                    else np.array([[x, y]])
                )
                return multivariate_normal.pdf(
                    points, mean=innervation_center_mean, cov=cov
                ).reshape(np.array(x).shape)

            # Use dblquad for integration (equivalent to MATLAB's integral2)
            in_circle_int = dblquad(
                probfun,
                -radius_mag,
                radius_mag,
                borderfun_neg,
                borderfun_pos,
            )[0]  # dblquad returns (integral, error)

            return 1 / in_circle_int

        # Parallel computation of out-of-circle coefficients
        results = []
        with tqdm(
            total=self._number_of_neurons,
            desc="Calculating out-of-circle coefficients",
            unit="MU",
        ) as pbar:
            for coeff in Parallel(
                n_jobs=n_jobs,
                return_as="generator",
                verbose=0,
                batch_size="auto",
            )(
                delayed(_compute_out_circle_coeff_single_mu)(
                    mu,
                    radius_magnitude,
                    self._innervation_center_positions__mm[mu].magnitude,
                    self.desired_innervation_areas__mm2[mu],
                    c,
                )
                for mu in range(self._number_of_neurons)
            ):
                results.append(coeff)
                pbar.update(1)

        out_circle_coeff = np.array(results)

        # Find nearest neighbors for suppression (equivalent to MATLAB's knnsearch)
        # Use magnitudes for sklearn compatibility
        if n_neighbours > 0:
            nbrs = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(
                self._muscle_fiber_centers__mm.magnitude
            )
            _, neighbours = nbrs.kneighbors(self._muscle_fiber_centers__mm.magnitude)
            neighbours = neighbours[:, 1:]  # Exclude self (equivalent to neighbours(:,2:end))

        # Pre-compute constant values for vectorized assignment (optimization)
        # A priori probabilities (constant for all fibers)
        apriori_probs = self.desired_number_of_innervated_fibers / self._number_of_muscle_fibers

        # Pre-compute means and covariances for all motor units
        mu_means = np.array(
            [
                self._innervation_center_positions__mm[mu].magnitude
                for mu in range(self._number_of_neurons)
            ]
        )  # Shape: (n_neurons, 2)

        mu_covs = np.array(
            [
                sigma(self.desired_innervation_areas__mm2[mu])
                for mu in range(self._number_of_neurons)
            ]
        )  # Shape: (n_neurons, 2, 2)

        # Pre-compute inverse covariances and determinants for faster PDF computation
        mu_cov_invs = np.array([np.linalg.pinv(cov) for cov in mu_covs])
        mu_cov_dets = np.array([np.linalg.det(cov) for cov in mu_covs])

        # Assignment procedure
        self._assignment = np.full(self._number_of_muscle_fibers, np.nan)
        randomized_mf = RANDOM_GENERATOR.permutation(self._number_of_muscle_fibers)

        for mf in tqdm(randomized_mf, desc="Assigning muscle fibers to motor neurons", unit="MF"):
            # Vectorized computation of likelihoods for all motor units
            # Compute differences: (n_neurons, 2)
            diffs = self._muscle_fiber_centers__mm[mf, :].magnitude - mu_means

            # Compute Mahalanobis distances efficiently
            # For each MU: (x - mean)^T * inv(cov) * (x - mean)
            mahal_dists = np.sum(
                np.sum(diffs[:, np.newaxis, :] * mu_cov_invs, axis=2) * diffs, axis=1
            )

            # Compute PDF values for all MUs at once
            clust_hoods = 1.0 / np.sqrt((2 * np.pi) ** 2 * mu_cov_dets) * np.exp(-0.5 * mahal_dists)
            clust_hoods *= out_circle_coeff

            # Compute posterior probabilities (vectorized)
            probs = apriori_probs * clust_hoods

            # Apply neighbor suppression
            if n_neighbours > 0:
                neighbor_assignments = self._assignment[neighbours[mf]]
                for mu in range(self._number_of_neurons):
                    if np.any(neighbor_assignments == mu):
                        probs[mu] = 0

            # Normalize probabilities
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                # Fallback if all probabilities are zero
                probs = np.ones(self._number_of_neurons) / self._number_of_neurons

            # Sample from the probability distribution (equivalent to MATLAB's randsample)
            self._assignment[mf] = RANDOM_GENERATOR.choice(self._number_of_neurons, p=probs)

        print(f"Assignment completed. {self._number_of_muscle_fibers} muscle fibers assigned.")

    def resulting_fiber_assignment(self, mu: int) -> Quantity__mm:
        """
        Get the muscle fiber positions assigned to a specific motor unit.

        Parameters
        ----------
        mu : int
            Motor unit index (0-based). Must be less than the total number of motor units.

        Returns
        -------
        Quantity__mm
            Array of shape (n_assigned_fibers, 2) containing the [x, y] coordinates
            (in mm) of all muscle fibers assigned to the specified motor unit.
            If no fibers are assigned to the motor unit, returns an empty array.

        Raises
        ------
        IndexError
            If mu is outside the valid range [0, n_motor_units-1].
        ValueError
            If the muscle fiber assignment has not been completed yet.

        Examples
        --------
        >>> fiber_positions = muscle.resulting_fiber_assignment(0)
        >>> print(f"Motor unit 0 has {len(fiber_positions)} fibers")
        >>> print(f"First fiber position: x={fiber_positions[0,0]:.2f}, y={fiber_positions[0,1]:.2f}")

        Notes
        -----
        This method should only be called after assign_mfs2mns() has been executed.
        The returned coordinates are in the muscle's coordinate system with the
        origin at the muscle center.
        """
        if self._assignment is None:
            raise ValueError(
                "Muscle fiber assignment not completed. "
                "Call assign_mfs2mns() first to assign fibers to motor units."
            )

        if self._muscle_fiber_centers__mm is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )

        if not (0 <= mu < len(self._recruitment_thresholds)):
            raise IndexError(
                f"Motor unit index {mu} is out of range. "
                f"Valid range is [0, {len(self._recruitment_thresholds) - 1}]."
            )

        return self._muscle_fiber_centers__mm[
            np.where(self._assignment == np.arange(len(self._recruitment_thresholds))[mu])[0]
        ]

    @property
    def resulting_number_of_innervated_fibers(self) -> np.ndarray:
        """
        Calculate the actual number of muscle fibers assigned to each motor unit.

        This property returns the final fiber counts after the assignment process,
        which may differ slightly from the desired counts due to the stochastic
        assignment algorithm and discrete fiber distribution.

        Returns
        -------
        np.ndarray
            Array of length n_motor_units where each element represents the actual
            number of muscle fibers assigned to the corresponding motor unit.
            The sum of all elements equals the total number of muscle fibers.

        Raises
        ------
        ValueError
            If muscle fiber assignment has not been completed yet.

        Examples
        --------
        >>> actual_counts = muscle.resulting_number_of_innervated_fibers
        >>> desired_counts = muscle.desired_number_of_innervated_fibers
        >>> print(f"Motor unit 0: desired {desired_counts[0]}, actual {actual_counts[0]}")

        Notes
        -----
        This property can be used to assess how well the assignment algorithm
        achieved the target fiber distribution. Large deviations may indicate
        the need to adjust assignment parameters or increase grid resolution.
        """
        if self._assignment is None:
            raise ValueError(
                "Muscle fiber assignment not completed. "
                "Call assign_mfs2mns() first to assign fibers to motor units."
            )

        return np.bincount(self._assignment.astype(int), minlength=self._number_of_neurons)

    @property
    def resulting_innervation_areas__mm2(self) -> Quantity__mm2:
        """
        Calculate the actual innervation areas for each motor unit based on assigned fibers.

        The innervation area is computed as the area of a circle that encompasses
        all muscle fibers assigned to a motor unit, centered on the motor unit's
        innervation center. This provides a measure of the spatial extent of each
        motor unit territory.

        Returns
        -------
        Quantity__mm2
            Array of length n_motor_units containing the innervation area (in mm²)
            for each motor unit. Areas are calculated as π × r², where r is the
            maximum distance from the innervation center to any assigned fiber.

        Raises
        ------
        ValueError
            If innervation_center_positions is None or assignment has not been completed.

        Examples
        --------
        >>> actual_areas = muscle.resulting_innervation_areas__mm2
        >>> desired_areas = muscle.desired_innervation_areas__mm2
        >>> for i, (actual, desired) in enumerate(zip(actual_areas, desired_areas)):
        ...     print(f"MU {i}: desired {desired:.2f} mm², actual {actual:.2f} mm²")

        Notes
        -----
        The resulting areas may differ from desired areas due to the discrete nature
        of fiber assignment and the constraint of the circular muscle boundary.
        Motor units near the muscle periphery may have smaller actual areas than
        desired due to boundary effects.
        """
        if self._innervation_center_positions__mm is None:
            raise ValueError(
                "Innervation center positions not computed. "
                "Call distribute_innervation_centers() first."
            )

        if self._assignment is None:
            raise ValueError(
                "Muscle fiber assignment not completed. "
                "Call assign_mfs2mns() first to assign fibers to motor units."
            )

        if self._muscle_fiber_centers__mm is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )

        areas = []
        for mu in range(self._number_of_neurons):
            # Calculate distances in mm
            distances = np.linalg.norm(
                self._muscle_fiber_centers__mm[self._assignment == mu]
                - self._innervation_center_positions__mm[mu],
                axis=-1,
            )
            # Get maximum distance and compute area
            max_distance = np.max(distances) if len(distances) > 0 else 0 * pq.mm
            area = np.pi * (max_distance**2)
            # Keep as quantity - extract magnitude for the array
            areas.append(area.magnitude if hasattr(area, "magnitude") else area)

        # Return as quantity array with mm^2 units
        return np.array(areas) * pq.mm**2

    # Property accessors for computed results
    @property
    def innervation_center_positions__mm(self) -> Quantity__mm:
        """
        Motor unit innervation center positions [x, y] in mm.

        Returns
        -------
        Quantity__mm
            Array of shape (n_motor_units, 2) containing [x, y] coordinates in mm.

        Raises
        ------
        ValueError
            If innervation centers have not been computed yet.
        """
        if self._innervation_center_positions__mm is None:
            raise ValueError(
                "Innervation center positions not computed. "
                "Call distribute_innervation_centers() first."
            )
        return self._innervation_center_positions__mm

    @property
    def muscle_fiber_centers__mm(self) -> Quantity__mm:
        """
        Muscle fiber center positions [x, y] in mm.

        Returns
        -------
        Quantity__mm
            Array of shape (n_fibers, 2) containing [x, y] coordinates in mm.

        Raises
        ------
        ValueError
            If muscle fiber centers have not been computed yet.
        """
        if self._muscle_fiber_centers__mm is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )
        return self._muscle_fiber_centers__mm

    @property
    def muscle_fiber_diameters__mm(self) -> Quantity__mm:
        """
        Muscle fiber diameters in mm.

        Returns
        -------
        Quantity__mm
            Array of muscle fiber diameters in mm.

        Raises
        ------
        ValueError
            If fiber properties have not been computed yet.
        """
        if self._muscle_fiber_diameters__mm is None:
            raise ValueError(
                "Muscle fiber properties not computed. "
                "Call _generate_fiber_properties() first (usually done automatically)."
            )
        return self._muscle_fiber_diameters__mm

    @property
    def muscle_fiber_conduction_velocities__mm_per_s(self) -> Quantity__mm_per_s:
        """
        Muscle fiber conduction velocities in mm/s.

        Returns
        -------
        Quantity__mm_per_s
            Array of muscle fiber conduction velocities in mm/s.

        Raises
        ------
        ValueError
            If fiber properties have not been computed yet.
        """
        if self._muscle_fiber_conduction_velocities__mm_per_s is None:
            raise ValueError(
                "Muscle fiber properties not computed. "
                "Call _generate_fiber_properties() first (usually done automatically)."
            )
        return self._muscle_fiber_conduction_velocities__mm_per_s

    @property
    def assignment(self) -> np.ndarray:
        """
        Motor unit assignment for each muscle fiber.

        Returns
        -------
        np.ndarray
            Array where each element indicates the motor unit index (0 to n_motor_units-1)
            assigned to that fiber.

        Raises
        ------
        ValueError
            If muscle fiber assignment has not been completed yet.
        """
        if self._assignment is None:
            raise ValueError(
                "Muscle fiber assignment not completed. "
                "Call assign_mfs2mns() first to assign fibers to motor units."
            )
        return self._assignment

    @property
    def number_of_muscle_fibers(self) -> int:
        """
        Total number of muscle fibers.

        Returns
        -------
        int
            Total number of muscle fibers.

        Raises
        ------
        ValueError
            If muscle fiber centers have not been computed yet.
        """
        if self._number_of_muscle_fibers is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )
        return self._number_of_muscle_fibers

    @property
    def muscle_border__mm(self) -> Quantity__mm:
        """
        Muscle boundary points for visualization.

        Returns
        -------
        Quantity__mm
            Array of boundary points for the circular muscle cross-section in mm.

        Raises
        ------
        ValueError
            If muscle fiber centers have not been computed yet.
        """
        if self._muscle_border__mm is None:
            raise ValueError(
                "Muscle fiber centers not computed. Call generate_muscle_fiber_centers() first."
            )
        return self._muscle_border__mm

    @property
    def recruitment_thresholds(self) -> RECRUITMENT_THRESHOLDS__ARRAY:
        """
        Motor unit recruitment thresholds.

        Returns
        -------
        np.ndarray
            Array of recruitment thresholds for each motor unit.
        """
        return self._recruitment_thresholds

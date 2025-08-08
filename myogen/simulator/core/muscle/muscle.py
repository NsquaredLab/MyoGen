from pathlib import Path

import inspect
import numpy as np
import pandas as pd
import skfmm
from scipy.integrate import dblquad
from scipy.stats import chi2, multivariate_normal
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from myogen import RANDOM_GENERATOR


def _perform_fast_marching(
    speed_map: np.ndarray, seed_points: np.ndarray
) -> np.ndarray:
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


class Muscle:
    """
    A muscle model based on the cylindrical description of the volume conductor by Farina et al. 2004 [1]_ and the motor unit distribution by Konstantin et al. 2020 [2]_.

    .. note::
        All default values are set to simulate the First Dorsal Interosseous (FDI) muscle. Values are pulled from the literature.

    Parameters
    ----------
    recruitment_thresholds : np.ndarray
        Array of recruitment thresholds for each motor unit (see `myogen.simulator.generate_mu_recruitment_thresholds`).
        Values range from 0 to 1 with the largest motor units having thresholds near 1.
    radius__mm : float, default=6.91
        Radius of the muscle cross-section in millimeters. Default is set to 6.91 mm as determined by Jacobson et al. 1992 [3]_.
    length__mm : float, default=30.0
        Length of the muscle in millimeters. Default is set to 30.0 mm as determined by no one.
    fiber_density__fibers_per_mm2 : float, default=400
        Density of muscle fibers per square millimeter. Default is set to 400 fibers/mm² as determined by no one.
    max_innervation_area_to_total_muscle_area__ratio : float, default=0.25
        Ratio defining the maximum territory size relative to total muscle area.
        Default is set to 0.25 as determined by no one but it is a good starting point.
        A value of 0.25 means the largest motor unit can innervate up to 25%
        of the total muscle cross-sectional area.
        Must be in range (0, 1].
    mean_conduction_velocity__m_s : float, default=4.2
        Mean conduction velocity in m/s. Default is set to 4.2 m/s as determined by Nishizono et al. 1990 [4]_.
        Experimental range determined by Nishizono et al. 1990 [4]_ is between 3.2 and 5.0 m/s.
    mean_fiber_length__mm : float, default=31.7
        Mean fiber length in mm. Default is set to 31.7 mm as determined by Jacobson et al. 1992 [3]_ (Table 1).
    var_fiber_length__mm : float, default=2.8
        Fiber length variance in mm. Default is set to 2.8 mm as determined by Jacobson et al. 1992 [3]_ (Table 1).
    radius_bone__mm : float, default=0
        Bone radius in mm. Default is set to 0 mm as the FDI muscle doesn't surround a bone.
    fat_thickness__mm : float, default=0.3
        Fat thickness in mm. Default is set to 0.3 mm as determined by Störchle et al. 2018 [5]_.
    skin_thickness__mm : float, default=1.29
        Skin thickness in mm. Default is set to the male skin thickness average of 1.29 mm as determined by Brodar 1960 [6]_.
    muscle_conductivity_radial__S_m : float, default=0.09
        Muscle conductivity in radial direction. Default is set to 0.09 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    muscle_conductivity_longitudinal__S_m : float, default=0.4
        Muscle conductivity in longitudinal direction. Default is set to 0.4 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    fat_conductivity__S_m : float, default=4.7E-2
        Fat conductivity. Default is set to 4.7E-2 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    skin_conductivity__S_m : float, default=4.88E-4
        Skin conductivity. Default is set to 4.88E-4 S/m as determined by Botelho et al. 2019 [7]_ (Table 1).
    grid_resolution : int, default=256
        Resolution of the computational grid used for innervation the muscle.
        Higher values provide more accurate spatial distribution but increase computational cost.
        Default is set to 256.
    autorun : bool, default=False
        If True, automatically executes the complete muscle simulation pipeline: innervation distribution, muscle fiber generation, and fiber-to-motor unit assignment.
        If False, these steps must be called manually.

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
        recruitment_thresholds: np.ndarray,
        radius__mm: float = 6.91,
        length__mm: float = 30.0,
        fiber_density__fibers_per_mm2: float = 400,
        max_innervation_area_to_total_muscle_area__ratio: float = 1 / 4,
        mean_conduction_velocity__m_s: float = 4.2,
        mean_fiber_length__mm: float = 31.7,
        var_fiber_length__mm: float = 2.8,
        radius_bone__mm: float = 0,
        fat_thickness__mm: float = 0.3,
        skin_thickness__mm: float = 1.29,
        muscle_conductivity_radial__S_m=0.09,
        muscle_conductivity_longitudinal__S_m=0.4,
        fat_conductivity__S_m=4.7e-2,
        skin_conductivity__S_m=4.88e-4,
        grid_resolution: int = 256,
        autorun: bool = False,
    ):
        # Muscle properties
        self.radius__mm = radius__mm
        self.length__mm = length__mm
        self.fiber_density__fibers_per_mm2 = fiber_density__fibers_per_mm2
        self.max_innervation_area_to_total_muscle_area__ratio = (
            max_innervation_area_to_total_muscle_area__ratio
        )
        self.mean_conduction_velocity__m_s = mean_conduction_velocity__m_s
        self.mean_fiber_length__mm = mean_fiber_length__mm
        self.var_fiber_length__mm = var_fiber_length__mm
        self.radius_bone__mm = radius_bone__mm
        self.fat_thickness__mm = fat_thickness__mm
        self.skin_thickness__mm = skin_thickness__mm
        self.muscle_conductivity_radial__S_m = muscle_conductivity_radial__S_m
        self.muscle_conductivity_longitudinal__S_m = (
            muscle_conductivity_longitudinal__S_m
        )
        self.fat_conductivity__S_m = fat_conductivity__S_m
        self.skin_conductivity__S_m = skin_conductivity__S_m
        self.muscle_area__mm2 = np.pi * np.power(self.radius__mm, 2)
        self.ma2max_ia = 1 / self.max_innervation_area_to_total_muscle_area__ratio

        # Simulation properties
        self.recruitment_thresholds = recruitment_thresholds
        self._number_of_neurons = len(recruitment_thresholds)
        self.grid_resolution = grid_resolution
        self.autorun = autorun

        # Simulation results
        self.innervation_center_positions: np.ndarray | None = None
        self.mf_centers: np.ndarray | None = None
        self.assignment: np.ndarray | None = None

        # Validate the ratio
        if not (0 < max_innervation_area_to_total_muscle_area__ratio <= 1):
            raise ValueError(
                '"max_innervation_area_to_total_muscle_area__ratio" must be in (0, 1].'
            )

        self.desired_innervation_areas__mm2 = (
            self.recruitment_thresholds
            / np.max(self.recruitment_thresholds)
            * self.muscle_area__mm2
            / self.ma2max_ia
        )

        self.desired_number_of_innervated_fibers = np.round(
            self.desired_innervation_areas__mm2
            / np.sum(self.desired_innervation_areas__mm2)
            * self.muscle_area__mm2
            * self.fiber_density__fibers_per_mm2
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
        """
        if self.mf_centers is None:
            raise ValueError("Muscle fiber centers must be generated first")

        n_fibers = len(self.mf_centers)

        # Generate muscle fiber diameters using log-normal distribution
        # Based on physiological measurements (Brooke & Kaiser, 1970)
        # Mean diameter ~50μm, range 20-80μm
        mean_diameter = 50e-3  # mm (50 μm)
        std_diameter = 15e-3  # mm (15 μm)

        self.mf_diameters = RANDOM_GENERATOR.lognormal(
            mean=np.log(mean_diameter), sigma=0.3, size=n_fibers
        )

        # Ensure diameters are within physiological range (20-80 μm)
        self.mf_diameters = np.clip(self.mf_diameters, 20e-3, 80e-3)

        # Generate conduction velocities based on fiber diameter
        # CV = k * diameter + c, where k ≈ 4.5-6.0 (m/s)/mm, c ≈ 0.5-1.0 m/s
        # Based on Hakansson (1956) and later studies
        k = 5.5  # (m/s)/mm
        c = 0.8  # m/s

        # Add some biological variability
        cv_base = k * (self.mf_diameters * 1000) + c  # Convert mm to m for diameter
        cv_noise = RANDOM_GENERATOR.normal(0, 0.2, n_fibers)  # 20% CV variation

        self.mf_cv = cv_base + cv_noise

        # Ensure velocities are within physiological range (2-6 m/s)
        self.mf_cv = np.clip(self.mf_cv, 2.0, 6.0)

        # Convert back to mm/s for consistency with the rest of the code
        self.mf_cv = self.mf_cv * 1000  # m/s to mm/s

    def distribute_innervation_centers(self) -> None:
        """
        Distribute innervation center positions using the fast marching method.

        This method implements an optimal packing algorithm to distribute motor unit
        innervation centers within the circular muscle cross-section. The algorithm
        uses the Fast Marching Method to ensure that each new innervation center is
        placed at the location that maximizes the minimum distance to all previously
        placed centers.

        Returns
        -------
        None
            Results are stored in self.innervation_center_positions as an array
            of shape (n_motor_units, 2) containing [x, y] coordinates in mm.

        Notes
        -----
        This method must be called before generate_muscle_fiber_centers() and
        assign_mfs2mns(). The resulting distribution approximates the optimal
        packing problem for circles, leading to realistic motor unit territory
        arrangements.
        """
        density_map = np.ones((self.grid_resolution, self.grid_resolution))
        X, Y = np.meshgrid(
            np.arange(self.grid_resolution),
            np.arange(self.grid_resolution),
        )
        density_map[
            np.sqrt(
                (X - self.grid_resolution / 2) ** 2
                + (Y - self.grid_resolution / 2) ** 2
            )
            > self.grid_resolution / 2 - 1
        ] = 1e-10

        vertices = np.zeros((2, self._number_of_neurons + 1))
        vertices[:, 0] = [1, 1]

        # MATLAB: for i = 2:(obj.N+1)
        for i in range(1, self._number_of_neurons + 1):
            # Use scikit-fmm for fast marching
            # Create speed map, avoiding division by zero
            ind = np.argmax(_perform_fast_marching(density_map.copy(), vertices[:, :i]))
            x, y = np.unravel_index(ind, (self.grid_resolution, self.grid_resolution))
            vertices[:, i] = [x, y]

        # MATLAB: obj.innervation_center_positions = vertices(:,end:-1:2)';
        # This takes columns from end down to 2 (1-indexed), then transposes
        # In Python: vertices[:, -1:0:-1] gives us columns from end down to 1 (0-indexed)
        self.innervation_center_positions = vertices[:, -1:0:-1].T

        # Only proceed if we have valid innervation_center_positions
        if (
            self.innervation_center_positions.shape[0] > 0
            and self.innervation_center_positions.shape[1] == 2
        ):
            center_offset = self.innervation_center_positions - self.grid_resolution / 2
            max_dist = np.max(
                np.sqrt(center_offset[:, 0] ** 2 + center_offset[:, 1] ** 2)
            )
            if max_dist > 0:  # Avoid division by zero
                self.innervation_center_positions = (
                    center_offset / max_dist * self.radius__mm
                )
            else:
                self.innervation_center_positions = (
                    center_offset  # Keep original if max_dist is 0
                )

    def generate_muscle_fiber_centers(self) -> None:
        """
        Generate muscle fiber center positions using a pre-computed Voronoi distribution.

        This method creates the spatial distribution of muscle fiber centers
        within the circular muscle cross-section. The distribution is based on a
        Voronoi tessellation pattern that mimics the natural packing of muscle fibers
        observed in histological studies.

        Returns
        -------
        None
            Results are stored in the following attributes:
                - self.mf_centers: Array of shape (n_fibers, 2) with fiber positions [x, y] in mm
                - self.number_of_muscle_fibers: Total number of muscle fibers
                - self.muscle_border: Array of border points for visualization

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
        self.number_of_muscle_fibers = int(
            np.rint((self.radius__mm**2) * np.pi * self.fiber_density__fibers_per_mm2)
        )

        self.mf_centers = pd.read_csv(
            Path(inspect.getfile(self.__class__)).parent / "voronoi_pi1e5.csv",
            header=None,
        ).values

        # Adjust the loaded innervation_center_positions to the expected number of fibers and muscle radius
        self.mf_centers = (self.mf_centers - 5) / 4  # 4 may be unnecessary here
        dists = np.sqrt(self.mf_centers[:, 0] ** 2 + self.mf_centers[:, 1] ** 2)
        sorted_indices = np.argsort(dists)

        if len(sorted_indices) >= self.number_of_muscle_fibers + 1:
            self.mf_centers = (
                self.mf_centers[sorted_indices[: self.number_of_muscle_fibers], :]
                / dists[sorted_indices[self.number_of_muscle_fibers]]
                * self.radius__mm
            )
        else:
            self.mf_centers = (
                self.mf_centers[sorted_indices, :]
                / dists[sorted_indices[-1]]
                * self.radius__mm
            )
            self.number_of_muscle_fibers = len(self.mf_centers)

        # Create muscle border for plotting
        phi_circle = np.linspace(0, 2 * np.pi, 1000)
        phi_circle = phi_circle[:-1]
        self.muscle_border = np.column_stack(
            [self.radius__mm * np.cos(phi_circle), self.radius__mm * np.sin(phi_circle)]
        )

    def assign_mfs2mns(self, n_neighbours: int = 3, conf: float = 0.999):
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

        Returns
        -------
        None
            Results are stored in self.assignment as an array of length n_fibers
            where each element indicates the motor unit index (0 to n_motor_units-1)
            assigned to that fiber.

        Raises
        ------
        ValueError
            If innervation_center_positions is None. Call distribute_innervation_centers()
            first.

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
        if self.innervation_center_positions is None:
            raise ValueError(
                "innervation_center_positions is None. Call distribute_innervation_centers() first."
            )

        # Out-of-muscle area compensation
        # Calculates how much of the MU's gaussian distribution is outside of the
        # muscle border and inflates the rest of the distribution according to it
        borderfun_pos = lambda x: np.real(np.sqrt(self.radius__mm**2 - x**2))
        borderfun_neg = lambda x: np.real(-np.sqrt(self.radius__mm**2 - x**2))
        out_circle_coeff = np.ones(self._number_of_neurons)

        c = chi2.ppf(conf, 2)
        sigma = lambda ia: np.eye(2) * ia / np.pi / c

        for mu in tqdm(
            range(self._number_of_neurons),
            desc="Calculating out-of-circle coefficients",
            unit="MU",
        ):
            # Create multivariate normal distribution for this motor unit
            mean = self.innervation_center_positions[mu]
            cov = sigma(self.desired_innervation_areas__mm2[mu])

            def probfun(y, x):
                points = (
                    np.column_stack([x.ravel(), y.ravel()])
                    if hasattr(x, "ravel")
                    else np.array([[x, y]])
                )
                return multivariate_normal.pdf(points, mean=mean, cov=cov).reshape(
                    np.array(x).shape
                )

            # Use dblquad for integration (equivalent to MATLAB's integral2)
            in_circle_int = dblquad(
                probfun, -self.radius__mm, self.radius__mm, borderfun_neg, borderfun_pos
            )[0]  # dblquad returns (integral, error)
            out_circle_coeff[mu] = 1 / in_circle_int

        # Find nearest neighbors for suppression (equivalent to MATLAB's knnsearch)
        if n_neighbours > 0:
            nbrs = NearestNeighbors(n_neighbors=n_neighbours + 1).fit(self.mf_centers)
            _, neighbours = nbrs.kneighbors(self.mf_centers)
            neighbours = neighbours[
                :, 1:
            ]  # Exclude self (equivalent to neighbours(:,2:end))

        # Assignment procedure
        self.assignment = np.full(self.number_of_muscle_fibers, np.nan)
        randomized_mf = RANDOM_GENERATOR.permutation(self.number_of_muscle_fibers)

        for mf in tqdm(
            randomized_mf, desc="Assigning muscle fibers to motor neurons", unit="MF"
        ):
            probs = np.zeros(self._number_of_neurons)

            for mu in range(self._number_of_neurons):
                # Suppression assignment if neighbours are from the same MU
                # Promotes intermingling
                if n_neighbours > 0 and np.any(self.assignment[neighbours[mf]] == mu):
                    probs[mu] = 0
                else:
                    # A priori probability of the assignment
                    apriori_prob = (
                        self.desired_number_of_innervated_fibers[mu]
                        / self.number_of_muscle_fibers
                    )

                    # Likelihood coming from clustered nature of mf distribution
                    # Use scipy's multivariate_normal.pdf (equivalent to MATLAB's mvnpdf)
                    mean = self.innervation_center_positions[mu]
                    cov = sigma(self.desired_innervation_areas__mm2[mu])
                    clust_hood = multivariate_normal.pdf(
                        self.mf_centers[mf, :], mean=mean, cov=cov
                    )
                    clust_hood = clust_hood * out_circle_coeff[mu]

                    # Final a posteriori probability
                    probs[mu] = apriori_prob * clust_hood

            # Normalize probabilities
            probs = probs / np.sum(probs)

            # should something bad happen with the probabilities, we should not crash
            probs = np.nan_to_num(probs, nan=0.0)
            if np.sum(probs) == 0:
                probs = np.ones(self._number_of_neurons) / self._number_of_neurons

            # Sample from the probability distribution (equivalent to MATLAB's randsample)
            self.assignment[mf] = RANDOM_GENERATOR.choice(
                self._number_of_neurons, p=probs
            )

        print(
            f"Assignment completed. {self.number_of_muscle_fibers} muscle fibers assigned."
        )

    def resulting_fiber_assignment(self, mu: int) -> np.ndarray:
        """
        Get the muscle fiber positions assigned to a specific motor unit.

        Parameters
        ----------
        mu : int
            Motor unit index (0-based). Must be less than the total number of motor units.

        Returns
        -------
        np.ndarray
            Array of shape (n_assigned_fibers, 2) containing the [x, y] coordinates
            (in mm) of all muscle fibers assigned to the specified motor unit.
            If no fibers are assigned to the motor unit, returns an empty array.

        Raises
        ------
        IndexError
            If mu is outside the valid range [0, n_motor_units-1].
        AttributeError
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
        origin at the muscle electrode_grid_center.
        """
        return self.mf_centers[
            np.where(
                self.assignment == np.arange(len(self.recruitment_thresholds))[mu]
            )[0]
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
        return np.bincount(
            self.assignment.astype(int), minlength=self._number_of_neurons
        )

    @property
    def resulting_innervation_areas__mm2(self) -> np.ndarray:
        """
        Calculate the actual innervation areas for each motor unit based on assigned fibers.

        The innervation area is computed as the area of a circle that encompasses
        all muscle fibers assigned to a motor unit, centered on the motor unit's
        innervation electrode_grid_center. This provides a measure of the spatial extent of each
        motor unit territory.

        Returns
        -------
        np.ndarray
            Array of length n_motor_units containing the innervation area (in mm²)
            for each motor unit. Areas are calculated as π × r², where r is the
            maximum distance from the innervation electrode_grid_center to any assigned fiber.

        Raises
        ------
        AttributeError
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
        if self.innervation_center_positions is None:
            raise AttributeError(
                "innervation_center_positions is None. Call distribute_innervation_centers() first."
            )

        return np.array(
            [
                np.pi
                * (
                    np.max(
                        np.linalg.norm(
                            self.mf_centers[self.assignment == mu]
                            - self.innervation_center_positions[mu],
                            axis=-1,
                        )
                    )
                    ** 2
                )
                for mu in range(self._number_of_neurons)
            ]
        )

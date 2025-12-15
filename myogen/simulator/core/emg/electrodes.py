"""
Electrode configuration framework for EMG simulation.
"""

from typing import Literal

import numpy as np
import numpy.matlib  # noqa
import quantities as pq
from scipy.spatial.transform import Rotation as R

from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__deg, Quantity__mm, Quantity__rad


@beartowertype
class SurfaceElectrodeArray:
    """
    Surface electrode array for EMG recording.

    Represents a grid of surface electrodes with configurable spacing,
    size, and differentiation modes.

    Parameters
    ----------
    num_rows : int
        Number of rows in the electrode array
    num_cols : int
        Number of columns in the electrode array
    inter_electrode_distances__mm : float
        Inter-electrode distances in mm.
    electrode_radius__mm : float, optional
        Radius of the electrodes in mm
    center_point__mm_deg : tuple[float, float]
        Position along z in mm and rotation around the muscle theta in degrees.
    bending_radius__mm : float, optional
        Bending radius around which the electrode grid is bent. Usually this is equal to the radius of the muscle.
    rotation_angle__deg : float, optional
        Rotation angle of the electrodes in degrees. This is the angle between the electrode grid and the muscle surface.
    differentiation_mode : {"monopolar", "bipolar_longitudinal", "bipolar_transversal", "laplacian"}
        Differentiation mode. Default is monopolar.

    Attributes
    ----------
    pos_z : np.ndarray
        Longitudinal electrode positions in mm, shape (num_rows, num_cols).
        Available after class initialization via `_create_electrode_grid()`.
    pos_theta : np.ndarray
        Angular electrode positions in radians, shape (num_rows, num_cols).
        Available after class initialization via `_create_electrode_grid()`.
    electrode_positions : tuple[np.ndarray, np.ndarray]
        Complete electrode position arrays (pos_z, pos_theta).
        Available after class initialization via `_create_electrode_grid()`.
    num_electrodes : int
        Total number of electrodes (num_rows * num_cols).
    num_channels : int
        Number of recording channels based on differentiation mode.
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        inter_electrode_distances__mm: Quantity__mm,
        electrode_radius__mm: Quantity__mm,
        center_point__mm_deg: tuple[Quantity__mm, Quantity__deg] = (0.0 * pq.mm, 0.0 * pq.deg),
        bending_radius__mm: Quantity__mm = 0.0 * pq.mm,
        rotation_angle__deg: Quantity__deg = 0.0 * pq.deg,
        differentiation_mode: Literal[
            "monopolar", "bipolar_longitudinal", "bipolar_transversal", "laplacian"
        ] = "monopolar",
    ):
        # Immutable public arguments - never modify these
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.center_point__mm_deg = center_point__mm_deg
        self.bending_radius__mm = bending_radius__mm
        self.rotation_angle__deg = rotation_angle__deg
        self.inter_electrode_distances__mm = inter_electrode_distances__mm
        self.electrode_radius__mm = electrode_radius__mm
        self.differentiation_mode = differentiation_mode

        # Private copies for internal modifications (extract magnitudes)
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._center_point__mm_deg = (
            float(center_point__mm_deg[0].rescale(pq.mm).magnitude),
            float(center_point__mm_deg[1].rescale(pq.deg).magnitude),
        )
        self._bending_radius__mm = float(bending_radius__mm.rescale(pq.mm).magnitude)
        self._rotation_angle__deg = float(rotation_angle__deg.rescale(pq.deg).magnitude)
        self._inter_electrode_distances__mm = float(
            inter_electrode_distances__mm.rescale(pq.mm).magnitude
        )
        self._electrode_radius__mm = float(electrode_radius__mm.rescale(pq.mm).magnitude)
        self._differentiation_mode = differentiation_mode

        self.num_electrodes = num_rows * num_cols

        # Handle zero bending radius
        if self._bending_radius__mm == 0:
            self._bending_radius__mm = np.finfo(np.float32).eps

        # Set up channel configuration based on differentiation mode
        if differentiation_mode == "monopolar":
            self.num_channels = self.num_electrodes
        elif differentiation_mode in ["bipolar_longitudinal", "bipolar_transversal"]:
            # For bipolar, we lose one channel per dimension
            if differentiation_mode == "bipolar_longitudinal":
                self.num_channels = max(1, num_rows - 1) * num_cols
            else:  # bipolar_transversal
                self.num_channels = num_rows * max(1, num_cols - 1)
        elif differentiation_mode == "laplacian":
            # For Laplacian, we lose border electrodes
            self.num_channels = max(1, num_rows - 2) * max(1, num_cols - 2)
        else:
            self.num_channels = self.num_electrodes

        # Create electrode grid in local coordinate system
        self._create_electrode_grid()

    def _create_electrode_grid(self) -> None:
        """Create electrode positions in local coordinate system.

        Results are stored in the `electrode_positions` property after execution.
        """
        _pos_z = np.zeros((self._num_rows, self._num_cols))
        if self._num_rows % 2 == 1:
            index_center = int((self._num_rows - 1) / 2)
            _pos_z[index_center, :] = self._center_point__mm_deg[0]
            for i in range(1, index_center + 1):
                _pos_z[index_center + i, :] = (
                    _pos_z[index_center + i - 1, :] + self._inter_electrode_distances__mm
                )
                _pos_z[index_center - i, :] = (
                    _pos_z[index_center - i + 1, :] - self._inter_electrode_distances__mm
                )
        else:
            index_center1 = int(self._num_rows / 2)
            index_center2 = index_center1 - 1
            _pos_z[index_center1, :] = (
                self._center_point__mm_deg[0] + self._inter_electrode_distances__mm / 2
            )
            _pos_z[index_center2, :] = (
                self._center_point__mm_deg[0] - self._inter_electrode_distances__mm / 2
            )
            for i in range(1, index_center2 + 1):
                _pos_z[index_center1 + i, :] = (
                    _pos_z[index_center1 + i - 1, :] + self._inter_electrode_distances__mm
                )
                _pos_z[index_center2 - i, :] = (
                    _pos_z[index_center2 - i + 1, :] - self._inter_electrode_distances__mm
                )

        _pos_theta = np.zeros((self._num_rows, self._num_cols))
        if self._num_cols % 2 == 1:
            index_center = int((self._num_cols - 1) / 2)
            _pos_theta[:, index_center] = self._center_point__mm_deg[1] * np.pi / 180
            for i in range(1, index_center + 1):
                _pos_theta[:, index_center + i] = (
                    _pos_theta[:, index_center + i - 1]
                    + self._inter_electrode_distances__mm / self._bending_radius__mm
                )
                _pos_theta[:, index_center - i] = (
                    _pos_theta[:, index_center - i + 1]
                    - self._inter_electrode_distances__mm / self._bending_radius__mm
                )
        else:
            index_center1 = int(self._num_cols / 2)
            index_center2 = index_center1 - 1
            _pos_theta[:, index_center1] = (
                self._center_point__mm_deg[1] * np.pi / 180
                + self._inter_electrode_distances__mm / 2 / self._bending_radius__mm
            )
            _pos_theta[:, index_center2] = (
                self._center_point__mm_deg[1] * np.pi / 180
                - self._inter_electrode_distances__mm / 2 / self._bending_radius__mm
            )
            for i in range(1, index_center2 + 1):
                _pos_theta[:, index_center1 + i] = (
                    _pos_theta[:, index_center1 + i - 1]
                    + self._inter_electrode_distances__mm / self._bending_radius__mm
                )
                _pos_theta[:, index_center2 - i] = (
                    _pos_theta[:, index_center2 - i + 1]
                    - self._inter_electrode_distances__mm / self._bending_radius__mm
                )

        ## Rotated detection system (Farina, 2004), eq (36)
        displacement = self._center_point__mm_deg[0] * np.ones(_pos_z.shape)
        _pos_z = (
            -self._bending_radius__mm * np.sin(self._rotation_angle__deg * np.pi / 180) * _pos_theta
            + np.cos(self._rotation_angle__deg * np.pi / 180) * (_pos_z - displacement)
            + displacement
        )
        _pos_theta = (
            np.cos(self._rotation_angle__deg * np.pi / 180) * _pos_theta
            + np.sin(self._rotation_angle__deg * np.pi / 180)
            * (_pos_z - displacement)
            / self._bending_radius__mm
        )

        # Store results privately
        self._pos_z = _pos_z
        self._pos_theta = _pos_theta

    @property
    def pos_z(self) -> Quantity__mm:
        """
        Longitudinal positions of electrodes in mm.

        Returns
        -------
        Quantity__mm
            Array of shape (num_rows, num_cols) containing z-coordinates
            of each electrode position in mm.

        Raises
        ------
        AttributeError
            If electrode grid has not been created. Run constructor first.
        """
        if not hasattr(self, "_pos_z"):
            raise AttributeError(
                "Electrode grid not computed. This should be automatically created "
                "during class initialization. Please check constructor execution."
            )
        return self._pos_z * pq.mm

    @property
    def pos_theta(self) -> Quantity__rad:
        """
        Angular positions of electrodes in radians.

        Returns
        -------
        Quantity__rad
            Array of shape (num_rows, num_cols) containing angular coordinates
            of each electrode position in radians.

        Raises
        ------
        AttributeError
            If electrode grid has not been created. Run constructor first.
        """
        if not hasattr(self, "_pos_theta"):
            raise AttributeError(
                "Electrode grid not computed. This should be automatically created "
                "during class initialization. Please check constructor execution."
            )
        return self._pos_theta * pq.rad

    @property
    def electrode_positions(self) -> tuple[Quantity__mm, Quantity__rad]:
        """
        Complete electrode position arrays (z, theta) in physical coordinates.

        Returns
        -------
        tuple[Quantity__mm, Quantity__rad]
            Tuple containing:
            - pos_z: Longitudinal positions in mm, shape (num_rows, num_cols)
            - pos_theta: Angular positions in radians, shape (num_rows, num_cols)

        Raises
        ------
        AttributeError
            If electrode grid has not been created. Run constructor first.
        """
        return (self.pos_z, self.pos_theta)

    def get_H_sf(
        self, ktheta_mesh_kzktheta: np.ndarray, kz_mesh_kzktheta: np.ndarray
    ) -> np.ndarray | float:
        """
        Get the spatial filter for the electrode array.

        Parameters
        ----------
        ktheta_mesh_kzktheta : np.ndarray
            Angular spatial frequency mesh
        kz_mesh_kzktheta : np.ndarray
            Longitudinal spatial frequency mesh

        Returns
        -------
        H_sf : np.ndarray or float
            Spatial filter for the specified differentiation mode
        """
        if self.differentiation_mode == "monopolar":
            H_sf = 1.0

        elif self.differentiation_mode == "bipolar_longitudinal":
            # Differential along muscle fiber direction (z-axis)
            # Apply coordinate transformation for rotation
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            kz_new = ktheta_mesh_kzktheta / self.bending_radius__mm * np.sin(
                alpha_rad
            ) + kz_mesh_kzktheta * np.cos(alpha_rad)
            # Spatial filter for longitudinal differential
            H_sf = np.exp(1j * kz_new) - np.exp(-1j * kz_new)

        elif self.differentiation_mode == "bipolar_transversal":
            # Differential around muscle circumference (theta-axis)
            # Apply coordinate transformation for rotation
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            ktheta_new = ktheta_mesh_kzktheta * np.cos(
                alpha_rad
            ) - kz_mesh_kzktheta * self.bending_radius__mm * np.sin(alpha_rad)
            # Spatial filter for transversal differential
            H_sf = np.exp(1j * ktheta_new / self.bending_radius__mm) - np.exp(
                -1j * ktheta_new / self.bending_radius__mm
            )

        elif self.differentiation_mode == "laplacian":
            # Laplacian (second-order spatial differential)
            # Combination of longitudinal and transversal second derivatives
            alpha_rad = self.rotation_angle__deg * np.pi / 180
            kz_new = ktheta_mesh_kzktheta / self.bending_radius__mm * np.sin(
                alpha_rad
            ) + kz_mesh_kzktheta * np.cos(alpha_rad)
            ktheta_new = ktheta_mesh_kzktheta * np.cos(
                alpha_rad
            ) - kz_mesh_kzktheta * self.bending_radius__mm * np.sin(alpha_rad)

            # Laplacian approximation: -k^2 in frequency domain
            k_total_sq = kz_new**2 + (ktheta_new / self.bending_radius__mm) ** 2
            H_sf = -k_total_sq

        return H_sf


@beartowertype
class IntramuscularElectrodeArray:
    """
    Intramuscular electrode array for EMG recording.

    Represents a linear array of intramuscular electrodes (needle electrodes)
    with configurable spacing and differentiation modes.

    Parameters
    ----------
    num_electrodes : int
        Number of electrodes in the array
    inter_electrode_distance__mm : float, default=0.5
        Inter-electrode distance in mm
    position__mm : tuple[float, float, float], default=(0.0, 0.0, 0.0)
        Position of the electrode array center in mm (x, y, z coordinates)
    orientation__rad : tuple[float, float, float], default=(0.0, 0.0, 0.0)
        Orientation of the electrode array in radians (roll, pitch, yaw)
    differentiation_mode : Literal["consecutive", "reference"], default="consecutive"
        Differentiation mode for recording
    trajectory_distance__mm : float, default=0.0
        Distance for trajectory movement in mm
    trajectory_steps : int, default=1
        Number of steps in the trajectory

    Attributes
    ----------
    electrode_positions : np.ndarray
        Current electrode positions in 3D space, shape (n_nodes * num_electrodes, 3).
        Available after set_linear_trajectory() execution.
    differential_matrix : np.ndarray
        Differential matrix for signal processing based on differentiation mode.
        Available after class initialization.
    trajectory_transforms : np.ndarray
        Transformation matrices for trajectory movement, shape (n_nodes, 6).
        Available after set_linear_trajectory() execution.
    initial_positions : np.ndarray
        Initial electrode positions after position/orientation setup, shape (num_electrodes, 3).
        Available after set_position() execution.
    num_channels : int
        Number of recording channels based on differentiation mode.
        Available after class initialization.
    num_points : int
        Alias for num_electrodes (compatibility).
    n_nodes : int
        Number of trajectory nodes.
    """

    def __init__(
        self,
        num_electrodes: int,
        inter_electrode_distance__mm: Quantity__mm = 0.5 * pq.mm,
        position__mm: tuple[Quantity__mm, Quantity__mm, Quantity__mm] = (
            0.0 * pq.mm,
            0.0 * pq.mm,
            0.0 * pq.mm,
        ),
        orientation__rad: tuple[Quantity__rad, Quantity__rad, Quantity__rad] = (
            0.0 * pq.rad,
            0.0 * pq.rad,
            0.0 * pq.rad,
        ),
        differentiation_mode: Literal["consecutive", "reference"] = "consecutive",
        trajectory_distance__mm: Quantity__mm = 0.0 * pq.mm,
        trajectory_steps: int = 1,
    ):
        # Immutable public arguments - never modify these
        self.num_electrodes = num_electrodes
        self.inter_electrode_distance__mm = inter_electrode_distance__mm
        self.position__mm = position__mm
        self.orientation__rad = orientation__rad
        self.differentiation_mode = differentiation_mode
        self.trajectory_distance__mm = trajectory_distance__mm
        self.trajectory_steps = trajectory_steps

        # Private copies for internal modifications (extract magnitudes)
        self._num_electrodes = num_electrodes
        self._inter_electrode_distance__mm = float(
            inter_electrode_distance__mm.rescale(pq.mm).magnitude
        )
        self._position__mm = (
            float(position__mm[0].rescale(pq.mm).magnitude),
            float(position__mm[1].rescale(pq.mm).magnitude),
            float(position__mm[2].rescale(pq.mm).magnitude),
        )
        self._orientation__rad = (
            float(orientation__rad[0].rescale(pq.rad).magnitude),
            float(orientation__rad[1].rescale(pq.rad).magnitude),
            float(orientation__rad[2].rescale(pq.rad).magnitude),
        )
        self._differentiation_mode = differentiation_mode
        self._trajectory_distance__mm = float(trajectory_distance__mm.rescale(pq.mm).magnitude)
        self._trajectory_steps = trajectory_steps

        self.num_points = num_electrodes  # Alias for compatibility
        self.n_nodes = trajectory_steps

        self._pts_origin = np.concatenate(
            [
                np.zeros((self._num_electrodes, 2)),
                np.arange(self._num_electrodes)[..., None] * self._inter_electrode_distance__mm,
            ],
            axis=-1,
        )

        self._normal_origin = []
        self._normals_init = []
        self._normals = []

        match differentiation_mode:
            case "consecutive":
                eye_mat = np.eye(self._pts_origin.shape[0] - 1, self._pts_origin.shape[0])
                self._diff_mat = eye_mat - np.roll(eye_mat, shift=1, axis=1)
            case "reference":
                self._diff_mat = np.roll(
                    np.eye(self._pts_origin.shape[0] - 1, self._pts_origin.shape[0]),
                    shift=1,
                    axis=1,
                )
                self._diff_mat[:, 0] = -1

        self._n_channels = self._diff_mat.shape[0]

        # Use public (Quantity) parameters for method calls
        self.set_position(position__mm=position__mm, orientation__rad=orientation__rad)
        self.set_linear_trajectory(
            distance__mm=trajectory_distance__mm, n_nodes=self._trajectory_steps
        )

    def set_position(
        self,
        position__mm: tuple[Quantity__mm, Quantity__mm, Quantity__mm],
        orientation__rad: tuple[Quantity__rad, Quantity__rad, Quantity__rad],
    ) -> None:
        """
        Set the position and orientation of the intramuscular electrode array.

        This method defines the spatial placement and angular orientation of the
        electrode array within the muscle volume. The array is first oriented
        according to the specified rotations and then translated to the target position.

        **Coordinate System:**
        - x-axis: radial direction (outward from muscle center)
        - y-axis: circumferential direction (around muscle)
        - z-axis: longitudinal direction (along muscle fibers)

        **Rotation Order:**
        Applied as: Roll (x) → Pitch (y) → Yaw (z) using Rodrigues rotation

        Parameters
        ----------
        position__mm : tuple[float, float, float]
            Center position of the electrode array in mm (x, y, z coordinates).
            This defines where the array center is placed within the muscle.
        orientation__rad : tuple[float, float, float]
            Orientation angles in radians (roll, pitch, yaw).
            - Roll: rotation around x-axis (radial tilt)
            - Pitch: rotation around y-axis (circumferential tilt)
            - Yaw: rotation around z-axis (longitudinal rotation)

        Notes
        -----
        Position and orientation changes affect all subsequent trajectory calculations.
        The electrode positions are recalculated based on the new transformation.

        Examples
        --------
        >>> # Place array at muscle center with 45° yaw rotation
        >>> array.set_position(
        ...     position__mm=(0.0, 0.0, 10.0),
        ...     orientation__rad=(0.0, 0.0, np.pi/4)
        ... )

        See Also
        --------
        set_linear_trajectory : Define trajectory movement parameters
        rodrigues_rot : Rodrigues rotation implementation
        """
        self._pts_init = np.copy(self._pts_origin)

        # Extract magnitude values for internal calculations
        orientation__rad_values = (
            float(orientation__rad[0].rescale(pq.rad).magnitude),
            float(orientation__rad[1].rescale(pq.rad).magnitude),
            float(orientation__rad[2].rescale(pq.rad).magnitude),
        )
        position__mm_values = np.array(
            [
                float(position__mm[0].rescale(pq.mm).magnitude),
                float(position__mm[1].rescale(pq.mm).magnitude),
                float(position__mm[2].rescale(pq.mm).magnitude),
            ]
        )

        self._pts_init = self.rodrigues_rot(self._pts_init, [1, 0, 0], orientation__rad_values[0])
        self._pts_init = self.rodrigues_rot(self._pts_init, [0, 1, 0], orientation__rad_values[1])
        self._pts_init = self.rodrigues_rot(self._pts_init, [0, 0, 1], orientation__rad_values[2])

        self._pts_init += np.matlib.repmat(position__mm_values[None], self._pts_init.shape[0], 1)
        self._pts = np.copy(self._pts_init)

    def rodrigues_rot(self, v, k, theta):
        """
        Apply Rodrigues rotation to vectors around an arbitrary axis.

        This method implements 3D rotation of points or vectors around an arbitrary
        axis using the Rodrigues rotation formula. It is used internally for
        electrode array positioning and trajectory calculations.

        **Mathematical Foundation:**
        Based on Rodrigues' rotation formula for rotating a vector v around
        axis k by angle theta: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))

        Parameters
        ----------
        v : array_like
            Vector(s) to rotate. Can be single vector (3,) or array of vectors (N, 3).
        k : array_like
            Rotation axis vector (3,). Will be normalized internally.
        theta : float
            Rotation angle in radians. Positive angles follow right-hand rule.

        Returns
        -------
        np.ndarray
            Rotated vector(s) with same shape as input v.

        Notes
        -----
        Uses scipy.spatial.transform.Rotation for numerical stability and efficiency.
        The rotation axis k is automatically normalized to unit length.

        Examples
        --------
        >>> # Rotate point 90° around z-axis
        >>> point = np.array([1.0, 0.0, 0.0])
        >>> rotated = array.rodrigues_rot(point, [0, 0, 1], np.pi/2)
        >>> # Result: approximately [0, 1, 0]
        """
        v = np.array(v.copy(), dtype=float)
        k = np.array(k.copy(), dtype=float)
        k = k / np.linalg.norm(k)  # normalize axis

        r = R.from_rotvec(k * theta)  # Create rotation from axis-angle
        return r.apply(v)  # Rotates v (works with (3,), (N, 3))

    def set_linear_trajectory(self, distance__mm: Quantity__mm, n_nodes: int | None = None) -> None:
        """
        Configure linear trajectory movement for the electrode array.

        This method sets up a linear movement path for the electrode array,
        simulating needle insertion or withdrawal. The trajectory is discretized
        into nodes for temporal interpolation during EMG simulation.

        **Trajectory Properties:**
        - Direction: Along the array's longitudinal axis (z-direction in local coordinates)
        - Movement: Linear progression from start to end position
        - Discretization: Evenly spaced nodes for smooth interpolation
        - Default step size: 0.5mm if n_nodes not specified

        Parameters
        ----------
        distance__mm : float
            Total trajectory distance in mm. Positive values move in the
            positive z-direction of the array's local coordinate system.
        n_nodes : int, optional
            Number of discrete trajectory nodes. If None, automatically
            calculated as max(ceil(distance/0.5), 1) for 0.5mm steps.

        Notes
        -----
        The trajectory is applied after position and orientation transformations.
        All trajectory transforms are calculated in the array's oriented coordinate system.

        Examples
        --------
        >>> # Set up 10mm insertion with default step size (~0.5mm)
        >>> array.set_linear_trajectory(distance__mm=10.0)

        >>> # Set up 5mm trajectory with specific number of nodes
        >>> array.set_linear_trajectory(distance__mm=5.0, n_nodes=20)

        See Also
        --------
        calc_observation_points : Calculate electrode positions along trajectory
        traj_mixing_mat : Generate mixing matrices for trajectory interpolation
        """
        # Extract magnitude value for internal calculations
        distance__mm_value = float(distance__mm.rescale(pq.mm).magnitude)

        if n_nodes is None:
            n_nodes = max(np.ceil(distance__mm_value / 0.5), 1)

        self.n_nodes = n_nodes
        self._trajectory_step = distance__mm_value / self.n_nodes

        self.traj_transforms = np.linspace(start=0, stop=distance__mm_value, num=self.n_nodes)
        self.traj_transforms = np.hstack(
            [
                np.zeros((max(self.traj_transforms.shape), 2)),
                self.traj_transforms.reshape(-1, 1),
                np.zeros((max(self.traj_transforms.shape), 3)),
            ]
        )

        # Use private orientation values (already extracted as floats)
        self.traj_transforms[:, :3] = self.rodrigues_rot(
            self.traj_transforms[:, :3], [1, 0, 0], self._orientation__rad[0]
        )
        self.traj_transforms[:, :3] = self.rodrigues_rot(
            self.traj_transforms[:, :3], [0, 1, 0], self._orientation__rad[1]
        )
        self.traj_transforms[:, :3] = self.rodrigues_rot(
            self.traj_transforms[:, :3], [0, 0, 1], self._orientation__rad[2]
        )

        self.calc_observation_points()

    def calc_observation_points(self) -> None:
        self.pts = np.concatenate(
            [
                self.rotate_and_translate(
                    self._pts_init,
                    self.traj_transforms[i, 3:],
                    self.traj_transforms[i, :3],
                )
                for i in range(self.n_nodes)
            ]
        )

        self._diff_mat = np.matlib.repmat(self._diff_mat, 1, self.n_nodes)

    def rotate_and_translate(self, pt, rpy, d):
        # pt: (N, 3) matrix
        # rpy: (3,) vector [roll, pitch, yaw]
        # d: (3,) vector translation

        pt = self.rodrigues_rot(pt, np.array([1, 0, 0]), rpy[0])  # roll
        pt = self.rodrigues_rot(pt, np.array([0, 1, 0]), rpy[1])  # pitch
        pt = self.rodrigues_rot(pt, np.array([0, 0, 1]), rpy[2])  # yaw

        return pt + d.reshape(1, 3)  # translation (broadcasted)

    def traj_mixing_fun(self, t, n_nodes, node) -> float:
        """
        Compute mixing weight for a specific trajectory node at given time.

        This function calculates the interpolation weight for a trajectory node based
        on the current time/position along the trajectory. Uses triangular weighting
        where nodes closer to the current time get higher weights.

        Parameters
        ----------
        t : float
            Current normalized time or position in trajectory (0.0 to 1.0).
        n_nodes : int
            Total number of nodes in the trajectory.
        node : int or array_like
            Node index(es) for which to calculate mixing weights.
            Can be scalar or array of node indices.

        Returns
        -------
        float or np.ndarray
            Mixing weight(s) for the specified node(s) at time t.
            Returns 0 for distant nodes, max weight 1 for closest node.
        """
        eps = np.finfo(float).eps
        return np.maximum(
            0,
            1 - (n_nodes - 1) * np.abs(t - (node - 1) / (n_nodes - 1 + eps)),
        )

    def traj_mixing_mat(self, t, n_nodes, n_channels) -> np.ndarray:
        """
        Generate mixing matrix for trajectory interpolation during EMG simulation.

        This method creates a diagonal mixing matrix that weights the contribution of
        different trajectory nodes during temporal interpolation. The matrix enables
        smooth transitions between electrode positions as the array moves along its
        trajectory during needle insertion or withdrawal.

        **Interpolation Strategy:**
        - Linear interpolation between adjacent trajectory nodes
        - Weights based on distance from current time/position to node positions
        - Diagonal matrix structure for efficient computation
        - Smooth transitions avoid discontinuities in EMG signals

        Parameters
        ----------
        t : float
            Current normalized time or position in trajectory (0.0 to 1.0).
            0.0 corresponds to trajectory start, 1.0 to trajectory end.
        n_nodes : int
            Total number of trajectory nodes for interpolation.
        n_channels : int
            Number of recording channels in the electrode array.
            Depends on differentiation mode and electrode count.

        Returns
        -------
        np.ndarray
            Diagonal mixing matrix with shape (n_nodes * n_channels, n_nodes * n_channels).
            Diagonal elements contain repeated mixing weights for each trajectory node,
            with each node's weight repeated n_channels times.

        Notes
        -----
        The mixing matrix enables temporal interpolation of EMG signals recorded
        at different trajectory positions. Higher weights are given to nodes
        closer to the current time/position parameter.

        Examples
        --------
        >>> # Get mixing weights for mid-trajectory position
        >>> mix_mat = array.traj_mixing_mat(t=0.5, n_nodes=10, n_channels=4)
        >>> # Matrix will weight middle nodes more heavily

        See Also
        --------
        traj_mixing_fun : Individual node mixing function
        set_linear_trajectory : Configure trajectory parameters
        """

        return np.diag(
            np.repeat(
                self.traj_mixing_fun(t, n_nodes, np.arange(1, n_nodes + 1)),
                n_channels,
            )
        )

    @property
    def electrode_positions(self) -> Quantity__mm:
        """
        Current electrode positions in 3D space (mm).

        Returns
        -------
        Quantity__mm
            Array of shape (n_nodes * num_electrodes, 3) containing x, y, z coordinates
            of each electrode position for all trajectory nodes, in mm.

        Raises
        ------
        AttributeError
            If trajectory has not been calculated. Run set_linear_trajectory() first.
        """
        if not hasattr(self, "pts"):
            raise AttributeError(
                "Electrode positions not computed. Run set_linear_trajectory() first "
                "to calculate trajectory and electrode positions."
            )
        return self.pts * pq.mm

    @property
    def differential_matrix(self) -> np.ndarray:
        """
        Differential matrix for signal processing based on differentiation mode.

        Returns
        -------
        np.ndarray
            Differential matrix for applying spatial differentiation to recorded signals.
            Shape depends on differentiation mode and number of trajectory nodes.

        Raises
        ------
        AttributeError
            If differential matrix has not been created. Run constructor first.
        """
        if not hasattr(self, "_diff_mat"):
            raise AttributeError(
                "Differential matrix not computed. This should be automatically created "
                "during class initialization. Please check constructor execution."
            )
        return self._diff_mat

    @property
    def trajectory_transforms(self) -> np.ndarray:
        """
        Transformation matrices for trajectory movement.

        Returns
        -------
        np.ndarray
            Array of shape (n_nodes, 6) containing translation and rotation parameters
            for each trajectory node. First 3 columns are translations (x, y, z),
            last 3 columns are rotations (roll, pitch, yaw).

        Raises
        ------
        AttributeError
            If trajectory has not been set. Run set_linear_trajectory() first.
        """
        if not hasattr(self, "traj_transforms"):
            raise AttributeError(
                "Trajectory transforms not computed. Run set_linear_trajectory() first "
                "to configure electrode trajectory movement."
            )
        return self.traj_transforms

    @property
    def initial_positions(self) -> Quantity__mm:
        """
        Initial electrode positions after position/orientation setup.

        Returns
        -------
        Quantity__mm
            Array of shape (num_electrodes, 3) containing initial x, y, z coordinates
            of electrodes before trajectory movement is applied, in mm.

        Raises
        ------
        AttributeError
            If initial positions have not been set. Run set_position() first.
        """
        if not hasattr(self, "_pts_init"):
            raise AttributeError(
                "Initial electrode positions not computed. Run set_position() first "
                "to configure electrode array placement and orientation."
            )
        return self._pts_init * pq.mm

    @property
    def num_channels(self) -> int:
        """
        Number of recording channels based on differentiation mode.

        Returns
        -------
        int
            Number of differential recording channels available from this electrode array.

        Raises
        ------
        AttributeError
            If channel count has not been calculated. Run constructor first.
        """
        if not hasattr(self, "_n_channels"):
            raise AttributeError(
                "Number of channels not computed. This should be automatically created "
                "during class initialization. Please check constructor execution."
            )
        return self._n_channels

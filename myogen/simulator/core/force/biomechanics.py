"""
Biomechanics module for joint angle dynamics and moment arm calculations.

This module provides tools for computing muscle lengths from joint angles,
moment arms, and joint torques - essential for linking neural activation
to movement mechanics in musculoskeletal simulations.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from myogen.utils.decorators import beartowertype
from myogen.utils.types import JOINT_ANGLE__ARRAY, MOMENT_ARM__MATRIX


@dataclass
class MuscleGeometry:
    """Geometric parameters for a muscle crossing a joint."""

    origin_coords: Tuple[float, float]  # (x, y) coordinates in cm
    insertion_coords: Tuple[float, float]  # (x, y) coordinates in cm
    optimal_length__cm: float
    max_length_change__cm: float  # Maximum length change from optimal


@dataclass
class JointGeometry:
    """Geometric parameters for a joint."""

    center_coords: Tuple[float, float]  # (x, y) coordinates in cm
    radius__cm: float  # Effective joint radius
    range_of_motion__degrees: Tuple[float, float]  # (min, max) joint angles


@beartowertype
class JointBiomechanics:
    """
    Joint biomechanics with muscle moment arms and length calculations.

    This class computes the relationship between joint angles and muscle lengths,
    moment arms, and resulting joint torques. It supports both simple geometric
    models and more complex biomechanical relationships.

    Parameters
    ----------
    joint_type : {"hinge", "ball_socket"}
        Type of joint for biomechanical calculations.
        "hinge" = single degree of freedom (e.g., elbow, knee)
        "ball_socket" = multi-degree of freedom (e.g., shoulder, hip)
    joint_geometry : JointGeometry
        Geometric parameters of the joint.
    muscle_geometries : list[MuscleGeometry]
        List of muscle geometry parameters for muscles crossing this joint.
    moment_arm_data__cm : MOMENT_ARM__MATRIX, optional
        Pre-computed moment arm data as function of joint angle.
        If None, computed from geometry. Shape: (n_angles, n_muscles).
    use_simplified_model : bool, default=True
        Whether to use simplified geometric model or detailed biomechanics.

    Attributes
    ----------
    n_muscles : int
        Number of muscles crossing this joint.
    joint_angles__degrees : np.ndarray
        Array of joint angles for moment arm calculations.
    moment_arms__cm : np.ndarray
        Moment arms for each muscle at each joint angle.
    """

    def __init__(
        self,
        joint_type: Literal["hinge", "ball_socket"],
        joint_geometry: JointGeometry,
        muscle_geometries: List[MuscleGeometry],
        moment_arm_data__cm: Optional[MOMENT_ARM__MATRIX] = None,
        use_simplified_model: bool = True,
    ) -> None:
        # Validate inputs
        if joint_type not in ["hinge", "ball_socket"]:
            raise ValueError(f"joint_type must be 'hinge' or 'ball_socket', got '{joint_type}'")

        if len(muscle_geometries) == 0:
            raise ValueError("At least one muscle geometry must be provided")

        if joint_geometry.radius__cm <= 0:
            raise ValueError(
                f"joint_geometry.radius__cm must be positive, got {joint_geometry.radius__cm}"
            )

        min_angle, max_angle = joint_geometry.range_of_motion__degrees
        if min_angle >= max_angle:
            raise ValueError(
                f"Joint range of motion invalid: min={min_angle}, max={max_angle}. "
                "Maximum angle must be greater than minimum angle."
            )

        # Store parameters (immutable public access)
        self.joint_type = joint_type
        self.joint_geometry = joint_geometry
        self.muscle_geometries = muscle_geometries
        self.use_simplified_model = use_simplified_model

        # Private copies for internal use
        self._joint_type = joint_type
        self._joint_geometry = joint_geometry
        self._muscle_geometries = muscle_geometries
        self._use_simplified = use_simplified_model

        # Derived properties
        self.n_muscles = len(muscle_geometries)

        # Set up angle array for calculations
        min_angle, max_angle = joint_geometry.range_of_motion__degrees
        self.joint_angles__degrees = np.linspace(min_angle, max_angle, 181)  # 1 degree resolution

        # Initialize or validate moment arm data
        if moment_arm_data__cm is not None:
            if moment_arm_data__cm.shape != (
                len(self.joint_angles__degrees),
                self.n_muscles,
            ):
                raise ValueError(
                    f"moment_arm_data__cm shape {moment_arm_data__cm.shape} does not match "
                    f"expected ({len(self.joint_angles__degrees)}, {self.n_muscles})"
                )
            self._moment_arms__cm = moment_arm_data__cm
        else:
            self._moment_arms__cm = self._compute_moment_arms()

        # Public access to moment arms
        self.moment_arms__cm = self._moment_arms__cm

    def _compute_moment_arms(self) -> np.ndarray:
        """
        Compute moment arms for all muscles across joint range of motion.

        Returns
        -------
        np.ndarray
            Moment arms with shape (n_angles, n_muscles).
        """
        moment_arms = np.zeros((len(self.joint_angles__degrees), self.n_muscles))

        for angle_idx, joint_angle in enumerate(self.joint_angles__degrees):
            for muscle_idx, muscle_geom in enumerate(self._muscle_geometries):
                moment_arms[angle_idx, muscle_idx] = self._compute_single_moment_arm(
                    joint_angle, muscle_geom
                )

        return moment_arms

    def _compute_single_moment_arm(
        self, joint_angle__degrees: float, muscle_geometry: MuscleGeometry
    ) -> float:
        """
        Compute moment arm for a single muscle at a given joint angle.

        Parameters
        ----------
        joint_angle__degrees : float
            Joint angle in degrees.
        muscle_geometry : MuscleGeometry
            Geometry parameters for the muscle.

        Returns
        -------
        float
            Moment arm in cm.
        """
        if self._use_simplified:
            # Simplified model: moment arm varies sinusoidally with joint angle
            # This is a common approximation for many joints
            angle_rad = np.radians(joint_angle__degrees)
            base_moment_arm = self._joint_geometry.radius__cm * 0.8  # 80% of joint radius

            # Vary moment arm with joint angle (sinusoidal approximation)
            moment_arm = base_moment_arm * np.abs(np.sin(angle_rad + np.pi / 4))

            # Ensure minimum moment arm for numerical stability
            return max(moment_arm, 0.1)  # minimum 1 mm moment arm
        else:
            # Detailed geometric calculation
            return self._compute_geometric_moment_arm(joint_angle__degrees, muscle_geometry)

    def _compute_geometric_moment_arm(
        self, joint_angle__degrees: float, muscle_geometry: MuscleGeometry
    ) -> float:
        """
        Compute moment arm using detailed geometric calculations.

        This method calculates the perpendicular distance from the joint center
        to the line of action of the muscle force.
        """
        angle_rad = np.radians(joint_angle__degrees)

        # Joint center coordinates
        jx, jy = self._joint_geometry.center_coords

        # Muscle origin (typically fixed)
        ox, oy = muscle_geometry.origin_coords

        # Muscle insertion rotates with joint
        base_ix, base_iy = muscle_geometry.insertion_coords

        # Rotate insertion point around joint center
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)

        # Translate to joint center, rotate, translate back
        ix_rel, iy_rel = base_ix - jx, base_iy - jy
        ix = jx + ix_rel * cos_angle - iy_rel * sin_angle
        iy = jy + ix_rel * sin_angle + iy_rel * cos_angle

        # Compute moment arm as perpendicular distance from joint to muscle line
        # Vector from origin to insertion
        muscle_vec_x = ix - ox
        muscle_vec_y = iy - oy
        muscle_length = np.sqrt(muscle_vec_x**2 + muscle_vec_y**2)

        if muscle_length < 1e-6:  # Avoid division by zero
            return 0.1

        # Unit vector along muscle
        muscle_unit_x = muscle_vec_x / muscle_length
        muscle_unit_y = muscle_vec_y / muscle_length

        # Vector from origin to joint
        joint_vec_x = jx - ox
        joint_vec_y = jy - oy

        # Project joint vector onto muscle vector
        projection = joint_vec_x * muscle_unit_x + joint_vec_y * muscle_unit_y

        # Perpendicular component gives moment arm
        perpendicular_x = joint_vec_x - projection * muscle_unit_x
        perpendicular_y = joint_vec_y - projection * muscle_unit_y

        moment_arm = np.sqrt(perpendicular_x**2 + perpendicular_y**2)

        return max(moment_arm, 0.1)  # minimum 1 mm moment arm

    def compute_muscle_length(self, joint_angle__degrees: float, muscle_index: int) -> float:
        """
        Compute muscle length from joint angle.

        Parameters
        ----------
        joint_angle__degrees : float
            Joint angle in degrees.
        muscle_index : int
            Index of the muscle (0 to n_muscles-1).

        Returns
        -------
        float
            Muscle length in cm.

        Raises
        ------
        ValueError
            If muscle_index is out of range.
        """
        if not 0 <= muscle_index < self.n_muscles:
            raise ValueError(
                f"muscle_index must be between 0 and {self.n_muscles - 1}, got {muscle_index}"
            )

        muscle_geom = self._muscle_geometries[muscle_index]

        if self._use_simplified:
            # Simplified model: length varies with joint angle
            angle_rad = np.radians(joint_angle__degrees)

            # Length change is proportional to joint rotation and moment arm
            moment_arm = self.get_moment_arm(joint_angle__degrees, muscle_index)

            # Reference angle (middle of range of motion)
            min_angle, max_angle = self._joint_geometry.range_of_motion__degrees
            ref_angle = (min_angle + max_angle) / 2

            angle_change = np.radians(joint_angle__degrees - ref_angle)
            length_change = moment_arm * angle_change

            muscle_length = muscle_geom.optimal_length__cm + length_change

            # Constrain to reasonable range
            min_length = muscle_geom.optimal_length__cm - muscle_geom.max_length_change__cm
            max_length = muscle_geom.optimal_length__cm + muscle_geom.max_length_change__cm

            return np.clip(muscle_length, min_length, max_length)
        else:
            # Detailed geometric calculation
            return self._compute_geometric_muscle_length(joint_angle__degrees, muscle_index)

    def _compute_geometric_muscle_length(
        self, joint_angle__degrees: float, muscle_index: int
    ) -> float:
        """Compute muscle length using detailed geometry."""
        angle_rad = np.radians(joint_angle__degrees)
        muscle_geom = self._muscle_geometries[muscle_index]

        # Joint center coordinates
        jx, jy = self._joint_geometry.center_coords

        # Muscle origin (typically fixed)
        ox, oy = muscle_geom.origin_coords

        # Muscle insertion rotates with joint
        base_ix, base_iy = muscle_geom.insertion_coords

        # Rotate insertion point
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        ix_rel, iy_rel = base_ix - jx, base_iy - jy
        ix = jx + ix_rel * cos_angle - iy_rel * sin_angle
        iy = jy + ix_rel * sin_angle + iy_rel * cos_angle

        # Distance from origin to insertion
        muscle_length = np.sqrt((ix - ox) ** 2 + (iy - oy) ** 2)

        return muscle_length

    def get_moment_arm(self, joint_angle__degrees: float, muscle_index: int) -> float:
        """
        Get moment arm for a muscle at a specific joint angle.

        Parameters
        ----------
        joint_angle__degrees : float
            Joint angle in degrees.
        muscle_index : int
            Index of the muscle.

        Returns
        -------
        float
            Moment arm in cm.
        """
        if not 0 <= muscle_index < self.n_muscles:
            raise ValueError(
                f"muscle_index must be between 0 and {self.n_muscles - 1}, got {muscle_index}"
            )

        # Interpolate from pre-computed moment arms
        return np.interp(
            joint_angle__degrees,
            self.joint_angles__degrees,
            self._moment_arms__cm[:, muscle_index],
        )

    def compute_joint_torque(
        self, muscle_forces__N: Union[float, np.ndarray], joint_angle__degrees: float
    ) -> float:
        """
        Compute net joint torque from muscle forces.

        Parameters
        ----------
        muscle_forces__N : float or np.ndarray
            Forces from each muscle in Newtons. If float, assumes single muscle.
            If array, must have length equal to n_muscles.
        joint_angle__degrees : float
            Current joint angle in degrees.

        Returns
        -------
        float
            Net joint torque in N⋅cm.

        Notes
        -----
        Positive torque indicates rotation in the positive joint angle direction.
        """
        forces = np.asarray(muscle_forces__N)

        if forces.ndim == 0:  # Single muscle
            if self.n_muscles != 1:
                raise ValueError(f"Single force provided but {self.n_muscles} muscles defined")
            forces = np.array([forces])
        elif len(forces) != self.n_muscles:
            raise ValueError(
                f"Force array length ({len(forces)}) must match number of muscles ({self.n_muscles})"
            )

        # Get moment arms for current joint angle
        moment_arms = np.array(
            [self.get_moment_arm(joint_angle__degrees, i) for i in range(self.n_muscles)]
        )

        # Compute torques (force × moment arm)
        muscle_torques = forces * moment_arms

        # Sum torques (considering muscle action directions would require additional info)
        # For now, assume all muscles act in same direction
        return np.sum(muscle_torques)

    def get_muscle_length_trajectory(
        self, joint_angle_trajectory__degrees: JOINT_ANGLE__ARRAY, muscle_index: int
    ) -> np.ndarray:
        """
        Compute muscle length trajectory from joint angle trajectory.

        Parameters
        ----------
        joint_angle_trajectory__degrees : JOINT_ANGLE__ARRAY
            Array of joint angles over time.
        muscle_index : int
            Index of the muscle.

        Returns
        -------
        np.ndarray
            Muscle length trajectory in cm.
        """
        return np.array(
            [
                self.compute_muscle_length(angle, muscle_index)
                for angle in joint_angle_trajectory__degrees
            ]
        )

    def get_biomechanical_summary(self) -> dict:
        """
        Get summary of biomechanical parameters.

        Returns
        -------
        dict
            Dictionary containing key biomechanical parameters.
        """
        min_angle, max_angle = self._joint_geometry.range_of_motion__degrees

        return {
            "joint_type": self._joint_type,
            "n_muscles": self.n_muscles,
            "joint_radius__cm": self._joint_geometry.radius__cm,
            "joint_range_of_motion__degrees": (min_angle, max_angle),
            "use_simplified_model": self._use_simplified,
            "moment_arm_range__cm": {
                f"muscle_{i}": (
                    np.min(self._moment_arms__cm[:, i]),
                    np.max(self._moment_arms__cm[:, i]),
                )
                for i in range(self.n_muscles)
            },
            "optimal_muscle_lengths__cm": [
                geom.optimal_length__cm for geom in self._muscle_geometries
            ],
        }

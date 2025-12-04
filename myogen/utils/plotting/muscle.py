from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from myogen.simulator import Muscle
from myogen.utils.decorators import beartowertype


@beartowertype
def plot_mf_centers(
    muscle_model: Muscle,
    ax: Axes,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot muscle fiber innervation center positions

    Parameters
    ----------
    muscle_model : Muscle
        Muscle model to plot
    ax : Axes
        Matplotlib axis to plot on
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    Axes
        The axis that was plotted on
    """
    if apply_default_formatting:
        # Draw muscle border
        ax.plot(*muscle_model.muscle_border__mm.T, "-", linewidth=2, color="#FF5944")
        ax.scatter(
            *muscle_model.muscle_fiber_centers__mm.T,
            s=3,
            color="#FF5944",
            edgecolors="black",
            linewidths=0.1,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Muscle fibers' center points")
    else:
        # Draw muscle border
        ax.plot(*muscle_model.muscle_border__mm.T, **kwargs)
        ax.scatter(*muscle_model.muscle_fiber_centers__mm.T, **kwargs)

    return ax


@beartowertype
def plot_innervation_areas_2d(
    muscle_model: Muscle,
    ax: Axes,
    indices_to_plot: Optional[np.ndarray] = None,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> Axes:
    """Plot 2D innervation areas

    Parameters
    ----------
    muscle_model : Muscle
        Muscle model to plot
    ax : Axes
        Matplotlib axis to plot on
    indices_to_plot : np.ndarray, optional
        Indices of motor neurons to plot. By default, all motor neurons are plotted.
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    Axes
        The axis that was plotted on
    """
    if indices_to_plot is None:
        indices_to_plot = np.arange(muscle_model._number_of_neurons)

    max_index = np.max(indices_to_plot)

    if apply_default_formatting:
        # Use current style's default color cycle (adapts to any plt.style)
        base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [base_colors[i % len(base_colors)] for i in range(len(indices_to_plot))]

        alphas = np.logspace(np.log10(0.1), np.log10(1.0), len(indices_to_plot))

        for (i, m), color, alpha in tqdm.tqdm(
            list(zip(enumerate(reversed(indices_to_plot)), colors, alphas)),
            desc="Plotting innervation areas",
            unit="MU",
        ):
            fiber_indices = np.where(muscle_model.assignment == m)[0]
            if len(fiber_indices) > 0:
                points = muscle_model.muscle_fiber_centers__mm[fiber_indices]

                if len(fiber_indices) > 2:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.plot(
                        *hull_points.T,
                        color=color,
                        linewidth=3 * alpha,
                        alpha=alpha,
                        zorder=-max_index + i,
                    )

                ax.scatter(
                    *points.T,
                    color=color,
                    s=15,
                    alpha=alpha,
                    zorder=-max_index + i,
                )

        # Draw muscle border
        ax.plot(
            *muscle_model.muscle_border__mm.T,
            "k-",
            linewidth=2,
            zorder=len(indices_to_plot) + 1,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title("Motor neuron innervation areas")
    else:
        # Basic plotting without default formatting
        for i, m in enumerate(indices_to_plot):
            fiber_indices = np.where(muscle_model.assignment == m)[0]
            if len(fiber_indices) > 0:
                points = muscle_model.muscle_fiber_centers__mm[fiber_indices]
                if len(fiber_indices) > 2:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    ax.plot(*hull_points.T, **kwargs)
                ax.scatter(*points.T, **kwargs)

        # Draw muscle border
        ax.plot(*muscle_model.muscle_border__mm.T, **kwargs)

    return ax


@beartowertype
def plot_innervation_areas_3d(
    muscle_model: Muscle,
    ax: Union[Axes, Axes3D],
    indices_to_plot: Optional[np.ndarray] = None,
    z_spacing: float = 1.0,
    stretch_factor: float = 2.0,
    apply_default_formatting: bool = True,
    **kwargs: Any,
) -> Union[Axes, Axes3D]:
    """Plot 3D stack of convex hulls along Z axis, forming a long cylinder

    Parameters
    ----------
    muscle_model : Muscle
        Muscle model with 2D fiber innervation center positions
    ax : Union[Axes, Axes3D]
        Matplotlib axis to plot on. Should be 3D axes for proper visualization.
    indices_to_plot : np.ndarray, optional
        Which motor neurons to plot. Defaults to all.
    z_spacing : float, optional
        Distance between stacked layers (in mm), by default 1.0
    stretch_factor : float, optional
        Multiplier for elongating the Z-axis to make it look like a muscle, by default 2.0
    apply_default_formatting : bool, optional
        Whether to apply default formatting to the plot, by default True
    **kwargs : Any
        Additional keyword arguments to pass to the plot function. Only used if apply_default_formatting is False.

    Returns
    -------
    Union[Axes, Axes3D]
        The axis that was plotted on

    Raises
    ------
    ValueError
        If muscle fiber centers are not 2D
    """
    # Create 3D axis if not provided
    if not hasattr(ax, "zaxis"):
        # Set clean white style before creating figure
        plt.style.use("default")
        fig = plt.figure(figsize=(12, 10), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")

        if apply_default_formatting:
            # Set clean white background and styling for 3D plot
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")  # type: ignore

            # Make grid subtle
            ax.grid(True, alpha=0.3, color="lightgray")  # type: ignore

            # Remove the background panes for a cleaner look
            try:
                ax.xaxis.pane.fill = False  # type: ignore
                ax.yaxis.pane.fill = False  # type: ignore
                ax.zaxis.pane.fill = False  # type: ignore
                ax.xaxis.pane.set_edgecolor("lightgray")  # type: ignore
                ax.yaxis.pane.set_edgecolor("lightgray")  # type: ignore
                ax.zaxis.pane.set_edgecolor("lightgray")  # type: ignore
                ax.xaxis.pane.set_alpha(0.1)  # type: ignore
                ax.yaxis.pane.set_alpha(0.1)  # type: ignore
                ax.zaxis.pane.set_alpha(0.1)  # type: ignore
            except AttributeError:
                # Fallback if the pane attributes are not available
                pass

    if indices_to_plot is None:
        indices_to_plot = np.arange(muscle_model._number_of_neurons)

    if apply_default_formatting:
        # Use a repeating palette of 10 colors
        base_colors = plt.cm.rainbow(np.linspace(0, 1, 10))  # type: ignore
        colors = [base_colors[i % 10] for i in range(len(indices_to_plot))]

        for i, m in enumerate(indices_to_plot):
            fiber_indices = np.where(muscle_model.assignment == m)[0]
            if len(fiber_indices) < 3:
                continue  # Need at least 3 points for a 2D hull

            points = muscle_model.muscle_fiber_centers__mm[fiber_indices]
            if points.shape[1] != 2:
                raise ValueError("Expected 2D points in mf_centers")

            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]

            z_value = i * z_spacing * stretch_factor
            verts3d = [[(x, y, z_value) for x, y in hull_pts]]

            poly = Poly3DCollection(
                verts3d, alpha=0.4, facecolor=colors[i], edgecolor="k", linewidths=0.5
            )
            ax.add_collection3d(poly)  # type: ignore

        ax.set_xlabel("x (mm)", fontsize=12)  # type: ignore
        ax.set_ylabel("y (mm)", fontsize=12)  # type: ignore
        ax.set_zlabel("Motor Unit Index (a. u.)", fontsize=12)  # type: ignore
        ax.set_title(
            "Motor Neuron Innervation Areas as Stacked Convex Hulls",
            fontsize=14,
            pad=20,
        )  # type: ignore

        ax.set_box_aspect([1, 1, stretch_factor])  # type: ignore
        ax.view_init(elev=25, azim=135)  # type: ignore
    else:
        # Basic plotting without default formatting
        for i, m in enumerate(indices_to_plot):
            fiber_indices = np.where(muscle_model.assignment == m)[0]
            if len(fiber_indices) < 3:
                continue

            points = muscle_model.muscle_fiber_centers__mm[fiber_indices]
            if points.shape[1] != 2:
                raise ValueError("Expected 2D points in mf_centers")

            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]

            z_value = i * z_spacing * stretch_factor
            verts3d = [[(x, y, z_value) for x, y in hull_pts]]

            poly = Poly3DCollection(verts3d, **kwargs)
            ax.add_collection3d(poly)  # type: ignore

    return ax


# Legacy function aliases for backward compatibility
def show_mf_centers(muscle_model: Muscle, ax: Axes) -> Axes:
    """Legacy function - use plot_mf_centers instead"""
    return plot_mf_centers(muscle_model, ax)


def show_innervation_areas_2d(
    muscle_model: Muscle,
    indices_to_plot: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Legacy function - use plot_innervation_areas_2d instead"""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))
    return plot_innervation_areas_2d(muscle_model, ax, indices_to_plot)


def show_innervation_areas_3d(
    muscle_model: Muscle,
    indices_to_plot: np.ndarray | None = None,
    ax: Axes3D | None = None,
    z_spacing: float = 1.0,
    stretch_factor: float = 2.0,
) -> Axes3D:
    """Legacy function - use plot_innervation_areas_3d instead"""
    if ax is None:
        plt.style.use("default")
        fig = plt.figure(figsize=(12, 10), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")  # type: ignore
    return plot_innervation_areas_3d(muscle_model, ax, indices_to_plot, z_spacing, stretch_factor)  # type: ignore

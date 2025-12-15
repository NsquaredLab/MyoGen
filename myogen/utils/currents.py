import numpy as np
import quantities as pq
from neo.core import AnalogSignal

from myogen.utils.decorators import beartowertype
from myogen.utils.types import (
    CURRENT__AnalogSignal,
    Quantity__Hz,
    Quantity__ms,
    Quantity__nA,
    Quantity__rad,
)


def _broadcast_and_validate(
    param_name: str,
    value: pq.Quantity | list[pq.Quantity],
    n_pools: int,
) -> list[pq.Quantity]:
    """Convert scalar or list of quantities to validated list.

    Handles three cases:
    1. Scalar value (non-array/list): broadcast to list of n_pools
    2. Scalar Quantity (0-dim array): broadcast to list of n_pools
    3. List/array with length n_pools: use as-is
    4. List/array with wrong length: raise ValueError

    Note: pq.Quantity is a subclass of np.ndarray, so scalar Quantities
    (e.g., 5.0 * pq.ms) are 0-dimensional arrays that need special handling.

    Parameters
    ----------
    param_name : str
        Name of the parameter (for error messages)
    value : pq.Quantity | list[pq.Quantity]
        The parameter value to broadcast/validate. Can be:
        - Scalar Quantity (e.g., 5.0 * pq.ms)
        - List of Quantities (e.g., [1*pq.ms, 2*pq.ms, 3*pq.ms])
        - Array of Quantities
    n_pools : int
        Expected length of the output list

    Returns
    -------
    list[pq.Quantity]
        List of length n_pools with Quantity values

    Raises
    ------
    ValueError
        If value is a list/array and its length doesn't match n_pools
    """
    # Check if value is a scalar (including 0-dimensional Quantity arrays)
    # This handles both non-array scalars and scalar Quantities
    is_scalar = not isinstance(value, (np.ndarray, list)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )

    if is_scalar:
        # Broadcast scalar to list of n_pools elements
        value_list = [value] * n_pools
    else:
        # Value is already a list or multi-element array
        value_list = value

        # Validate that length matches expected n_pools
        if len(value_list) != n_pools:
            raise ValueError(
                f"Length of {param_name} ({len(value_list)}) must match n_pools ({n_pools})"
            )

    return value_list  # type: ignore


def _broadcast_and_validate_float(
    param_name: str,
    value: float | list[float],
    n_pools: int,
) -> list[float]:
    """Convert scalar or list of floats to validated list.

    Parameters
    ----------
    param_name : str
        Name of the parameter (for error messages)
    value : float | list[float]
        The parameter value to broadcast/validate
    n_pools : int
        Expected length of the output list

    Returns
    -------
    list[float]
        List of length n_pools with float values

    Raises
    ------
    ValueError
        If value is a list and its length doesn't match n_pools
    """
    if isinstance(value, (list, np.ndarray)):
        if len(value) != n_pools:
            raise ValueError(
                f"Length of {param_name} ({len(value)}) must match n_pools ({n_pools})"
            )
        return list(value)
    else:
        return [value] * n_pools


@beartowertype
def create_sinusoidal_current(
    n_pools: int,
    t_points: int,
    timestep__ms: Quantity__ms,
    amplitudes__nA: Quantity__nA | list[Quantity__nA],
    frequencies__Hz: Quantity__Hz | list[Quantity__Hz],
    offsets__nA: Quantity__nA | list[Quantity__nA],
    phases__rad: Quantity__rad | list[Quantity__rad] = 0.0 * pq.rad,
) -> CURRENT__AnalogSignal:
    """Create a matrix of sinusoidal currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : Quantity__ms
        Time step in milliseconds as a Quantity
    amplitudes__nA : Quantity__nA | list[Quantity__nA]
        Amplitude(s) of the sinusoidal current(s) in nanoamperes.
    frequencies__Hz : Quantity__Hz | list[Quantity__Hz]
        Frequency(s) of the sinusoidal current(s) in Hertz.
    offsets__nA : Quantity__nA | list[Quantity__nA]
        DC offset(s) to add to the sinusoidal current(s) in nanoamperes.
    phases__rad : Quantity__rad | list[Quantity__rad]
        Phase(s) of the sinusoidal current(s) in radians.

    Raises
    ------
    ValueError
        If the amplitudes, frequencies, offsets, or phases are lists and the length of the parameters does not match n_pools

    Notes
    -----
    If a parameter is provided as a single Quantity, it is broadcasted to all pools.
    If provided as a list, its length must match n_pools.

    Returns
    -------
    INPUT_CURRENT__AnalogSignal
        Analog signal of shape (t_points, n_pools) * pq.nA containing sinusoidal currents
    """
    # Convert timestep to milliseconds for time array
    timestep_ms = timestep__ms.magnitude
    t = np.arange(0, t_points * timestep_ms, timestep_ms)

    # Convert quantities to lists of floats in expected units
    amplitudes_list = _broadcast_and_validate("amplitudes__nA", amplitudes__nA, n_pools)
    frequencies_list = _broadcast_and_validate("frequencies__Hz", frequencies__Hz, n_pools)
    offsets_list = _broadcast_and_validate("offsets__nA", offsets__nA, n_pools)
    phases_list = _broadcast_and_validate("phases__rad", phases__rad, n_pools)

    return AnalogSignal(
        signal=np.stack(
            [
                (
                    amplitudes_list[i].magnitude
                    * np.sin(
                        2 * np.pi * frequencies_list[i].magnitude * t / 1000
                        + phases_list[i].magnitude
                    )
                    + offsets_list[i].magnitude
                )
                for i in range(n_pools)
            ],
            axis=-1,
        )
        * pq.nA,
        t_start=0 * pq.s,
        sampling_period=timestep__ms.rescale(pq.s),
    )


@beartowertype
def create_sawtooth_current(
    n_pools: int,
    t_points: int,
    timestep__ms: Quantity__ms,
    amplitudes__nA: Quantity__nA | list[Quantity__nA],
    frequencies__Hz: Quantity__Hz | list[Quantity__Hz],
    offsets__nA: Quantity__nA | list[Quantity__nA] = 0.0 * pq.nA,
    widths__ratio: float | list[float] = 0.5,
    phases__rad: Quantity__rad | list[Quantity__rad] = 0.0 * pq.rad,
) -> CURRENT__AnalogSignal:
    """Create a matrix of sawtooth currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : Quantity__ms
        Time step in milliseconds as a Quantity
    amplitudes__nA : Quantity__nA | list[Quantity__nA]
        Amplitude(s) of the sawtooth current(s) in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    frequencies__Hz : Quantity__Hz | list[Quantity__Hz]
        Frequency(s) of the sawtooth current(s) in Hertz.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    offsets__nA : Quantity__nA | list[Quantity__nA]
        DC offset(s) to add to the sawtooth current(s) in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    widths__ratio : float | list[float]
        Width(s) of the rising edge as proportion of period (0 to 1).

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    phases__rad : Quantity__rad | list[Quantity__rad]
        Phase(s) of the sawtooth current(s) in radians.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    INPUT_CURRENT__AnalogSignal
        Analog signal of shape (t_points, n_pools) * pq.nA containing sawtooth currents
    """
    t = np.arange(0, t_points * timestep__ms.magnitude, timestep__ms.magnitude)

    # Convert parameters to lists and validate
    amplitudes_list = _broadcast_and_validate("amplitudes__nA", amplitudes__nA, n_pools)
    frequencies_list = _broadcast_and_validate("frequencies__Hz", frequencies__Hz, n_pools)
    offsets_list = _broadcast_and_validate("offsets__nA", offsets__nA, n_pools)
    widths_list = _broadcast_and_validate_float("widths__ratio", widths__ratio, n_pools)
    phases_list = _broadcast_and_validate("phases__rad", phases__rad, n_pools)

    return AnalogSignal(
        signal=np.stack(
            [
                (
                    amplitudes_list[i].magnitude
                    * np.where(
                        (
                            (
                                2 * np.pi * frequencies_list[i].magnitude * t / 1000
                                + phases_list[i].magnitude
                            )
                            / (2 * np.pi)
                        )
                        % 1
                        < widths_list[i],
                        (
                            (
                                2 * np.pi * frequencies_list[i].magnitude * t / 1000
                                + phases_list[i].magnitude
                            )
                            / (2 * np.pi)
                        )
                        % 1
                        / widths_list[i],
                        (
                            1
                            - (
                                (
                                    2 * np.pi * frequencies_list[i].magnitude * t / 1000
                                    + phases_list[i].magnitude
                                )
                                / (2 * np.pi)
                            )
                            % 1
                        )
                        / (1 - widths_list[i]),
                    )
                    + offsets_list[i].magnitude
                )
                for i in range(n_pools)
            ],
            axis=-1,
        )
        * pq.nA,
        t_start=0 * pq.s,
        sampling_period=timestep__ms.rescale(pq.s),
    )


@beartowertype
def create_step_current(
    n_pools: int,
    t_points: int,
    timestep__ms: Quantity__ms,
    step_heights__nA: Quantity__nA | list[Quantity__nA],
    step_durations__ms: Quantity__ms | list[Quantity__ms],
    offsets__nA: Quantity__nA | list[Quantity__nA] = 0.0 * pq.nA,
) -> CURRENT__AnalogSignal:
    """Create a matrix of step currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : Quantity__ms
        Time step in milliseconds as a Quantity
    step_heights__nA : Quantity__nA | list[Quantity__nA]
        Step height(s) for the current(s) in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools
    step_durations__ms : Quantity__ms | list[Quantity__ms]
        Step duration(s) in milliseconds as Quantities.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools
    offsets__nA : Quantity__nA | list[Quantity__nA]
        DC offset(s) to add to the step current(s) in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    INPUT_CURRENT__AnalogSignal
        Analog signal of shape (t_points, n_pools) * pq.nA containing step currents
    """
    # Convert parameters to lists and validate
    step_heights_list = _broadcast_and_validate("step_heights__nA", step_heights__nA, n_pools)
    step_durations_list = _broadcast_and_validate("step_durations__ms", step_durations__ms, n_pools)
    offsets_list = _broadcast_and_validate("offsets__nA", offsets__nA, n_pools)

    def create_step_for_pool(i: int) -> np.ndarray:
        current = np.zeros(t_points)
        duration_points = int(step_durations_list[i].magnitude / timestep__ms.magnitude)
        if duration_points > 0:
            end_idx = min(duration_points, t_points)
            current[:end_idx] = step_heights_list[i].magnitude
        return current + offsets_list[i].magnitude

    return AnalogSignal(
        signal=np.stack([create_step_for_pool(i) for i in range(n_pools)], axis=-1) * pq.nA,
        t_start=0 * pq.s,
        sampling_period=timestep__ms.rescale(pq.s),
    )


@beartowertype
def create_ramp_current(
    n_pools: int,
    t_points: int,
    timestep__ms: Quantity__ms,
    start_currents__nA: Quantity__nA | list[Quantity__nA],
    end_currents__nA: Quantity__nA | list[Quantity__nA],
    offsets__nA: Quantity__nA | list[Quantity__nA] = 0.0 * pq.nA,
) -> CURRENT__AnalogSignal:
    """Create a matrix of ramp currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : Quantity__ms
        Time step in milliseconds as a Quantity
    start_currents__nA : Quantity__nA | list[Quantity__nA]
        Starting current(s) for the ramp in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    end_currents__nA : Quantity__nA | list[Quantity__nA]
        Ending current(s) for the ramp in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools

    offsets__nA : Quantity__nA | list[Quantity__nA]
        DC offset(s) to add to the ramp current(s) in nanoamperes.

        Must be:
            - Single Quantity: used for all pools
            - List of Quantities: must match n_pools


    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    INPUT_CURRENT__AnalogSignal
        Analog signal of shape (t_points, n_pools) * pq.nA containing ramp currents
    """
    # Convert parameters to lists and validate
    start_currents_list = _broadcast_and_validate("start_currents__nA", start_currents__nA, n_pools)
    end_currents_list = _broadcast_and_validate("end_currents__nA", end_currents__nA, n_pools)
    offsets_list = _broadcast_and_validate("offsets__nA", offsets__nA, n_pools)

    return AnalogSignal(
        signal=np.stack(
            [
                np.linspace(
                    start_currents_list[i].magnitude, end_currents_list[i].magnitude, t_points
                )
                + offsets_list[i].magnitude
                for i in range(n_pools)
            ],
            axis=-1,
        )
        * pq.nA,
        t_start=0 * pq.s,
        sampling_period=timestep__ms.rescale(pq.s),
    )


@beartowertype
def create_trapezoid_current(
    n_pools: int,
    t_points: int,
    timestep__ms: Quantity__ms,
    amplitudes__nA: Quantity__nA | list[Quantity__nA],
    rise_times__ms: Quantity__ms | list[Quantity__ms] = 100.0 * pq.ms,
    plateau_times__ms: Quantity__ms | list[Quantity__ms] = 200.0 * pq.ms,
    fall_times__ms: Quantity__ms | list[Quantity__ms] = 100.0 * pq.ms,
    offsets__nA: Quantity__nA | list[Quantity__nA] = 0.0 * pq.nA,
    delays__ms: Quantity__ms | list[Quantity__ms] = 0.0 * pq.ms,
) -> CURRENT__AnalogSignal:
    """Create a matrix of trapezoidal currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : float
        Time step in milliseconds
    amplitudes__nA : float | list[float]
        Amplitude(s) of the trapezoidal current(s) in nano Amperes.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    rise_times__ms : float | list[float]
        Duration(s) of the rising phase in milliseconds.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    plateau_times__ms : float | list[float]
        Duration(s) of the plateau phase in milliseconds.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    fall_times__ms : float | list[float]
        Duration(s) of the falling phase in milliseconds.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    offsets__nA : float | list[float]
        DC offset(s) to add to the trapezoidal current(s) in nano Amperes.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    delays__ms : float | list[float]
        Delay(s) before starting the trapezoid in milliseconds.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    INPUT_CURRENT__AnalogSignal
        Analog signal of shape (t_points, n_pools) * pq.nA containing trapezoidal currents
    """
    # Convert parameters to lists and validate
    amplitudes_list = _broadcast_and_validate("amplitudes__nA", amplitudes__nA, n_pools)
    rise_times_list = _broadcast_and_validate("rise_times__ms", rise_times__ms, n_pools)
    plateau_times_list = _broadcast_and_validate("plateau_times__ms", plateau_times__ms, n_pools)
    fall_times_list = _broadcast_and_validate("fall_times__ms", fall_times__ms, n_pools)
    offsets_list = _broadcast_and_validate("offsets__nA", offsets__nA, n_pools)
    delays_list = _broadcast_and_validate("delays__ms", delays__ms, n_pools)

    def create_trapezoid_for_pool(i: int):
        # Calculate indices for each phase
        delay_points = int(delays_list[i].magnitude / timestep__ms.magnitude)
        rise_points = int(rise_times_list[i].magnitude / timestep__ms.magnitude)
        plateau_points = int(plateau_times_list[i].magnitude / timestep__ms.magnitude)
        fall_points = int(fall_times_list[i].magnitude / timestep__ms.magnitude)

        # Create the base trapezoid shape
        trapezoid = np.zeros(t_points)

        # Calculate start indices for each phase
        rise_start = delay_points
        plateau_start = rise_start + rise_points
        fall_start = plateau_start + plateau_points
        end_idx = fall_start + fall_points

        # Ensure we don't exceed array bounds
        if rise_start < t_points:
            # Rising phase (linear ramp up)
            rise_end = min(plateau_start, t_points)
            if rise_end > rise_start:
                points_to_fill = rise_end - rise_start
                trapezoid[rise_start:rise_end] = np.linspace(0, 1, points_to_fill)

            # Plateau phase (constant)
            if plateau_start < t_points:
                plateau_end = min(fall_start, t_points)
                if plateau_end > plateau_start:
                    trapezoid[plateau_start:plateau_end] = 1

                # Falling phase (linear ramp down)
                if fall_start < t_points:
                    fall_end = min(end_idx, t_points)
                    if fall_end > fall_start:
                        points_to_fill = fall_end - fall_start
                        trapezoid[fall_start:fall_end] = np.linspace(1, 0, points_to_fill)

        return amplitudes_list[i].magnitude * trapezoid + offsets_list[i].magnitude

    return AnalogSignal(
        signal=np.stack([create_trapezoid_for_pool(i) for i in range(n_pools)], axis=-1) * pq.nA,
        sampling_period=timestep__ms.rescale(pq.s),
    )

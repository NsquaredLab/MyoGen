from typing import cast

import numpy as np
from beartype import beartype

from myogen.utils.types import CORTICAL_INPUT__MATRIX


@beartype
def create_sinusoidal_cortical_input(
    n_pools: int,
    t_points: int,
    timestep__ms: float,
    amplitudes__pps: float | list[float],
    frequencies__Hz: float | list[float],
    offsets__pps: float | list[float],
    phases__rad: float | list[float] = 0.0,
) -> CORTICAL_INPUT__MATRIX:
    """Create a matrix of sinusoidal cortical input for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep__ms : float
        Time step in milliseconds
    amplitudes__pps : float | list[float]
        Amplitude(s) of the sinusoidal current(s) in pulses per second.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    frequencies__Hz : float | list[float]
        Frequency(s) of the sinusoidal current(s) in Hertz.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    offsets__pps : float | list[float]
        DC offset(s) to add to the sinusoidal current(s) in pulses per second.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    phases__rad : float | list[float]
        Phase(s) of the sinusoidal current(s) in radians.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    Raises
    ------
    ValueError
        If the amplitudes, frequencies, offsets, or phases are lists and the length of the parameters does not match n_pools

    Returns
    -------
    CORTICAL_INPUT__MATRIX
        Matrix of shape (n_pools, t_points) containing sinusoidal currents
    """
    t = np.arange(0, t_points * timestep__ms, timestep__ms)

    # Convert parameters to lists
    amplitudes_list = cast(
        list[float],
        [amplitudes__pps] * n_pools
        if isinstance(amplitudes__pps, float)
        else amplitudes__pps,
    )
    frequencies_list = cast(
        list[float],
        [frequencies__Hz] * n_pools
        if isinstance(frequencies__Hz, float)
        else frequencies__Hz,
    )
    offsets_list = cast(
        list[float],
        [offsets__pps] * n_pools if isinstance(offsets__pps, float) else offsets__pps,
    )
    phases_list = cast(
        list[float],
        [phases__rad] * n_pools if isinstance(phases__rad, float) else phases__rad,
    )

    if len(amplitudes_list) != n_pools:
        raise ValueError(
            f"Length of amplitudes__pps ({len(amplitudes_list)}) must match n_pools ({n_pools})"
        )
    if len(frequencies_list) != n_pools:
        raise ValueError(
            f"Length of frequencies__Hz ({len(frequencies_list)}) must match n_pools ({n_pools})"
        )
    if len(offsets_list) != n_pools:
        raise ValueError(
            f"Length of offsets__pps ({len(offsets_list)}) must match n_pools ({n_pools})"
        )
    if len(phases_list) != n_pools:
        raise ValueError(
            f"Length of phases__rad ({len(phases_list)}) must match n_pools ({n_pools})"
        )

    return np.array(
        [
            (
                amplitudes_list[i]
                * np.sin(2 * np.pi * frequencies_list[i] * t / 1000 + phases_list[i])
                + offsets_list[i]
            )
            for i in range(n_pools)
        ]
    )


@beartype
def create_sawtooth_cortical_input(
    n_pools: int,
    t_points: int,
    timestep_ms: float,
    amplitudes__pps: float | list[float],
    frequencies__Hz: float | list[float],
    offsets__pps: float | list[float] = 0.0,
    widths: float | list[float] = 0.5,
    phases__rad: float | list[float] = 0.0,
) -> CORTICAL_INPUT__MATRIX:
    """Create a matrix of sawtooth currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep_ms : float
        Time step in milliseconds
    amplitudes__muV : float | list[float]
        Amplitude(s) of the sawtooth current(s) in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    frequencies__Hz : float | list[float]
        Frequency(s) of the sawtooth current(s) in Hertz.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    offsets__muV : float | list[float]
        DC offset(s) to add to the sawtooth current(s) in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    widths : float | list[float]
        Width(s) of the rising edge as proportion of period (0 to 1).

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    phases__rad : float | list[float]
        Phase(s) of the sawtooth current(s) in radians.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    CORTICAL_INPUT__MATRIX
        Matrix of shape (n_pools, t_points) containing sawtooth currents
    """
    t = np.arange(0, t_points * timestep_ms, timestep_ms)

    # Convert parameters to lists
    amplitudes_list = cast(
        list[float],
        [amplitudes__pps] * n_pools
        if isinstance(amplitudes__pps, float)
        else amplitudes__pps,
    )
    frequencies_list = cast(
        list[float],
        [frequencies__Hz] * n_pools
        if isinstance(frequencies__Hz, float)
        else frequencies__Hz,
    )
    offsets_list = cast(
        list[float],
        [offsets__pps] * n_pools if isinstance(offsets__pps, float) else offsets__pps,
    )
    widths_list = cast(
        list[float], [widths] * n_pools if isinstance(widths, float) else widths
    )
    phases_list = cast(
        list[float],
        [phases__rad] * n_pools if isinstance(phases__rad, float) else phases__rad,
    )

    if len(amplitudes_list) != n_pools:
        raise ValueError(
            f"Length of amplitudes__pps ({len(amplitudes_list)}) must match n_pools ({n_pools})"
        )
    if len(frequencies_list) != n_pools:
        raise ValueError(
            f"Length of frequencies__Hz ({len(frequencies_list)}) must match n_pools ({n_pools})"
        )
    if len(offsets_list) != n_pools:
        raise ValueError(
            f"Length of offsets__pps ({len(offsets_list)}) must match n_pools ({n_pools})"
        )
    if len(widths_list) != n_pools:
        raise ValueError(
            f"Length of widths ({len(widths_list)}) must match n_pools ({n_pools})"
        )
    if len(phases_list) != n_pools:
        raise ValueError(
            f"Length of phases__rad ({len(phases_list)}) must match n_pools ({n_pools})"
        )

    cortical_input__matrix = np.zeros((n_pools, t_points))

    for i in range(n_pools):
        phase_t = 2 * np.pi * frequencies_list[i] * t / 1000 + phases_list[i]
        sawtooth = (phase_t / (2 * np.pi)) % 1
        sawtooth = np.where(
            sawtooth < widths_list[i],
            sawtooth / widths_list[i],
            (1 - sawtooth) / (1 - widths_list[i]),
        )
        cortical_input__matrix[i] = amplitudes_list[i] * sawtooth + offsets_list[i]

    return cortical_input__matrix


@beartype
def create_step_cortical_input(
    n_pools: int,
    t_points: int,
    timestep_ms: float,
    step_heights__pps: float | list[float],
    step_durations__ms: float | list[float],
    offsets__pps: float | list[float] = 0.0,
) -> CORTICAL_INPUT__MATRIX:
    """Create a matrix of step currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep_ms : float
        Time step in milliseconds.
    step_heights__muV : float | list[float]
        Step height(s) for the current(s) in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    step_durations__ms : float | list[float]
        Step duration(s) in milliseconds.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools
    offsets__muV : float | list[float]
        DC offset(s) to add to the step current(s) in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    INPUT_CURRENT__MATRIX
        Matrix of shape (n_pools, t_points) containing step currents
    """
    # Convert parameters to lists
    step_heights_list = cast(
        list[float],
        [step_heights__pps] * n_pools
        if isinstance(step_heights__pps, float)
        else step_heights__pps,
    )
    step_durations_list = cast(
        list[float],
        [step_durations__ms] * n_pools
        if isinstance(step_durations__ms, float)
        else step_durations__ms,
    )
    offsets_list = cast(
        list[float],
        [offsets__pps] * n_pools if isinstance(offsets__pps, float) else offsets__pps,
    )

    if len(step_heights_list) != n_pools:
        raise ValueError(
            f"Length of step_heights__pps ({len(step_heights_list)}) must match n_pools ({n_pools})"
        )
    if len(step_durations_list) != n_pools:
        raise ValueError(
            f"Length of step_durations__ms ({len(step_durations_list)}) must match n_pools ({n_pools})"
        )
    if len(offsets_list) != n_pools:
        raise ValueError(
            f"Length of offsets__pps ({len(offsets_list)}) must match n_pools ({n_pools})"
        )

    cortical_input__matrix = np.zeros((n_pools, t_points))

    for i in range(n_pools):
        current = np.zeros(t_points)

        # Create step: constant value for duration, then back to zero
        duration_points = int(step_durations_list[i] / timestep_ms)
        if duration_points > 0:
            end_idx = min(duration_points, t_points)
            current[:end_idx] = step_heights_list[i]

        cortical_input__matrix[i] = current + offsets_list[i]

    return cortical_input__matrix


@beartype
def create_ramp_cortical_input(
    n_pools: int,
    t_points: int,
    start_fr__pps: float | list[float],
    end_fr__pps: float | list[float],
    offsets__pps: float | list[float] = 0.0,
) -> CORTICAL_INPUT__MATRIX:
    """Create a matrix of ramp currents for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    start_currents__muV : float | list[float]
        Starting current(s) for the ramp in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    end_currents__muV : float | list[float]
        Ending current(s) for the ramp in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools

    offsets__muV : float | list[float]
        DC offset(s) to add to the ramp current(s) in microvolts.

        Must be:
            - Single float: used for all pools
            - List of floats: must match n_pools


    Raises
    ------
    ValueError
        If the parameters are lists and the length of the parameters does not match n_pools

    Returns
    -------
    CORTICAL_INPUT__MATRIX
        Matrix of shape (n_pools, t_points) containing ramp firing rates
    """
    # Convert parameters to lists
    start_fr_list = cast(
        list[float],
        [start_fr__pps] * n_pools
        if isinstance(start_fr__pps, float)
        else start_fr__pps,
    )
    end_fr_list = cast(
        list[float],
        [end_fr__pps] * n_pools if isinstance(end_fr__pps, float) else end_fr__pps,
    )
    offsets_list = cast(
        list[float],
        [offsets__pps] * n_pools if isinstance(offsets__pps, float) else offsets__pps,
    )

    if len(start_fr_list) != n_pools:
        raise ValueError(
            f"Length of start_fr__pps ({len(start_fr_list)}) must match n_pools ({n_pools})"
        )
    if len(end_fr_list) != n_pools:
        raise ValueError(
            f"Length of end_fr__pps ({len(end_fr_list)}) must match n_pools ({n_pools})"
        )
    if len(offsets_list) != n_pools:
        raise ValueError(
            f"Length of offsets__pps ({len(offsets_list)}) must match n_pools ({n_pools})"
        )

    cortical_input__matrix = np.zeros((n_pools, t_points))

    for i in range(n_pools):
        ramp = np.linspace(start_fr_list[i], end_fr_list[i], t_points)
        cortical_input__matrix[i] = ramp + offsets_list[i]

    return cortical_input__matrix


@beartype
def create_trapezoid_cortical_input(
    n_pools: int,
    t_points: int,
    timestep_ms: float,
    amplitudes__pps: float | list[float],
    rise_times__ms: float | list[float] = 100.0,
    plateau_times__ms: float | list[float] = 200.0,
    fall_times__ms: float | list[float] = 100.0,
    offsets__pps: float | list[float] = 0.0,
    delays__ms: float | list[float] = 0.0,
) -> CORTICAL_INPUT__MATRIX:
    """Create a matrix of trapezoidal firing rates for multiple pools.

    Parameters
    ----------
    n_pools : int
        Number of current pools to generate
    t_points : int
        Number of time points
    timestep_ms : float
        Time step in milliseconds
    amplitudes__pps : float | list[float]
        Amplitude(s) of the trapezoidal current(s) in pps.

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
    offsets__pps : float | list[float]
        DC offset(s) to add to the trapezoidal current(s) in pulses per second.

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
    CORTICAL_INPUT__MATRIX
        Matrix of shape (n_pools, t_points) containing trapezoidal firing rates
    """
    # Convert parameters to lists
    amplitudes_list = cast(
        list[float],
        [amplitudes__pps] * n_pools
        if isinstance(amplitudes__pps, float)
        else amplitudes__pps,
    )
    rise_times_list = cast(
        list[float],
        [rise_times__ms] * n_pools
        if isinstance(rise_times__ms, float)
        else rise_times__ms,
    )
    plateau_times_list = cast(
        list[float],
        [plateau_times__ms] * n_pools
        if isinstance(plateau_times__ms, float)
        else plateau_times__ms,
    )
    fall_times_list = cast(
        list[float],
        [fall_times__ms] * n_pools
        if isinstance(fall_times__ms, float)
        else fall_times__ms,
    )
    offsets_list = cast(
        list[float],
        [offsets__pps] * n_pools if isinstance(offsets__pps, float) else offsets__pps,
    )
    delays_list = cast(
        list[float],
        [delays__ms] * n_pools if isinstance(delays__ms, float) else delays__ms,
    )

    if len(amplitudes_list) != n_pools:
        raise ValueError(
            f"Length of amplitudes__pps ({len(amplitudes_list)}) must match n_pools ({n_pools})"
        )
    if len(rise_times_list) != n_pools:
        raise ValueError(
            f"Length of rise_times__ms ({len(rise_times_list)}) must match n_pools ({n_pools})"
        )
    if len(plateau_times_list) != n_pools:
        raise ValueError(
            f"Length of plateau_times__ms ({len(plateau_times_list)}) must match n_pools ({n_pools})"
        )
    if len(fall_times_list) != n_pools:
        raise ValueError(
            f"Length of fall_times__ms ({len(fall_times_list)}) must match n_pools ({n_pools})"
        )
    if len(offsets_list) != n_pools:
        raise ValueError(
            f"Length of offsets__pps ({len(offsets_list)}) must match n_pools ({n_pools})"
        )
    if len(delays_list) != n_pools:
        raise ValueError(
            f"Length of delays__ms ({len(delays_list)}) must match n_pools ({n_pools})"
        )

    cortical_input__matrix = np.zeros((n_pools, t_points))

    for i in range(n_pools):
        # Calculate indices for each phase
        delay_points = int(delays_list[i] / timestep_ms)
        rise_points = int(rise_times_list[i] / timestep_ms)
        plateau_points = int(plateau_times_list[i] / timestep_ms)
        fall_points = int(fall_times_list[i] / timestep_ms)

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
                        trapezoid[fall_start:fall_end] = np.linspace(
                            1, 0, points_to_fill
                        )

        cortical_input__matrix[i] = amplitudes_list[i] * trapezoid + offsets_list[i]

    return cortical_input__matrix

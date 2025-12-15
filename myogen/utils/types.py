"""
Type definitions for MyoGen with Beartype validation.

This module provides type aliases for physical quantities, neural signals, and data structures
used throughout MyoGen simulations. All types include runtime validation constraints using
Beartype's vale system to ensure data integrity and dimensional correctness.

Type Categories
---------------
- Physical Quantities: Time, angles, electrical properties, lengths, velocities
- Signal Types: Neo-based analog signals and blocks for neural data
- Array Types: NumPy arrays for matrices and multi-dimensional data structures
"""

from typing import Annotated, TypeAlias

import numpy as np
import numpy.typing as npt
import quantities as pq
from beartype.vale import Is, IsAttr, IsEqual
from neo.core.analogsignal import AnalogSignal
from neo.core.block import Block


def __make_quantity_type(reference_unit: pq.Quantity):
    return Annotated[
        pq.Quantity,
        IsAttr[
            "dimensionality",
            IsAttr["unicode", IsEqual[reference_unit.dimensionality.unicode]],
        ],
    ]


pps = pq.UnitQuantity(
    "pulses per second", pq.s**-1, symbol="pps", u_symbol="pps", doc="pulses per second"
)

# ===============================
# TIME UNITS
# ===============================

Quantity__s: TypeAlias = __make_quantity_type(pq.s)  # type: ignore
"""Physical quantity type for time in seconds."""

Quantity__ms: TypeAlias = __make_quantity_type(pq.ms)  # type: ignore
"""Physical quantity type for time in milliseconds."""

# ===============================
# ANGLES
# ===============================

Quantity__rad: TypeAlias = __make_quantity_type(pq.rad)  # type: ignore
"""Physical quantity type for angles in radians."""

Quantity__deg: TypeAlias = __make_quantity_type(pq.deg)  # type: ignore
"""Physical quantity type for angles in degrees."""

# ===============================
# ELECTRICAL POTENTIAL
# ===============================

Quantity__mV: TypeAlias = __make_quantity_type(pq.mV)  # type: ignore
"""Physical quantity type for electrical potential in millivolts."""

Quantity__uV: TypeAlias = __make_quantity_type(pq.uV)  # type: ignore
"""Physical quantity type for electrical potential in microvolts."""

# ===============================
# ELECTRICAL CURRENT
# ===============================

Quantity__nA: TypeAlias = __make_quantity_type(pq.nA)  # type: ignore
"""Physical quantity type for electrical current in nanoamperes."""

# ===============================
# ELECTRICAL CONDUCTANCE
# ===============================

Quantity__uS: TypeAlias = __make_quantity_type(pq.uS)  # type: ignore
"""Physical quantity type for electrical conductance in microsiemens."""

Quantity__S_per_m: TypeAlias = __make_quantity_type(pq.S / pq.m)  # type: ignore
"""Physical quantity type for conductivity in siemens per meter."""

# ===============================
# FREQUENCY
# ===============================

Quantity__Hz: TypeAlias = __make_quantity_type(pq.Hz)  # type: ignore
"""Physical quantity type for frequency in hertz."""

Quantity__pps: TypeAlias = __make_quantity_type(pps)  # type: ignore
"""Physical quantity type for firing rate in pulses per second."""

# ===============================
# LENGTH & AREAS
# ===============================

Quantity__mm: TypeAlias = __make_quantity_type(pq.mm)  # type: ignore
"""Physical quantity type for length in millimeters."""

Quantity__m: TypeAlias = __make_quantity_type(pq.m)  # type: ignore
"""Physical quantity type for length in meters."""

Quantity__mm2: TypeAlias = __make_quantity_type(pq.mm**2)  # type: ignore
"""Physical quantity type for area in square millimeters."""

Quantity__per_mm2: TypeAlias = __make_quantity_type(pq.mm**-2)  # type: ignore
"""Physical quantity type for density per square millimeter."""

# ===============================
# VELOCITY
# ===============================

Quantity__m_per_s: TypeAlias = __make_quantity_type(pq.m / pq.s)  # type: ignore
"""Physical quantity type for velocity in meters per second."""

Quantity__mm_per_s: TypeAlias = __make_quantity_type(pq.mm / pq.s)  # type: ignore
"""Physical quantity type for velocity in millimeters per second."""

CURRENT__AnalogSignal = Annotated[
    AnalogSignal, Is[lambda x: x.units == pq.nA and x.sampling_period.units == pq.s]
]
"""Neo AnalogSignal for input currents in nanoamperes with time in seconds.
Shape: (time_points, n_channels)"""

FORCE__AnalogSignal = Annotated[
    AnalogSignal, Is[lambda x: (x.units == pq.dimensionless) or (x.units == pq.N)]
]
"""Neo AnalogSignal for force measurements in newtons or dimensionless units.
Shape: (time_points, n_channels)"""

CORTICAL_INPUT__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]
"""2D floating-point array for cortical input patterns.
Shape: (n_motor_units, n_timesteps)"""

SPIKE_TRAIN__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.segments) > 0
        and all(hasattr(seg, "spiketrains") for seg in x.segments)
        and all(len(seg.spiketrains) > 0 for seg in x.segments)
    ],
]
"""Neo Block containing spike train data organized by motor unit pools.
Structure: segments (motor pools) → spiketrains (individual neurons)"""

SURFACE_MUAP__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.groups) > 0
        and all("ElectrodeArray_" in grp.name for grp in x.groups)
        and all(hasattr(grp, "segments") for grp in x.groups)
        and all(len(grp.segments) > 0 for grp in x.groups)
        and all("MUAP_" in seg.name for grp in x.groups for seg in grp.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(
                len(signal.shape) == 3 for signal in seg.analogsignals
            )  # (samples, rows, columns)
            for grp in x.groups
            for seg in grp.segments
        )
    ],
]
"""Neo Block containing surface motor unit action potentials (MUAPs).
Structure: groups (electrode arrays) → segments (MUAP indices) → analogsignals (samples × rows × columns)"""

SURFACE_EMG__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.groups) > 0
        and all(hasattr(grp, "segments") for grp in x.groups)
        and all(len(grp.segments) > 0 for grp in x.groups)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(
                len(signal.shape) == 3 for signal in seg.analogsignals
            )  # (samples, rows, columns)
            for grp in x.groups
            for seg in grp.segments
        )
    ],
]
"""Neo Block containing surface EMG signals.
Structure: groups (electrode arrays) → segments (motor pools) → analogsignals (time × rows × columns)"""

INTRAMUSCULAR_MUAP__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and all("MUAP_" in seg.name for seg in x.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 2 for signal in seg.analogsignals)
            for seg in x.segments
        )
    ],
]
"""Neo Block containing intramuscular motor unit action potentials (MUAPs).
Structure: segments (MUAP indices) → analogsignals (samples × electrodes)"""

INTRAMUSCULAR_EMG__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and all("Pool_" in seg.name for seg in x.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 2 for signal in seg.analogsignals)
            for seg in x.segments
        )
    ],
]
"""Neo Block containing intramuscular EMG signals.
Structure: segments (motor pools) → analogsignals (time × electrodes)"""

RECRUITMENT_THRESHOLDS__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]
"""1D array of recruitment threshold values for motor units.
Shape: (n_motor_units,)"""

JOINT_ANGLE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]
"""1D array representing joint angle trajectory over time.
Shape: (n_timesteps,)"""

MOMENT_ARM__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]
"""2D array of moment arms as a function of joint angle.
Shape: (n_angle_samples, n_muscles)"""

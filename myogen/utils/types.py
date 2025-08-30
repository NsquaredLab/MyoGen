from typing import Annotated

import numpy as np
import numpy.typing as npt
import quantities as pq
from beartype.typing import Sequence
from beartype.vale import Is
from neo.core.analogsignal import AnalogSignal
from neo.core.block import Block
from neo.core.spiketrainlist import SpikeTrainList

# Type aliases for numpy arrays with specific dimensions

# Current neo.AnalogSignal: (time_points, input_currents) * pq.nA
CURRENT__AnalogSignal = Annotated[AnalogSignal, Is[lambda x: x.units == pq.nA]]

FORCE__AnalogSignal = Annotated[
    AnalogSignal, Is[lambda x: (x.units == pq.dimensionless) or (x.units == pq.N)]
]

# Cortical input matrix: (mu_pools, time_points)
CORTICAL_INPUT__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

# Spike train tensor: (pools, neurons_per_pool, time_points)
SPIKE_TRAIN__TENSOR = Sequence[SpikeTrainList]

# Spike train neo.Block: (mu_pools - segments; time_points, spike_train - spiketrains)
SPIKE_TRAIN__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.segments) > 0
        and all(hasattr(seg, "spiketrains") for seg in x.segments)
        and all(len(seg.spiketrains) > 0 for seg in x.segments)
    ],
]

# Surface MUAP neo.Block: (electrode_array - groups; muap_index - segments; muap_samples, electrode_grid_rows, electrode_grid_columns - analogsignals)
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

# Surface EMG neo.Block: (electrode_array - groups; mu_pools - segments; time_points, electrode_grid_rows, electrode_grid_columns - analogsignals)
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

# Intramuscular MUAP neo.Block: (muap_index - segments; muap_samples, n_electrodes - analogsignals)
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

# Intramuscular EMG neo.Block: (mu_pools - segments; time_points, electrode_grid_rows, electrode_grid_columns - analogsignals)
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


# Intramuscular MUAP shape tensor: (muap_index, n_electrodes, muap_samples)
INTRAMUSCULAR_MUAP_SHAPE__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 3],
]

# Intramuscular EMG tensor: (mu_pools, n_electrodes, time_points)
INTRAMUSCULAR_EMG__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 3],
]

# Recruitment thresholds array: (n_motor_units,)
RECRUITMENT_THRESHOLDS__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# ============================================================================
# Proprioceptive Feedback Types (Phase 1 Integration)
# ============================================================================

# Spindle afferent output: (time_steps, 2) for [Ia_firing, II_firing]
SPINDLE_OUTPUT__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2 and x.shape[1] == 2],
]

# GTO afferent output: (time_steps,) for Ib_firing_rate
GTO_OUTPUT__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# Combined afferent firing rates: (time_steps, 3) for [Ia, II, Ib]
AFFERENT_FIRING__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2 and x.shape[1] == 3],
]

# Muscle mechanical state: (time_steps, 3) for [length, velocity, acceleration]
MUSCLE_STATE__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2 and x.shape[1] == 3],
]

# Joint angle trajectory: (time_steps,)
JOINT_ANGLE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# Muscle force output: (time_steps,)
MUSCLE_FORCE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# Gamma motor neuron drives: (time_steps, 2) for [dynamic, static]
GAMMA_DRIVE__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2 and x.shape[1] == 2],
]

# Descending motor commands: (time_steps, motor_unit_count)
DESCENDING_DRIVE__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

# Alpha motor neuron drive: (time_steps,)
ALPHA_MN_DRIVE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# Moment arms vs joint angle: (angle_samples, muscle_count)
MOMENT_ARM__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

# External force disturbances: (time_steps,)
FORCE_DISTURBANCE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# Joint torque output: (time_steps,)
JOINT_TORQUE__ARRAY = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 1],
]

# ============================================================================
# Parameter Dictionary Types (Phase 1 Integration)
# ============================================================================

# Hill muscle model parameters
HILL_MUSCLE_PARAMS = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Muscle spindle sensitivity parameters
SPINDLE_PARAMS = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Golgi tendon organ detection parameters
GTO_PARAMS = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Spinal neural population configuration
SPINAL_POPULATION_CONFIG = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Proprioceptive system configuration
PROPRIOCEPTION_CONFIG = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Biomechanical state for Hill muscle simulation
HILL_MUSCLE_STATE = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Joint biomechanics result containing torque and state information
JOINT_BIOMECHANICS_RESULT = Annotated[
    dict,
    Is[lambda x: isinstance(x, dict)],
]

# Force integration result combining multiple force models
FORCE_INTEGRATION_RESULT = Annotated[
    tuple,
    Is[lambda x: isinstance(x, tuple) and len(x) == 2],
]

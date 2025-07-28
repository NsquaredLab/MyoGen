from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype import beartype, BeartypeConf
from beartype.vale import Is


# See https://beartype.readthedocs.io/en/latest/api_decor/#beartype.BeartypeConf.is_pep484_tower
beartowertype = beartype(conf=BeartypeConf(is_pep484_tower=True))

# Type aliases for numpy arrays with specific dimensions

# Input current matrix: (input_currents, time_points)
INPUT_CURRENT__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

# Spike train matrix: (pools, neurons_per_pool, time_points)
SPIKE_TRAIN__MATRIX = Annotated[
    npt.NDArray[np.bool_],
    Is[lambda x: x.ndim == 3],
]

# MUAP shape tensor: (muap_index, electrode_grid_rows, electrode_grid_columns, muap_samples)
MUAP_SHAPE__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 4],
]

# Surface EMG tensor: (mu_pools, electrode_grid_rows, electrode_grid_columns, time)
SURFACE_EMG__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 4],
]

# Intramuscular EMG tensor: (mu_pools, n_electrodes, time)
INTRAMUSCULAR_EMG__TENSOR = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 3],
]

# Cortical input matrix: (mu_pools, time_points)
CORTICAL_INPUT__MATRIX = Annotated[
    npt.NDArray[np.floating],
    Is[lambda x: x.ndim == 2],
]

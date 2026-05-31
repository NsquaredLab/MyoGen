"""MyoGen utility functions and helpers."""

from myogen.utils.continuous_saver import ContinuousSaver, convert_chunks_to_neo
from myogen.utils.emg_noise import (
    add_realistic_noise,
    calibrate_baseline_drift_profile,
    calibrate_realistic_noise_profile,
    generate_realistic_noise,
    estimate_noise_rms_from_recording,
    tune_noise_profile_with_optuna,
)
from myogen.utils.neo import (
    create_grid_signal,
    signal_to_grid,
    get_electrode,
    get_row,
    get_column,
)

# NWB utilities - import lazily to avoid requiring optional dependencies
def export_to_nwb(*args, **kwargs):
    """Export to NWB format. Requires: pip install myogen[nwb]"""
    from myogen.utils.nwb import export_to_nwb as _export
    return _export(*args, **kwargs)

def export_simulation_to_nwb(*args, **kwargs):
    """Export simulation to NWB format. Requires: pip install myogen[nwb]"""
    from myogen.utils.nwb import export_simulation_to_nwb as _export
    return _export(*args, **kwargs)

def validate_nwb(*args, **kwargs):
    """Validate NWB file. Requires: pip install myogen[nwb]"""
    from myogen.utils.nwb import validate_nwb as _validate
    return _validate(*args, **kwargs)


__all__ = [
    "ContinuousSaver",
    "convert_chunks_to_neo",
    # Grid signal utilities (NWB-compatible)
    "create_grid_signal",
    "signal_to_grid",
    "get_electrode",
    "get_row",
    "get_column",
    # NWB export utilities (optional dependency)
    "export_to_nwb",
    "export_simulation_to_nwb",
    "validate_nwb",
    # Colored EMG noise model (iEMG-calibrated)
    "add_realistic_noise",
    "calibrate_baseline_drift_profile",
    "calibrate_realistic_noise_profile",
    "generate_realistic_noise",
    "estimate_noise_rms_from_recording",
    "tune_noise_profile_with_optuna",
]

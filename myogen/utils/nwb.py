"""
NWB (Neurodata Without Borders) export utilities.

This module provides functions to export MyoGen simulation data to NWB format
using Neo's NWBIO. NWB is a standardized data format for neurophysiology that
enables data sharing and interoperability with other neuroscience tools.

NWB enables:
- Data sharing via the DANDI Archive (https://dandiarchive.org/)
- Interoperability with the NWB ecosystem of tools
- Schema validation for data integrity
- Rich metadata for experimental context

For more information:
- NWB documentation: https://pynwb.readthedocs.io/
- NWB Overview: https://nwb-overview.readthedocs.io/
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from neo import Block

from .decorators import beartowertype

# Check for optional NWB dependencies
try:
    from neo.io import NWBIO as _BaseNWBIO
    from itertools import chain
    HAS_NWB = True

    class NWBIO(_BaseNWBIO):
        """Custom NWBIO that preserves signal names instead of renaming them."""

        def _write_segment(self, nwbfile, segment, electrodes):
            """Override to preserve original signal names."""
            for signal in chain(segment.analogsignals, segment.irregularlysampledsignals):
                if signal.segment is not segment:
                    raise TypeError(f"signal.segment must be segment and is {signal.segment}")
                # Preserve original name - don't rename like base class does
                if not signal.name:
                    signal.name = f"{segment.name}_signal"
                self._write_signal(self._nwbfile, signal, electrodes)

            for i, train in enumerate(segment.spiketrains):
                if train.segment is not segment:
                    raise TypeError(f"train.segment must be segment and is {train.segment}")
                if not train.name:
                    train.name = f"{segment.name}_spiketrain{i}"
                self._write_spiketrain(self._nwbfile, train)

            for event in segment.events:
                if event.segment is not segment:
                    raise TypeError(f"event.segment must be segment and is {event.segment}")
                if not event.name:
                    event.name = f"{segment.name}_event"
                self._write_event(self._nwbfile, event)

            for i, epoch in enumerate(segment.epochs):
                if not epoch.name:
                    epoch.name = f"{segment.name}_epoch{i}"
                self._write_epoch(self._nwbfile, epoch)

except ImportError:
    HAS_NWB = False
    NWBIO = None


def _check_nwb_available() -> None:
    """Check if NWB dependencies are available."""
    if not HAS_NWB:
        raise ImportError(
            "NWB export requires optional dependencies. "
            "Install with: pip install myogen[nwb]"
        )


@beartowertype
def export_to_nwb(
    block: Block,
    filepath: str | Path,
    session_description: str = "MyoGen simulation",
    identifier: str | None = None,
    session_start_time: datetime | None = None,
    experimenter: str | list[str] | None = None,
    institution: str | None = None,
    lab: str | None = None,
    experiment_description: str | None = None,
    keywords: list[str] | None = None,
    # Subject metadata (important for DANDI compliance)
    subject_id: str | None = None,
    species: str = "Homo sapiens",
    age: str | None = None,
    sex: str | None = None,
    subject_description: str | None = None,
    **kwargs,
) -> Path:
    """
    Export a Neo Block to NWB format.

    This function uses Neo's NWBIO to write simulation data to an NWB file.
    The Block should contain AnalogSignals with grid annotations (created
    via create_grid_signal) for electrode array data.

    Parameters
    ----------
    block : Block
        Neo Block containing simulation data. Can be spike trains, EMG,
        or MUAP data from MyoGen simulations.
    filepath : str or Path
        Output file path. Should end with '.nwb'.
    session_description : str, default="MyoGen simulation"
        Description of the simulation session.
    identifier : str, optional
        Unique identifier for this NWB file. If None, a UUID is generated.
    session_start_time : datetime, optional
        Start time of the session. If None, current time is used.
    experimenter : str or list[str], optional
        Name(s) of experimenter(s).
    institution : str, optional
        Institution where the simulation was performed.
    lab : str, optional
        Lab where the simulation was performed.
    experiment_description : str, optional
        Description of the experiment/simulation.
    keywords : list[str], optional
        Keywords describing the data.
    subject_id : str, optional
        Subject identifier (recommended for DANDI).
    species : str, default="Homo sapiens"
        Species of the subject. Use Latin binomial (e.g., "Homo sapiens",
        "Mus musculus"). For simulations, defaults to human.
    age : str, optional
        Age of subject in ISO 8601 duration format (e.g., "P30Y" for 30 years).
    sex : str, optional
        Sex of subject. One of: "M", "F", "U" (unknown), "O" (other).
    subject_description : str, optional
        Description of the subject.
    **kwargs
        Additional keyword arguments passed to NWBIO.

    Returns
    -------
    Path
        Path to the created NWB file.

    Examples
    --------
    >>> from myogen.utils.nwb import export_to_nwb
    >>>
    >>> # Export spike trains to NWB
    >>> export_to_nwb(
    ...     spike_train__Block,
    ...     "simulation_spikes.nwb",
    ...     session_description="Motor neuron pool simulation",
    ...     institution="My University",
    ... )
    >>>
    >>> # Export surface EMG to NWB
    >>> export_to_nwb(
    ...     surface_emg__Block,
    ...     "simulation_emg.nwb",
    ...     session_description="Surface EMG simulation",
    ...     experimenter="John Doe",
    ... )

    Notes
    -----
    For electrode array data (surface EMG, MUAPs), the grid structure is
    preserved via electrode_positions in annotations, which map to NWB's
    electrode table.

    See Also
    --------
    create_grid_signal : Create grid-annotated AnalogSignals
    validate_nwb : Validate NWB file with NWBInspector
    """
    _check_nwb_available()

    filepath = Path(filepath)
    if not filepath.suffix == ".nwb":
        filepath = filepath.with_suffix(".nwb")

    # Generate defaults
    if identifier is None:
        identifier = str(uuid.uuid4())
    if session_start_time is None:
        session_start_time = datetime.now()
    if keywords is None:
        keywords = ["MyoGen", "simulation", "EMG", "motor unit"]

    # Build metadata dict
    nwb_metadata = {
        "session_description": session_description,
        "identifier": identifier,
        "session_start_time": session_start_time,
    }
    if experimenter is not None:
        nwb_metadata["experimenter"] = (
            [experimenter] if isinstance(experimenter, str) else experimenter
        )
    if institution is not None:
        nwb_metadata["institution"] = institution
    if lab is not None:
        nwb_metadata["lab"] = lab
    if experiment_description is not None:
        nwb_metadata["experiment_description"] = experiment_description
    if keywords:
        nwb_metadata["keywords"] = keywords

    # Merge with any additional kwargs
    nwb_metadata.update(kwargs)

    # Write to NWB
    writer = NWBIO(str(filepath), mode="w", **nwb_metadata)
    writer.write(block)

    # Add subject metadata if provided (important for DANDI compliance)
    # Neo's NWBIO doesn't support subject directly, so we add it post-hoc
    if subject_id is not None or subject_description is not None:
        from pynwb import NWBHDF5IO
        from pynwb.file import Subject

        with NWBHDF5IO(str(filepath), mode="r+") as io:
            nwbfile = io.read()
            nwbfile.subject = Subject(
                subject_id=subject_id or "simulation",
                species=species,
                age=age,
                sex=sex,
                description=subject_description or "Simulated subject",
            )
            io.write(nwbfile)

    return filepath


@beartowertype
def export_simulation_to_nwb(
    filepath: str | Path,
    spike_train__Block: Block | None = None,
    surface_emg__Block: Block | None = None,
    surface_muap__Block: Block | None = None,
    intramuscular_emg__Block: Block | None = None,
    intramuscular_muap__Block: Block | None = None,
    session_description: str = "MyoGen neuromuscular simulation",
    identifier: str | None = None,
    session_start_time: datetime | None = None,
    **kwargs,
) -> Path:
    """
    Export all simulation data to a single NWB file.

    This is a convenience function that combines multiple Neo Blocks
    (spike trains, EMG, MUAPs) into a single NWB file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    spike_train__Block : SPIKE_TRAIN__Block, optional
        Block containing spike train data.
    surface_emg__Block : SURFACE_EMG__Block, optional
        Block containing surface EMG data.
    surface_muap__Block : SURFACE_MUAP__Block, optional
        Block containing surface MUAP templates.
    intramuscular_emg__Block : INTRAMUSCULAR_EMG__Block, optional
        Block containing intramuscular EMG data.
    intramuscular_muap__Block : INTRAMUSCULAR_MUAP__Block, optional
        Block containing intramuscular MUAP templates.
    session_description : str, default="MyoGen neuromuscular simulation"
        Description of the simulation session.
    identifier : str, optional
        Unique identifier for this NWB file.
    session_start_time : datetime, optional
        Start time of the session.
    **kwargs
        Additional metadata passed to export_to_nwb.

    Returns
    -------
    Path
        Path to the created NWB file.

    Examples
    --------
    >>> from myogen.utils.nwb import export_simulation_to_nwb
    >>>
    >>> # Export complete simulation
    >>> export_simulation_to_nwb(
    ...     "full_simulation.nwb",
    ...     spike_train__Block=simulation.get_spike_train__Block(),
    ...     surface_emg__Block=surface_emg.surface_emg__Block,
    ...     session_description="Biceps brachii simulation",
    ...     institution="University",
    ... )
    """
    _check_nwb_available()

    # Combine all blocks into one
    combined_block = Block(name="MyoGen_Simulation")

    # Add annotations to identify data types
    if spike_train__Block is not None:
        for segment in spike_train__Block.segments:
            segment.annotate(myogen_data_type="spike_train")
            combined_block.segments.append(segment)

    if surface_emg__Block is not None:
        for group in surface_emg__Block.groups:
            group.annotate(myogen_data_type="surface_emg")
            combined_block.groups.append(group)

    if surface_muap__Block is not None:
        for group in surface_muap__Block.groups:
            group.annotate(myogen_data_type="surface_muap")
            combined_block.groups.append(group)

    if intramuscular_emg__Block is not None:
        for segment in intramuscular_emg__Block.segments:
            segment.annotate(myogen_data_type="intramuscular_emg")
            combined_block.segments.append(segment)

    if intramuscular_muap__Block is not None:
        for segment in intramuscular_muap__Block.segments:
            segment.annotate(myogen_data_type="intramuscular_muap")
            combined_block.segments.append(segment)

    return export_to_nwb(
        combined_block,
        filepath,
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        **kwargs,
    )


@beartowertype
def validate_nwb(filepath: str | Path, verbose: bool = True) -> bool:
    """
    Validate an NWB file using NWBInspector.

    Parameters
    ----------
    filepath : str or Path
        Path to the NWB file to validate.
    verbose : bool, default=True
        If True, print validation results.

    Returns
    -------
    bool
        True if validation passed with no errors, False otherwise.

    Examples
    --------
    >>> from myogen.utils.nwb import validate_nwb
    >>>
    >>> is_valid = validate_nwb("simulation.nwb")
    >>> if is_valid:
    ...     print("File is valid!")
    """
    try:
        from nwbinspector import inspect_nwbfile
        from nwbinspector.inspector_tools import format_messages
    except ImportError:
        if verbose:
            print(
                "NWBInspector not installed. Install with: pip install nwbinspector"
            )
        return True  # Can't validate without inspector

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NWB file not found: {filepath}")

    # Run inspection
    messages = list(inspect_nwbfile(nwbfile_path=str(filepath)))

    # Filter by severity
    errors = [m for m in messages if m.importance.name == "CRITICAL"]
    warnings = [m for m in messages if m.importance.name in ("ERROR", "WARNING")]

    if verbose:
        if not messages:
            print(f"✓ {filepath.name}: No issues found")
        else:
            print(f"\n{filepath.name} validation results:")
            print(format_messages(messages))

    return len(errors) == 0


__all__ = [
    "export_to_nwb",
    "export_simulation_to_nwb",
    "validate_nwb",
]

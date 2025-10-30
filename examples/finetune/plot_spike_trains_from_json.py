r"""
Plot Spike Train Raster from JSON Configuration
================================================

This script loads a JSON configuration file from optimization results,
reconstructs the motor neuron pool and descending drive, simulates spike
trains, and generates a raster plot for specified neurons.

The script supports both DD optimization and force optimization JSON formats
and allows flexible neuron selection by indices or ranges.

Usage:
------
# Plot specific neurons
python plot_spike_trains_from_json.py \
    --json-file results/dd_optimization/FIFTEEN_gamma3.0_dd_optimized_params.json \
    --neurons 0 5 10 15 20

# Plot neuron range
python plot_spike_trains_from_json.py \
    --json-file results/force_optimization/THIRTY_gamma0.5_dd_optimized_params_force_5pct.json \
    --neurons 0-9

# Mix ranges and individual indices
python plot_spike_trains_from_json.py \
    --json-file results/dd_optimization/TWENTY_gamma1.0_dd_optimized_params.json \
    --neurons 0-5 10 15-20 25
"""

import os

os.environ["MPLBACKEND"] = "Agg"
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neo import SpikeTrain, Segment
import quantities as pq
from neuron import h
import seaborn as sns
import scienceplots  # noqa

import myogen
from myogen import RANDOM_GENERATOR
from myogen.simulator import RecruitmentThresholds
from myogen.simulator.neuron.populations import AlphaMN__Pool, DescendingDrive__Pool
from myogen.simulator.neuron import Network
from myogen.utils.nmodl import load_nmodl_mechanisms

# Import helper functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from helper import calculate_firing_rate_statistics

##############################################################################
# Configure Matplotlib Style
##############################################################################

plt.style.use(["science", "nature"])
sns.set_context("paper", font_scale=2)

# Disable LaTeX rendering
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.figsize"] = (12, 8)

# Keep text editable in SVG/PDF exports
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Set font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "Roboto", "DejaVu Sans"]

# Remove top and right spines
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False

# Make ticks and axis lines thicker
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["xtick.major.width"] = 2.0
plt.rcParams["ytick.major.width"] = 2.0

# Remove minor ticks
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = False

##############################################################################
# Configuration
##############################################################################

# Simulation constants
DEFAULT_SIMULATION_TIME_MS = 10000.0
DEFAULT_TIMESTEP_MS = 0.1
DEFAULT_SYNAPTIC_WEIGHT = 0.05  # μS
DEFAULT_N_MOTOR_UNITS = 100

# Recruitment threshold parameters
DEFAULT_RECRUITMENT_RANGE = 100
DEFAULT_DELUCA_SLOPE = 5
DEFAULT_KONSTANTIN_MAX_THRESHOLD = 1.0
DEFAULT_RECRUITMENT_MODE = "combined"

# Neural simulation
DEFAULT_SPIKE_THRESHOLD = 50  # mV

##############################################################################
# JSON Loading and Parameter Extraction
##############################################################################


def load_json_config(json_path):
    """
    Load JSON configuration file.

    Parameters
    ----------
    json_path : Path
        Path to JSON config file.

    Returns
    -------
    dict
        Configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If JSON file doesn't exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON config file not found: {json_path}")

    with open(json_path, "r") as f:
        config = json.load(f)

    return config


def extract_parameters(config):
    """
    Extract simulation parameters from JSON config.

    Handles both DD optimization and force optimization formats.

    Parameters
    ----------
    config : dict
        Configuration dictionary from JSON.

    Returns
    -------
    dict
        Dictionary with extracted parameters:
        - dd_neurons: int
        - conn_probability: float
        - dd_drive__Hz: float
        - gamma_shape: float
        - gfluctdv_enabled: bool
        - gfluctdv_noise_amplitude: float or None
        - synaptic_weight: float
        - muscle_name: str (if extractable from filename)
        - gamma_str: str (if extractable)
    """
    # Try force optimization format first
    if "dd_parameters" in config:
        dd_params = config["dd_parameters"]
        gfluctdv_enabled = config.get("gfluctdv_enabled", False)
    # Fall back to DD optimization format
    elif "best_trial" in config:
        dd_params = config["best_trial"]
        gfluctdv_enabled = config.get("input_parameters", {}).get(
            "gfluctdv_enabled", False
        )
    else:
        raise ValueError(
            "Unknown JSON format: expected 'dd_parameters' or 'best_trial' key"
        )

    # Extract core parameters
    parameters = {
        "dd_neurons": dd_params["dd_neurons"],
        "conn_probability": dd_params["conn_probability"],
        "dd_drive__Hz": dd_params["dd_drive__Hz"],
        "gamma_shape": float(dd_params["gamma_shape"]),
        "gfluctdv_enabled": gfluctdv_enabled,
        "gfluctdv_noise_amplitude": dd_params.get("gfluctdv_noise_amplitude", None),
        "synaptic_weight": dd_params.get("synaptic_weight", DEFAULT_SYNAPTIC_WEIGHT),
    }

    return parameters


def extract_muscle_gamma_from_filename(json_path):
    """
    Extract muscle name and gamma value from JSON filename.

    Expected patterns:
    - {MUSCLE}_gamma{X}_dd_optimized_params.json
    - {MUSCLE}_gamma{X}_dd_optimized_params_force_{Y}pct.json

    Parameters
    ----------
    json_path : Path
        Path to JSON file.

    Returns
    -------
    tuple
        (muscle_name, gamma_str) or (None, None) if not extractable.
    """
    filename = json_path.stem

    # Split by underscore
    parts = filename.split("_")

    try:
        # Find muscle (first part before "gamma")
        muscle_name = None
        gamma_str = None

        for i, part in enumerate(parts):
            if part.startswith("gamma"):
                if i > 0:
                    muscle_name = parts[0]
                gamma_str = part
                break

        return muscle_name, gamma_str
    except (ValueError, IndexError):
        return None, None


##############################################################################
# Neuron Selection Parsing
##############################################################################


def parse_neuron_selection(neuron_args, max_neurons):
    """
    Parse neuron selection arguments.

    Supports:
    - Individual indices: ["0", "5", "10"]
    - Ranges: ["0-10"]
    - Mixed: ["0-5", "10", "15-20"]

    Parameters
    ----------
    neuron_args : list of str
        List of neuron selection strings.
    max_neurons : int
        Maximum number of neurons in pool (for validation).

    Returns
    -------
    list of int
        Sorted list of unique neuron indices.

    Raises
    ------
    ValueError
        If any index is out of range or format is invalid.
    """
    neuron_indices = set()

    for arg in neuron_args:
        if "-" in arg:
            # Range notation: "0-10"
            try:
                start, end = arg.split("-")
                start_idx = int(start)
                end_idx = int(end)

                if start_idx > end_idx:
                    raise ValueError(f"Invalid range: {arg} (start > end)")

                neuron_indices.update(range(start_idx, end_idx + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format '{arg}': {e}")
        else:
            # Individual index: "5"
            try:
                idx = int(arg)
                neuron_indices.add(idx)
            except ValueError:
                raise ValueError(f"Invalid neuron index: '{arg}'")

    # Validate indices
    invalid_indices = [idx for idx in neuron_indices if idx < 0 or idx >= max_neurons]
    if invalid_indices:
        raise ValueError(
            f"Neuron indices out of range [0, {max_neurons-1}]: {invalid_indices}"
        )

    return sorted(list(neuron_indices))


##############################################################################
# Simulation Reconstruction
##############################################################################


def create_motor_neuron_pool(
    n_motor_units, gfluctdv_enabled, gfluctdv_noise_amplitude
):
    """
    Create motor neuron pool with recruitment thresholds.

    Parameters
    ----------
    n_motor_units : int
        Number of motor units.
    gfluctdv_enabled : bool
        Whether to enable fluctuating conductance.
    gfluctdv_noise_amplitude : float or None
        Noise amplitude for Gfluctdv.

    Returns
    -------
    AlphaMN__Pool
        Motor neuron pool.
    """
    # Create recruitment thresholds
    recruitment_thresholds, _ = RecruitmentThresholds(
        N=n_motor_units,
        recruitment_range__ratio=DEFAULT_RECRUITMENT_RANGE,
        deluca__slope=DEFAULT_DELUCA_SLOPE,
        konstantin__max_threshold__ratio=DEFAULT_KONSTANTIN_MAX_THRESHOLD,
        mode=DEFAULT_RECRUITMENT_MODE,
    )

    # Create motor neuron pool
    motor_neuron_pool = AlphaMN__Pool(
        recruitment_thresholds__array=recruitment_thresholds,
        config_file="alpha_mn_default.yaml",
    )

    # Apply Gfluctdv if enabled
    if gfluctdv_enabled and gfluctdv_noise_amplitude is not None:
        for cell in motor_neuron_pool:
            cell.insert_Gfluctdv()
            for d in cell.dend:
                d.std_e_Gfluctdv = gfluctdv_noise_amplitude
                d.std_i_Gfluctdv = gfluctdv_noise_amplitude

    return motor_neuron_pool


def create_descending_drive_pool(dd_neurons, gamma_shape, timestep_ms):
    """
    Create descending drive pool.

    Parameters
    ----------
    dd_neurons : int
        Number of descending drive neurons.
    gamma_shape : float
        Gamma process shape parameter.
    timestep_ms : float
        Simulation timestep in milliseconds.

    Returns
    -------
    DescendingDrive__Pool
        Descending drive pool.
    """
    descending_drive_pool = DescendingDrive__Pool(
        n=dd_neurons,
        timestep__ms=timestep_ms,
        process_type="gamma",
        shape=gamma_shape,
    )

    return descending_drive_pool


def setup_network(
    motor_neuron_pool, descending_drive_pool, conn_probability, synaptic_weight
):
    """
    Setup network connections between DD and motor neurons.

    Parameters
    ----------
    motor_neuron_pool : AlphaMN__Pool
        Motor neuron pool.
    descending_drive_pool : DescendingDrive__Pool
        Descending drive pool.
    conn_probability : float
        Connection probability.
    synaptic_weight : float
        Synaptic weight in μS.

    Returns
    -------
    tuple
        (network, dd_netcons) - Network object and descending drive NetCons.
    """
    network = Network({"DD": descending_drive_pool, "aMN": motor_neuron_pool})

    # Connect DD to motor neurons
    network.connect(
        source="DD",
        target="aMN",
        probability=conn_probability,
        weight__μS=synaptic_weight,
    )

    # Connect external input to DD
    network.connect_from_external(source="cortical_input", target="DD", weight__μS=1.0)

    dd_netcons = network.get_netcons("cortical_input", "DD")

    return network, dd_netcons


def setup_spike_recording(motor_neuron_pool):
    """
    Setup spike recording for motor neurons.

    Parameters
    ----------
    motor_neuron_pool : AlphaMN__Pool
        Motor neuron pool.

    Returns
    -------
    list of h.Vector
        Spike time recorders for each motor neuron.
    """
    spike_recorders = []

    for cell in motor_neuron_pool:
        spike_recorder = h.Vector()
        nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
        nc.threshold = DEFAULT_SPIKE_THRESHOLD
        nc.record(spike_recorder)
        spike_recorders.append(spike_recorder)

    return spike_recorders


def run_simulation(
    motor_neuron_pool,
    descending_drive_pool,
    dd_netcons,
    dd_drive__Hz,
    simulation_time_ms,
    timestep_ms,
):
    """
    Run NEURON simulation.

    Parameters
    ----------
    motor_neuron_pool : AlphaMN__Pool
        Motor neuron pool.
    descending_drive_pool : DescendingDrive__Pool
        Descending drive pool.
    dd_netcons : list
        Descending drive NetCons.
    dd_drive__Hz : float
        Descending drive frequency in Hz.
    simulation_time_ms : float
        Simulation duration in milliseconds.
    timestep_ms : float
        Simulation timestep in milliseconds.
    """
    # Create drive signal
    time_points = int(simulation_time_ms / timestep_ms)
    drive_signal = np.ones(time_points) * dd_drive__Hz + np.clip(
        RANDOM_GENERATOR.normal(0, 1.0, size=time_points), 0, None
    )

    # Setup NEURON simulation
    h.load_file("stdrun.hoc")
    h.dt = timestep_ms
    h.tstop = simulation_time_ms

    # Initialize voltages
    for section, voltage in zip(*motor_neuron_pool.get_initialization_data()):
        section.v = voltage
    for section, voltage in zip(*descending_drive_pool.get_initialization_data()):
        section.v = voltage

    h.finitialize()

    # Run simulation loop
    step_counter = 0
    while h.t < h.tstop:
        current_drive = drive_signal[min(step_counter, len(drive_signal) - 1)]
        for dd_cell in descending_drive_pool:
            if dd_cell.integrate(current_drive):
                if h.t < h.tstop:
                    dd_netcons[dd_cell.pool__ID].event(h.t + 1)
        h.fadvance()
        step_counter += 1


def convert_to_neo_format(spike_recorders, simulation_time_ms, timestep_ms):
    """
    Convert spike recordings to Neo format.

    Parameters
    ----------
    spike_recorders : list of h.Vector
        Spike time recorders.
    simulation_time_ms : float
        Simulation duration in milliseconds.
    timestep_ms : float
        Simulation timestep in milliseconds.

    Returns
    -------
    neo.Segment
        Neo Segment containing spike trains.
    """
    dt_s = timestep_ms / 1000.0

    segment = Segment(name="Motor Neurons")
    segment.spiketrains = [
        SpikeTrain(
            recorder.as_numpy() / 1000 * pq.s,
            t_stop=simulation_time_ms / 1000 * pq.s,
            sampling_rate=(1 / dt_s * (pq.Hz)),
            sampling_period=dt_s * pq.s,
            name=f"MN_{i}",
        )
        for i, recorder in enumerate(spike_recorders)
    ]

    return segment


##############################################################################
# Raster Plot Generation
##############################################################################


def create_raster_plot(
    spiketrains,
    neuron_indices,
    muscle_name,
    gamma_str,
    dd_drive__Hz,
    conn_probability,
    output_path,
):
    """
    Create raster plot for selected neurons.

    Parameters
    ----------
    spiketrains : list of neo.SpikeTrain
        All spike trains.
    neuron_indices : list of int
        Neuron indices to plot.
    muscle_name : str or None
        Muscle name for title.
    gamma_str : str or None
        Gamma string for title.
    dd_drive__Hz : float
        Descending drive frequency.
    conn_probability : float
        Connection probability.
    output_path : Path
        Output file path.
    """
    # Fixed figure size: 7.16 inches wide, 1.5 inches tall
    fig, ax = plt.subplots(figsize=(7.16, 1.5))

    # Calculate per-neuron statistics including CV
    per_neuron_stats = calculate_firing_rate_statistics(
        spiketrains, return_per_neuron=True
    )

    # Create CV lookup dictionary using MU_ID as key
    cv_lookup = {}
    if per_neuron_stats is not None and "CV_ISI" in per_neuron_stats.columns:
        for idx, row in per_neuron_stats.iterrows():
            cv_lookup[row["MU_ID"]] = row["CV_ISI"]

    # Color map for neurons
    colors = plt.cm.get_cmap("rainbow")(np.linspace(0, 1, len(neuron_indices)))

    # Plot selected neurons as vertical spike lines using consecutive y-positions
    y_positions = []
    y_labels = []

    for idx, mu_id in enumerate(neuron_indices):
        if mu_id < len(spiketrains):
            spiketrain = spiketrains[mu_id]
            if len(spiketrain) > 0:
                spike_times = spiketrain.rescale(pq.s).magnitude

                # Get CV for this neuron from lookup
                cv = cv_lookup.get(mu_id, None)
                cv_str = f"{cv:.3f}" if cv is not None and not np.isnan(cv) else "N/A"

                # Use idx (0, 1, 2, 3...) for y-position instead of actual mu_id
                y_pos = idx

                # Draw vertical lines for each spike at consecutive positions
                ax.vlines(
                    spike_times,
                    y_pos - 0.3,
                    y_pos + 0.3,
                    colors=colors[idx],
                    linewidth=1.0,
                    alpha=0.8,
                )

                # Add CV text on the right side of the plot
                ax.text(
                    1.01,
                    y_pos,
                    f"CV={cv_str}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9,
                    verticalalignment="center",
                    color=colors[idx],
                )

                y_positions.append(y_pos)
                y_labels.append(str(mu_id))

    # Calculate overall statistics
    stats = calculate_firing_rate_statistics(spiketrains, return_per_neuron=False)

    # Format plot
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Motor\nNeuron\nID", fontsize=12)

    # Set y-ticks to consecutive positions but labels to actual neuron IDs
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, len(y_positions) - 0.5)

    ax.tick_params(axis="both", labelsize=10)
    sns.despine(ax=ax, offset=10, trim=True)
    plt.tight_layout(pad=0.3)

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


##############################################################################
# Main Execution
##############################################################################


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Plot spike train raster from JSON configuration"
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        required=True,
        help="Path to JSON config file (DD or force optimization)",
    )
    parser.add_argument(
        "--neurons",
        type=str,
        nargs="+",
        required=True,
        help="Neuron indices to plot (e.g., '0 5 10' or '0-10' or '0-5 10-15')",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=DEFAULT_SIMULATION_TIME_MS,
        help=f"Simulation duration in ms (default: {DEFAULT_SIMULATION_TIME_MS})",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output plot file (default: auto-generated)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "svg", "pdf"],
        help="Output format (default: jpg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-motor-units",
        type=int,
        default=DEFAULT_N_MOTOR_UNITS,
        help=f"Number of motor units (default: {DEFAULT_N_MOTOR_UNITS})",
    )

    args = parser.parse_args()

    # Set random seed
    myogen.set_random_seed(args.seed)

    # Load JSON config
    config = load_json_config(args.json_file)
    params = extract_parameters(config)
    muscle_name, gamma_str = extract_muscle_gamma_from_filename(args.json_file)

    print(
        f"Loaded config: {muscle_name or 'Unknown'} | {gamma_str or 'Unknown gamma'}"
    )

    # Parse neuron selection
    neuron_indices = parse_neuron_selection(args.neurons, args.n_motor_units)
    print(f"Neurons to plot: {neuron_indices}")

    # Setup NEURON
    load_nmodl_mechanisms()
    h.secondorder = 2

    print(
        f"Simulating: {args.n_motor_units} MUs | DD={params['dd_drive__Hz']:.1f} Hz | Conn={params['conn_probability']*100:.0f}%"
    )

    # Create motor neuron pool
    motor_neuron_pool = create_motor_neuron_pool(
        args.n_motor_units,
        params["gfluctdv_enabled"],
        params["gfluctdv_noise_amplitude"],
    )

    # Create descending drive pool
    descending_drive_pool = create_descending_drive_pool(
        params["dd_neurons"], params["gamma_shape"], DEFAULT_TIMESTEP_MS
    )

    # Setup network
    network, dd_netcons = setup_network(
        motor_neuron_pool,
        descending_drive_pool,
        params["conn_probability"],
        params["synaptic_weight"],
    )

    # Setup spike recording
    spike_recorders = setup_spike_recording(motor_neuron_pool)

    # Run simulation
    run_simulation(
        motor_neuron_pool,
        descending_drive_pool,
        dd_netcons,
        params["dd_drive__Hz"],
        args.simulation_time,
        DEFAULT_TIMESTEP_MS,
    )

    # Convert to Neo format
    segment = convert_to_neo_format(
        spike_recorders, args.simulation_time, DEFAULT_TIMESTEP_MS
    )

    # Calculate statistics
    stats = calculate_firing_rate_statistics(segment.spiketrains, return_per_neuron=False)
    print(
        f"Simulation complete: {args.simulation_time/1000:.1f}s | {stats['n_active']} active MUs"
    )

    # Generate output filename if not specified
    if args.output_file is None:
        neuron_str = "_".join(map(str, neuron_indices[:5]))  # First 5 neurons
        if len(neuron_indices) > 5:
            neuron_str += "_etc"
        output_file = (
            Path("results")
            / "raster_plots"
            / f"{muscle_name or 'Unknown'}_{gamma_str or 'gamma'}_{neuron_str}.{args.output_format}"
        )
    else:
        output_file = args.output_file

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create raster plot
    create_raster_plot(
        segment.spiketrains,
        neuron_indices,
        muscle_name,
        gamma_str,
        params["dd_drive__Hz"],
        params["conn_probability"],
        output_file,
    )

    print(f"Saved: {output_file} ✅")


if __name__ == "__main__":
    main()

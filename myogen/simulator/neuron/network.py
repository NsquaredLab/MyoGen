"""
Neural Network Connectivity Module

This module provides network connectivity functionality for MyoGen's neuron models,
integrating with both the legacy NEURON-based populations and the modern MyoGen API.
"""

from math import pi, sqrt
from typing import Callable, Optional

from neuron import h
from scipy.constants import R

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron.pops import (
    AffIa__Pool,
    AffII__Pool,
    AlphaMN__Pool,
    DescendingDrive__Pool,
    GII__Pool,
)

# Network Constants
MOTOR_NEURON_CONNECTION = "aMN->Muscle"
DEFAULT_SYNAPTIC_WEIGHT = 0.6  # μS
DEFAULT_SPIKE_THRESHOLD = -10.0  # mV
DEFAULT_SYNAPTIC_DELAY = 1.0  # ms
EXTERNAL_INPUT_LABEL = "Spindle"
EXTERNAL_TARGET_LABEL = "Muscle"


# Helper functions for create_netcon
def _create_basic_netcon(source_neuron, target_neuron) -> h.NetCon:
    """
    Create the basic NEURON NetCon object based on neuron types.

    Handles the three main connection patterns:
    1. Compartmental neuron (has soma)
    2. External input (source_neuron is None)
    3. Point process neuron (has ns attribute)
    """
    if hasattr(source_neuron, "soma"):
        return h.NetCon(
            source_neuron.soma(0.5)._ref_v, target_neuron, sec=source_neuron.soma
        )
    elif source_neuron is None:
        return h.NetCon(None, target_neuron.ns)
    else:
        return h.NetCon(source_neuron.ns, target_neuron)


def _setup_muscle_activation(
    netcon: h.NetCon, muscle_callback: Optional[Callable], muscle, source_neuron
):
    """
    Setup muscle activation callback for motor neuron connections.

    Creates a wrapper function that calls the muscle_callback with the appropriate
    parameters when the source neuron fires.
    """
    if callable(muscle_callback) and muscle is not None:

        def muscle_activation_wrapper():
            return muscle_callback(
                source_neuron.class_ID, muscle, 1 + source_neuron.axonDelay
            )

        netcon.record(muscle_activation_wrapper)


def _setup_spike_recording(netcon: h.NetCon, id_vector, spike_vector, neuron_id):
    """
    Setup spike recording for post-simulation analysis.

    Records spike times and neuron IDs when all required parameters are provided.
    """
    if neuron_id is not None and id_vector is not None and spike_vector is not None:
        netcon.record(spike_vector, id_vector, neuron_id)


def _apply_default_synaptic_params(netcon: h.NetCon, source_neuron):
    """
    Apply default synaptic parameters to the NetCon.

    Sets default weight, threshold, and delay, with optional axonal delay addition.
    """
    netcon.weight[0] = DEFAULT_SYNAPTIC_WEIGHT
    netcon.threshold = DEFAULT_SPIKE_THRESHOLD
    netcon.delay = DEFAULT_SYNAPTIC_DELAY  # 1ms synaptic + axon delay

    # Add axonal delay if source neuron has it
    if hasattr(source_neuron, "axonDelay"):
        netcon.delay = (
            DEFAULT_SYNAPTIC_DELAY + source_neuron.axonDelay
        )  # 1ms synaptic + axon delay


def create_netcon(
    source_neuron,
    target_neuron,
    muscle_callback=None,
    neuron_id=None,
    id_vector=None,
    spike_vector=None,
    muscle=None,
):
    """
    Create a single NEURON NetCon (network connection) between two neurons.

    This function handles the complexity of NEURON's heterogeneous neuron types by detecting
    the source neuron architecture and creating the appropriate NetCon object. It also
    sets up optional muscle activation callbacks and spike recording for EMG simulation.

    REFACTORING NOTES:
    - Function does too many things: connection creation, muscle activation, spike recording
    - Complex conditional logic for neuron types could be simplified with polymorphism
    - Muscle activation logic tightly couples neural and muscle systems
    - Parameter validation missing (e.g., muscle_callback callable check happens after NetCon creation)

    Parameters
    ----------
    source_neuron : neuron object or None
        Source neuron. Can be:
        - Compartmental neuron (has 'soma' attribute)
        - Point process neuron (has 'ns' NetStim attribute)
        - None (for external stimulation)
    target_neuron : neuron object
        Target neuron or synapse object
    muscle_callback : callable, optional
        Muscle activation callback function. Called when source fires.
        Expected signature: muscle_callback(recruitment_id, muscle, delay_time)
    neuron_id : int, optional
        Source neuron ID for spike recording. Required if id_vector/spike_vector provided.
    id_vector : h.Vector, optional
        NEURON Vector to record neuron IDs that spike. Used with spike_vector.
    spike_vector : h.Vector, optional
        NEURON Vector to record spike times. Used with id_vector.
    muscle : object, optional
        Muscle object for motor neuron connections. Required if muscle_callback provided.

    Returns
    -------
    h.NetCon
        NEURON NetCon object with default synaptic properties:
        - weight[0] = 0.6 μS
        - threshold = -10 mV
        - delay = 1 ms + axonal delay (if available)

    Notes
    -----
    Connection types created based on source neuron:
    1. Compartmental (has soma): NetCon(soma.v, target, sec=soma)
    2. External input (src=None): NetCon(None, target.ns)
    3. Point process: NetCon(src.ns, target)

    Muscle activation: When foo and muscle are provided, creates callback
    that activates muscle fibers using source recruitment ID and axonal delay.

    Spike recording: When i, idvec, spkvec all provided, records spike times
    and neuron IDs for post-simulation analysis.
    """
    # Create the basic NEURON NetCon based on neuron types
    netcon = _create_basic_netcon(source_neuron, target_neuron)

    # Setup optional muscle activation callback
    _setup_muscle_activation(netcon, muscle_callback, muscle, source_neuron)

    # Setup optional spike recording
    _setup_spike_recording(netcon, id_vector, spike_vector, neuron_id)

    # Apply default synaptic parameters
    _apply_default_synaptic_params(netcon, source_neuron)

    return netcon


# Helper functions for connect_populations
def _connect_population_to_population(
    source_pop: str,
    target_pop: str,
    populations: dict,
    connection_probability: float,
    **kwargs,
) -> list:
    """
    Create connections between two neural populations with probabilistic connectivity.

    Implements sparse connectivity where each source-target neuron pair connects
    with the specified probability.
    """
    connections = []
    for source_neuron in populations[source_pop]:
        for target_neuron in populations[target_pop]:
            if RANDOM_GENERATOR.uniform() < connection_probability:
                target_synapse = RANDOM_GENERATOR.choice(target_neuron.synlist)
                netcon = create_netcon(
                    source_neuron,
                    target_synapse,
                    muscle_callback=kwargs.get("muscle_callback"),
                    id_vector=kwargs.get("id_vector"),
                    spike_vector=kwargs.get("spike_vector"),
                    neuron_id=source_neuron.global_ID,
                )

                # Apply custom synaptic parameters if provided
                if kwargs.get("synaptic_weight") is not None:
                    netcon.weight[0] = kwargs.get("synaptic_weight")
                if kwargs.get("spike_threshold") is not None:
                    netcon.threshold = kwargs.get("spike_threshold")

                connections.append(netcon)
    return connections


def _connect_population_to_external(
    source_pop: str, populations: dict, **kwargs
) -> list:
    """
    Create connections from a neural population to an external target (e.g., muscle).

    All neurons in the source population connect to the external target.
    """
    connections = []
    for source_neuron in populations[source_pop]:
        external_target = None
        netcon = create_netcon(
            source_neuron,
            external_target,
            muscle_callback=kwargs.get("muscle_callback"),
            id_vector=kwargs.get("id_vector"),
            muscle=kwargs.get("muscle"),
            spike_vector=kwargs.get("spike_vector"),
            neuron_id=source_neuron.global_ID,
        )

        # Apply custom spike threshold if provided
        if kwargs.get("spike_threshold") is not None:
            netcon.threshold = kwargs.get("spike_threshold")

        connections.append(netcon)
    return connections


def _connect_external_to_population(
    target_pop: Optional[str], populations: dict, **kwargs
) -> list:
    """
    Create connections from an external source (e.g., spindle) to a neural population.

    External input connects to all neurons in the target population.
    """
    connections = []
    external_source = None
    for target_neuron in populations[target_pop]:
        netcon = create_netcon(external_source, target_neuron)
        connections.append(netcon)
    return connections


def connect_populations(
    populations: dict,
    source_pop: Optional[str],
    target_pop: Optional[str],
    connection_probability: float,
    muscle_callback: Optional[Callable] = None,
    id_vector=None,
    spike_vector=None,
    muscle=None,
    synaptic_weight: Optional[float] = None,
    spike_threshold: Optional[float] = None,
):
    """
    Create probabilistic network connections between neural populations.

    Implements sparse connectivity patterns between source and target populations,
    mimicking biological neural networks where connections are probabilistic rather
    than all-to-all. Handles three connection scenarios: population-to-population,
    population-to-external (e.g., muscle), and external-to-population.

    REFACTORING NOTES:
    - Complex nested conditional logic makes function hard to understand and test
    - Three different connection patterns should be separate functions:
      * connect_populations(source_pop, target_pop, connection_probability, ...)
      * connect_to_external(source_pop, external_target, ...)
      * connect_from_external(external_source, target_pop, ...)
    - Random number generation not seeded/controlled for reproducibility
    - Side effects: modifies global random state, creates NetCons with recording
    - Inconsistent parameter passing to create_netcon (sometimes neuron_id=sc.global_ID, sometimes not)

    Parameters
    ----------
    populations : dict
        Dictionary of neural populations. Keys are population names (str),
        values are lists of neuron objects.
    source_pop : str or None
        Name of source population in populations. None for external input.
    target_pop : str or None
        Name of target population in populations. None for external target (e.g., muscle).
    connection_probability : float
        Connection probability (0.0 to 1.0). Fraction of possible connections made.
        Only applies to source→target population connections.
    muscle_callback : callable, optional
        Muscle activation callback function for motor neuron connections.
        Passed to create_netcon for muscle activation recording.
    id_vector : h.Vector, optional
        NEURON Vector for recording neuron IDs that spike.
    spike_vector : h.Vector, optional
        NEURON Vector for recording spike times.
    muscle : object, optional
        Muscle object for motor neuron→muscle connections.
    synaptic_weight : float, optional
        Synaptic weight override. If None, uses create_netcon default (0.6).
    spike_threshold : float, optional
        Spike threshold override. If None, uses create_netcon default (-10).

    Returns
    -------
    list of h.NetCon
        List of NEURON NetCon objects representing all created connections.
        Empty list if no connections made.

    Notes
    -----
    Connection patterns:
    1. Population→Population (source and target not None):
       - Nested loops over all neuron pairs
       - Each pair connects with probability 'prob'
       - Random synapse selection on target neuron

    2. Population→External (target is None):
       - All source neurons connect to external target
       - No probability filtering (100% connection rate)
       - Used for motor neuron→muscle connections

    3. External→Population (source is None):
       - External input connects to all target neurons
       - No probability, weights, or thresholds applied
       - Simple stimulus input connections

    Random synapse selection uses np.random.choice(tg.synlist) which assumes
    target neurons have a 'synlist' attribute containing available synapses.

    Global ID usage: Uses sc.global_ID for spike recording, assuming source
    neurons have this attribute for unique identification.
    """
    # Parameter validation
    if not 0.0 <= connection_probability <= 1.0:
        raise ValueError(
            f"Connection probability must be between 0.0 and 1.0, got {connection_probability}"
        )

    if source_pop is not None and source_pop not in populations:
        raise ValueError(
            f"Source population '{source_pop}' not found in population dictionary"
        )

    if target_pop is not None and target_pop not in populations:
        raise ValueError(
            f"Target population '{target_pop}' not found in population dictionary"
        )

    # Prepare keyword arguments for helper functions
    connection_kwargs = {
        "muscle_callback": muscle_callback,
        "id_vector": id_vector,
        "spike_vector": spike_vector,
        "muscle": muscle,
        "synaptic_weight": synaptic_weight,
        "spike_threshold": spike_threshold,
    }

    # Route to appropriate connection type function
    if source_pop is not None and target_pop is not None:
        # Population to population connection
        return _connect_population_to_population(
            source_pop,
            target_pop,
            populations,
            connection_probability,
            **connection_kwargs,
        )
    elif source_pop is not None and target_pop is None:
        # Population to external target (e.g., muscle)
        return _connect_population_to_external(
            source_pop, populations, **connection_kwargs
        )
    else:
        # External source to population (e.g., spindle input)
        return _connect_external_to_population(
            target_pop, populations, **connection_kwargs
        )


# Helper functions for print_network_connections
def _get_connection_source_name(netcon: h.NetCon) -> str:
    """
    Get a descriptive name for the connection source.

    Handles different source types:
    - Regular neurons with presynaptic cell
    - External inputs (preloc != -1)
    - Unknown external inputs (default to EXTERNAL_INPUT_LABEL)
    """
    source_cell = netcon.pre()

    if source_cell is None:
        presynaptic_location = netcon.preloc()
        if presynaptic_location != -1:
            # NEURON introspection - this affects global state
            source_cell = h.cas()
            h.pop_section()
            return str(source_cell)
        else:
            return EXTERNAL_INPUT_LABEL

    return str(source_cell)


def _get_connection_target_name(netcon: h.NetCon) -> str:
    """
    Get a descriptive name for the connection target.

    Handles different target types:
    - Postsynaptic cells
    - Synapses
    - External targets (default to EXTERNAL_TARGET_LABEL)
    """
    target_cell = netcon.postcell()

    if target_cell is None:
        target_cell = netcon.syn()

    if target_cell is None:
        return EXTERNAL_TARGET_LABEL

    return str(target_cell)


def _format_connection_info(netcon: h.NetCon, connection_index: int) -> str:
    """
    Format connection information as a string for display.

    Returns a formatted string: "NC[index]: source -> target"
    """
    source_name = _get_connection_source_name(netcon)
    target_name = _get_connection_target_name(netcon)
    return "NC[{}]: {} -> {}".format(connection_index, source_name, target_name)


def print_network_connections(network_connections):
    """
    Print a human-readable summary of all network connections.

    Iterates through a dictionary of NetCon lists and displays each connection
    with source and target information. Handles various NEURON object types
    and provides fallback labels for special connection types (Muscle, Spindle).

    REFACTORING NOTES:
    - Function name violates Python naming convention (should be print_nc_list)
    - Hardcoded string labels ("Muscle", "Spindle") should be configurable
    - Complex conditional logic for determining source/target names
    - Side effect function (prints) should return formatted strings instead
    - No error handling for malformed NetCon objects
    - Uses global NEURON state (h.cas(), h.pop_section()) unsafely
    - Should use logging instead of print statements

    Parameters
    ----------
    network_connections : dict
        Dictionary where keys are connection names (str) and values are
        lists of h.NetCon objects created by connect_populations().

    Returns
    -------
    None
        Prints connection information to stdout. No return value.

    Notes
    -----
    Connection display format: "NC[index]: source -> target"

    Source identification logic:
    1. If nc.pre() exists: Use the presynaptic object
    2. If nc.preloc() != -1: Use current access section (h.cas())
    3. Otherwise: Label as "Spindle" (external input)

    Target identification logic:
    1. If nc.postcell() exists: Use the postsynaptic cell
    2. If nc.syn() exists: Use the synapse object
    3. Otherwise: Label as "Muscle" (external target)

    The h.cas() and h.pop_section() calls are NEURON-specific functions
    for accessing the current section stack, which may have side effects
    on global NEURON state.

    Examples
    --------
    >>> ncD = {"aMN->Muscle": [nc1, nc2], "Input->aMN": [nc3]}
    >>> printNClist(ncD)
    NC[0]: <neuron_obj> -> Muscle
    NC[1]: <neuron_obj> -> Muscle
    NC[2]: Spindle -> <target_neuron>
    """
    connection_index = 0
    for connection_group in network_connections.values():
        for netcon in connection_group:
            # Format and print connection information
            connection_info = _format_connection_info(netcon, connection_index)
            print(connection_info)
            connection_index += 1


# Helper functions for create_network
def _extract_connection_parameters(
    connection_name: str,
    connection_config: dict,
    muscle_callback: Optional[Callable],
    muscle,
) -> tuple:
    """
    Extract connection parameters from configuration and determine callback/muscle settings.

    Returns:
        tuple: (callback_function, muscle_object, custom_threshold)
    """
    # Determine if this is a motor neuron connection requiring muscle activation
    if connection_name == MOTOR_NEURON_CONNECTION:
        callback_function = muscle_callback
        muscle_object = muscle
    else:
        callback_function = None
        muscle_object = None

    # Extract custom threshold if specified
    custom_threshold = connection_config.get("threshold")
    if custom_threshold is not None:
        print("Definiu o Threshold para")

    return callback_function, muscle_object, custom_threshold


def _setup_spike_vectors(connection_config: dict, id_vector, spike_vector) -> tuple:
    """
    Setup spike recording vectors for the connection's source population.

    Returns:
        tuple: (source_id_vector, source_spike_vector)
    """
    source_population = connection_config.get("source")

    if id_vector is not None and source_population in id_vector:
        source_id_vector = id_vector[source_population]
        source_spike_vector = (
            spike_vector[source_population] if spike_vector is not None else None
        )

        return source_id_vector, source_spike_vector

    return None, None


def create_network(
    populations: dict[str, list],
    connections_config: dict,
    id_vector=None,
    spike_vector=None,
    muscle_callback: Optional[Callable] = None,
    spike_save: Optional[list] = None,
    muscle=None,
):
    """
    Create a complete neural network from population and connection specifications.

    High-level network builder that orchestrates the creation of all neural connections
    based on structured parameter dictionaries. Handles special cases like motor neuron
    to muscle connections and spike recording setup for different population types.

    REFACTORING NOTES:
    - Side effects: print statements should use logging
    - Complex parameter extraction logic should be extracted to helper functions
    - Missing parameter validation (connections_config structure, population existence)
    - Reference to external thesis (pg 104 [Elias PhD]) should be in module docstring

    Parameters
    ----------
    populations : dict
        Dictionary of neural populations. Keys are population names (str),
        values are lists of neuron objects. Same as populations in connect_populations.
    connections_config : dict
        Network connectivity specification. Each key is a connection name (str),
        each value is a dict containing:
        - "source": source population name (str) or None
        - "target": target population name (str) or None
        - "connP": connection probability (float, 0-1)
        - "w": synaptic weight (float, optional)
        - "threshold": spike threshold (float, optional)
    id_vector : h.Vector or dict, optional
        Spike recording ID vectors. Can be:
        - Single h.Vector for all populations
        - Dict mapping population names to h.Vector objects
        - None to disable spike ID recording
    spike_vector : h.Vector or dict, optional
        Spike recording time vectors. Structure must match id_vector.
    muscle_callback : callable, optional
        Muscle activation callback for motor neuron connections.
        Applied only to connections named MOTOR_NEURON_CONNECTION.
    spike_save : list, optional
        Legacy parameter for compatibility. Not actively used.
    muscle : object, optional
        Muscle object for motor neuron to muscle connections.
        Required if muscle_callback is provided and motor neuron connections exist.

    Returns
    -------
    dict
        Dictionary mapping connection names (str) to lists of h.NetCon objects.
        Keys match connection_params keys, values are NetCon lists from genPopNC.

    Notes
    -----
    Special connection handling:
    - "aMN->Muscle" connections get muscle activation (foo) and muscle object
    - All other connections use standard neural-to-neural parameters

    Spike recording logic:
    - If idvec/spkvec contain source population name, extracts vectors for that population
    - Otherwise, uses None (no spike recording for that connection)

    Connection creation delegates to genPopNC for actual NetCon generation,
    this function primarily handles parameter routing and special cases.

    Examples
    --------
    >>> pop_params = {"aMN": [neuron1, neuron2], "Ia": [sensory1]}
    >>> conn_params = {
    ...     "Ia->aMN": {"source": "Ia", "target": "aMN", "connP": 0.3, "w": 0.8},
    ...     "aMN->Muscle": {"source": "aMN", "target": None, "connP": 1.0}
    ... }
    >>> network = genNetwork__Adapted(pop_params, conn_params, muscle=my_muscle)
    """
    # Parameter validation
    if not isinstance(populations, dict):
        raise TypeError("populations must be a dictionary")

    if not isinstance(connections_config, dict):
        raise TypeError("connections_config must be a dictionary")

    # Handle mutable default parameter
    if spike_save is None:
        spike_save = []

    # weights and connections at table 7, pg 104 [Elias PhD tesis]
    print("Generating network")
    network_connections_dict = {}

    for connection_name, connection_config in connections_config.items():
        # Extract connection parameters and muscle settings
        callback_function, muscle_object, custom_threshold = (
            _extract_connection_parameters(
                connection_name, connection_config, muscle_callback, muscle
            )
        )

        # Setup spike recording vectors for this connection's source population
        source_id_vector, source_spike_vector = _setup_spike_vectors(
            connection_config, id_vector, spike_vector
        )

        # Create connections for this connection group
        network_connections_dict[connection_name] = connect_populations(
            populations=populations,
            source_pop=connection_config["source"],
            target_pop=connection_config["target"],
            connection_probability=connection_config["connP"],
            synaptic_weight=connection_config["w"],
            spike_threshold=custom_threshold,
            id_vector=source_id_vector,
            spike_vector=source_spike_vector,
            muscle_callback=callback_function,
            muscle=muscle_object,
        )

    print("network created")
    return network_connections_dict


if __name__ == "__main__":
    from myogen import setup_myogen

    setup_myogen()

    timestep__ms = 0.05

    dd__pool = DescendingDrive__Pool(
        n=2, poisson_random_process_order=16, timestep__ms=timestep__ms
    )

    # Alpha motor neuron parameters
    n_type1 = 2
    n_type2 = 2
    n_alpha_mn = n_type1 + n_type2

    alphaMN__pool = AlphaMN__Pool(
        n=n_alpha_mn,
        model="Powers2017",
        mode="active",
        axon_velocities=(44, 53),
        axon_length=0.9,  # cm
        gamma=1.0,
        # Soma parameters
        soma_length_range=(2952, 3665, 0.3),
        soma_diameter_range=(22, 30, 0.3),
        soma_capacitance_range=(1.35546, 1.87853, 0.3),
        soma_passive_conductance_range=(8.11e-5, 3.77e-4, 0.3),
        soma_passive_reversal_range=(-71, -72, 0.3),
        soma_na3rp_conductance_range=(0.01, 0.022, 0.3),
        soma_naps_conductance_range=(2.6e-5, 2e-5, 0.3),
        soma_kdrrl_conductance_range=(0.015, 0.02, 0.3),
        soma_mahp_ca_conductance_range=(6.4e-6, 1.015e-5, 0.075),
        soma_mahp_k_conductance_range=(4.5e-4, 6e-4, 0.3),
        soma_mahp_tau_range=(90, 30, 0.3),
        soma_gh_conductance_range=(3e-5, 2.3e-4, 0.3),
        # Dendrite parameters
        dendrite_length_range=(1794.13, 2226.91, 0.3),
        dendrite_diameter_range=(8.73071, 11.9055, 0.3),
        dendrite_passive_conductance_range=(7.93e-5, 1.75e-4, 0.3),
        dendrite_passive_reversal_range=(-71, -72, 0.3),
        dendrite_resistance_range=(51.038, 40.755, 0.3),
        dendrite_capacitance_range=(0.867781, 0.880407, 0.3),
        dendrite_gh_conductance_range=(3e-5, 2.3e-4, 0.3),
        # Ca channel parameters - 4 dendrites
        dendrite_ca_conductance_ranges=(
            (8.5e-5, 1.18e-4, 0.3),
            (9.5e-5, 1.28e-4, 0.3),
            (1e-4, 1.38e-4, 0.3),
            (1.15e-4, 1.53e-4, 0.3),
        ),
        dendrite_ca_theta_m_range=(-42, -39, 0.3),
        dendrite_ca_theta_h_range=(10, -10, 0.3),
    )

    ia_pool = AffIa__Pool(
        n=2,
        poisson_random_process_order=25,
        recruitment_thresholds=(0, 150),
        axon_velocities=(62, 67),
        axon_length__m=1.0,  # cm
        timestep__ms=timestep__ms,
    )

    ii_pool = AffII__Pool(
        n=2,
        poisson_random_process_order=25,
        recruitment_thresholds=(0, 50),
        axon_velocities=(30, 35),
        axon_length__m=1.0,  # cm
        timestep__ms=timestep__ms,
    )

    def gIIL(i):
        Amu = 81390 + 3113  # [um^2] Bui et al.(2003) IaIn data.
        Aci = 1.96 * (891.5 + 46.141) / sqrt(8)
        A = [Amu - Aci, Amu + Aci]
        D = [sqrt(A[0] / pi), sqrt(A[1] / pi)]
        return D[i]

    gii_pool = GII__Pool(
        n=2,
        soma_length_range__μm=(gIIL(0), gIIL(1)),
        soma_diameter_range=(gIIL(0), gIIL(1)),
        passive_conductance_range=(3e-5, 7e-5),
        na3rp_conductance_range=(0.003, 0.01),
        kdrrl_conductance_range=(0.015, 0.015),
        mahp_ca_conductance_range=(3e-6, 3e-6),
        mahp_k_conductance_range=(5e-4, 5e-4),
        mahp_tau_range=(60, 70),
        gh_conductance_range=(2.5e-5, 2.5e-5),
        axon_length=0.5,  # cm
        axon_velocities=(10, 10),
    )

    # Interneuron parameters
    group_ii_interneuron_params = {"n": 2}  # Number of group II interneurons

    # Population parameter dictionary
    population_params = {
        "DD": dd__pool,  # Descending drive
        "aMN": alphaMN__pool,  # Alpha motoneurones
        "Ia": ia_pool,  # Afferent Ia
        "II": ii_pool,  # Afferent II
        "gII": gii_pool,  # Group II interneurones
    }

    # Connection Parameters
    connection_params = {
        "DD->aMN": {
            "connP": 0.9,  # Connection probability
            "source": "DD",
            "target": "aMN",
            "w": 0.6,  # Synaptic weight
        },
        "DD->gII": {
            "connP": 0.9,  # Connection probability
            "source": "DD",
            "target": "gII",
            "w": 0.6,  # Synaptic weight
        },
        "aMN->Muscle": {
            "connP": 1.0,  # Connection probability
            "source": "aMN",
            "target": None,  # Muscle target
            "w": 1.0,  # Synaptic weight
        },
        "gII->aMN": {
            "connP": 0.9,  # Connection probability
            "source": "gII",
            "target": "aMN",
            "w": 0.3,  # [uS] Synaptic weight
        },
        "Spindle->Ia": {
            "connP": 1.0,  # Connection probability
            "source": None,  # External spindle input
            "target": "Ia",
            "w": 0.8,  # [uS] Synaptic weight
        },
        "Ia->aMN": {
            "connP": 0.9,  # Connection probability
            "source": "Ia",
            "target": "aMN",
            "w": 0.6,  # [uS] Synaptic weight
        },
        "Spindle->II": {
            "connP": 1.0,  # Connection probability
            "source": None,  # External spindle input
            "target": "II",
            "w": 0.5,  # [uS] Synaptic weight
        },
        "II->gII": {
            "connP": 0.3,  # Connection probability
            "source": "II",
            "target": "gII",
            "w": 0.4,  # [uS] Synaptic weight
        },
    }

    def foo():
        print("oi")

    spike_id_vector = h.Vector()
    spike_time_vector = h.Vector()

    network_connections_dict = create_network(
        populations=population_params,
        connections_config=connection_params,
        id_vector=spike_id_vector,
        spike_vector=spike_time_vector,
        muscle_callback=foo,
        spike_save=[],
        muscle=None,
    )

    print_network_connections(network_connections_dict)

"""
Neural Network Connectivity Module

This module provides network connectivity functionality for MyoGen's neuron models,
integrating with both the legacy NEURON-based populations and the modern MyoGen API.
"""

from typing import Callable, Optional

import quantities as pq
from neuron import h

from myogen import RANDOM_GENERATOR
from myogen.simulator.neuron.populations import (
    AffIa__Pool,
    AffII__Pool,
    AlphaMN__Pool,
    DescendingDrive__Pool,
    GII__Pool,
    _Pool,
)
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__mV, Quantity__ms, Quantity__uS

# Network Constants
MOTOR_NEURON_CONNECTION = "aMN->Muscle"

DEFAULT_SYNAPTIC_WEIGHT = 0.6 * pq.uS  # uS
DEFAULT_SPIKE_THRESHOLD = -10.0 * pq.mV  # mV
DEFAULT_SYNAPTIC_DELAY = 1.0 * pq.ms  # ms
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
        return h.NetCon(source_neuron.soma(0.5)._ref_v, target_neuron, sec=source_neuron.soma)
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
                source_neuron.pool__ID, muscle, 1 + float(source_neuron.axon_delay__ms)
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
    if hasattr(source_neuron, "axon_delay__ms") and source_neuron.axon_delay__ms is not None:
        netcon.delay = (
            DEFAULT_SYNAPTIC_DELAY + source_neuron.axon_delay__ms
        )  # 1ms synaptic + axon delay


def _create_netcon(
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
        - weight[0] = 0.6 uS
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
    deterministic: bool = False,
    **kwargs,
) -> list:
    """
    Create connections between two neural populations with probabilistic or deterministic connectivity.

    Implements sparse connectivity using either:
    - Probabilistic: Each source-target pair connects with specified probability (default)
    - Deterministic: Each source connects to exactly N targets where N = probability × n_targets

    Parameters
    ----------
    deterministic : bool, optional
        If True, each source neuron connects to exactly int(connection_probability × n_targets)
        randomly selected target neurons. If False, uses probabilistic sampling. Default False.
    """
    connections = []
    target_neurons = populations[target_pop]

    if deterministic and connection_probability < 1.0:
        # Deterministic connectivity: each source connects to exact number of targets
        n_connections = int(connection_probability * len(target_neurons))

        for source_neuron in populations[source_pop]:
            # Randomly select exactly n_connections target neurons
            selected_targets = RANDOM_GENERATOR.choice(
                target_neurons, size=n_connections, replace=False
            )

            for target_neuron in selected_targets:
                target_synapse = RANDOM_GENERATOR.choice(target_neuron.synapse__list)
                netcon = _create_netcon(
                    source_neuron,
                    target_synapse,
                    muscle_callback=kwargs.get("muscle_callback"),
                    id_vector=kwargs.get("id_vector"),
                    spike_vector=kwargs.get("spike_vector"),
                    neuron_id=source_neuron.global__ID,
                )

                # Apply custom synaptic parameters if provided
                if kwargs.get("synaptic_weight") is not None:
                    netcon.weight[0] = kwargs.get("synaptic_weight")
                if kwargs.get("spike_threshold") is not None:
                    netcon.threshold = kwargs.get("spike_threshold")

                connections.append(netcon)
    else:
        # Probabilistic connectivity: each pair has probability of connecting
        for source_neuron in populations[source_pop]:
            for target_neuron in target_neurons:
                if RANDOM_GENERATOR.uniform() < connection_probability:
                    target_synapse = RANDOM_GENERATOR.choice(target_neuron.synapse__list)
                    netcon = _create_netcon(
                        source_neuron,
                        target_synapse,
                        muscle_callback=kwargs.get("muscle_callback"),
                        id_vector=kwargs.get("id_vector"),
                        spike_vector=kwargs.get("spike_vector"),
                        neuron_id=source_neuron.global__ID,
                    )

                    # Apply custom synaptic parameters if provided
                    if kwargs.get("synaptic_weight") is not None:
                        netcon.weight[0] = kwargs.get("synaptic_weight")
                    if kwargs.get("spike_threshold") is not None:
                        netcon.threshold = kwargs.get("spike_threshold")

                    connections.append(netcon)
    return connections


def _connect_population_to_external(source_pop: str, populations: dict, **kwargs) -> list:
    """
    Create connections from a neural population to an external target (e.g., muscle).

    All neurons in the source population connect to the external target.
    """
    connections = []
    for source_neuron in populations[source_pop]:
        external_target = None
        netcon = _create_netcon(
            source_neuron,
            external_target,
            muscle_callback=kwargs.get("muscle_callback"),
            id_vector=kwargs.get("id_vector"),
            muscle=kwargs.get("muscle"),
            spike_vector=kwargs.get("spike_vector"),
            neuron_id=source_neuron.global__ID,
        )

        # Apply custom spike threshold if provided
        if kwargs.get("spike_threshold") is not None:
            netcon.threshold = kwargs.get("spike_threshold")

        connections.append(netcon)
    return connections


def _connect_external_to_population(target_pop: Optional[str], populations: dict, **kwargs) -> list:
    """
    Create connections from an external source (e.g., spindle) to a neural population.

    External input connects to all neurons in the target population.
    """
    connections = []
    external_source = None
    for target_neuron in populations[target_pop]:
        # Note: For external connections, we don't record the external source spikes,
        # but we can still record target neuron spikes if they're the source in other connections
        netcon = _create_netcon(
            external_source,
            target_neuron,
            id_vector=kwargs.get("id_vector"),
            spike_vector=kwargs.get("spike_vector"),
            neuron_id=None,  # External source has no ID to record
        )
        connections.append(netcon)
    return connections


def _connect_one_to_one(
    source_pop: str,
    target_pop: str,
    populations: dict,
    connection_probability: float = 1.0,
    **kwargs,
) -> list:
    """
    Create one-to-one connections between two neural populations.

    Connects source[i] to target[i] for all matching indices with specified probability.
    This is useful for creating independent noise inputs where each target neuron receives
    input from exactly one source neuron (e.g., independent Poisson processes to motor neurons).

    Parameters
    ----------
    source_pop : str
        Name of source population in populations dict.
    target_pop : str
        Name of target population in populations dict.
    populations : dict
        Dictionary of neural populations.
    connection_probability : float, optional
        Probability that each source[i] -> target[i] connection is made, by default 1.0.
        Must be between 0.0 and 1.0.
    **kwargs
        Additional keyword arguments passed to _create_netcon:
        - muscle_callback: Muscle activation callback
        - id_vector: Spike recording ID vector
        - spike_vector: Spike recording time vector
        - synaptic_weight: Synaptic weight override
        - spike_threshold: Spike threshold override

    Returns
    -------
    list of h.NetCon
        List of NEURON NetCon objects for connections that were created.

    Raises
    ------
    ValueError
        If source and target populations have different sizes, or if
        connection_probability is not in [0.0, 1.0].

    Notes
    -----
    Requires source and target populations to have equal size. Each source neuron
    may connect to the target neuron at the same index position, depending on
    connection_probability.
    """
    source_neurons = populations[source_pop]
    target_neurons = populations[target_pop]

    if len(source_neurons) != len(target_neurons):
        raise ValueError(
            f"One-to-one connection requires equal population sizes. "
            f"Source '{source_pop}' has {len(source_neurons)} neurons, "
            f"target '{target_pop}' has {len(target_neurons)} neurons."
        )

    if not 0.0 <= connection_probability <= 1.0:
        raise ValueError(
            f"Connection probability must be between 0.0 and 1.0, got {connection_probability}"
        )

    connections = []
    for source_neuron, target_neuron in zip(source_neurons, target_neurons):
        # Check if this pair should be connected
        if RANDOM_GENERATOR.uniform() < connection_probability:
            target_synapse = RANDOM_GENERATOR.choice(target_neuron.synapse__list)
            netcon = _create_netcon(
                source_neuron,
                target_synapse,
                muscle_callback=kwargs.get("muscle_callback"),
                id_vector=kwargs.get("id_vector"),
                spike_vector=kwargs.get("spike_vector"),
                neuron_id=source_neuron.global__ID,
            )

            # Apply custom synaptic parameters if provided
            if kwargs.get("synaptic_weight") is not None:
                netcon.weight[0] = kwargs.get("synaptic_weight")
            if kwargs.get("spike_threshold") is not None:
                netcon.threshold = kwargs.get("spike_threshold")

            connections.append(netcon)

    return connections


def _connect_populations(
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
    deterministic: bool = False,
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
    - Inconsistent parameter passing to create_netcon (sometimes neuron_id=sc.global__ID, sometimes not)

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

    Random synapse selection uses np.random.choice(tg.synapse__list) which assumes
    target neurons have a 'synapse__list' attribute containing available synapses.

    Global ID usage: Uses sc.global__ID for spike recording, assuming source
    neurons have this attribute for unique identification.
    """
    # Parameter validation
    if not 0.0 <= connection_probability <= 1.0:
        raise ValueError(
            f"Connection probability must be between 0.0 and 1.0, got {connection_probability}"
        )

    if source_pop is not None and source_pop not in populations:
        raise ValueError(f"Source population '{source_pop}' not found in population dictionary")

    if target_pop is not None and target_pop not in populations:
        raise ValueError(f"Target population '{target_pop}' not found in population dictionary")

    # Prepare keyword arguments for helper functions
    connection_kwargs = {
        "muscle_callback": muscle_callback,
        "id_vector": id_vector,
        "spike_vector": spike_vector,
        "muscle": muscle,
        "synaptic_weight": synaptic_weight,
        "spike_threshold": spike_threshold,
        "deterministic": deterministic,
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
        return _connect_population_to_external(source_pop, populations, **connection_kwargs)
    else:
        # External source to population (e.g., spindle input)
        return _connect_external_to_population(target_pop, populations, **connection_kwargs)


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


def _print_network_connections(network_connections):
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
        source_spike_vector = spike_vector[source_population] if spike_vector is not None else None

        return source_id_vector, source_spike_vector

    return None, None


def _create_network(
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
        callback_function, muscle_object, custom_threshold = _extract_connection_parameters(
            connection_name, connection_config, muscle_callback, muscle
        )

        # Setup spike recording vectors for this connection's source population
        source_id_vector, source_spike_vector = _setup_spike_vectors(
            connection_config, id_vector, spike_vector
        )

        # Create connections for this connection group
        network_connections_dict[connection_name] = _connect_populations(
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


@beartowertype
class Network:
    """
    Modern neural network builder with intuitive connection API.

    Provides a clean, discoverable interface for creating neural network connections
    while maintaining compatibility with existing NEURON-based infrastructure.
    """

    def __init__(self, populations: dict[str, _Pool], spike_recording: Optional[dict] = None):
        """
        Initialize network with neural populations.

        Parameters
        ----------
        populations : dict[str, Union[list, Any]]
            Dictionary mapping population names to Pool objects or lists of neuron objects.
            Pool objects will have .neurons extracted, lists used directly.
            Example: {"alpha_mn": alphaMN_pool, "ia": ia_pool}
        spike_recording : dict, optional
            Dictionary containing 'idvec' and 'spkvec' for spike recording.
            Example: {"idvec": {"aMN": h.Vector()}, "spkvec": {"aMN": h.Vector()}}
        """
        self.populations: dict[str, _Pool] = populations
        self.connections = []
        self._netcons_by_connection: dict[tuple[str, str], list[h.NetCon]] = {}
        self.spike_recording = spike_recording

    def setup_spike_recording(self):
        """
        Set up spike recording NetCons for all neurons in all populations.

        This creates additional NetCons specifically for recording spikes from neurons
        that might not have outgoing connections but still need spike recording for analysis.
        """
        if not self.spike_recording:
            return

        for pop_name, population in self.populations.items():
            # Skip non-neuron populations (like gMN which is a config dict)
            if not hasattr(population, "__iter__") or isinstance(population, dict):
                continue

            # Get spike recording vectors for this population
            id_vector = self.spike_recording.get("idvec", {}).get(pop_name)
            spike_vector = self.spike_recording.get("spkvec", {}).get(pop_name)

            if id_vector is not None and spike_vector is not None:
                # Create NetCons for spike recording (no target, just recording)
                recording_netcons = []
                for neuron in population:
                    # Skip if this is not actually a neuron object
                    if not hasattr(neuron, "soma") and not hasattr(neuron, "ns"):
                        continue

                    # Create a NetCon from the neuron to None (just for recording)
                    if hasattr(neuron, "soma"):
                        # Compartmental neuron
                        nc = h.NetCon(neuron.soma(0.5)._ref_v, None, sec=neuron.soma)
                    else:
                        # Point process neuron
                        nc = h.NetCon(neuron.ns, None)

                    # Set up spike recording with population-specific threshold
                    if hasattr(population, "spike_threshold__mV"):
                        nc.threshold = population.spike_threshold__mV
                    else:
                        nc.threshold = DEFAULT_SPIKE_THRESHOLD  # Fallback for populations without explicit threshold
                    nc.record(spike_vector, id_vector, neuron.global__ID)
                    recording_netcons.append(nc)

                # Store these recording NetCons separately
                connection_key = (pop_name, "spike_recording")
                self._netcons_by_connection[connection_key] = recording_netcons

    def connect(
        self,
        source: str,
        target: str,
        probability: float = 1.0,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
        deterministic: bool = False,
    ) -> list:
        """
        Connect two neural populations with specified parameters.

        Parameters
        ----------
        source : str
            Name of source population (must exist in populations dict).
        target : str
            Name of target population (must exist in populations dict).
        probability : float, optional
            Connection probability between 0.0 and 1.0, by default 1.0.
            Each source-target neuron pair connects with this probability (if deterministic=False).
            If deterministic=True, each source connects to exactly int(probability × n_targets) targets.
        weight__uS : float, optional
            Synaptic weight in microsiemens, by default 0.6.
        delay__ms : float, optional
            Synaptic delay in milliseconds, by default 1.0.
        threshold__mV : float, optional
            Spike threshold in millivolts, by default -10.0.
        deterministic : bool, optional
            If True, each source neuron connects to exactly int(probability × n_targets) randomly
            selected target neurons. If False, uses probabilistic sampling. Default False.

        Returns
        -------
        list[h.NetCon]
            List of created NEURON NetCon objects for this connection group.

        Raises
        ------
        ValueError
            If source or target populations don't exist, or probability out of range.
        """
        # Validation
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be 0.0-1.0, got {probability}")

        # Extract numeric values (handle both Quantity objects and plain floats)
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)

        # Extract spike recording vectors for source population
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)

        # Create connections using existing infrastructure
        netcons = _connect_populations(
            populations=self.populations,
            source_pop=source,
            target_pop=target,
            connection_probability=probability,
            synaptic_weight=weight_value,
            spike_threshold=threshold_value,
            id_vector=id_vector,
            spike_vector=spike_vector,
            deterministic=deterministic,
        )

        self.connections.append(
            {
                "type": "neural",
                "source": source,
                "target": target,
                "probability": probability,
                "weight__uS": weight_value,
                "delay__ms": delay_value,
                "threshold__mV": threshold_value,
            }
        )
        self._netcons_by_connection[(source, target)] = netcons

        return netcons

    def connect_to_muscle(
        self,
        source: str,
        muscle,
        activation_callback: Callable,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
    ) -> list:
        """
        Connect a neural population to a muscle with activation callback.

        Parameters
        ----------
        source : str
            Name of motor neuron population.
        muscle : object
            Muscle object for force generation.
        activation_callback : Callable
            Function called when motor neurons fire.
            Expected signature: callback(neuron_id, muscle, delay_time)
        weight__uS : float, optional
            Synaptic weight in microsiemens, by default 1.0.
        threshold__mV : float, optional
            Spike threshold in millivolts, by default -10.0.

        Returns
        -------
        list[h.NetCon]
            List of motor neuron to muscle NetCon objects.
        """
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")

        # Extract numeric values (handle both Quantity objects and plain floats)
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)

        # Extract spike recording vectors for source population
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)

        # Create muscle connections using existing infrastructure
        netcons = _connect_populations(
            populations=self.populations,
            source_pop=source,
            target_pop=None,  # External target
            connection_probability=1.0,  # All motor neurons connect
            muscle_callback=activation_callback,
            muscle=muscle,
            synaptic_weight=weight_value,
            spike_threshold=threshold_value,
            id_vector=id_vector,
            spike_vector=spike_vector,
        )

        self.connections.append(
            {
                "type": "muscle",
                "source": source,
                "target": "muscle",
                "muscle": muscle,
                "callback": activation_callback,
                "weight__uS": weight_value,
                "threshold__mV": threshold_value,
            }
        )
        self._netcons_by_connection[(source, "muscle")] = netcons

        return netcons

    def connect_from_external(
        self,
        source: str,
        target: str,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
    ) -> list:
        """
        Connect external input source to a neural population.

        Parameters
        ----------
        source : str
            Name/label for external input source (e.g., "spindle", "cortical_drive").
        target : str
            Name of target neural population.
        weight__uS : Quantity__uS, optional
            Synaptic weight in microsiemens, by default 0.8.
        delay__ms : float, optional
            Synaptic delay in milliseconds, by default 1.0.
        threshold__mV : float, optional
            Spike threshold in millivolts, by default -10.0.

        Returns
        -------
        list[h.NetCon]
            List of external to neural NetCon objects.
        """
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")

        # Extract numeric values (handle both Quantity objects and plain floats)
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)

        # Create external NetCons manually to maintain individual access
        from neuron import h

        netcons = []
        target_neurons = self.populations[target]

        for target_neuron in target_neurons:
            # Create NetCon from None (external source) to target neuron
            nc = h.NetCon(None, target_neuron.ns)
            nc.weight[0] = weight_value
            nc.delay = delay_value
            nc.threshold = threshold_value
            netcons.append(nc)

        self.connections.append(
            {
                "type": "external",
                "source": source,
                "target": target,
                "weight__uS": weight_value,
                "delay__ms": delay_value,
                "threshold__mV": threshold_value,
            }
        )
        self._netcons_by_connection[(source, target)] = netcons

        return netcons

    def connect_one_to_one(
        self,
        source: str,
        target: str,
        probability: float = 1.0,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
    ) -> list:
        """
        Connect two neural populations with one-to-one mapping.

        Creates individual connections between source[i] and target[i] for each neuron
        pair at matching indices with specified probability. This is particularly useful
        for modeling independent noise sources (e.g., independent Poisson drives) where
        each target neuron should receive input from exactly one source neuron.

        Parameters
        ----------
        source : str
            Name of source population (must exist in populations dict).
        target : str
            Name of target population (must exist in populations dict).
        probability : float, optional
            Probability that each source[i] -> target[i] connection is made, by default 1.0.
            Must be between 0.0 and 1.0.
        weight__uS : Quantity__uS, optional
            Synaptic weight in microsiemens, by default 0.6.
        delay__ms : Quantity__ms, optional
            Synaptic delay in milliseconds, by default 1.0.
        threshold__mV : Quantity__mV, optional
            Spike threshold in millivolts, by default -10.0.

        Returns
        -------
        list[h.NetCon]
            List of created NEURON NetCon objects for connections that were made.

        Raises
        ------
        ValueError
            If source or target populations don't exist, have different sizes,
            or probability is not in [0.0, 1.0].

        Examples
        --------
        >>> # Create independent noise for each motor neuron
        >>> noise_pool = DescendingDrive__Pool(n=10, poisson_batch_size=16, timestep__ms=0.05)
        >>> mn_pool = AlphaMN__Pool(n=10)
        >>> network = Network({"noise": noise_pool, "mn": mn_pool})
        >>> network.connect_one_to_one("noise", "mn", weight__uS=0.5)
        """
        # Validation
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be 0.0-1.0, got {probability}")

        # Extract numeric values (handle both Quantity objects and plain floats)
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)

        # Extract spike recording vectors for source population
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)

        # Create one-to-one connections using new helper function
        netcons = _connect_one_to_one(
            source_pop=source,
            target_pop=target,
            populations=self.populations,
            connection_probability=probability,
            synaptic_weight=weight_value,
            spike_threshold=threshold_value,
            id_vector=id_vector,
            spike_vector=spike_vector,
        )

        self.connections.append(
            {
                "type": "one_to_one",
                "source": source,
                "target": target,
                "probability": probability,
                "weight__uS": weight_value,
                "delay__ms": delay_value,
                "threshold__mV": threshold_value,
            }
        )
        self._netcons_by_connection[(source, target)] = netcons

        return netcons

    def get_connections(self) -> list[dict]:
        """Get list of all connection specifications."""
        return self.connections.copy()

    def get_netcons(self, source: Optional[str] = None, target: Optional[str] = None) -> list:
        """
        Get NEURON NetCon objects with optional filtering by source and target.

        Parameters
        ----------
        source : str, optional
            Filter by source population/input name. If None, returns NetCons from all sources.
        target : str, optional
            Filter by target population name. If None, returns NetCons to all targets.

        Returns
        -------
        list[h.NetCon]
            List of matching NetCon objects.
        """
        if source is None and target is None:
            all_netcons = []
            for netcon_list in self._netcons_by_connection.values():
                all_netcons.extend(netcon_list)
            return all_netcons

        matching_netcons = []

        for (
            conn_source,
            conn_target,
        ), netcon_list in self._netcons_by_connection.items():
            source_matches = source is None or conn_source == source
            target_matches = target is None or conn_target == target

            if source_matches and target_matches:
                matching_netcons.extend(netcon_list)

        return matching_netcons

    def print_network(self):
        """Print a summary of network structure."""
        print(f"Network with {len(self.populations)} populations:")
        for name, neurons in self.populations.items():
            print(f"  {name}: {len(neurons)} neurons")

        print(f"\nConnections ({len(self.connections)}):")
        for i, conn in enumerate(self.connections):
            if conn["type"] == "neural":
                print(
                    f"  {i + 1}. {conn['source']} → {conn['target']} "
                    f"(p={conn['probability']}, w={conn['weight__uS']}uS)"
                )
            elif conn["type"] == "muscle":
                print(f"  {i + 1}. {conn['source']} → muscle (w={conn['weight__uS']}uS)")
            elif conn["type"] == "external":
                print(f"  {i + 1}. {conn['source']} → {conn['target']} (w={conn['weight__uS']}uS)")
            elif conn["type"] == "one_to_one":
                print(
                    f"  {i + 1}. {conn['source']} → {conn['target']} [1-to-1] "
                    f"(p={conn['probability']}, w={conn['weight__uS']}uS)"
                )


if __name__ == "__main__":
    from myogen import setup_myogen

    setup_myogen()

    timestep__ms = 0.05

    dd__pool = DescendingDrive__Pool(n=2, poisson_batch_size=16, timestep__ms=timestep__ms)

    n_type1 = 2
    n_type2 = 2
    n_alpha_mn = n_type1 + n_type2

    alphaMN__pool = AlphaMN__Pool(n=n_alpha_mn)

    ia_pool = AffIa__Pool(n=2, timestep__ms=timestep__ms)

    ii_pool = AffII__Pool(n=2, timestep__ms=timestep__ms)

    gii_pool = GII__Pool(n=2)

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

    network_connections_dict = _create_network(
        populations=population_params,
        connections_config=connection_params,
        id_vector=spike_id_vector,
        spike_vector=spike_time_vector,
        muscle_callback=foo,
        spike_save=[],
        muscle=None,
    )

    _print_network_connections(network_connections_dict)

    # Example using new Network class
    print("\n" + "=" * 50)
    print("Testing new Network class:")
    print("=" * 50)

    # Create network with same populations
    network = Network(population_params)

    # Add connections with clean API using unit conventions
    network.connect("Ia", "aMN", probability=0.9, weight__uS=0.6)
    network.connect("gII", "aMN", probability=0.9, weight__uS=0.3)
    network.connect_to_muscle("aMN", muscle=None, activation_callback=foo, weight__uS=1.0)
    network.connect_from_external("spindle", "Ia", weight__uS=0.8)

    # Print network summary
    network.print_network()

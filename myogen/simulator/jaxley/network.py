"""
Neural Network Connectivity Module - Jaxley Backend

This module provides network connectivity functionality for MyoGen's Jaxley-based
neuron models, maintaining full API compatibility with the NEURON version while
using Jaxley's connection infrastructure.

Key Features:
- Population-level connectivity patterns
- Probabilistic and deterministic connection strategies
- One-to-one connection mapping
- Spike recording management
- External input/output connections
- Full API compatibility with NEURON version

Architectural note — connectivity metadata vs. live synaptic transmission:
    ``JaxleyConnection`` and the ``Network`` class store connectivity *specifications*
    (source/target neurons, weights, delays) but do **not** register synapses in a
    Jaxley computation graph. Actual synaptic transmission in simulation examples is
    implemented analytically: DD/Ia spike times are convolved with exponential kernels
    to produce conductance waveforms, which are then injected as currents via
    ``jx.integrate()``. This approach is necessary because Jaxley's native synapse
    system requires a single monolithic ``jx.Network`` of all cells, which is
    incompatible with our per-cell ``jx.integrate()`` architecture that supports
    heterogeneous pools of independently-sized motor neurons.
"""

from typing import Callable, Optional, Any
import numpy as np
import quantities as pq

import myogen
from myogen.utils.decorators import beartowertype
from myogen.utils.types import Quantity__mV, Quantity__ms, Quantity__uS

# Network Constants
MOTOR_NEURON_CONNECTION = "aMN->Muscle"
DEFAULT_SYNAPTIC_WEIGHT = 0.6 * pq.uS
DEFAULT_SPIKE_THRESHOLD = -10.0 * pq.mV
DEFAULT_SYNAPTIC_DELAY = 1.0 * pq.ms
EXTERNAL_INPUT_LABEL = "Spindle"
EXTERNAL_TARGET_LABEL = "Muscle"
INHIBITORY_REVERSAL_THRESHOLD = -40.0  # mV


class JaxleyConnection:
    """
    Jaxley-compatible connection object that mimics NEURON NetCon API.
    
    This class provides a lightweight connection representation for Jaxley
    networks while maintaining API compatibility with NEURON's NetCon.
    """
    
    def __init__(
        self,
        source_neuron,
        target_neuron,
        weight: float = 0.6,
        delay: float = 1.0,
        threshold: float = -10.0,
        synapse_index: int = 0,
    ):
        self.source = source_neuron
        self.target = target_neuron
        self.weight = [weight]  # List to match NEURON's weight[0] API
        self.delay = delay
        self.threshold = threshold
        self.synapse_index = synapse_index
        self._muscle_callback = None
        self._spike_recording = {"idvec": None, "spkvec": None, "neuron_id": None}
    
    def record(self, *args):
        """
        Record spikes or setup muscle activation callback.
        
        Supports two calling conventions:
        1. record(callback_func) - muscle activation callback
        2. record(spike_vector, id_vector, neuron_id) - spike recording
        """
        if len(args) == 1 and callable(args[0]):
            # Muscle activation callback
            self._muscle_callback = args[0]
        elif len(args) == 3:
            # Spike recording
            self._spike_recording = {
                "spkvec": args[0],
                "idvec": args[1],
                "neuron_id": args[2],
            }
    
    def pre(self):
        """Get presynaptic (source) neuron."""
        return self.source
    
    def postcell(self):
        """Get postsynaptic (target) neuron."""
        return self.target if self.target is not None else None
    
    def syn(self):
        """Get target synapse."""
        if self.target is not None and hasattr(self.target, 'synapse__list'):
            if 0 <= self.synapse_index < len(self.target.synapse__list):
                return self.target.synapse__list[self.synapse_index]
        return None
    
    def preloc(self):
        """Get presynaptic location (-1 for external)."""
        return -1 if self.source is None else 0


def _select_synapse(target_neuron, inhibitory: bool = False) -> tuple:
    """
    Select appropriate synapse from target neuron based on connection type.
    
    Parameters
    ----------
    target_neuron : neuron object
        Target neuron with synapse__list attribute
    inhibitory : bool, optional
        If True, select inhibitory synapse (reversal < -40 mV).
        If False, select excitatory synapse (reversal >= -40 mV).
    
    Returns
    -------
    tuple
        (synapse_dict, synapse_index) for connection creation
    """
    synapse_list = target_neuron.synapse__list
    
    if len(synapse_list) == 1:
        return synapse_list[0], 0
    
    # Filter synapses by type
    if inhibitory:
        matching_synapses = [
            (syn, i) for i, syn in enumerate(synapse_list)
            if syn.get('e', 0) < INHIBITORY_REVERSAL_THRESHOLD
        ]
    else:
        matching_synapses = [
            (syn, i) for i, syn in enumerate(synapse_list)
            if syn.get('e', 0) >= INHIBITORY_REVERSAL_THRESHOLD
        ]
    
    if matching_synapses:
        syn, idx = myogen.RANDOM_GENERATOR.choice(matching_synapses)
        return syn, idx
    else:
        # Fallback to random selection
        idx = myogen.RANDOM_GENERATOR.choice(len(synapse_list))
        return synapse_list[idx], idx


def _create_netcon(
    source_neuron,
    target_neuron,
    muscle_callback=None,
    neuron_id=None,
    id_vector=None,
    spike_vector=None,
    muscle=None,
    synapse_index=0,
) -> JaxleyConnection:
    """
    Create a Jaxley-compatible network connection.
    
    Maintains API compatibility with NEURON's create_netcon while using
    Jaxley's connection infrastructure.
    """
    netcon = JaxleyConnection(
        source_neuron,
        target_neuron,
        weight=float(DEFAULT_SYNAPTIC_WEIGHT.magnitude),
        delay=float(DEFAULT_SYNAPTIC_DELAY.magnitude),
        threshold=float(DEFAULT_SPIKE_THRESHOLD.magnitude),
        synapse_index=synapse_index,
    )
    
    # Setup muscle activation callback
    if callable(muscle_callback) and muscle is not None:
        def muscle_activation_wrapper():
            delay = 1.0
            if hasattr(source_neuron, "axon_delay__ms") and source_neuron.axon_delay__ms is not None:
                delay = 1.0 + float(source_neuron.axon_delay__ms.magnitude if hasattr(source_neuron.axon_delay__ms, 'magnitude') else source_neuron.axon_delay__ms)
            return muscle_callback(source_neuron.pool__ID, muscle, delay)
        
        netcon.record(muscle_activation_wrapper)
    
    # Setup spike recording
    if neuron_id is not None and id_vector is not None and spike_vector is not None:
        netcon.record(spike_vector, id_vector, neuron_id)
    
    # Add axonal delay if present
    if hasattr(source_neuron, "axon_delay__ms") and source_neuron.axon_delay__ms is not None:
        axon_delay = float(source_neuron.axon_delay__ms.magnitude if hasattr(source_neuron.axon_delay__ms, 'magnitude') else source_neuron.axon_delay__ms)
        netcon.delay = float(DEFAULT_SYNAPTIC_DELAY.magnitude) + axon_delay
    
    return netcon


def _connect_population_to_population(
    source_pop: str,
    target_pop: str,
    populations: dict,
    connection_probability: float,
    deterministic: bool = False,
    inhibitory: bool = False,
    **kwargs,
) -> list:
    """Create connections between two neural populations."""
    connections = []
    target_neurons = populations[target_pop]
    
    if deterministic and connection_probability < 1.0:
        # Deterministic: each source connects to exact number of targets
        n_connections = int(connection_probability * len(target_neurons))
        
        for source_neuron in populations[source_pop]:
            selected_targets = myogen.RANDOM_GENERATOR.choice(
                target_neurons, size=n_connections, replace=False
            )
            
            for target_neuron in selected_targets:
                target_synapse, syn_idx = _select_synapse(target_neuron, inhibitory=inhibitory)
                netcon = _create_netcon(
                    source_neuron,
                    target_neuron,
                    muscle_callback=kwargs.get("muscle_callback"),
                    id_vector=kwargs.get("id_vector"),
                    spike_vector=kwargs.get("spike_vector"),
                    neuron_id=source_neuron.global__ID,
                    synapse_index=syn_idx,
                )
                
                # Apply custom parameters
                if kwargs.get("synaptic_weight") is not None:
                    netcon.weight[0] = kwargs.get("synaptic_weight")
                if kwargs.get("spike_threshold") is not None:
                    netcon.threshold = kwargs.get("spike_threshold")
                
                connections.append(netcon)
    else:
        # Probabilistic: each pair has probability of connecting
        for source_neuron in populations[source_pop]:
            for target_neuron in target_neurons:
                if myogen.RANDOM_GENERATOR.uniform() < connection_probability:
                    target_synapse, syn_idx = _select_synapse(target_neuron, inhibitory=inhibitory)
                    netcon = _create_netcon(
                        source_neuron,
                        target_neuron,
                        muscle_callback=kwargs.get("muscle_callback"),
                        id_vector=kwargs.get("id_vector"),
                        spike_vector=kwargs.get("spike_vector"),
                        neuron_id=source_neuron.global__ID,
                        synapse_index=syn_idx,
                    )
                    
                    # Apply custom parameters
                    if kwargs.get("synaptic_weight") is not None:
                        netcon.weight[0] = kwargs.get("synaptic_weight")
                    if kwargs.get("spike_threshold") is not None:
                        netcon.threshold = kwargs.get("spike_threshold")
                    
                    connections.append(netcon)
    
    return connections


def _connect_population_to_external(source_pop: str, populations: dict, **kwargs) -> list:
    """Create connections from neural population to external target (muscle)."""
    connections = []
    for source_neuron in populations[source_pop]:
        netcon = _create_netcon(
            source_neuron,
            None,  # External target
            muscle_callback=kwargs.get("muscle_callback"),
            id_vector=kwargs.get("id_vector"),
            muscle=kwargs.get("muscle"),
            spike_vector=kwargs.get("spike_vector"),
            neuron_id=source_neuron.global__ID,
        )
        
        if kwargs.get("spike_threshold") is not None:
            netcon.threshold = kwargs.get("spike_threshold")
        
        connections.append(netcon)
    
    return connections


def _connect_external_to_population(target_pop: Optional[str], populations: dict, **kwargs) -> list:
    """Create connections from external source to neural population."""
    connections = []
    for target_neuron in populations[target_pop]:
        netcon = _create_netcon(
            None,  # External source
            target_neuron,
            id_vector=kwargs.get("id_vector"),
            spike_vector=kwargs.get("spike_vector"),
            neuron_id=None,
        )
        connections.append(netcon)
    
    return connections


def _connect_one_to_one(
    source_pop: str,
    target_pop: str,
    populations: dict,
    connection_probability: float = 1.0,
    inhibitory: bool = False,
    **kwargs,
) -> list:
    """Create one-to-one connections between populations."""
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
        if myogen.RANDOM_GENERATOR.uniform() < connection_probability:
            target_synapse, syn_idx = _select_synapse(target_neuron, inhibitory=inhibitory)
            netcon = _create_netcon(
                source_neuron,
                target_neuron,
                muscle_callback=kwargs.get("muscle_callback"),
                id_vector=kwargs.get("id_vector"),
                spike_vector=kwargs.get("spike_vector"),
                neuron_id=source_neuron.global__ID,
                synapse_index=syn_idx,
            )
            
            # Apply custom parameters
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
    inhibitory: bool = False,
) -> list:
    """
    Create network connections between populations.
    
    Maintains full API compatibility with NEURON version while using
    Jaxley connection infrastructure.
    """
    # Validation
    if not 0.0 <= connection_probability <= 1.0:
        raise ValueError(
            f"Connection probability must be between 0.0 and 1.0, got {connection_probability}"
        )
    
    if source_pop is not None and source_pop not in populations:
        raise ValueError(f"Source population '{source_pop}' not found")
    
    if target_pop is not None and target_pop not in populations:
        raise ValueError(f"Target population '{target_pop}' not found")
    
    # Connection kwargs
    connection_kwargs = {
        "muscle_callback": muscle_callback,
        "id_vector": id_vector,
        "spike_vector": spike_vector,
        "muscle": muscle,
        "synaptic_weight": synaptic_weight,
        "spike_threshold": spike_threshold,
        "deterministic": deterministic,
        "inhibitory": inhibitory,
    }
    
    # Route to appropriate connection function
    if source_pop is not None and target_pop is not None:
        return _connect_population_to_population(
            source_pop, target_pop, populations, connection_probability, **connection_kwargs
        )
    elif source_pop is not None and target_pop is None:
        return _connect_population_to_external(source_pop, populations, **connection_kwargs)
    else:
        return _connect_external_to_population(target_pop, populations, **connection_kwargs)


def _print_network_connections(network_connections):
    """Print human-readable summary of network connections."""
    connection_index = 0
    for connection_group in network_connections.values():
        for netcon in connection_group:
            source = netcon.pre()
            target = netcon.postcell()
            
            source_name = EXTERNAL_INPUT_LABEL if source is None else str(source)
            target_name = EXTERNAL_TARGET_LABEL if target is None else str(target)
            
            print(f"NC[{connection_index}]: {source_name} -> {target_name}")
            connection_index += 1


@beartowertype
class Network:
    """
    Neural network builder for Jaxley backend.
    
    Provides identical API to NEURON version while using Jaxley's
    connection infrastructure internally.
    """
    
    def __init__(self, populations: dict, spike_recording: Optional[dict] = None):
        """
        Initialize network with neural populations.
        
        Parameters
        ----------
        populations : dict
            Dictionary mapping population names to neuron lists or Pool objects.
        spike_recording : dict, optional
            Spike recording configuration with 'idvec' and 'spkvec'.
        """
        self.populations = populations
        self.connections = []
        self._netcons_by_connection = {}
        self.spike_recording = spike_recording
    
    def setup_spike_recording(self):
        """Setup spike recording for all neurons in all populations."""
        if not self.spike_recording:
            return
        
        for pop_name, population in self.populations.items():
            if not hasattr(population, "__iter__") or isinstance(population, dict):
                continue
            
            id_vector = self.spike_recording.get("idvec", {}).get(pop_name)
            spike_vector = self.spike_recording.get("spkvec", {}).get(pop_name)
            
            if id_vector is not None and spike_vector is not None:
                recording_netcons = []
                for neuron in population:
                    if not hasattr(neuron, "cell") and not hasattr(neuron, "ns"):
                        continue
                    
                    # Create recording-only connection
                    nc = JaxleyConnection(neuron, None)
                    nc.threshold = float(DEFAULT_SPIKE_THRESHOLD.magnitude)
                    nc.record(spike_vector, id_vector, neuron.global__ID)
                    recording_netcons.append(nc)
                
                connection_key = (pop_name, "spike_recording")
                self._netcons_by_connection[connection_key] = recording_netcons
    
    @beartowertype
    def connect(
        self,
        source: str,
        target: str,
        probability: float = 1.0,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
        deterministic: bool = False,
        inhibitory: bool = False,
    ) -> list:
        """
        Connect two neural populations.
        
        Parameters
        ----------
        source : str
            Source population name
        target : str
            Target population name
        probability : float
            Connection probability (0.0-1.0)
        weight__uS : Quantity__uS
            Synaptic weight in microsiemens
        delay__ms : Quantity__ms
            Synaptic delay in milliseconds
        threshold__mV : Quantity__mV
            Spike threshold in millivolts
        deterministic : bool
            Use deterministic connectivity
        inhibitory : bool
            Connect to inhibitory synapses
        
        Returns
        -------
        list
            List of created connections
        """
        # Validation
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be 0.0-1.0, got {probability}")
        
        # Extract values
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)
        
        # Get spike recording vectors
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)
        
        # Create connections
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
            inhibitory=inhibitory,
        )
        
        self.connections.append({
            "type": "neural",
            "source": source,
            "target": target,
            "probability": probability,
            "weight__uS": weight_value,
            "delay__ms": delay_value,
            "threshold__mV": threshold_value,
            "inhibitory": inhibitory,
        })
        self._netcons_by_connection[(source, target)] = netcons
        
        return netcons
    
    @beartowertype
    def connect_to_muscle(
        self,
        source: str,
        muscle,
        activation_callback: Callable,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
    ) -> list:
        """
        Connect neural population to muscle with activation callback.
        
        Parameters
        ----------
        source : str
            Motor neuron population name
        muscle : object
            Muscle object
        activation_callback : Callable
            Callback for motor neuron spikes
        weight__uS : Quantity__uS
            Synaptic weight
        threshold__mV : Quantity__mV
            Spike threshold
        
        Returns
        -------
        list
            List of muscle connections
        """
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")
        
        # Extract values
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)
        
        # Get spike recording
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)
        
        # Create connections
        netcons = _connect_populations(
            populations=self.populations,
            source_pop=source,
            target_pop=None,
            connection_probability=1.0,
            muscle_callback=activation_callback,
            muscle=muscle,
            synaptic_weight=weight_value,
            spike_threshold=threshold_value,
            id_vector=id_vector,
            spike_vector=spike_vector,
        )
        
        self.connections.append({
            "type": "muscle",
            "source": source,
            "target": "muscle",
            "muscle": muscle,
            "callback": activation_callback,
            "weight__uS": weight_value,
            "threshold__mV": threshold_value,
        })
        self._netcons_by_connection[(source, "muscle")] = netcons
        
        return netcons
    
    @beartowertype
    def connect_from_external(
        self,
        source: str,
        target: str,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
    ) -> list:
        """
        Connect external input to neural population.
        
        Parameters
        ----------
        source : str
            External source label
        target : str
            Target population name
        weight__uS : Quantity__uS
            Synaptic weight
        delay__ms : Quantity__ms
            Synaptic delay
        threshold__mV : Quantity__mV
            Spike threshold
        
        Returns
        -------
        list
            List of external connections
        """
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")
        
        # Extract values
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)
        
        # Create external connections
        netcons = []
        target_neurons = self.populations[target]
        
        for target_neuron in target_neurons:
            nc = JaxleyConnection(None, target_neuron)
            nc.weight[0] = weight_value
            nc.delay = delay_value
            nc.threshold = threshold_value
            netcons.append(nc)
        
        self.connections.append({
            "type": "external",
            "source": source,
            "target": target,
            "weight__uS": weight_value,
            "delay__ms": delay_value,
            "threshold__mV": threshold_value,
        })
        self._netcons_by_connection[(source, target)] = netcons
        
        return netcons
    
    @beartowertype
    def connect_one_to_one(
        self,
        source: str,
        target: str,
        probability: float = 1.0,
        weight__uS: Quantity__uS = DEFAULT_SYNAPTIC_WEIGHT,
        delay__ms: Quantity__ms = DEFAULT_SYNAPTIC_DELAY,
        threshold__mV: Quantity__mV = DEFAULT_SPIKE_THRESHOLD,
        inhibitory: bool = False,
    ) -> list:
        """
        Connect populations with one-to-one mapping.
        
        Parameters
        ----------
        source : str
            Source population name
        target : str
            Target population name
        probability : float
            Connection probability for each pair
        weight__uS : Quantity__uS
            Synaptic weight
        delay__ms : Quantity__ms
            Synaptic delay
        threshold__mV : Quantity__mV
            Spike threshold
        inhibitory : bool
            Connect to inhibitory synapses
        
        Returns
        -------
        list
            List of one-to-one connections
        """
        # Validation
        if source not in self.populations:
            raise ValueError(f"Source population '{source}' not found")
        if target not in self.populations:
            raise ValueError(f"Target population '{target}' not found")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be 0.0-1.0, got {probability}")
        
        # Extract values
        weight_value = getattr(weight__uS, 'magnitude', weight__uS)
        delay_value = getattr(delay__ms, 'magnitude', delay__ms)
        threshold_value = getattr(threshold__mV, 'magnitude', threshold__mV)
        
        # Get spike recording
        id_vector = None
        spike_vector = None
        if self.spike_recording:
            id_vector = self.spike_recording.get("idvec", {}).get(source)
            spike_vector = self.spike_recording.get("spkvec", {}).get(source)
        
        # Create connections
        netcons = _connect_one_to_one(
            source_pop=source,
            target_pop=target,
            populations=self.populations,
            connection_probability=probability,
            synaptic_weight=weight_value,
            spike_threshold=threshold_value,
            id_vector=id_vector,
            spike_vector=spike_vector,
            inhibitory=inhibitory,
        )
        
        self.connections.append({
            "type": "one_to_one",
            "source": source,
            "target": target,
            "probability": probability,
            "weight__uS": weight_value,
            "delay__ms": delay_value,
            "threshold__mV": threshold_value,
            "inhibitory": inhibitory,
        })
        self._netcons_by_connection[(source, target)] = netcons
        
        return netcons
    
    def get_connections(self) -> list[dict]:
        """Get list of all connection specifications."""
        return self.connections.copy()
    
    def get_netcons(self, source: Optional[str] = None, target: Optional[str] = None) -> list:
        """
        Get connection objects with optional filtering.
        
        Parameters
        ----------
        source : str, optional
            Filter by source population
        target : str, optional
            Filter by target population
        
        Returns
        -------
        list
            List of matching connections
        """
        if source is None and target is None:
            # Return all connections
            all_netcons = []
            for netcon_list in self._netcons_by_connection.values():
                all_netcons.extend(netcon_list)
            return all_netcons
        
        # Filter by source and/or target
        matching_netcons = []
        for (src, tgt), netcon_list in self._netcons_by_connection.items():
            if (source is None or src == source) and (target is None or tgt == target):
                matching_netcons.extend(netcon_list)
        
        return matching_netcons
    
    def print_summary(self):
        """Print network summary."""
        print(f"Network with {len(self.populations)} populations:")
        for pop_name, pop in self.populations.items():
            n_neurons = len(pop) if hasattr(pop, '__len__') else '?'
            print(f"  {pop_name}: {n_neurons} neurons")
        
        print(f"\n{len(self.connections)} connection groups:")
        for conn in self.connections:
            conn_type = conn.get('type', 'unknown')
            source = conn.get('source', '?')
            target = conn.get('target', '?')
            
            if conn_type == 'neural':
                prob = conn.get('probability', 1.0)
                weight = conn.get('weight__uS', 0)
                inh = " (inhibitory)" if conn.get('inhibitory', False) else ""
                print(f"  {source} -> {target}: p={prob:.2f}, w={weight:.2f}uS{inh}")
            elif conn_type == 'muscle':
                print(f"  {source} -> muscle")
            elif conn_type == 'external':
                print(f"  {source} (external) -> {target}")
            elif conn_type == 'one_to_one':
                prob = conn.get('probability', 1.0)
                print(f"  {source} -> {target}: one-to-one (p={prob:.2f})")

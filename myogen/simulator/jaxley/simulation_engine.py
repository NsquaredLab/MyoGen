"""
Jaxley Simulation Engine for MyoGen.

This module provides the core simulation infrastructure for running
biophysical neural network simulations using Jaxley.

Key Features:
- JAX-accelerated neural simulation
- Spike detection and recording
- Synaptic event handling
- Current injection support
- Network-level simulation control
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import jaxley as jx
import myogen


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SpikeRecord:
    """Container for recorded spikes."""
    times: List[float] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)
    
    def add_spike(self, time: float, neuron_id: int):
        """Record a spike."""
        self.times.append(time)
        self.ids.append(neuron_id)
    
    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return np.array(self.times), np.array(self.ids)
    
    def clear(self):
        """Clear all recorded spikes."""
        self.times.clear()
        self.ids.clear()


@dataclass
class VoltageRecord:
    """Container for voltage traces."""
    times: np.ndarray = None
    voltages: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def add_trace(self, neuron_id: int, voltage: np.ndarray):
        """Add a voltage trace for a neuron."""
        self.voltages[neuron_id] = voltage


@dataclass
class SimulationConfig:
    """Configuration for Jaxley simulation."""
    dt: float = 0.025  # Time step (ms)
    duration: float = 1000.0  # Simulation duration (ms)
    record_voltage: bool = False
    record_spikes: bool = True
    spike_threshold: float = 0.0  # mV
    initial_voltage: float = -70.0  # mV


# =============================================================================
# SPIKE DETECTION
# =============================================================================

def detect_spikes(
    voltage: jnp.ndarray,
    threshold: float = 0.0,
    refractory: int = 5,
) -> jnp.ndarray:
    """
    Detect spikes using threshold crossing.
    
    Parameters
    ----------
    voltage : jnp.ndarray
        Voltage trace, shape (n_timesteps,) or (n_neurons, n_timesteps).
    threshold : float
        Spike threshold in mV.
    refractory : int
        Minimum samples between spikes.
    
    Returns
    -------
    jnp.ndarray
        Boolean array of spike times.
    """
    # Threshold crossings (upward)
    above = voltage > threshold
    below_before = jnp.roll(voltage, 1, axis=-1) <= threshold
    crossings = above & below_before
    
    # Zero out first sample (no valid crossing)
    if crossings.ndim == 1:
        crossings = crossings.at[0].set(False)
    else:
        crossings = crossings.at[:, 0].set(False)
    
    return crossings


def apply_refractory(
    spikes: jnp.ndarray,
    refractory_samples: int,
) -> jnp.ndarray:
    """Apply refractory period to spike train."""
    def scan_fn(carry, spike):
        counter, last_spike = carry
        # Can spike if counter is 0
        can_spike = counter == 0
        new_spike = spike & can_spike
        # Reset counter on spike, otherwise decrement
        new_counter = jnp.where(new_spike, refractory_samples, jnp.maximum(counter - 1, 0))
        return (new_counter, new_spike), new_spike
    
    init_carry = (0, False)
    _, filtered_spikes = jax.lax.scan(scan_fn, init_carry, spikes)
    return filtered_spikes


# =============================================================================
# CURRENT INJECTION
# =============================================================================

@dataclass
class CurrentInjection:
    """
    Specification for current injection.
    
    Parameters
    ----------
    amplitude : float or array
        Current amplitude in nA. Can be constant or time-varying.
    start : float
        Start time in ms.
    duration : float
        Duration in ms.
    location : str
        Injection location ("soma", "dendrite", etc.).
    """
    amplitude: Union[float, np.ndarray]
    start: float = 0.0
    duration: float = float('inf')
    location: str = "soma"
    
    def get_current(self, t: float) -> float:
        """Get current at time t."""
        if t < self.start or t >= self.start + self.duration:
            return 0.0
        
        if isinstance(self.amplitude, (int, float)):
            return float(self.amplitude)
        else:
            # Time-varying amplitude
            idx = int((t - self.start) / 0.025)  # Assume dt=0.025
            if idx < len(self.amplitude):
                return float(self.amplitude[idx])
            return 0.0


# =============================================================================
# SYNAPTIC EVENT HANDLING
# =============================================================================

@dataclass
class SynapticEvent:
    """
    A synaptic activation event.
    
    Parameters
    ----------
    time : float
        Event time in ms.
    target_id : int
        Target neuron ID.
    weight : float
        Synaptic weight (uS).
    synapse_type : str
        "excitatory" or "inhibitory".
    """
    time: float
    target_id: int
    weight: float
    synapse_type: str = "excitatory"


class SynapticEventQueue:
    """Priority queue for synaptic events."""
    
    def __init__(self):
        self.events: List[SynapticEvent] = []
    
    def add_event(self, event: SynapticEvent):
        """Add event and maintain time ordering."""
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)
    
    def pop_events_until(self, t: float) -> List[SynapticEvent]:
        """Get and remove all events up to time t."""
        to_process = []
        while self.events and self.events[0].time <= t:
            to_process.append(self.events.pop(0))
        return to_process
    
    def clear(self):
        """Clear all events."""
        self.events.clear()


# =============================================================================
# NETWORK SIMULATOR
# =============================================================================

class JaxleyNetworkSimulator:
    """
    Network-level simulator for Jaxley neural populations.
    
    Manages multiple cell populations, connectivity, and simulation.
    
    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        self.populations: Dict[str, List[Any]] = {}
        self.connections: List[Dict] = []
        self.spike_records: Dict[str, SpikeRecord] = {}
        self.voltage_records: Dict[str, VoltageRecord] = {}
        self.current_injections: Dict[str, List[CurrentInjection]] = {}
        self.event_queue = SynapticEventQueue()
        
        self._time = 0.0
        self._initialized = False
    
    def add_population(
        self,
        name: str,
        cells: List[Any],
        record_spikes: bool = True,
        record_voltage: bool = False,
        spike_threshold: float = None,
    ):
        """
        Add a neural population.
        
        Parameters
        ----------
        name : str
            Population name.
        cells : list
            List of Jaxley cells or cell wrappers.
        record_spikes : bool
            Whether to record spikes.
        record_voltage : bool
            Whether to record voltage traces.
        spike_threshold : float, optional
            Spike threshold (default from config).
        """
        self.populations[name] = cells
        
        if record_spikes:
            self.spike_records[name] = SpikeRecord()
        
        if record_voltage:
            self.voltage_records[name] = VoltageRecord()
        
        # Store threshold
        if spike_threshold is not None:
            for cell in cells:
                if hasattr(cell, 'spike_threshold'):
                    cell.spike_threshold = spike_threshold
    
    def connect(
        self,
        source_pop: str,
        target_pop: str,
        weight: float,
        delay: float = 1.0,
        probability: float = 1.0,
        synapse_type: str = "excitatory",
    ):
        """
        Create connections between populations.
        
        Parameters
        ----------
        source_pop : str
            Source population name.
        target_pop : str
            Target population name.
        weight : float
            Synaptic weight in uS.
        delay : float
            Synaptic delay in ms.
        probability : float
            Connection probability [0, 1].
        synapse_type : str
            "excitatory" or "inhibitory".
        """
        connection = {
            "source": source_pop,
            "target": target_pop,
            "weight": weight,
            "delay": delay,
            "probability": probability,
            "synapse_type": synapse_type,
            "connections": [],  # (source_idx, target_idx) pairs
        }
        
        # Generate connection pairs
        source_cells = self.populations.get(source_pop, [])
        target_cells = self.populations.get(target_pop, [])
        
        for i, src in enumerate(source_cells):
            for j, tgt in enumerate(target_cells):
                if myogen.RANDOM_GENERATOR.random() < probability:
                    connection["connections"].append((i, j))
        
        self.connections.append(connection)
    
    def inject_current(
        self,
        population: str,
        neuron_ids: Union[int, List[int]],
        injection: CurrentInjection,
    ):
        """
        Add current injection to neurons.
        
        Parameters
        ----------
        population : str
            Target population name.
        neuron_ids : int or list
            Neuron indices to inject into.
        injection : CurrentInjection
            Current injection specification.
        """
        if population not in self.current_injections:
            self.current_injections[population] = []
        
        if isinstance(neuron_ids, int):
            neuron_ids = [neuron_ids]
        
        for nid in neuron_ids:
            self.current_injections[population].append((nid, injection))
    
    def _process_spikes(self, t: float):
        """Process spikes and generate synaptic events."""
        for pop_name, record in self.spike_records.items():
            cells = self.populations[pop_name]
            threshold = self.config.spike_threshold
            
            for i, cell in enumerate(cells):
                # Get voltage (assumes cell has voltage attribute or method)
                if hasattr(cell, 'get_voltage'):
                    v = cell.get_voltage()
                elif hasattr(cell, 'v'):
                    v = cell.v
                else:
                    continue
                
                # Simple threshold crossing detection
                if hasattr(cell, '_prev_voltage'):
                    if cell._prev_voltage <= threshold < v:
                        record.add_spike(t, i)
                        
                        # Generate synaptic events
                        self._generate_synaptic_events(pop_name, i, t)
                
                cell._prev_voltage = v
    
    def _generate_synaptic_events(self, source_pop: str, source_idx: int, t: float):
        """Generate synaptic events from a spike."""
        for conn in self.connections:
            if conn["source"] == source_pop:
                for src_i, tgt_j in conn["connections"]:
                    if src_i == source_idx:
                        event = SynapticEvent(
                            time=t + conn["delay"],
                            target_id=tgt_j,
                            weight=conn["weight"],
                            synapse_type=conn["synapse_type"],
                        )
                        self.event_queue.add_event(event)
    
    def _apply_synaptic_events(self, t: float, pop_name: str):
        """Apply pending synaptic events to population."""
        events = self.event_queue.pop_events_until(t)
        
        for event in events:
            # Find target population from connections
            for conn in self.connections:
                if conn["target"] == pop_name:
                    cells = self.populations[pop_name]
                    if 0 <= event.target_id < len(cells):
                        cell = cells[event.target_id]
                        
                        # Apply synaptic conductance
                        if hasattr(cell, 'apply_synaptic_input'):
                            cell.apply_synaptic_input(
                                event.weight,
                                event.synapse_type,
                            )
    
    def _get_current_injection(self, pop_name: str, neuron_id: int, t: float) -> float:
        """Get total injected current for a neuron at time t."""
        total = 0.0
        if pop_name in self.current_injections:
            for nid, inj in self.current_injections[pop_name]:
                if nid == neuron_id:
                    total += inj.get_current(t)
        return total
    
    def initialize(self):
        """Initialize simulation state."""
        self._time = 0.0
        
        # Initialize cells
        for pop_name, cells in self.populations.items():
            for cell in cells:
                if hasattr(cell, 'set'):
                    cell.set("v", self.config.initial_voltage)
                cell._prev_voltage = self.config.initial_voltage
        
        # Clear records
        for record in self.spike_records.values():
            record.clear()
        self.event_queue.clear()
        
        self._initialized = True
    
    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step.
        
        Parameters
        ----------
        dt : float, optional
            Time step (default from config).
        """
        if not self._initialized:
            self.initialize()
        
        dt = dt or self.config.dt
        
        # Apply synaptic events
        for pop_name in self.populations:
            self._apply_synaptic_events(self._time, pop_name)
        
        # Process spikes and generate events
        self._process_spikes(self._time)
        
        # Advance time
        self._time += dt
    
    def run(
        self,
        duration: Optional[float] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Parameters
        ----------
        duration : float, optional
            Simulation duration (default from config).
        progress_callback : callable, optional
            Called with progress fraction.
        
        Returns
        -------
        dict
            Simulation results including spike records.
        """
        duration = duration or self.config.duration
        dt = self.config.dt
        n_steps = int(duration / dt)
        
        self.initialize()
        
        for step in range(n_steps):
            self.step(dt)
            
            if progress_callback and step % 100 == 0:
                progress_callback(step / n_steps)
        
        # Compile results
        results = {
            "time": np.arange(0, duration, dt),
            "spikes": {},
            "voltages": {},
        }
        
        for name, record in self.spike_records.items():
            results["spikes"][name] = record.to_arrays()
        
        for name, record in self.voltage_records.items():
            results["voltages"][name] = record.voltages
        
        return results
    
    def get_spike_times(self, population: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get spike times and IDs for a population."""
        if population in self.spike_records:
            return self.spike_records[population].to_arrays()
        return np.array([]), np.array([])


# =============================================================================
# SINGLE CELL SIMULATOR
# =============================================================================

class JaxleyCellSimulator:
    """
    Simulator for single Jaxley cells.
    
    Provides a simpler interface for simulating individual cells
    with current injection.
    
    Parameters
    ----------
    cell : jx.Cell
        Jaxley cell to simulate.
    dt : float
        Time step in ms.
    """
    
    def __init__(self, cell: jx.Cell, dt: float = 0.025):
        self.cell = cell
        self.dt = dt
        self._time = 0.0
        self._voltage_trace = []
        self._spike_times = []
    
    def inject_step_current(
        self,
        amplitude: float,
        start: float,
        duration: float,
    ) -> CurrentInjection:
        """Create a step current injection."""
        return CurrentInjection(
            amplitude=amplitude,
            start=start,
            duration=duration,
        )
    
    def simulate(
        self,
        duration: float,
        i_ext: Union[float, np.ndarray, CurrentInjection] = 0.0,
        record_voltage: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate cell for given duration.
        
        Parameters
        ----------
        duration : float
            Simulation duration in ms.
        i_ext : float, array, or CurrentInjection
            External current.
        record_voltage : bool
            Whether to record voltage.
        
        Returns
        -------
        dict
            Results with 'time', 'voltage', 'spikes'.
        """
        n_steps = int(duration / self.dt)
        times = np.arange(0, duration, self.dt)
        
        # Prepare current
        if isinstance(i_ext, CurrentInjection):
            currents = np.array([i_ext.get_current(t) for t in times])
        elif isinstance(i_ext, np.ndarray):
            currents = i_ext
        else:
            currents = np.full(n_steps, float(i_ext))
        
        # Initialize
        self._voltage_trace = []
        self._spike_times = []
        self._time = 0.0
        
        # Note: Actual Jaxley simulation would use jx.integrate()
        # This is a placeholder showing the interface
        
        results = {
            "time": times,
            "voltage": np.zeros(n_steps),  # Placeholder
            "spikes": np.array([]),
            "current": currents,
        }
        
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_firing_rate(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    neuron_id: int,
    window: float = 100.0,
    dt: float = 1.0,
    duration: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous firing rate for a neuron.
    
    Parameters
    ----------
    spike_times : array
        Spike times in ms.
    spike_ids : array
        Neuron IDs for each spike.
    neuron_id : int
        Neuron to analyze.
    window : float
        Smoothing window in ms.
    dt : float
        Output time step.
    duration : float, optional
        Total duration (default: max spike time + window).
    
    Returns
    -------
    times : array
        Time points.
    rates : array
        Firing rates in Hz.
    """
    # Get spikes for this neuron
    mask = spike_ids == neuron_id
    neuron_spikes = spike_times[mask]
    
    if len(neuron_spikes) == 0:
        if duration is None:
            return np.array([0]), np.array([0])
        times = np.arange(0, duration, dt)
        return times, np.zeros_like(times)
    
    # Time bins
    if duration is None:
        duration = neuron_spikes.max() + window
    times = np.arange(0, duration, dt)
    
    # Compute rate using sliding window
    rates = np.zeros(len(times))
    half_window = window / 2
    
    for i, t in enumerate(times):
        count = np.sum((neuron_spikes >= t - half_window) & (neuron_spikes < t + half_window))
        rates[i] = count * 1000.0 / window  # Convert to Hz
    
    return times, rates


def compute_isi_statistics(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    neuron_id: int,
) -> Dict[str, float]:
    """
    Compute interspike interval statistics.
    
    Returns
    -------
    dict
        'mean_isi', 'std_isi', 'cv_isi', 'mean_rate'
    """
    mask = spike_ids == neuron_id
    neuron_spikes = np.sort(spike_times[mask])
    
    if len(neuron_spikes) < 2:
        return {
            "mean_isi": np.nan,
            "std_isi": np.nan,
            "cv_isi": np.nan,
            "mean_rate": 0.0,
        }
    
    isis = np.diff(neuron_spikes)
    
    return {
        "mean_isi": np.mean(isis),
        "std_isi": np.std(isis),
        "cv_isi": np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else np.nan,
        "mean_rate": 1000.0 / np.mean(isis),  # Hz
    }

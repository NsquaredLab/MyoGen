"""
Jaxley Integration Module for MyoGen.

This module provides the core simulation infrastructure that connects
Jaxley cells to the MyoGen framework, including:
- Current injection with time-varying waveforms
- Spike detection and recording
- Multi-cell network simulation
- JIT-compiled simulation loops

This is the bridge between Jaxley's `jx.integrate()` and MyoGen's API.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

import jaxley as jx


# =============================================================================
# CURRENT STIMULI
# =============================================================================

def step_current(
    delay: float,
    duration: float,
    amplitude: float,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    """
    Create a step current injection waveform.
    
    Parameters
    ----------
    delay : float
        Time before current onset (ms).
    duration : float
        Duration of current pulse (ms).
    amplitude : float
        Current amplitude (nA).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    
    Returns
    -------
    jnp.ndarray
        Current waveform array.
    """
    return jx.step_current(delay, duration, amplitude, dt, t_max)


def ramp_current(
    delay: float,
    duration: float,
    start_amp: float,
    end_amp: float,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    """
    Create a linearly ramping current injection waveform.
    
    Parameters
    ----------
    delay : float
        Time before ramp onset (ms).
    duration : float
        Duration of ramp (ms).
    start_amp : float
        Starting amplitude (nA).
    end_amp : float
        Ending amplitude (nA).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    
    Returns
    -------
    jnp.ndarray
        Current waveform array.
    """
    n_steps = int(t_max / dt) + 1
    current = jnp.zeros(n_steps)
    
    start_idx = int(delay / dt)
    end_idx = int((delay + duration) / dt)
    ramp_len = end_idx - start_idx
    
    if ramp_len > 0:
        ramp = jnp.linspace(start_amp, end_amp, ramp_len)
        current = current.at[start_idx:end_idx].set(ramp)
    
    return current


def noisy_current(
    mean: float,
    std: float,
    dt: float,
    t_max: float,
    seed: int = 0,
) -> jnp.ndarray:
    """
    Create a noisy (Gaussian) current injection waveform.
    
    Parameters
    ----------
    mean : float
        Mean current amplitude (nA).
    std : float
        Standard deviation of noise (nA).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    seed : int
        Random seed.
    
    Returns
    -------
    jnp.ndarray
        Noisy current waveform array.
    """
    n_steps = int(t_max / dt) + 1
    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, shape=(n_steps,))
    return mean + std * noise


def custom_current(
    waveform: np.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    """
    Create current from a custom waveform array.
    
    Parameters
    ----------
    waveform : np.ndarray
        Custom current waveform (nA). Will be resampled if needed.
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    
    Returns
    -------
    jnp.ndarray
        Current waveform array.
    """
    n_steps = int(t_max / dt) + 1
    
    if len(waveform) == n_steps:
        return jnp.array(waveform)
    else:
        # Resample to match simulation timesteps
        old_t = np.linspace(0, t_max, len(waveform))
        new_t = np.linspace(0, t_max, n_steps)
        resampled = np.interp(new_t, old_t, waveform)
        return jnp.array(resampled)


def sinusoidal_current(
    frequency: float,
    amplitude: float,
    offset: float,
    dt: float,
    t_max: float,
    phase: float = 0.0,
) -> jnp.ndarray:
    """
    Create a sinusoidal current injection waveform.
    
    Parameters
    ----------
    frequency : float
        Frequency in Hz.
    amplitude : float
        Amplitude of oscillation (nA).
    offset : float
        DC offset / mean current (nA).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    phase : float
        Initial phase (radians).
    
    Returns
    -------
    jnp.ndarray
        Current waveform array.
    """
    n_steps = int(t_max / dt) + 1
    t = jnp.arange(n_steps) * dt / 1000.0  # Convert to seconds for Hz
    return offset + amplitude * jnp.sin(2 * jnp.pi * frequency * t + phase)


# =============================================================================
# SPIKE DETECTION
# =============================================================================

def detect_spikes(
    voltage: jnp.ndarray,
    threshold: float = 0.0,
    dt: float = 0.025,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Detect spikes from voltage trace using threshold crossing.
    
    Parameters
    ----------
    voltage : jnp.ndarray
        Voltage trace, shape (n_timesteps,) or (n_recordings, n_timesteps).
    threshold : float
        Spike threshold in mV.
    dt : float
        Time step in ms.
    
    Returns
    -------
    spike_times : jnp.ndarray
        Times of detected spikes (ms).
    spike_indices : jnp.ndarray
        Indices of spike times in voltage array.
    """
    # Handle 1D and 2D arrays
    if voltage.ndim == 1:
        voltage = voltage[jnp.newaxis, :]
    
    # Find threshold crossings (upward)
    above = voltage > threshold
    below_before = jnp.concatenate([
        jnp.zeros((voltage.shape[0], 1), dtype=bool),
        voltage[:, :-1] <= threshold
    ], axis=1)
    
    crossings = above & below_before
    
    # Get spike indices and times
    spike_indices = jnp.where(crossings)
    spike_times = spike_indices[1] * dt
    
    return spike_times, spike_indices


def compute_firing_rate(
    spike_times: jnp.ndarray,
    t_max: float,
    window: float = 100.0,
    dt: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute instantaneous firing rate from spike times.
    
    Parameters
    ----------
    spike_times : jnp.ndarray
        Spike times in ms.
    t_max : float
        Total time in ms.
    window : float
        Smoothing window in ms.
    dt : float
        Output time resolution in ms.
    
    Returns
    -------
    times : jnp.ndarray
        Time points.
    rates : jnp.ndarray
        Firing rates in Hz.
    """
    times = jnp.arange(0, t_max, dt)
    rates = jnp.zeros(len(times))
    half_window = window / 2
    
    # Count spikes in sliding window
    for i, t in enumerate(times):
        count = jnp.sum((spike_times >= t - half_window) & (spike_times < t + half_window))
        rates = rates.at[i].set(count * 1000.0 / window)
    
    return times, rates


def compute_isi(spike_times: jnp.ndarray) -> Dict[str, float]:
    """
    Compute interspike interval statistics.
    
    Parameters
    ----------
    spike_times : jnp.ndarray
        Spike times in ms.
    
    Returns
    -------
    dict
        ISI statistics: mean, std, cv, mean_rate.
    """
    if len(spike_times) < 2:
        return {
            "mean_isi": float('nan'),
            "std_isi": float('nan'),
            "cv_isi": float('nan'),
            "mean_rate": 0.0,
        }
    
    isis = jnp.diff(jnp.sort(spike_times))
    mean_isi = float(jnp.mean(isis))
    std_isi = float(jnp.std(isis))
    
    return {
        "mean_isi": mean_isi,
        "std_isi": std_isi,
        "cv_isi": std_isi / mean_isi if mean_isi > 0 else float('nan'),
        "mean_rate": 1000.0 / mean_isi if mean_isi > 0 else 0.0,
    }


def compute_firing_rate_from_spikes(
    spike_times: jnp.ndarray,
    t_start: float = 0.0,
    t_stop: Optional[float] = None,
) -> float:
    """
    Compute mean firing rate from spike times.
    
    Parameters
    ----------
    spike_times : jnp.ndarray
        Spike times in ms.
    t_start : float
        Start time for rate calculation (ms).
    t_stop : float, optional
        Stop time for rate calculation (ms). If None, uses max spike time.
    
    Returns
    -------
    float
        Mean firing rate in Hz.
    """
    if len(spike_times) == 0:
        return 0.0
    
    # Filter to time window
    if t_stop is None:
        t_stop = float(jnp.max(spike_times))
    
    in_window = (spike_times >= t_start) & (spike_times <= t_stop)
    n_spikes = int(jnp.sum(in_window))
    
    duration_s = (t_stop - t_start) / 1000.0  # Convert ms to s
    
    if duration_s <= 0:
        return 0.0
    
    return n_spikes / duration_s


def compute_isi_cv(spike_times: jnp.ndarray) -> float:
    """
    Compute coefficient of variation (CV) of interspike intervals.
    
    CV = std(ISI) / mean(ISI)
    
    Parameters
    ----------
    spike_times : jnp.ndarray
        Spike times in ms.
    
    Returns
    -------
    float
        CV of ISI. Returns NaN if fewer than 2 spikes.
    """
    if len(spike_times) < 2:
        return float('nan')
    
    isis = jnp.diff(jnp.sort(spike_times))
    mean_isi = float(jnp.mean(isis))
    std_isi = float(jnp.std(isis))
    
    if mean_isi <= 0:
        return float('nan')
    
    return std_isi / mean_isi


# =============================================================================
# SINGLE CELL SIMULATION
# =============================================================================

@dataclass
class CellSimulationResult:
    """Container for single cell simulation results."""
    time: jnp.ndarray
    voltage: jnp.ndarray
    current: jnp.ndarray
    spike_times: jnp.ndarray
    spike_count: int
    firing_rate: float
    isi_stats: Dict[str, float]
    
    @property
    def has_spikes(self) -> bool:
        return self.spike_count > 0


def simulate_cell(
    cell: jx.Cell,
    current: Optional[jnp.ndarray] = None,
    dt: float = 0.025,
    t_max: float = 100.0,
    record_loc: Tuple[int, float] = (0, 0.0),
    stim_loc: Tuple[int, float] = (0, 0.0),
    spike_threshold: float = 0.0,
    initial_voltage: float = -70.0,
) -> CellSimulationResult:
    """
    Simulate a single Jaxley cell with current injection.
    
    Parameters
    ----------
    cell : jx.Cell
        Jaxley cell to simulate.
    current : jnp.ndarray, optional
        Current injection waveform (nA). If None, no current is injected.
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    record_loc : tuple
        (branch_idx, location) for voltage recording.
    stim_loc : tuple
        (branch_idx, location) for current injection.
    spike_threshold : float
        Threshold for spike detection (mV).
    initial_voltage : float
        Initial membrane voltage (mV).
    
    Returns
    -------
    CellSimulationResult
        Simulation results including voltage, spikes, etc.
    """
    # Clear previous recordings and stimuli
    cell.delete_recordings()
    cell.delete_stimuli()
    
    # Set initial voltage
    cell.set("v", initial_voltage)
    
    # Set up recording
    branch_idx, loc = record_loc
    cell.branch(branch_idx).loc(loc).record("v")
    
    # Apply stimulus if provided
    if current is not None:
        stim_branch, stim_loc_val = stim_loc
        cell.branch(stim_branch).loc(stim_loc_val).stimulate(current)
    else:
        current = jnp.zeros(int(t_max / dt) + 1)
    
    # Run simulation
    voltages = jx.integrate(cell, delta_t=dt, t_max=t_max)
    
    # Extract voltage trace (shape: n_recordings x n_timesteps)
    voltage = voltages[0] if voltages.ndim > 1 else voltages
    
    # Create time vector
    time = jnp.arange(0, t_max + dt, dt)[:len(voltage)]
    
    # Detect spikes
    spike_times, _ = detect_spikes(voltage, threshold=spike_threshold, dt=dt)
    spike_count = len(spike_times)
    
    # Compute statistics
    duration_s = t_max / 1000.0
    firing_rate = spike_count / duration_s if duration_s > 0 else 0.0
    isi_stats = compute_isi(spike_times)
    
    return CellSimulationResult(
        time=time,
        voltage=voltage,
        current=current[:len(time)],
        spike_times=spike_times,
        spike_count=spike_count,
        firing_rate=firing_rate,
        isi_stats=isi_stats,
    )


def simulate_cell_with_data_stimulate(
    cell: jx.Cell,
    current: jnp.ndarray,
    dt: float = 0.025,
    t_max: float = 100.0,
    stim_loc: Tuple[int, float] = (0, 0.0),
    params: Optional[Dict] = None,
) -> jnp.ndarray:
    """
    Simulate cell using data_stimulate (JIT-compatible).
    
    This version uses data_stimulate which is compatible with
    JAX transformations (jit, grad, vmap).
    
    Parameters
    ----------
    cell : jx.Cell
        Jaxley cell to simulate.
    current : jnp.ndarray
        Current injection waveform (nA).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    stim_loc : tuple
        (branch_idx, location) for current injection.
    params : dict, optional
        Optional parameters to pass to integrate.
    
    Returns
    -------
    jnp.ndarray
        Voltage recordings.
    """
    # Set up data stimulus
    data_stimuli = None
    stim_branch, stim_loc_val = stim_loc
    data_stimuli = cell.branch(stim_branch).loc(stim_loc_val).data_stimulate(
        current, data_stimuli
    )
    
    # Run simulation
    if params is not None:
        return jx.integrate(cell, params=params, data_stimuli=data_stimuli, 
                           delta_t=dt, t_max=t_max)
    else:
        return jx.integrate(cell, data_stimuli=data_stimuli, delta_t=dt, t_max=t_max)


# =============================================================================
# POPULATION SIMULATION
# =============================================================================

@dataclass
class PopulationSimulationResult:
    """Container for population simulation results."""
    time: jnp.ndarray
    voltages: Dict[int, jnp.ndarray]
    spike_times: Dict[int, jnp.ndarray]
    spike_ids: jnp.ndarray
    all_spike_times: jnp.ndarray
    total_spike_count: int
    mean_firing_rate: float
    
    def get_neuron_spikes(self, neuron_id: int) -> jnp.ndarray:
        """Get spike times for a specific neuron."""
        return self.spike_times.get(neuron_id, jnp.array([]))
    
    def get_neuron_voltage(self, neuron_id: int) -> jnp.ndarray:
        """Get voltage trace for a specific neuron."""
        return self.voltages.get(neuron_id, jnp.array([]))


def simulate_population(
    cells: List[jx.Cell],
    currents: Optional[List[jnp.ndarray]] = None,
    dt: float = 0.025,
    t_max: float = 100.0,
    spike_threshold: float = 0.0,
    record_voltage: bool = True,
) -> PopulationSimulationResult:
    """
    Simulate a population of Jaxley cells.
    
    Parameters
    ----------
    cells : list of jx.Cell
        List of Jaxley cells to simulate.
    currents : list of jnp.ndarray, optional
        Current injection for each cell. If None, no current is injected.
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    spike_threshold : float
        Threshold for spike detection (mV).
    record_voltage : bool
        Whether to record full voltage traces.
    
    Returns
    -------
    PopulationSimulationResult
        Population simulation results.
    """
    n_cells = len(cells)
    
    # Handle currents
    if currents is None:
        n_steps = int(t_max / dt) + 1
        currents = [jnp.zeros(n_steps) for _ in range(n_cells)]
    
    # Storage
    voltages = {}
    spike_times = {}
    all_spikes = []
    all_ids = []
    
    # Simulate each cell
    for i, (cell, current) in enumerate(zip(cells, currents)):
        result = simulate_cell(
            cell=cell,
            current=current,
            dt=dt,
            t_max=t_max,
            spike_threshold=spike_threshold,
        )
        
        if record_voltage:
            voltages[i] = result.voltage
        
        spike_times[i] = result.spike_times
        all_spikes.extend(result.spike_times.tolist())
        all_ids.extend([i] * len(result.spike_times))
    
    # Create time vector
    time = jnp.arange(0, t_max + dt, dt)
    
    # Compute stats
    total_spikes = len(all_spikes)
    duration_s = t_max / 1000.0
    mean_rate = (total_spikes / n_cells / duration_s) if n_cells > 0 and duration_s > 0 else 0.0
    
    return PopulationSimulationResult(
        time=time,
        voltages=voltages,
        spike_times=spike_times,
        spike_ids=jnp.array(all_ids),
        all_spike_times=jnp.array(all_spikes),
        total_spike_count=total_spikes,
        mean_firing_rate=mean_rate,
    )


# =============================================================================
# NETWORK SIMULATION WITH SYNAPSES
# =============================================================================

@dataclass
class SynapticConnection:
    """Specification for a synaptic connection."""
    source_cell_idx: int
    target_cell_idx: int
    target_branch: int = 0
    target_loc: float = 0.5
    weight: float = 0.001  # uS
    delay: float = 1.0  # ms
    reversal: float = 0.0  # mV (0 for excitatory, -75 for inhibitory)
    tau1: float = 0.2  # ms (rise time)
    tau2: float = 2.0  # ms (decay time)


class NetworkSimulator:
    """
    Simulator for networks of Jaxley cells with synaptic connections.
    
    This provides a manual time-stepping approach for network simulation
    where spikes from source cells activate synapses on target cells.
    """
    
    def __init__(
        self,
        cells: List[jx.Cell],
        connections: List[SynapticConnection],
        dt: float = 0.025,
    ):
        self.cells = cells
        self.connections = connections
        self.dt = dt
        self.n_cells = len(cells)
        
        # State tracking
        self._time = 0.0
        self._voltages = {i: -70.0 for i in range(self.n_cells)}
        self._prev_voltages = {i: -70.0 for i in range(self.n_cells)}
        self._pending_spikes = []  # (time, connection_idx)
        
        # Records
        self.spike_times = {i: [] for i in range(self.n_cells)}
        self.voltage_history = {i: [] for i in range(self.n_cells)}
    
    def reset(self):
        """Reset simulation state."""
        self._time = 0.0
        self._voltages = {i: -70.0 for i in range(self.n_cells)}
        self._prev_voltages = {i: -70.0 for i in range(self.n_cells)}
        self._pending_spikes = []
        self.spike_times = {i: [] for i in range(self.n_cells)}
        self.voltage_history = {i: [] for i in range(self.n_cells)}
    
    def _detect_spike(self, cell_idx: int, threshold: float = 0.0) -> bool:
        """Check if cell just crossed threshold (upward)."""
        v = self._voltages[cell_idx]
        v_prev = self._prev_voltages[cell_idx]
        return v > threshold and v_prev <= threshold
    
    def _schedule_synaptic_events(self, source_idx: int):
        """Schedule synaptic events from a spiking source cell."""
        for conn_idx, conn in enumerate(self.connections):
            if conn.source_cell_idx == source_idx:
                event_time = self._time + conn.delay
                self._pending_spikes.append((event_time, conn_idx))
    
    def _process_synaptic_events(self) -> Dict[int, float]:
        """Process pending synaptic events and return synaptic currents."""
        synaptic_currents = {i: 0.0 for i in range(self.n_cells)}
        
        # Get events that should be processed now
        remaining = []
        for event_time, conn_idx in self._pending_spikes:
            if event_time <= self._time:
                conn = self.connections[conn_idx]
                target = conn.target_cell_idx
                
                # Add synaptic conductance as current
                # I_syn = g * (V - E_rev)
                v = self._voltages[target]
                i_syn = conn.weight * (v - conn.reversal)
                synaptic_currents[target] += i_syn
            else:
                remaining.append((event_time, conn_idx))
        
        self._pending_spikes = remaining
        return synaptic_currents
    
    def step(
        self,
        external_currents: Optional[Dict[int, float]] = None,
        spike_threshold: float = 0.0,
    ):
        """
        Advance simulation by one time step.
        
        Parameters
        ----------
        external_currents : dict, optional
            External current for each cell {cell_idx: current_nA}.
        spike_threshold : float
            Threshold for spike detection (mV).
        """
        if external_currents is None:
            external_currents = {}
        
        # Process synaptic events
        synaptic_currents = self._process_synaptic_events()
        
        # Update each cell
        for i in range(self.n_cells):
            # Save previous voltage
            self._prev_voltages[i] = self._voltages[i]
            
            # Total current
            i_ext = external_currents.get(i, 0.0)
            i_syn = synaptic_currents.get(i, 0.0)
            total_current = i_ext + i_syn
            
            # Note: In a full implementation, this would integrate the cell
            # using Jaxley. For now, we use a placeholder.
            # The actual integration would be:
            # voltages = jx.integrate(self.cells[i], ...)
            
            # Detect spikes
            if self._detect_spike(i, spike_threshold):
                self.spike_times[i].append(self._time)
                self._schedule_synaptic_events(i)
            
            # Record voltage
            self.voltage_history[i].append(self._voltages[i])
        
        # Advance time
        self._time += self.dt
    
    def run(
        self,
        t_max: float,
        external_currents: Optional[Dict[int, jnp.ndarray]] = None,
        spike_threshold: float = 0.0,
    ) -> PopulationSimulationResult:
        """
        Run network simulation.
        
        Parameters
        ----------
        t_max : float
            Total simulation time (ms).
        external_currents : dict, optional
            Time-varying currents {cell_idx: current_array}.
        spike_threshold : float
            Threshold for spike detection (mV).
        
        Returns
        -------
        PopulationSimulationResult
            Simulation results.
        """
        self.reset()
        n_steps = int(t_max / self.dt)
        
        for step in range(n_steps):
            # Get current for this step
            step_currents = {}
            if external_currents:
                for cell_idx, current in external_currents.items():
                    if step < len(current):
                        step_currents[cell_idx] = float(current[step])
            
            self.step(step_currents, spike_threshold)
        
        # Compile results
        time = jnp.arange(0, t_max, self.dt)
        voltages = {i: jnp.array(v) for i, v in self.voltage_history.items()}
        spike_times_dict = {i: jnp.array(st) for i, st in self.spike_times.items()}
        
        all_spikes = []
        all_ids = []
        for i, st in spike_times_dict.items():
            all_spikes.extend(st.tolist())
            all_ids.extend([i] * len(st))
        
        total = len(all_spikes)
        mean_rate = (total / self.n_cells / (t_max / 1000.0)) if self.n_cells > 0 else 0.0
        
        return PopulationSimulationResult(
            time=time,
            voltages=voltages,
            spike_times=spike_times_dict,
            spike_ids=jnp.array(all_ids),
            all_spike_times=jnp.array(all_spikes),
            total_spike_count=total,
            mean_firing_rate=mean_rate,
        )


# =============================================================================
# JIT-COMPILED SIMULATION FUNCTIONS
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3))
def jit_simulate_cell(
    cell_params: Dict,
    current: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    """
    JIT-compiled cell simulation (for parameter optimization).
    
    Note: This requires the cell to be configured with trainable parameters.
    
    Parameters
    ----------
    cell_params : dict
        Cell parameters (from cell.get_parameters()).
    current : jnp.ndarray
        Current injection waveform.
    dt : float
        Time step.
    t_max : float
        Total simulation time.
    
    Returns
    -------
    jnp.ndarray
        Voltage recordings.
    """
    # This is a placeholder - actual implementation depends on cell setup
    # The real implementation would use jx.integrate with params
    pass


def sweep_current_amplitudes(
    cell: jx.Cell,
    amplitudes: List[float],
    delay: float = 10.0,
    duration: float = 500.0,
    dt: float = 0.025,
    t_max: float = 600.0,
) -> List[CellSimulationResult]:
    """
    Sweep current injection amplitudes (FI curve measurement).
    
    Parameters
    ----------
    cell : jx.Cell
        Jaxley cell to simulate.
    amplitudes : list of float
        Current amplitudes to test (nA).
    delay : float
        Current onset delay (ms).
    duration : float
        Current duration (ms).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    
    Returns
    -------
    list of CellSimulationResult
        Results for each amplitude.
    """
    results = []
    
    for amp in amplitudes:
        current = step_current(delay, duration, amp, dt, t_max)
        result = simulate_cell(
            cell=cell,
            current=current,
            dt=dt,
            t_max=t_max,
        )
        results.append(result)
    
    return results


def compute_fi_curve(
    cell: jx.Cell,
    amplitudes: List[float],
    delay: float = 100.0,
    duration: float = 500.0,
    dt: float = 0.025,
    t_max: float = 700.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frequency-current (FI) curve.
    
    Parameters
    ----------
    cell : jx.Cell
        Jaxley cell to characterize.
    amplitudes : list of float
        Current amplitudes to test (nA).
    delay : float
        Current onset delay (ms).
    duration : float
        Current duration (ms).
    dt : float
        Time step (ms).
    t_max : float
        Total simulation time (ms).
    
    Returns
    -------
    currents : np.ndarray
        Current amplitudes (nA).
    rates : np.ndarray
        Firing rates (Hz).
    """
    results = sweep_current_amplitudes(
        cell=cell,
        amplitudes=amplitudes,
        delay=delay,
        duration=duration,
        dt=dt,
        t_max=t_max,
    )
    
    currents = np.array(amplitudes)
    rates = np.array([r.firing_rate for r in results])
    
    return currents, rates

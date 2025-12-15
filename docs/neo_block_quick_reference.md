# Neo Block Quick Reference

Quick reference guide for accessing data from MyoGen's Neo Block structures.

## Block Types & Access Patterns

### 1. SPIKE_TRAIN__Block

**Structure**: Block → Segments (motor pools) → SpikeTrain objects

```python
# Load
spike_train__Block = joblib.load("spike_trains.pkl")

# Access
motor_pool = spike_train__Block.segments[pool_idx]
spiketrain = motor_pool.spiketrains[neuron_idx]

# Extract data
spike_times__s = spiketrain.magnitude  # numpy array of spike times
n_spikes = len(spiketrain)
duration = spiketrain.t_stop
sampling_rate = spiketrain.sampling_rate
```

### 2. SURFACE_MUAP__Block

**Structure**: Block → Groups (electrode arrays) → Segments (MUAP indices) → AnalogSignals (3D)

```python
# Load
muaps__Block = joblib.load("surface_muaps.pkl")

# Access
electrode_array = muaps__Block.groups[array_idx]
muap_segment = electrode_array.segments[muap_idx]
muap_signal = muap_segment.analogsignals[0]

# Extract data
muap_array = muap_signal.magnitude  # Shape: (time, rows, cols)
time__s = muap_signal.times.magnitude
electrode_muap = muap_signal[:, row, col]  # Specific electrode
```

### 3. SURFACE_EMG__Block

**Structure**: Block → Groups (electrode arrays) → Segments (motor pools) → AnalogSignals (3D)

```python
# Load
surface_emg__Block = joblib.load("surface_emg.pkl")

# Access
electrode_array = surface_emg__Block.groups[array_idx]
motor_pool = electrode_array.segments[pool_idx]
emg_signal = motor_pool.analogsignals[0]

# Extract data
emg_array = emg_signal.magnitude  # Shape: (time, rows, cols)
time__s = emg_signal.times.magnitude
electrode_emg = emg_signal[:, row, col]  # Specific electrode
```

### 4. INTRAMUSCULAR_MUAP__Block

**Structure**: Block → Segments (MUAP indices) → AnalogSignals (2D)

```python
# Load
im_muaps__Block = joblib.load("intramuscular_muaps.pkl")

# Access
muap_segment = im_muaps__Block.segments[muap_idx]
muap_signal = muap_segment.analogsignals[0]

# Extract data
muap_array = muap_signal.magnitude  # Shape: (time, electrodes)
time__s = muap_signal.times.magnitude
electrode_muap = muap_signal[:, electrode_idx]  # Specific electrode
```

### 5. INTRAMUSCULAR_EMG__Block

**Structure**: Block → Segments (motor pools) → AnalogSignals (2D)

```python
# Load
im_emg__Block = joblib.load("intramuscular_emg.pkl")

# Access
motor_pool = im_emg__Block.segments[pool_idx]
emg_signal = motor_pool.analogsignals[0]

# Extract data
emg_array = emg_signal.magnitude  # Shape: (time, electrodes)
time__s = emg_signal.times.magnitude
electrode_emg = emg_signal[:, electrode_idx]  # Specific electrode
```

## Common Metadata Access

All AnalogSignal objects have these properties:

```python
signal.magnitude          # NumPy array of values
signal.times.magnitude    # Time axis (in seconds)
signal.sampling_rate      # Sampling frequency
signal.sampling_period    # Time between samples
signal.t_start           # Start time
signal.t_stop            # Stop time
signal.duration          # Total duration
signal.units             # Physical units (mV, uV, etc.)
signal.shape             # Array dimensions
signal.name              # Signal name
signal.description       # Description
signal.annotations       # Custom annotations dict
```

## Iteration Patterns

### Iterate through spike trains
```python
for segment in spike_train__Block.segments:
    for spiketrain in segment.spiketrains:
        spike_times = spiketrain.magnitude
        # Process spike times
```

### Iterate through surface MUAPs
```python
for group in muaps__Block.groups:
    for segment in group.segments:
        muap_signal = segment.analogsignals[0]
        # Process MUAP
```

### Iterate through intramuscular data
```python
for segment in im_emg__Block.segments:
    emg_signal = segment.analogsignals[0]
    # Process EMG
```

## Common Operations

### Calculate RMS amplitude
```python
signal_array = emg_signal.magnitude
rms = np.sqrt(np.mean(signal_array ** 2))
```

### Calculate peak-to-peak amplitude
```python
signal_array = emg_signal.magnitude
ptp = np.ptp(signal_array)
```

### Calculate firing rate
```python
import elephant.statistics
firing_rate = elephant.statistics.mean_firing_rate(spiketrain)
```

### Extract time window
```python
# For spike trains
windowed_train = spiketrain.time_slice(t_start, t_stop)

# For analog signals
start_idx = int(t_start * emg_signal.sampling_rate.magnitude)
end_idx = int(t_stop * emg_signal.sampling_rate.magnitude)
windowed_signal = emg_signal[start_idx:end_idx]
```

### Convert to NumPy
```python
# Spike trains
spike_times = spiketrain.magnitude  # 1D array

# Analog signals
signal_array = emg_signal.magnitude  # 2D or 3D array
time_array = emg_signal.times.magnitude  # 1D time axis
```

## Shape Reference

| Block Type | Dimensionality | Shape Format |
|-----------|----------------|--------------|
| SPIKE_TRAIN | 1D | (n_spikes,) |
| SURFACE_MUAP | 3D | (time, rows, cols) |
| SURFACE_EMG | 3D | (time, rows, cols) |
| INTRAMUSCULAR_MUAP | 2D | (time, electrodes) |
| INTRAMUSCULAR_EMG | 2D | (time, electrodes) |

## Example: Complete Workflow

```python
import joblib
import numpy as np
import matplotlib.pyplot as plt
from myogen.utils.types import SPIKE_TRAIN__Block, SURFACE_EMG__Block

# 1. Load data
spike_trains = joblib.load("spike_trains.pkl")
surface_emg = joblib.load("surface_emg.pkl")

# 2. Extract spike times
motor_pool = spike_trains.segments[0]
neuron_5 = motor_pool.spiketrains[5]
spike_times = neuron_5.magnitude

# 3. Extract EMG from center electrode
emg_signal = surface_emg.groups[0].segments[0].analogsignals[0]
center_row = emg_signal.shape[1] // 2
center_col = emg_signal.shape[2] // 2
electrode_emg = emg_signal[:, center_row, center_col]

# 4. Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Spike raster
ax1.scatter(spike_times, [0]*len(spike_times), s=10, c='black')
ax1.set_ylabel('Neuron 5')
ax1.set_ylim(-0.5, 0.5)

# EMG signal
time = emg_signal.times.magnitude
ax2.plot(time, electrode_emg.magnitude)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel(f'EMG ({electrode_emg.units})')

plt.tight_layout()
plt.show()
```

## Tips

1. **Always use `.magnitude`** to extract NumPy arrays from Quantity objects
2. **Check `.shape`** before processing to understand dimensions
3. **Use `.units`** for correct axis labels in plots
4. **Access `.times.magnitude`** for time axis in seconds
5. **Segment names** contain useful info (e.g., "MUAP_5", "Pool_0")
6. **Group names** identify electrode arrays (e.g., "ElectrodeArray_0")

## See Also

- [Full documentation](neo_block_structures.md)
- [Example code](../examples/basic/10_extract_data_from_neo_blocks.py)
- [Neo documentation](https://neo.readthedocs.io/)

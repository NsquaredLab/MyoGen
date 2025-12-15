# Neo Block Data Structures in MyoGen

## Overview

MyoGen uses **Neo Block** objects as the primary data containers for storing simulation results. Neo is a Python package for working with electrophysiology data, providing standardized data structures that are widely used in neuroscience.

This guide explains the hierarchical structure of each Block type used in MyoGen and provides practical code examples for extracting data.

## Table of Contents

1. [SPIKE_TRAIN__Block](#spike_train__block)
2. [SURFACE_MUAP__Block](#surface_muap__block)
3. [SURFACE_EMG__Block](#surface_emg__block)
4. [INTRAMUSCULAR_MUAP__Block](#intramuscular_muap__block)
5. [INTRAMUSCULAR_EMG__Block](#intramuscular_emg__block)
6. [Common Operations](#common-operations)

---

## SPIKE_TRAIN__Block

### Structure

```
Block
└── Segments (motor pools/neuron populations)
    └── SpikeTrain objects (individual neurons)
```

### Description

The `SPIKE_TRAIN__Block` stores neural firing patterns from motor neuron pools. Each segment represents a motor pool (or neuron population), and each spiketrain within a segment represents the firing times of an individual neuron.

### Type Definition

```python
SPIKE_TRAIN__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.segments) > 0
        and all(hasattr(seg, "spiketrains") for seg in x.segments)
        and all(len(seg.spiketrains) > 0 for seg in x.segments)
    ],
]
```

**Structure**: segments (motor pools) → spiketrains (individual neurons)

### Code Examples

#### Loading and Accessing Spike Trains

```python
import joblib
from myogen.utils.types import SPIKE_TRAIN__Block

# Load spike train block
spike_train__Block: SPIKE_TRAIN__Block = joblib.load("spike_trains.pkl")

# Access the first motor pool segment
motor_pool = spike_train__Block.segments[0]
print(f"Motor pool name: {motor_pool.name}")
print(f"Number of neurons: {len(motor_pool.spiketrains)}")

# Access individual neuron spike train
neuron_0 = motor_pool.spiketrains[0]
print(f"Neuron name: {neuron_0.name}")
print(f"Number of spikes: {len(neuron_0)}")
print(f"Spike times (first 5): {neuron_0[:5]}")
print(f"Sampling rate: {neuron_0.sampling_rate}")
print(f"Simulation duration: {neuron_0.t_stop}")
```

#### Iterating Through All Neurons

```python
# Iterate through all motor pools and neurons
for pool_idx, segment in enumerate(spike_train__Block.segments):
    print(f"\nMotor Pool {pool_idx}: {segment.name}")

    for neuron_idx, spiketrain in enumerate(segment.spiketrains):
        print(f"  Neuron {neuron_idx}: {len(spiketrain)} spikes")

        # Extract spike times as numpy array
        spike_times__s = spiketrain.magnitude  # In seconds
        print(f"    First spike: {spike_times__s[0]:.4f} s")
        print(f"    Last spike: {spike_times__s[-1]:.4f} s")
```

#### Calculating Firing Rate Statistics

```python
import numpy as np
import elephant.statistics

# Get all spike trains from first motor pool
motor_pool = spike_train__Block.segments[0]
spike_trains = motor_pool.spiketrains

# Calculate mean firing rates for each neuron
firing_rates = []
for spiketrain in spike_trains:
    if len(spiketrain) > 0:
        rate = elephant.statistics.mean_firing_rate(spiketrain)
        firing_rates.append(rate.magnitude)

print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")
print(f"Std firing rate: {np.std(firing_rates):.2f} Hz")
print(f"Min/Max: {np.min(firing_rates):.2f} - {np.max(firing_rates):.2f} Hz")
```

#### Extracting Spike Times for Analysis

```python
# Extract spike times for all neurons in a motor pool
import numpy as np

motor_pool = spike_train__Block.segments[0]
n_neurons = len(motor_pool.spiketrains)

# Create list of spike times for each neuron
all_spike_times = []
for spiketrain in motor_pool.spiketrains:
    spike_times__s = spiketrain.magnitude  # Convert to numpy array
    all_spike_times.append(spike_times__s)

# Example: Find neurons that fire at specific time
target_time = 5.0  # 5 seconds
tolerance = 0.01  # ±10 ms

active_neurons = []
for neuron_idx, spike_times in enumerate(all_spike_times):
    if np.any(np.abs(spike_times - target_time) < tolerance):
        active_neurons.append(neuron_idx)

print(f"Neurons firing near {target_time}s: {active_neurons}")
```

---

## SURFACE_MUAP__Block

### Structure

```
Block
└── Groups (electrode arrays)
    └── Segments (MUAP indices)
        └── AnalogSignals (3D: samples × rows × columns)
```

### Description

The `SURFACE_MUAP__Block` stores Motor Unit Action Potential (MUAP) templates for surface electrode arrays. Each group represents an electrode array, each segment represents a MUAP from a specific motor unit, and each analogsignal is a 3D array representing the electrical potential across the electrode grid over time.

### Type Definition

```python
SURFACE_MUAP__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.groups) > 0
        and all("ElectrodeArray_" in grp.name for grp in x.groups)
        and all(hasattr(grp, "segments") for grp in x.groups)
        and all(len(grp.segments) > 0 for grp in x.groups)
        and all("MUAP_" in seg.name for grp in x.groups for seg in grp.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 3 for signal in seg.analogsignals)
            for grp in x.groups
            for seg in grp.segments
        )
    ],
]
```

**Structure**: groups (electrode arrays) → segments (MUAP indices) → analogsignals (samples × rows × columns)

### Code Examples

#### Loading and Accessing MUAPs

```python
import joblib
import numpy as np
from myogen.utils.types import SURFACE_MUAP__Block

# Load MUAP block
muaps__Block: SURFACE_MUAP__Block = joblib.load("surface_muaps.pkl")

# Access the first electrode array
electrode_array = muaps__Block.groups[0]
print(f"Electrode array name: {electrode_array.name}")
print(f"Number of MUAPs: {len(electrode_array.segments)}")

# Access a specific MUAP (e.g., motor unit 5)
muap_segment = electrode_array.segments[5]
print(f"MUAP name: {muap_segment.name}")

# Get the MUAP signal (3D grid)
muap_signal = muap_segment.analogsignals[0]
print(f"MUAP shape: {muap_signal.shape}")
print(f"  - Time samples: {muap_signal.shape[0]}")
print(f"  - Electrode rows: {muap_signal.shape[1]}")
print(f"  - Electrode columns: {muap_signal.shape[2]}")
print(f"Sampling rate: {muap_signal.sampling_rate}")
print(f"Units: {muap_signal.units}")
```

#### Extracting MUAP from Specific Electrode

```python
# Get MUAP from motor unit 5, electrode at row 2, column 3
electrode_array = muaps__Block.groups[0]
muap_signal = electrode_array.segments[5].analogsignals[0]

# Extract signal from specific electrode position
row, col = 2, 3
electrode_muap = muap_signal[:, row, col]  # Time series at this electrode

print(f"MUAP amplitude at electrode ({row},{col}):")
print(f"  Peak-to-peak: {np.ptp(electrode_muap.magnitude):.2f} {electrode_muap.units}")
print(f"  Max: {np.max(electrode_muap.magnitude):.2f} {electrode_muap.units}")
print(f"  Min: {np.min(electrode_muap.magnitude):.2f} {electrode_muap.units}")

# Get time axis
time_axis__s = muap_signal.times.magnitude  # In seconds
```

#### Iterating Through All MUAPs and Electrodes

```python
# Iterate through all electrode arrays and MUAPs
for array_idx, group in enumerate(muaps__Block.groups):
    print(f"\nElectrode Array {array_idx}: {group.name}")

    for muap_idx, segment in enumerate(group.segments):
        muap_signal = segment.analogsignals[0]
        n_samples, n_rows, n_cols = muap_signal.shape

        print(f"  MUAP {muap_idx}: {n_samples} samples, {n_rows}×{n_cols} grid")

        # Calculate peak amplitude across all electrodes
        peak_amplitude = np.max(np.abs(muap_signal.magnitude))
        print(f"    Peak amplitude: {peak_amplitude:.2f} {muap_signal.units}")
```

#### Visualizing MUAP Spatial Distribution

```python
import matplotlib.pyplot as plt

# Get a specific MUAP
electrode_array = muaps__Block.groups[0]
muap_signal = electrode_array.segments[5].analogsignals[0]

# Find time of peak activity
peak_time_idx = np.argmax(np.abs(muap_signal.magnitude[:, :, :]))
peak_time_idx = np.unravel_index(peak_time_idx, muap_signal.shape)[0]

# Extract 2D spatial map at peak time
spatial_map = muap_signal[peak_time_idx, :, :].magnitude

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(spatial_map, cmap='seismic', aspect='auto')
plt.colorbar(label=f'Amplitude ({muap_signal.units})')
plt.title(f'MUAP Spatial Distribution at Peak (t={peak_time_idx})')
plt.xlabel('Electrode Column')
plt.ylabel('Electrode Row')
plt.show()
```

---

## SURFACE_EMG__Block

### Structure

```
Block
└── Groups (electrode arrays)
    └── Segments (motor pools)
        └── AnalogSignals (3D: time × rows × columns)
```

### Description

The `SURFACE_EMG__Block` stores synthesized surface EMG signals. Unlike MUAPs (which are templates), this contains the actual EMG signal generated by convolving MUAPs with spike trains. Each group represents an electrode array, each segment represents a motor pool, and each analogsignal is a 3D time-series across the electrode grid.

### Type Definition

```python
SURFACE_EMG__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and len(x.groups) > 0
        and all(hasattr(grp, "segments") for grp in x.groups)
        and all(len(grp.segments) > 0 for grp in x.groups)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 3 for signal in seg.analogsignals)
            for grp in x.groups
            for seg in grp.segments
        )
    ],
]
```

**Structure**: groups (electrode arrays) → segments (motor pools) → analogsignals (time × rows × columns)

### Code Examples

#### Loading and Accessing Surface EMG

```python
import joblib
import numpy as np
from myogen.utils.types import SURFACE_EMG__Block

# Load surface EMG block
surface_emg__Block: SURFACE_EMG__Block = joblib.load("surface_emg.pkl")

# Access the first electrode array
electrode_array = surface_emg__Block.groups[0]
print(f"Electrode array name: {electrode_array.name}")
print(f"Number of motor pools: {len(electrode_array.segments)}")

# Access the first motor pool segment
motor_pool = electrode_array.segments[0]
print(f"Motor pool name: {motor_pool.name}")

# Get the EMG signal
emg_signal = motor_pool.analogsignals[0]
print(f"EMG signal shape: {emg_signal.shape}")
print(f"  - Time samples: {emg_signal.shape[0]}")
print(f"  - Electrode rows: {emg_signal.shape[1]}")
print(f"  - Electrode columns: {emg_signal.shape[2]}")
print(f"Sampling rate: {emg_signal.sampling_rate}")
print(f"Duration: {emg_signal.t_stop}")
```

#### Extracting EMG from Specific Electrode

```python
# Get EMG from first motor pool, electrode at row 2, column 2
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]

# Extract time series from specific electrode
row, col = 2, 2
electrode_emg = emg_signal[:, row, col]  # Shape: (time_samples,)

print(f"EMG from electrode ({row},{col}):")
print(f"  RMS amplitude: {np.sqrt(np.mean(electrode_emg.magnitude**2)):.2f} {electrode_emg.units}")
print(f"  Peak-to-peak: {np.ptp(electrode_emg.magnitude):.2f} {electrode_emg.units}")

# Get time axis
time_axis__s = emg_signal.times.magnitude  # In seconds
```

#### Plotting EMG Signal

```python
import matplotlib.pyplot as plt

# Extract EMG from center electrode
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]
n_samples, n_rows, n_cols = emg_signal.shape
center_row, center_col = n_rows // 2, n_cols // 2

# Get time series
time__s = emg_signal.times.magnitude
emg_values = emg_signal[:, center_row, center_col].magnitude

# Plot
plt.figure(figsize=(12, 4))
plt.plot(time__s, emg_values, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel(f'Amplitude ({emg_signal.units})')
plt.title(f'Surface EMG at Electrode ({center_row},{center_col})')
plt.grid(True, alpha=0.3)
plt.show()
```

#### Calculating Average EMG Across Electrode Grid

```python
# Get EMG signal
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]

# Calculate spatial average across all electrodes
average_emg = np.mean(emg_signal.magnitude, axis=(1, 2))  # Average over rows and columns

# Calculate RMS over time windows
window_size = int(0.1 * emg_signal.sampling_rate.magnitude)  # 100ms windows
n_windows = len(average_emg) // window_size

rms_values = []
for i in range(n_windows):
    window_data = average_emg[i*window_size:(i+1)*window_size]
    rms = np.sqrt(np.mean(window_data**2))
    rms_values.append(rms)

print(f"Average RMS amplitude: {np.mean(rms_values):.2f} {emg_signal.units}")
```

---

## INTRAMUSCULAR_MUAP__Block

### Structure

```
Block
└── Segments (MUAP indices)
    └── AnalogSignals (2D: samples × electrodes)
```

### Description

The `INTRAMUSCULAR_MUAP__Block` stores MUAP templates for intramuscular electrodes. Each segment represents a MUAP from a specific motor unit, and each analogsignal is a 2D array with time samples across multiple electrode contacts.

### Type Definition

```python
INTRAMUSCULAR_MUAP__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and all("MUAP_" in seg.name for seg in x.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 2 for signal in seg.analogsignals)
            for seg in x.segments
        )
    ],
]
```

**Structure**: segments (MUAP indices) → analogsignals (samples × electrodes)

### Code Examples

#### Loading and Accessing Intramuscular MUAPs

```python
import joblib
import numpy as np
from myogen.utils.types import INTRAMUSCULAR_MUAP__Block

# Load intramuscular MUAP block
muaps__Block: INTRAMUSCULAR_MUAP__Block = joblib.load("intramuscular_muaps.pkl")

print(f"Number of MUAPs: {len(muaps__Block.segments)}")

# Access a specific MUAP (e.g., motor unit 3)
muap_segment = muaps__Block.segments[3]
print(f"MUAP name: {muap_segment.name}")

# Get the MUAP signal
muap_signal = muap_segment.analogsignals[0]
print(f"MUAP shape: {muap_signal.shape}")
print(f"  - Time samples: {muap_signal.shape[0]}")
print(f"  - Number of electrodes: {muap_signal.shape[1]}")
print(f"Sampling rate: {muap_signal.sampling_rate}")
```

#### Extracting MUAP from Specific Electrode Contact

```python
# Get MUAP from motor unit 3, electrode contact 0
muap_signal = muaps__Block.segments[3].analogsignals[0]

# Extract signal from first electrode contact
electrode_idx = 0
electrode_muap = muap_signal[:, electrode_idx]

print(f"MUAP at electrode {electrode_idx}:")
print(f"  Peak amplitude: {np.max(np.abs(electrode_muap.magnitude)):.2f} {electrode_muap.units}")
print(f"  Duration: {muap_signal.t_stop}")

# Get time axis
time_axis__s = muap_signal.times.magnitude
```

#### Comparing MUAPs Across Motor Units

```python
import matplotlib.pyplot as plt

# Plot MUAPs from first 5 motor units at electrode 0
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

for mu_idx in range(min(5, len(muaps__Block.segments))):
    muap_signal = muaps__Block.segments[mu_idx].analogsignals[0]
    time__s = muap_signal.times.magnitude

    # Plot from first electrode
    axes[mu_idx].plot(time__s, muap_signal[:, 0].magnitude)
    axes[mu_idx].set_ylabel(f'MU {mu_idx}\n({muap_signal.units})')
    axes[mu_idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Intramuscular MUAPs - Electrode 0')
plt.tight_layout()
plt.show()
```

---

## INTRAMUSCULAR_EMG__Block

### Structure

```
Block
└── Segments (motor pools)
    └── AnalogSignals (2D: time × electrodes)
```

### Description

The `INTRAMUSCULAR_EMG__Block` stores synthesized intramuscular EMG signals. Each segment represents a motor pool, and each analogsignal is a 2D time-series across electrode contacts.

### Type Definition

```python
INTRAMUSCULAR_EMG__Block = Annotated[
    Block,
    Is[
        lambda x: isinstance(x, Block)
        and all("Pool_" in seg.name for seg in x.segments)
        and all(
            hasattr(seg, "analogsignals")
            and len(seg.analogsignals) > 0
            and all(hasattr(signal, "shape") for signal in seg.analogsignals)
            and all(len(signal.shape) == 2 for signal in seg.analogsignals)
            for seg in x.segments
        )
    ],
]
```

**Structure**: segments (motor pools) → analogsignals (time × electrodes)

### Code Examples

#### Loading and Accessing Intramuscular EMG

```python
import joblib
import numpy as np
from myogen.utils.types import INTRAMUSCULAR_EMG__Block

# Load intramuscular EMG block
im_emg__Block: INTRAMUSCULAR_EMG__Block = joblib.load("intramuscular_emg.pkl")

print(f"Number of motor pools: {len(im_emg__Block.segments)}")

# Access the first motor pool
motor_pool = im_emg__Block.segments[0]
print(f"Motor pool name: {motor_pool.name}")

# Get the EMG signal
emg_signal = motor_pool.analogsignals[0]
print(f"EMG shape: {emg_signal.shape}")
print(f"  - Time samples: {emg_signal.shape[0]}")
print(f"  - Number of electrodes: {emg_signal.shape[1]}")
print(f"Sampling rate: {emg_signal.sampling_rate}")
print(f"Duration: {emg_signal.t_stop}")
```

#### Extracting EMG from Specific Electrode Contact

```python
# Get EMG from first motor pool, electrode contact 2
emg_signal = im_emg__Block.segments[0].analogsignals[0]

# Extract time series from specific electrode
electrode_idx = 2
electrode_emg = emg_signal[:, electrode_idx]

print(f"EMG from electrode {electrode_idx}:")
print(f"  RMS amplitude: {np.sqrt(np.mean(electrode_emg.magnitude**2)):.2f} {electrode_emg.units}")
print(f"  Peak-to-peak: {np.ptp(electrode_emg.magnitude):.2f} {electrode_emg.units}")
```

#### Calculating Differential EMG Between Adjacent Contacts

```python
# Calculate single differential EMG between adjacent electrode contacts
emg_signal = im_emg__Block.segments[0].analogsignals[0]
n_samples, n_electrodes = emg_signal.shape

# Single differential (subtract adjacent electrodes)
differential_emg = np.zeros((n_samples, n_electrodes - 1))

for i in range(n_electrodes - 1):
    differential_emg[:, i] = (emg_signal[:, i+1] - emg_signal[:, i]).magnitude

print(f"Differential EMG shape: {differential_emg.shape}")
print(f"RMS of differential channel 0: {np.sqrt(np.mean(differential_emg[:, 0]**2)):.2f}")
```

---

## Common Operations

### Saving and Loading Blocks

```python
import joblib

# Save a block
joblib.dump(spike_train__Block, "spike_trains.pkl")
joblib.dump(surface_emg__Block, "surface_emg.pkl")

# Load a block
spike_train__Block = joblib.load("spike_trains.pkl")
surface_emg__Block = joblib.load("surface_emg.pkl")
```

### Converting to NumPy Arrays

```python
# Spike trains to numpy array of spike times
motor_pool = spike_train__Block.segments[0]
spike_times_list = [st.magnitude for st in motor_pool.spiketrains]

# EMG signal to numpy array
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]
emg_array = emg_signal.magnitude  # Shape: (time, rows, cols)
time_array = emg_signal.times.magnitude  # Time axis in seconds
```

### Accessing Metadata

```python
# All Neo objects have metadata
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]

print(f"Sampling rate: {emg_signal.sampling_rate}")
print(f"Sampling period: {emg_signal.sampling_period}")
print(f"Start time: {emg_signal.t_start}")
print(f"Stop time: {emg_signal.t_stop}")
print(f"Duration: {emg_signal.duration}")
print(f"Units: {emg_signal.units}")
print(f"Name: {emg_signal.name}")
print(f"Description: {emg_signal.description}")

# Access custom annotations (if any)
print(f"Annotations: {emg_signal.annotations}")
```

### Type Validation

```python
from myogen.utils.types import (
    SPIKE_TRAIN__Block,
    SURFACE_EMG__Block,
    SURFACE_MUAP__Block,
    INTRAMUSCULAR_EMG__Block,
    INTRAMUSCULAR_MUAP__Block
)

# These type annotations provide runtime validation with Beartype
def process_spike_trains(spike_train__Block: SPIKE_TRAIN__Block) -> None:
    """Function with automatic type validation."""
    # Beartype automatically validates that spike_train__Block matches
    # the SPIKE_TRAIN__Block structure requirements
    pass

# Example: This will raise an error if the structure is invalid
# process_spike_trains(surface_emg__Block)  # TypeError!
```

### Checking Block Structure

```python
# Check what type of block you have
from neo.core import Block

if isinstance(data, Block):
    # Check for spike trains
    if hasattr(data, 'segments') and len(data.segments) > 0:
        first_segment = data.segments[0]

        if hasattr(first_segment, 'spiketrains') and len(first_segment.spiketrains) > 0:
            print("This is a SPIKE_TRAIN__Block")

        elif hasattr(first_segment, 'analogsignals') and len(first_segment.analogsignals) > 0:
            signal_shape = first_segment.analogsignals[0].shape

            if len(signal_shape) == 2:
                if "MUAP_" in first_segment.name:
                    print("This is an INTRAMUSCULAR_MUAP__Block")
                elif "Pool_" in first_segment.name:
                    print("This is an INTRAMUSCULAR_EMG__Block")

    # Check for groups (surface recordings)
    if hasattr(data, 'groups') and len(data.groups) > 0:
        first_group = data.groups[0]
        first_segment = first_group.segments[0]

        if "MUAP_" in first_segment.name:
            print("This is a SURFACE_MUAP__Block")
        else:
            print("This is a SURFACE_EMG__Block")
```

### Iterating with Progress Tracking

```python
from tqdm import tqdm

# Process all MUAPs with progress bar
muaps__Block = joblib.load("surface_muaps.pkl")

for group in tqdm(muaps__Block.groups, desc="Electrode Arrays"):
    for segment in tqdm(group.segments, desc="MUAPs", leave=False):
        muap_signal = segment.analogsignals[0]

        # Process MUAP
        peak_amplitude = np.max(np.abs(muap_signal.magnitude))
        # ... do something with peak_amplitude
```

---

## Key Takeaways

1. **Neo Blocks** are hierarchical containers with different structures for different data types

2. **Access patterns**:
   - Spike trains: `block.segments[pool_idx].spiketrains[neuron_idx]`
   - Surface signals: `block.groups[array_idx].segments[idx].analogsignals[0]`
   - Intramuscular signals: `block.segments[idx].analogsignals[0]`

3. **Dimensionality**:
   - Spike trains: 1D array of spike times
   - Surface signals: 3D (time × rows × columns)
   - Intramuscular signals: 2D (time × electrodes)

4. **Units and metadata**: All signals carry units (via `quantities` package) and metadata (sampling rate, duration, etc.)

5. **Type safety**: MyoGen's type annotations provide automatic validation when using `@beartowertype` decorator

---

## Additional Resources

- [Neo Documentation](https://neo.readthedocs.io/)
- [Quantities Package](https://python-quantities.readthedocs.io/)
- [Elephant (Electrophysiology Analysis)](https://elephant.readthedocs.io/)
- [MyoGen Examples](/examples/)

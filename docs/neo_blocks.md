# Neo Blocks in MyoGen

MyoGen uses Neo Block objects to store simulation results with automatic unit tracking and metadata.

## Block Types

| Block Type | What it stores | Shape | Access Pattern |
|------------|---------------|-------|----------------|
| SPIKE_TRAIN__Block | Spike times | `(n_spikes,)` | `block.segments[pool].spiketrains[neuron]` |
| SURFACE_MUAP__Block | MUAP templates (surface grid) | `(time, rows, cols)` | `block.groups[array].segments[mu].analogsignals[0]` |
| SURFACE_EMG__Block | EMG signals (surface grid) | `(time, rows, cols)` | `block.groups[array].segments[pool].analogsignals[0]` |
| INTRAMUSCULAR_MUAP__Block | MUAP templates (needle) | `(time, electrodes)` | `block.segments[mu].analogsignals[0]` |
| INTRAMUSCULAR_EMG__Block | EMG signals (needle) | `(time, electrodes)` | `block.segments[pool].analogsignals[0]` |

## Structure Diagrams

```
SPIKE_TRAIN__Block                 SURFACE_EMG__Block / SURFACE_MUAP__Block
──────────────────                 ────────────────────────────────────────
Block                              Block
└── Segment[pool_idx]              └── Group[array_idx]
    └── SpikeTrain[neuron_idx]         └── Segment[pool/muap_idx]
        → [t1, t2, t3, ...]                └── AnalogSignal[0]
                                               → 3D array (time × rows × cols)

INTRAMUSCULAR_EMG__Block / INTRAMUSCULAR_MUAP__Block
────────────────────────────────────────────────────
Block
└── Segment[pool/muap_idx]
    └── AnalogSignal[0]
        → 2D array (time × electrodes)
```

## Quick Start

```python
import joblib

# Load
data = joblib.load("results.pkl")

# Spike trains
spike_times = data.segments[0].spiketrains[0].magnitude  # numpy array

# Surface EMG/MUAP (3D grid)
signal = data.groups[0].segments[0].analogsignals[0]
electrode_trace = signal[:, row, col].magnitude  # single electrode
time = signal.times.magnitude

# Intramuscular EMG/MUAP (2D array)
signal = data.segments[0].analogsignals[0]
electrode_trace = signal[:, electrode_idx].magnitude
```

## Common Operations

```python
# Metadata
signal.sampling_rate      # e.g., 2048 Hz
signal.t_start / t_stop   # Start/stop time
signal.units              # e.g., mV, µV
signal.shape              # Array dimensions

# Convert to numpy
values = signal.magnitude
time = signal.times.magnitude

# Slice time window
windowed = spiketrain.time_slice(t_start, t_stop)

# Calculate RMS
rms = np.sqrt(np.mean(signal.magnitude ** 2))

# Firing rate (requires elephant)
import elephant.statistics
rate = elephant.statistics.mean_firing_rate(spiketrain)
```

## Iteration Examples

```python
# All spike trains
for segment in block.segments:
    for st in segment.spiketrains:
        print(f"{st.name}: {len(st)} spikes")

# All surface MUAPs
for group in block.groups:
    for segment in group.segments:
        muap = segment.analogsignals[0]
        print(f"{segment.name}: shape {muap.shape}")

# All intramuscular signals
for segment in block.segments:
    signal = segment.analogsignals[0]
    print(f"{segment.name}: shape {signal.shape}")
```

## Key Differences

| | Surface | Intramuscular |
|--|---------|---------------|
| **Electrodes** | 2D grid (rows × cols) | 1D array |
| **Signal shape** | 3D: `(time, rows, cols)` | 2D: `(time, electrodes)` |
| **Has Groups?** | Yes (multiple arrays) | No |

| | MUAP | EMG |
|--|------|-----|
| **Duration** | Short (~10-50ms) | Long (full simulation) |
| **Content** | Template per motor unit | Full signal (MUAPs convolved with spikes) |
| **Segments named** | `MUAP_0`, `MUAP_1`... | `Pool_0`, `Pool_1`... |

## Troubleshooting

**`AttributeError: 'Block' object has no attribute 'groups'`**
→ This is a spike train or intramuscular block. Use `block.segments` instead.

**`IndexError` when accessing electrodes**
→ Check shape first: `print(signal.shape)`

**Values seem wrong**
→ Check units: `print(signal.units)` and use `signal.rescale(pq.mV)` if needed.

## See Also

- Example: `examples/01_basic/12_extract_data_from_neo_blocks.py`
- Neo docs: https://neo.readthedocs.io/
- Elephant docs: https://elephant.readthedocs.io/

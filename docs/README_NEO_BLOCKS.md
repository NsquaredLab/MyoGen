# Working with Neo Blocks in MyoGen

This guide helps you understand and extract data from MyoGen's Neo Block structures.

## 📚 Documentation Files

1. **[neo_block_structures.md](neo_block_structures.md)** - Comprehensive guide
   - Detailed explanation of each Block type
   - Complete code examples with real-world use cases
   - Common operations and best practices
   - ~15 min read

2. **[neo_block_quick_reference.md](neo_block_quick_reference.md)** - Quick reference
   - Concise access patterns for each Block type
   - Common code snippets
   - Shape and structure summary table
   - ~3 min read

3. **[neo_block_structure_diagrams.txt](neo_block_structure_diagrams.txt)** - Visual diagrams
   - ASCII art hierarchical structures
   - Data flow visualization
   - Memory considerations
   - ~5 min read

4. **[Example Code](../examples/basic/10_extract_data_from_neo_blocks.py)** - Executable examples
   - Runnable Python script demonstrating all extraction patterns
   - Generates analysis plots
   - Can be run after completing other examples

## 🚀 Quick Start

### 1. Understanding the Block Types

MyoGen uses 5 different Neo Block types:

| Block Type | Purpose | Dimensions |
|-----------|---------|------------|
| `SPIKE_TRAIN__Block` | Neural firing times | 1D spike times |
| `SURFACE_MUAP__Block` | Surface MUAP templates | 3D electrode grid |
| `SURFACE_EMG__Block` | Surface EMG signals | 3D electrode grid |
| `INTRAMUSCULAR_MUAP__Block` | Intramuscular MUAP templates | 2D electrode array |
| `INTRAMUSCULAR_EMG__Block` | Intramuscular EMG signals | 2D electrode array |

### 2. Basic Access Patterns

```python
import joblib

# Spike trains
spike_trains = joblib.load("spike_trains.pkl")
spiketrain = spike_trains.segments[0].spiketrains[0]
spike_times = spiketrain.magnitude  # NumPy array

# Surface EMG
surface_emg = joblib.load("surface_emg.pkl")
emg_signal = surface_emg.groups[0].segments[0].analogsignals[0]
electrode_data = emg_signal[:, row, col]  # Time series at electrode

# Intramuscular EMG
im_emg = joblib.load("intramuscular_emg.pkl")
emg_signal = im_emg.segments[0].analogsignals[0]
electrode_data = emg_signal[:, electrode_idx]  # Time series
```

### 3. Running the Example

```bash
# Make sure you have run the prerequisite examples first
cd examples/basic

# Run spike train generation
python 02_simulate_spike_trains_descending_drive.py

# Run surface EMG simulation
python 05_simulate_surface_emg.py

# Run the extraction example
python 10_extract_data_from_neo_blocks.py
```

## 📖 Common Use Cases

### Extract and Plot Spike Raster

```python
motor_pool = spike_train__Block.segments[0]

for neuron_idx, spiketrain in enumerate(motor_pool.spiketrains):
    spike_times = spiketrain.magnitude
    plt.scatter(spike_times, [neuron_idx]*len(spike_times), s=1, c='black')
```

### Calculate EMG RMS

```python
emg_signal = surface_emg__Block.groups[0].segments[0].analogsignals[0]
electrode_emg = emg_signal[:, row, col].magnitude

rms = np.sqrt(np.mean(electrode_emg ** 2))
```

### Compute Firing Rate

```python
import elephant.statistics

spiketrain = spike_train__Block.segments[0].spiketrains[0]
firing_rate = elephant.statistics.mean_firing_rate(spiketrain)
```

## 🔍 Finding What You Need

**Want to understand the structure?**
→ Read [neo_block_structures.md](neo_block_structures.md)

**Need quick code snippets?**
→ Check [neo_block_quick_reference.md](neo_block_quick_reference.md)

**Want visual diagrams?**
→ See [neo_block_structure_diagrams.txt](neo_block_structure_diagrams.txt)

**Need working examples?**
→ Run [10_extract_data_from_neo_blocks.py](../examples/basic/10_extract_data_from_neo_blocks.py)

## ❓ FAQ

**Q: Why are there different Block types?**

A: Each represents a different stage of the EMG simulation pipeline:
- Spike trains → when neurons fire
- MUAPs → what individual motor units look like
- EMG → the combined signal from all motor units

**Q: What's the difference between surface and intramuscular?**

A:
- Surface: 2D grid of electrodes on the skin (3D data: time × rows × cols)
- Intramuscular: Linear array of electrodes in the muscle (2D data: time × electrodes)

**Q: What's the difference between MUAP and EMG blocks?**

A:
- MUAP: Templates for individual motor units (short duration)
- EMG: Full signal from convolving MUAPs with spike trains (long duration)

**Q: How do I access metadata like sampling rate?**

A: All signals have metadata properties:
```python
signal.sampling_rate    # Frequency
signal.t_start         # Start time
signal.t_stop          # Stop time
signal.units           # Physical units
```

**Q: Why use Neo instead of plain NumPy arrays?**

A: Neo provides:
- Standardized structure for electrophysiology data
- Automatic unit tracking (mV, µV, seconds, etc.)
- Metadata storage (sampling rate, duration, etc.)
- Compatibility with analysis tools like Elephant
- Integration with neuroscience workflows

## 🛠️ Troubleshooting

**Issue**: `AttributeError: 'Block' object has no attribute 'groups'`

**Solution**: This is likely a spike train or intramuscular block. Those use `segments` not `groups`.

---

**Issue**: `IndexError` when accessing electrodes

**Solution**: Check the shape first:
```python
print(signal.shape)  # e.g., (150000, 13, 5)
# Then access within bounds: signal[:, 0:13, 0:5]
```

---

**Issue**: Values seem wrong (very large or very small)

**Solution**: Check the units:
```python
print(signal.units)  # e.g., mV or µV
signal.rescale(pq.mV)  # Convert to desired unit
```

## 📚 Additional Resources

- [Neo Documentation](https://neo.readthedocs.io/)
- [Elephant (Electrophysiology Analysis)](https://elephant.readthedocs.io/)
- [Quantities (Unit Handling)](https://python-quantities.readthedocs.io/)
- [MyoGen Examples](/examples/)

## 💡 Tips

1. **Always extract `.magnitude`** to get NumPy arrays from Quantity objects
2. **Check `.shape`** before processing to understand dimensions
3. **Use `.units`** for proper axis labels in plots
4. **Segment names** contain useful information (e.g., "MUAP_5", "Pool_0")
5. **Type annotations** provide runtime validation with Beartype

---

**Need help?** Check the full documentation or run the example code to see it all in action!

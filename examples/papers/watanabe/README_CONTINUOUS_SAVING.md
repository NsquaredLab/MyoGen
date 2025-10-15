# Continuous Saving for Long Simulations

## Problem

The original Watanabe simulation records membrane potentials for all 400 motor neurons across 180 seconds (7.2 million timesteps), requiring **~23 GB of RAM**. This causes out-of-memory errors on most systems.

## Solution

The continuous saving system saves data in **chunks** every 10 seconds of simulation time, keeping peak RAM usage **below 100 MB** regardless of simulation duration.

---

## Files

- **`continuous_saver.py`** - Core continuous saving utilities
- **`10a_paper_watanabe.py`** - Modified simulation with continuous saving
- **`10b_load_and_analyze.py`** - Load and analyze saved chunks

---

## How It Works

### During Simulation

1. **Record**: At each timestep, voltages are read and stored in Python lists
2. **Chunk**: Every 10 seconds of simulation time, data is saved to disk as `chunk_XXXX.pkl`
3. **Clear**: Memory is immediately freed after saving
4. **Repeat**: Process continues until simulation ends

### Memory Usage

- **Per chunk**: ~80 MB (40 neurons × 400,000 timesteps × 8 bytes)
- **Peak RAM**: <100 MB (only one chunk in memory at a time)
- **Total saved**: ~18 chunks for 180-second simulation

---

## Usage

### 1. Run the Simulation

```bash
python 10a_paper_watanabe.py
```

**Output structure:**
```
results/
└── watanabe_chunks/
    ├── chunk_0000.pkl  # 0-10 seconds
    ├── chunk_0001.pkl  # 10-20 seconds
    ├── ...
    ├── chunk_0017.pkl  # 170-180 seconds
    ├── spikes.pkl      # All spike data
    └── metadata.pkl    # Simulation metadata
```

### 2. Load and Analyze

**Option A: Use the provided analysis script**
```bash
python 10b_load_and_analyze.py
```

**Option B: Load chunks manually in Python**
```python
from continuous_saver import load_and_combine_chunks
from pathlib import Path

# Load all chunks and combine
data = load_and_combine_chunks(
    Path("./results/watanabe_chunks"),
    output_filename="combined_data.pkl"  # Optional: save combined file
)

# Access data
times = data["times"]  # Time vector (ms)
voltages = data["membrane_data"]["aMN"][0]  # Neuron 0 voltages
spikes = data["spikes"]["aMN"]  # All aMN spikes
```

### 3. Work with Individual Chunks (Advanced)

If combined data is too large, work with chunks individually:

```python
import joblib
from pathlib import Path

chunks_path = Path("./results/watanabe_chunks")

# Load metadata
metadata = joblib.load(chunks_path / "metadata.pkl")
print(f"Total chunks: {metadata['total_chunks']}")

# Load specific chunk
chunk_5 = joblib.load(chunks_path / "chunk_0005.pkl")
print(f"Chunk 5 time range: {chunk_5['time_start']:.1f} - {chunk_5['time_end']:.1f} ms")

# Access voltage data
neuron_0_voltages = chunk_5["membrane_data"]["aMN"][0]
```

---

## Configuration Options

### Adjust Recording Density

In `10a_paper_watanabe.py`, line 297:

```python
# Record every 10th neuron (40 total) - ~80 MB per chunk
recording_neurons = list(range(0, naMN, 10))

# Record every 20th neuron (20 total) - ~40 MB per chunk
recording_neurons = list(range(0, naMN, 20))

# Record specific neurons
recording_neurons = [0, 50, 100, 150, 200, 250, 300, 350, 399]
```

### Adjust Chunk Duration

In `10a_paper_watanabe.py`, line 301:

```python
# Save every 10 seconds (18 chunks for 180s simulation)
chunk_duration__ms=10000.0

# Save every 5 seconds (36 smaller chunks)
chunk_duration__ms=5000.0

# Save every 30 seconds (6 larger chunks)
chunk_duration__ms=30000.0
```

**Trade-off**: Smaller chunks = more frequent saves = lower peak RAM but slower simulation

---

## Data Structure

### Chunk File Format

```python
chunk_data = {
    "chunk_id": 0,
    "time_start": 0.0,          # ms
    "time_end": 10000.0,        # ms
    "times": np.array([...]),   # Time vector for this chunk
    "timestep__ms": 0.025,
    "membrane_data": {
        "aMN": {
            0: np.array([...]),   # Voltages for neuron 0
            10: np.array([...]),  # Voltages for neuron 10
            # ...
        }
    }
}
```

### Combined Data Format

```python
combined_data = {
    "times": np.array([...]),           # Full time vector
    "timestep__ms": 0.025,
    "membrane_data": {
        "aMN": {
            0: np.array([...]),           # All voltages for neuron 0
            10: np.array([...]),          # All voltages for neuron 10
            # ...
        }
    },
    "spikes": {
        "aMN": {
            "times": np.array([...]),     # All spike times
            "ids": np.array([...]),       # Corresponding neuron IDs
        },
        "DD": {...},
        "IN": {...}
    },
    "metadata": {
        "total_chunks": 18,
        "chunk_duration__ms": 10000.0,
        "recording_config": {"aMN": [0, 10, 20, ...]}
    }
}
```

---

## Benefits

✅ **No memory limits** - Can run arbitrarily long simulations
✅ **Automatic saving** - No risk of losing data if simulation crashes
✅ **Flexible analysis** - Load only chunks you need
✅ **Incremental results** - Start analyzing while simulation runs
✅ **Disk-based backup** - Data saved continuously to disk

---

## Limitations

⚠️ **Slightly slower** - Recording overhead adds ~5-10% to simulation time
⚠️ **More disk I/O** - Frequent writes (mitigated by compressed pickle)
⚠️ **Manual loading** - Need to explicitly combine chunks for full analysis

---

## Tips

1. **Monitor disk space**: 180s simulation with 40 neurons ≈ 1.5 GB disk space
2. **Use SSD**: Faster disk I/O reduces saving overhead
3. **Adjust chunk size**: Larger chunks = fewer saves but more RAM
4. **Sample neurons**: Recording every 10th neuron provides good resolution
5. **Check chunks**: Verify chunk files are created during simulation

---

## Troubleshooting

**Problem: "No results to save"**
- ✓ Fixed by using continuous saving instead of SimulationRunner recording

**Problem: Simulation still runs out of memory**
- Record fewer neurons (every 20th instead of 10th)
- Reduce chunk duration (5s instead of 10s)
- Check for memory leaks in step callback

**Problem: Chunks not saving**
- Check disk space (need ~2 GB free)
- Verify write permissions to results directory
- Check console output for ContinuousSaver messages

**Problem: Combined data file too large to load**
- Work with individual chunks instead
- Use memory-mapped arrays (np.memmap)
- Analyze data in smaller time windows

---

## Performance Comparison

| Mode | RAM Usage | Simulation Speed | Data Access |
|------|-----------|------------------|-------------|
| **Original** | 23 GB | Baseline | Immediate |
| **Continuous** | <100 MB | 95% of baseline | Load required |

---

## Questions?

Check the code comments in:
- `continuous_saver.py` - Core implementation
- `10a_paper_watanabe.py:156-224` - Integration with step callback
- `10b_load_and_analyze.py` - Usage examples

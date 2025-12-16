<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span>Welcome to</span>
    <img src="https://raw.githubusercontent.com/NsquaredLab/MyoGen/main/docs/source/_static/myogen_logo.png" height="100" alt="MyoGen Logo">
  </h1>

  <h2>The modular and extandable simulation toolkit for neurophysiology</h2>

  [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://nsquaredlab.github.io/MyoGen/)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Version](https://img.shields.io/badge/version-0.6.5-orange.svg)](https://github.com/NsquaredLab/MyoGen)

  [Installation](https://nsquaredlab.github.io/MyoGen/#installation) •
  [Documentation](https://nsquaredlab.github.io/MyoGen/) •
  [Examples](https://nsquaredlab.github.io/MyoGen/examples.html) •
  [How to Cite](https://nsquaredlab.github.io/MyoGen/#how-to-cite)
</div>

# Overview

MyoGen is a **modular and extensible neuromuscular simulation framework** for generating physiologically grounded motor-unit activity, muscle force, and surface EMG signals.  

It supports end-to-end modeling of the neuromuscular pathway, from descending neural drive and spinal motor neuron dynamics to muscle activation and bioelectric signal formation at the electrode level.
MyoGen is designed for algorithm validation, hypothesis-driven research, and education, providing configurable building blocks that can be independently combined and extended.

# Highlights

🧬 **Biophysically inspired neuron models** — NEURON-based motor neurons with validated calcium dynamics and membrane properties

🎯 **Everything is inspectable** — Complete access to every motor unit, spike time, fiber location etc. for rigorous algorithm testing

⚡️ **Vectorized & parallel** — Multi-core CPU processing with NumPy/Numba vectorization for fast computation

🔬 **End-to-end simulation** — From motor unit recruitment to high-density surface EMG in a single framework

📊 **Reproducible science** — Deterministic random seeds and standardized Neo Block outputs for exact replication

🧰 **Comprehensive toolkit** — Surface EMG, intramuscular EMG, force generation, and spinal network modeling

# Installation

> [!WARNING]
> **Windows users**: Install [NEURON 8.2.6](https://github.com/neuronsimulator/nrn/releases/download/8.2.6/nrn-8.2.6.w64-mingw-py-38-39-310-311-312-setup.exe) before installing MyoGen

```bash
uv add MyoGen
# or
pip install MyoGen
```

NEURON mechanisms compile automatically during installation.

---

**Prerequisites**: Python ≥3.12, Linux/Windows/macOS

> [!IMPORTANT]
> **System Requirements**:
> - **Linux/macOS**: OpenMPI or MPICH (install via package manager)
> - **Windows**: NEURON 8.2.6 required (see warning above)

```bash
# Install MPI (Linux)
sudo apt-get install libopenmpi-dev  # Ubuntu/Debian
# or
sudo yum install openmpi-devel       # RHEL/CentOS

# Install MPI (macOS)
brew install open-mpi
```

---

## From Source (for development)

```bash
# Clone and install
git clone https://github.com/NsquaredLab/MyoGen.git
cd MyoGen
uv sync

# Activate environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Compile NEURON mechanisms (required for editable installs)
uv run poe setup_myogen
```

> [!TIP]
> Install [uv](https://docs.astral.sh/uv/) first

## Optional Dependencies

**GPU acceleration** (5-10× speedup for convolutions):

```bash
pip install cupy-cuda12x
```


# Quick Start

Generate motor unit action potentials (MUAPs):

```python
from myogen import simulator
import quantities as pq

# 1. Generate recruitment thresholds (100 motor units)
thresholds, _ = simulator.RecruitmentThresholds(
    N=100,
    recruitment_range__ratio=50,
    mode="fuglevand"
)

# 2. Create muscle model with fiber distribution
muscle = simulator.Muscle(
    recruitment_thresholds=thresholds,
    radius_bone__mm=1.0 * pq.mm,
    fiber_density__fibers_per_mm2=400 * pq.mm**-2,
    fat_thickness__mm=10 * pq.mm,
    autorun=True
)

# 3. Set up surface electrode array
electrode_array = simulator.SurfaceElectrodeArray(
    num_rows=5,
    num_cols=5,
    inter_electrode_distances__mm=5 * pq.mm,
    electrode_radius__mm=5 * pq.mm,
    bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm,
)

# 4. Create surface EMG simulator
surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    electrode_arrays=[electrode_array],
    sampling_frequency__Hz=2048.0,
    MUs_to_simulate=[0, 1, 2, 3, 4]  # First 5 motor units
)

# 5. Simulate MUAPs (parallel processing)
muaps = surface_emg.simulate_muaps(n_jobs=-2)
```

**Access MUAP data**:

```python
import numpy as np

# Get MUAP from motor unit 0
muap_signal = muaps.groups[0].segments[0].analogsignals[0]
print(f"MUAP shape: {muap_signal.shape}")  # (time, rows, cols)

# Extract from specific electrode (row 2, col 2)
electrode_muap = muap_signal[:, 2, 2]
peak_amplitude = np.max(np.abs(electrode_muap.magnitude))
print(f"Peak amplitude: {peak_amplitude:.3f} {electrode_muap.units}")
```

**For full EMG simulation** with spike trains, see [examples](https://nsquaredlab.github.io/MyoGen/examples.html)

# Documentation

📖 **[Read the full documentation](https://nsquaredlab.github.io/MyoGen/)**

- [User Guide](https://nsquaredlab.github.io/MyoGen/neo_blocks_guide.html) — Working with simulation outputs
- [API Reference](https://nsquaredlab.github.io/MyoGen/api/) — Complete class documentation
- [Examples](examples/) — Step-by-step tutorials from recruitment to EMG

# How to Cite

If you use MyoGen in your research, please cite:

TBD

# Contributing

Contributions welcome! See [issues](https://github.com/NsquaredLab/MyoGen/issues) if you want to add a feature or fix a bug.

# License

MyoGen is AGPL licensed. See [LICENSE](https://github.com/NsquaredLab/MyoGen/LICENSE.md) for details.

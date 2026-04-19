<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span>Welcome to</span>
    <img src="https://raw.githubusercontent.com/NsquaredLab/MyoGen/main/docs/source/_static/myogen_logo.png" height="100" alt="MyoGen Logo">
  </h1>

  <h2>The modular and extensible simulation toolkit for neurophysiology</h2>

  [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://nsquaredlab.github.io/MyoGen/)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Version](https://img.shields.io/badge/version-0.9.0-orange.svg)](https://github.com/NsquaredLab/MyoGen)

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

📦 **NWB export** — Optional export to [Neurodata Without Borders](https://www.nwb.org/) format for data sharing via DANDI

🧰 **Comprehensive toolkit** — Surface EMG, intramuscular EMG, force generation, and spinal network modeling

# Installation

> **Requires Python 3.12+** — Check your version with `python --version`

## System Requirements

| Platform | Before Installing MyoGen |
|----------|--------------------------|
| **Windows** | [NEURON 8.2.7](https://github.com/neuronsimulator/nrn/releases/download/8.2.7/nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe) - Download, run installer, select "Add to PATH" |
| **Linux** | `sudo apt install libopenmpi-dev` (Ubuntu/Debian) or `sudo dnf install openmpi-devel` (Fedora) |
| **macOS** | `brew install open-mpi` |

> [!CAUTION]
>
> ## Windows Users: Prerequisites
>
> **You MUST install the following before installing MyoGen on Windows:**
>
> ### 1. Visual C++ Build Tools
>
> Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
>
>During installation, select these components:
>
> - MSVC Build Tools for x64/x86 (Latest)
> - MSVC v143 – VS 2022 C++ x64/x86 build tools
> - Windows 11 SDK (latest)
> - C++ core desktop features
>
> ### 2. NEURON Simulator
>
> 1. **Download**: [NEURON 8.2.7 Installer](https://github.com/neuronsimulator/nrn/releases/download/8.2.7/nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe)
> 2. **Run the installer** and select **"Add to PATH"** when prompted
> 3. **Restart your terminal** (close and reopen)
> 4. Then continue with the installation below

---

## Step 1: Install uv (Package Manager)

We use [uv](https://docs.astral.sh/uv/) - a fast Python package manager. Install it first:

**Windows** (open PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/macOS**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, **restart your terminal** (close and reopen it).

---

## Step 2: Create a New Project

Open a terminal and navigate to where you want your project:

```bash
# Create a new folder for your project
mkdir my_emg_project
cd my_emg_project

# Initialize a Python project
uv init

# Add MyoGen to your project
uv add myogen
```

That's it! MyoGen is now installed and ready to use.

---

## Step 3: Verify Installation

Create a test file to make sure everything works:

```bash
# Create a test script
uv run python -c "from myogen import simulator; print('MyoGen installed successfully!')"
```

If you see `MyoGen installed successfully!` - you're all set!

---

## Alternative: pip install

If you prefer pip over uv:

```bash
pip install myogen
```

---

## For Developers (From Source)

```bash
git clone https://github.com/NsquaredLab/MyoGen.git
cd MyoGen
uv sync
uv run poe setup_myogen
```

## Optional: GPU Acceleration

For 5-10× faster convolutions (requires NVIDIA GPU):

```bash
uv add cupy-cuda12x
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


```{eval-rst}
----

Package Structure
-----------------

.. code-block:: text

   MyoGen/
   ├── myogen/              # Main package source code
   │   ├── simulator/       # Core simulation functionality
   │   │   ├── core/        # Core simulation components
   │   │   │   ├── emg/     # EMG signal generation
   │   │   │   ├── muscle/  # Muscle modeling
   │   │   │   └── spike_train/ # Motor neuron simulation
   │   │   └── ...
   │   ├── utils/           # Utility functions and tools
   │   │   ├── plotting/    # Visualization utilities
   │   │   ├── currents.py  # Current generation
   │   │   └── nmodl.py     # NMODL file handling
   │   └── ...
   ├── examples/            # Example scripts and tutorials
   ├── docs/                # Documentation source
   ├── pyproject.toml       # Project metadata and dependencies
   └── uv.lock              # Pinned versions of dependencies



.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Documentation

   api/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   neo_blocks_guide

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials
   :hidden:

   examples
```

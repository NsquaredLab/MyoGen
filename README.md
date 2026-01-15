<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span>Welcome to</span>
    <img src="https://raw.githubusercontent.com/NsquaredLab/MyoGen/main/docs/source/_static/myogen_logo.png" height="100" alt="MyoGen Logo">
  </h1>

  <h2>The modular and extandable simulation toolkit for neurophysiology</h2>

  [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://nsquaredlab.github.io/MyoGen/)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Version](https://img.shields.io/badge/version-0.8.5-orange.svg)](https://github.com/NsquaredLab/MyoGen)

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
| **Windows** | [NEURON 8.2.6](https://github.com/neuronsimulator/nrn/releases/download/8.2.6/nrn-8.2.6.w64-mingw-py-38-39-310-311-312-setup.exe) - Download, run installer, select "Add to PATH" |
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
> 1. **Download**: [NEURON 8.2.6 Installer](https://github.com/neuronsimulator/nrn/releases/download/8.2.6/nrn-8.2.6.w64-mingw-py-38-39-310-311-312-setup.exe)
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

# Documentation

📖 **[Read the full documentation](https://nsquaredlab.github.io/MyoGen/)**

- [User Guide](https://nsquaredlab.github.io/MyoGen/neo_blocks_guide.html) — Working with simulation outputs
- [API Reference](https://nsquaredlab.github.io/MyoGen/api/) — Complete class documentation
- [Examples](examples/) — Step-by-step tutorials from recruitment to EMG

# How to Cite

If you use MyoGen in your research, please cite:

```bibtex
@article{simpetru_molinari_2026_myogen,
  title   = {MyoGen: Unified Biophysical Modeling of Human Neuromotor Activity and Resulting Signals},
  author  = {S{\^i}mpetru, Raul C. and Molinari, Ricardo G. and Rohlf, Devon R. and
             Batichotti, Rebeka L. and Watanabe, Renato N. and
             Elias, Leonardo A. and Del Vecchio, Alessandro},
  journal = {bioRxiv},
  note    = {preprint},
  year    = {2026},
  doi     = {10.64898/2026.01.01.697284},
  url     = {https://www.biorxiv.org/content/10.64898/2026.01.01.697284}
}
```

# Contributing

Contributions welcome! See [issues](https://github.com/NsquaredLab/MyoGen/issues) if you want to add a feature or fix a bug.

# License

MyoGen is AGPL licensed. See [LICENSE](https://github.com/NsquaredLab/MyoGen/blob/main/LICENSE.md) for details.

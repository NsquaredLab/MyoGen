# Getting Started

!!! info "Requires Python 3.12+"
    Check your version with `python --version`.

## System requirements

NEURON and an MPI library must be available before installing MyoGen.

| Platform | Before installing MyoGen |
|----------|--------------------------|
| **Windows** | [NEURON 8.2.7](https://github.com/neuronsimulator/nrn/releases/download/8.2.7/nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe) — download, run the installer, select "Add to PATH". |
| **Linux** | `sudo apt install libopenmpi-dev` (Ubuntu/Debian) or `sudo dnf install openmpi-devel` (Fedora). |
| **macOS** | `brew install open-mpi`. |

!!! warning "Windows prerequisites"
    You **must** install the following before installing MyoGen on Windows:

    1. **Visual C++ Build Tools** — install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with: MSVC Build Tools for x64/x86 (latest), MSVC v143 (VS 2022 C++ x64/x86), Windows 11 SDK (latest), and C++ core desktop features.
    2. **NEURON 8.2.7** — run the [installer](https://github.com/neuronsimulator/nrn/releases/download/8.2.7/nrn-8.2.7.w64-mingw-py-39-310-311-312-313-setup.exe), select **"Add to PATH"**, then restart your terminal.

## Install

We recommend [uv](https://docs.astral.sh/uv/), a fast Python package manager.

=== "uv (recommended)"

    ```bash
    # install uv (Linux/macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # ...or Windows PowerShell:
    # powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # in a fresh project
    uv init my_emg_project && cd my_emg_project
    uv add myogen
    ```

=== "pip"

    ```bash
    pip install myogen
    ```

=== "from source (developers)"

    ```bash
    git clone https://github.com/NsquaredLab/MyoGen.git
    cd MyoGen
    uv sync
    uv run poe setup_myogen
    ```

Verify the install:

```bash
uv run python -c "from myogen import simulator; print('MyoGen installed successfully!')"
```

!!! tip "Optional: GPU acceleration"
    For 5–10× faster convolutions on an NVIDIA GPU: `uv add cupy-cuda12x`.

## Quick start

Generate motor-unit action potentials and surface EMG:

```python
from myogen import simulator
import quantities as pq

# 1. Recruitment thresholds for 100 motor units
thresholds, _ = simulator.RecruitmentThresholds(
    N=100, recruitment_range__ratio=50, mode="fuglevand"
)

# 2. Muscle model with fiber distribution
muscle = simulator.Muscle(
    recruitment_thresholds=thresholds,
    radius_bone__mm=1.0 * pq.mm,
    fiber_density__fibers_per_mm2=400 * pq.mm**-2,
    fat_thickness__mm=10 * pq.mm,
    autorun=True,
)

# 3. Surface electrode array
electrode_array = simulator.SurfaceElectrodeArray(
    num_rows=5,
    num_cols=5,
    inter_electrode_distance__mm=5 * pq.mm,
    electrode_radius__mm=5 * pq.mm,
    bending_radius__mm=muscle.radius__mm + muscle.skin_thickness__mm + muscle.fat_thickness__mm,
)

# 4. Surface EMG simulator
surface_emg = simulator.SurfaceEMG(
    muscle_model=muscle,
    electrode_arrays=[electrode_array],
    sampling_frequency__Hz=2048.0,
    MUs_to_simulate=[0, 1, 2, 3, 4],
)
```

For the full pipeline — spike trains, force, intramuscular EMG, spinal networks
— browse the [Examples](auto_examples/01_basic/index.md) gallery.

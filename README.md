<div align="center">
  <h1 style="display: flex; align-items: center; justify-content: center; gap: 10px;">
    <span>Welcome to</span>
    <img src="https://raw.githubusercontent.com/NsquaredLab/MyoGen/main/docs/source/_static/myogen_logo.png" height="100" alt="MyoGen Logo">
  </h1>

  <h2>The modular and extensible simulation toolkit for neurophysiology</h2>

  [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://nsquaredlab.github.io/MyoGen/)
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![Version](https://img.shields.io/badge/version-0.10.1-orange.svg)](https://github.com/NsquaredLab/MyoGen)

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

# Contributors

MyoGen is authored by **Raul C. Sîmpetru** and **Ricardo G. Molinari**. It is a joint project of the [Nsquared Lab](https://nsquared.tf.fau.de/) (Neuromuscular Physiology and Neural Interfacing Laboratory) at FAU Erlangen-Nürnberg, the [NER Lab](https://www.ceb.unicamp.br/pesquisa/laboratorios/laboratorio-de-pesquisa-em-neuroengenharia/) (Neural Engineering Research Laboratory) at the University of Campinas, and the [BMClab](https://bmclab.pesquisa.ufabc.edu.br/) (Biomechanics and Motor Control Laboratory) at the Federal University of ABC.

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/en/reference/emoji-key/)):

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-8-orange.svg?style=flat-square)](#contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RaulSimpetru"><img src="https://avatars.githubusercontent.com/u/26602941?v=4?s=100" width="100px;" alt="Raul C. Sîmpetru"/><br /><sub><b>Raul C. Sîmpetru</b></sub></a><br /><a href="https://github.com/NsquaredLab/MyoGen/commits?author=RaulSimpetru" title="Code">💻</a> <a href="#ideas-RaulSimpetru" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-RaulSimpetru" title="Maintenance">🚧</a> <a href="#projectManagement-RaulSimpetru" title="Project Management">📆</a> <a href="#research-RaulSimpetru" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/molinaris"><img src="https://avatars.githubusercontent.com/u/18554447?v=4?s=100" width="100px;" alt="Ricardo G. Molinari"/><br /><sub><b>Ricardo G. Molinari</b></sub></a><br /><a href="#ideas-molinaris" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-molinaris" title="Maintenance">🚧</a> <a href="#research-molinaris" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DRohlf"><img src="https://avatars.githubusercontent.com/u/199583526?v=4?s=100" width="100px;" alt="Devon R. Rohlf"/><br /><sub><b>Devon R. Rohlf</b></sub></a><br /><a href="https://github.com/NsquaredLab/MyoGen/commits?author=DRohlf" title="Code">💻</a> <a href="#maintenance-DRohlf" title="Maintenance">🚧</a> <a href="#research-DRohlf" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/rnwatanabe"><img src="https://avatars.githubusercontent.com/u/12960874?v=4?s=100" width="100px;" alt="Renato N. Watanabe"/><br /><sub><b>Renato N. Watanabe</b></sub></a><br /><a href="https://github.com/NsquaredLab/MyoGen/commits?author=rnwatanabe" title="Code">💻</a> <a href="#ideas-rnwatanabe" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-rnwatanabe" title="Mentoring">🧑‍🏫</a> <a href="#research-rnwatanabe" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alecsdelvecchio"><img src="https://www.nsquared.tf.fau.de/files/2021/05/0811_FAU_TECHFAK_FATHER-AND-SUN_20231018_1420-scaled-e1709914326501.jpg?s=100" width="100px;" alt="Alessandro Del Vecchio"/><br /><sub><b>Alessandro Del Vecchio</b></sub></a><br /><a href="#ideas-alecsdelvecchio" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-alecsdelvecchio" title="Mentoring">🧑‍🏫</a> <a href="#research-alecsdelvecchio" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leoelias-unicamp"><img src="https://avatars.githubusercontent.com/u/16341631?v=4?s=100" width="100px;" alt="Leonardo A. Elias"/><br /><sub><b>Leonardo A. Elias</b></sub></a><br /><a href="#ideas-leoelias-unicamp" title="Ideas, Planning, & Feedback">🤔</a> <a href="#mentoring-leoelias-unicamp" title="Mentoring">🧑‍🏫</a> <a href="#research-leoelias-unicamp" title="Research">🔬</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joaohbbittar"><img src="https://avatars.githubusercontent.com/u/140083565?v=4?s=100" width="100px;" alt="João Bittar"/><br /><sub><b>João Bittar</b></sub></a><br /><a href="https://github.com/NsquaredLab/MyoGen/issues?q=author%3Ajoaohbbittar" title="Bug reports">🐛</a> <a href="#ideas-joaohbbittar" title="Ideas, Planning, & Feedback">🤔</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/veylantis"><img src="https://avatars.githubusercontent.com/u/93032025?v=4?s=100" width="100px;" alt="BraveUnicorn"/><br /><sub><b>BraveUnicorn</b></sub></a><br /><a href="https://github.com/NsquaredLab/MyoGen/commits?author=veylantis" title="Code">💻</a></td>
    </tr>
  </tbody>
  <tfoot>
    <tr>
      <td align="center" size="13px" colspan="7">
        <img src="https://raw.githubusercontent.com/all-contributors/all-contributors-cli/1b8533af435da9854653492b1327a23a4dbd0a10/assets/logo-small.svg">
          <a href="https://all-contributors.js.org/docs/en/bot/usage">Add your contributions</a>
        </img>
      </td>
    </tr>
  </tfoot>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

# License

MyoGen is AGPL licensed. See [LICENSE](https://github.com/NsquaredLab/MyoGen/blob/main/LICENSE.md) for details.

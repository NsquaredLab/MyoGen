<div align="center" markdown>
<img src="images/myogen_logo.png" height="100" alt="MyoGen logo" markdown>

# MyoGen

**The modular and extensible simulation toolkit for neurophysiology**

[Getting Started](getting-started.md){ .md-button .md-button--primary } ·
[Examples](auto_examples/01_basic/index.md){ .md-button } ·
[API Reference](api/index.md){ .md-button }
</div>

## Overview

MyoGen is a **modular and extensible neuromuscular simulation framework** for
generating physiologically grounded motor-unit activity, muscle force, and
surface EMG signals.

It supports end-to-end modeling of the neuromuscular pathway — from descending
neural drive and spinal motor-neuron dynamics to muscle activation and
bioelectric signal formation at the electrode level. MyoGen is designed for
algorithm validation, hypothesis-driven research, and education, providing
configurable building blocks that can be combined and extended independently.

## Highlights

- 🧬 **Biophysically inspired neuron models** — NEURON-based motor neurons with validated calcium dynamics and membrane properties.
- 🎯 **Everything is inspectable** — complete access to every motor unit, spike time, and fiber location for rigorous algorithm testing.
- ⚡️ **Vectorized & parallel** — multi-core CPU processing with NumPy/Numba vectorization for fast computation.
- 🔬 **End-to-end simulation** — from motor-unit recruitment to high-density surface EMG in a single framework.
- 📊 **Reproducible science** — deterministic random seeds and standardized Neo Block outputs for exact replication.
- 📦 **NWB export** — optional export to [Neurodata Without Borders](https://www.nwb.org/) for data sharing via DANDI.
- 🧰 **Comprehensive toolkit** — surface EMG, intramuscular EMG, force generation, and spinal-network modeling.

Head to [Getting Started](getting-started.md) to install MyoGen and run your
first simulation, browse the [Examples](auto_examples/01_basic/index.md)
gallery, or dive into the [API Reference](api/index.md).

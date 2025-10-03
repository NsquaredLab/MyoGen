.. _watanabe-reproduction:

========================================
Watanabe et al. - Spinal Network Model
========================================

Reproduction of spinal motor neuron network modeling and corticomuscular coherence analysis from Watanabe et al.

Reference
---------

**Paper**: Watanabe, R. N., & Kohn, A. F. (Year). TBD - Full citation

**Key Findings**: This paper demonstrated how spinal motor neuron networks transform descending cortical drive into muscle force output, with specific frequency-dependent coherence patterns between cortical input and motor unit activity.

Overview
--------

This reproduction demonstrates a complete neuromuscular modeling pipeline:

1. **Spinal Network Simulation** - Alpha motor neurons with physiological connectivity
2. **Descending Drive** - Time-varying sinusoidal modulation patterns
3. **Force Generation** - Fuglevand force model with gain modulation
4. **Coherence Analysis** - Corticomuscular coherence across different frequency bands

The simulation reproduces key figures from the paper showing:

- Descending drive power spectra (Panels A-C)
- Corticomuscular coherence (Panels D-F)
- Force timeseries (Panel G)
- Motor neuron raster plots (Panel H)

Workflow
--------

The reproduction is split into three modular scripts:

**1. Spike Train Generation** (``10a_watanabe_spike_trains.py``)

   - Generate 180 seconds of motor neuron activity
   - Three phases with different descending drive patterns
   - Saves: ``spinal_network_results.pkl`` (Neo Block)

**2. Force Computation** (``10b_watanabe_compute_force.py``)

   - Load spike trains
   - Compute muscle force using Fuglevand model
   - Saves: ``watanabe_force_results.pkl`` (Neo Block)
   - Runtime: ~5-10 minutes

**3. Visualization** (``10c_watanabe_visualize.py``)

   - Load spike trains and force
   - Generate all paper figures
   - Compute power spectra and coherence
   - Saves: Multiple PNG figures
   - Runtime: ~2-3 minutes

Execution
---------

Run scripts in order:

.. code-block:: bash

    # 1. Generate spike trains (run once)
    python 10a_watanabe_spike_trains.py

    # 2. Compute force (run once, or when changing force parameters)
    python 10b_watanabe_compute_force.py

    # 3. Generate visualizations (run multiple times for figure tweaking)
    python 10c_watanabe_visualize.py

Outputs
-------

**Data files** (saved to ``results/``):

- ``spinal_network_results.pkl`` - Complete spike train data
- ``watanabe_force_results.pkl`` - Force output with metadata

**Figures** (saved to ``results/``):

- DDdrive power spectra (3 panels)
- Corticomuscular coherence plots (3 panels)
- Force timeseries
- Full duration raster plot
- Per-window analysis (9 supplementary figures)

Computational Requirements
--------------------------

- **Memory**: ~2-4 GB RAM
- **Time**: Total ~10-15 minutes for full pipeline
- **Storage**: ~50-100 MB for results

The modular structure allows skipping expensive re-computation when experimenting with visualization or force parameters.

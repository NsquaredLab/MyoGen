.. _paper-reproductions:

====================
Paper Reproductions
====================

This gallery contains complete reproductions of published neuromuscular modeling studies using MyoGen.

Each reproduction demonstrates MyoGen's ability to replicate established research findings and provides validated reference implementations for advanced modeling techniques.

Purpose
-------

These examples serve multiple purposes:

- **Validation**: Demonstrate that MyoGen can reproduce published results
- **Reference**: Provide working implementations of established methods
- **Education**: Show complete workflows from simulation to publication-quality figures
- **Benchmarking**: Establish performance baselines for comparison

Structure
---------

Each paper reproduction is organized into separate scripts:

1. **Simulation** - Generate spike trains and neural activity
2. **Analysis** - Compute derived signals (force, coherence, etc.)
3. **Visualization** - Reproduce paper figures

This modular structure allows you to:

- Re-run expensive simulations independently
- Quickly iterate on visualizations
- Modify analysis parameters without full re-simulation

Available Reproductions
-----------------------

Currently available paper reproductions:

.. card:: Watanabe & Kohn (2015)
   :link: watanabe-reproduction
   :link-type: ref

   **Fast Oscillatory Commands from the Motor Cortex Can Be Decoded by the Spinal Cord for Force Control**

   J. Neurosci. 35(40):13687-13697. `DOI: 10.1523/JNEUROSCI.1950-15.2015 <https://doi.org/10.1523/JNEUROSCI.1950-15.2015>`_

   Demonstrates how spinal motor neuron pools decode frequency oscillations from descending cortical drive for force modulation.

   **MyoGen Components**: ``AlphaMN__Pool``, ``DescendingDrive__Pool``, ``Network``, ``ForceModel``, ``SimulationRunner``, ``ContinuousSaver``

   **Workflow**: 6-step pipeline from baseline force → optimization → simulation → analysis → force computation → visualization

   See :ref:`watanabe-reproduction` for complete documentation.

More reproductions will be added over time as MyoGen capabilities expand.

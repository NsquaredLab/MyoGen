.. _finetune-examples:

==============================
Parameter Optimization & Tuning
==============================

This gallery contains examples for fine-tuning and optimizing neuromuscular simulation parameters to match specific physiological targets or experimental data.

These examples demonstrate advanced techniques for parameter optimization, model calibration, and validation against target firing rates, force profiles, and other physiological metrics.

Overview
--------

The parameter tuning examples cover:

1. **Descending Drive Optimization** - Calibrating input patterns for target firing rates
2. **Force Profile Matching** - Optimizing parameters to achieve desired force trajectories
3. **Firing Rate Statistics** - Extracting and analyzing inter-spike intervals (ISI) and coefficient of variation (CV)
4. **Multi-Muscle Comparison** - Comparing optimized parameters across different muscle models

Workflow & Dependencies
-----------------------

The examples have specific prerequisites and should be run in sequence:

**Option A: Firing Rate Optimization Workflow**

1. ``00_optimize_dd_for_target_firing_rate.py`` - Generate baseline optimized DD parameters
2. ``01_compute_force_from_optimized_dd.py`` - Calculate force from optimized parameters (optional)
3. ``03_extract_isi_and_cv_per_ramps.py --use-baseline`` - Extract ISI/CV statistics using baseline parameters
4. ``04_plot_isi_cv_multi_muscle_comparison.py`` - Generate comparison plots

**Option B: Force Optimization Workflow**

1. ``00_optimize_dd_for_target_firing_rate.py`` - Generate baseline optimized DD parameters (if not already done)
2. ``02_optimize_dd_for_target_force.py --target-force-pct 30`` - Optimize for specific force level (e.g., 30% MVC)
3. ``03_extract_isi_and_cv_per_ramps.py --mvc-level 30`` - Extract ISI/CV statistics using force-optimized parameters
4. ``04_plot_isi_cv_multi_muscle_comparison.py`` - Generate comparison plots

.. note::
   Example 03 requires optimization results from EITHER example 00 (with ``--use-baseline``) OR example 02 (without the flag).
   If you see a ``FileNotFoundError``, ensure you've run the appropriate prerequisite example first.

Use Cases
---------

These examples are particularly useful for:

- Matching simulation outputs to experimental recordings
- Calibrating models for specific muscle types or pathological conditions
- Validating model behavior against physiological constraints
- Developing standardized parameter sets for reproducible research


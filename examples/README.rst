.. _examples-index:

==================
Examples
==================

Welcome to the MyoGen examples gallery! These examples demonstrate the complete neuromuscular simulation pipeline, from basic tutorials to full paper reproductions.

Example Galleries
-----------------

Examples are organized into two main galleries:

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Basic Tutorials
        :link: auto_examples/basic/index
        :link-type: doc
        :class-card: sd-text-black sd-bg-light

        **Getting Started with MyoGen**

        Fundamental tutorials covering core functionality and workflow patterns.

        Perfect for learning the basics of neuromuscular simulation.

        +++
        :bdg-primary:`Beginner Friendly` :bdg-info:`10 Examples`

    .. grid-item-card:: Paper Reproductions
        :link: auto_examples/papers/watanabe/index
        :link-type: doc
        :class-card: sd-text-black sd-bg-light

        **Scientific Validation**

        Complete reproductions of published neuromuscular modeling studies.

        Demonstrates MyoGen's ability to replicate research findings.

        +++
        :bdg-warning:`Advanced` :bdg-success:`Validated`

Simulation Pipeline Overview
-----------------------------

.. mermaid::
   :caption: MyoGen Simulation Workflow

    %%{init: {
      "theme": "base",
      "themeVariables": {
        "fontSize": "1.1em",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "primaryColor": "#ffffff",
        "primaryTextColor": "#1f2937",
        "primaryBorderColor": "#e5e7eb",
        "lineColor": "#6b7280",
        "background": "#ffffff",
        "mainBkg": "#ffffff",
        "secondaryColor": "#f9fafb",
        "tertiaryColor": "#f3f4f6",
        "clusterBkg": "#f8fafc",
        "clusterBorder": "#cbd5e1",
        "edgeLabelBackground": "#ffffff",
        "cScale0": "#1f2937",
        "cScale1": "#1f2937",
        "cScale2": "#1f2937",
        "clusterTextSize": "1.5em"
      },
      "flowchart": {
        "curve": "linear",
        "nodeSpacing": 50,
        "rankSpacing": 80
      }
    }}%%

   flowchart TD
        S((Start))
        A["Recruitment<br/>Thresholds"]

        B["Injected Current"]
        C["Cortical Input"]

        D["Muscle Model"]

        E["Surface MUAPs"]
        F["Surface EMG"]
        G["Intramuscular EMG"]
        J["Intramuscular MUAPs"]

        H["Current Generation"]
        I["Force Model"]

        subgraph neural_drive ["<b>⚡ Neural Drive</b>"]
            B
            C
        end

        subgraph utilities ["<b>🔧 Utilities</b>"]
            H
        end

        subgraph physiology ["<b>💪 Anatomy Model</b>"]
            S --> A --> D & I
        end

        subgraph emg ["<b>📊 EMG Generation</b>"]
            direction TB
            E --> F
            J --> G
        end


        H --> neural_drive
        D --> emg

        neural_drive --> emg

        classDef start fill:#f0f9ff,stroke:#0369a1,stroke-width:3px,color:#0c4a6e
        classDef foundation fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#1e3a8a
        classDef neural fill:#ede9fe,stroke:#7c3aed,stroke-width:2px,color:#5b21b6
        classDef physiology fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e
        classDef emg fill:#d1fae5,stroke:#059669,stroke-width:2px,color:#064e3b
        classDef utility fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#991b1b
        classDef default fill:#f3f4f6,stroke:#6b7280,stroke-width:2px,color:#374151

        class S start
        class A foundation
        class B,C neural
        class D,I physiology
        class E,F,G,J emg
        class H utility

        style neural_drive fill:#f3f0ff,stroke:#8b5cf6,stroke-width:2px,stroke-dasharray: 5 5
        style utilities fill:#fef2f2,stroke:#ef4444,stroke-width:2px,stroke-dasharray: 5 5
        style physiology fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,stroke-dasharray: 5 5
        style emg fill:#ecfdf5,stroke:#10b981,stroke-width:2px,stroke-dasharray: 5 5

Quick Start
-----------

**New to MyoGen?** Start with the basic tutorials:

1. :ref:`sphx_glr_auto_examples_basic_00_simulate_recruitment_thresholds.py` - Create recruitment thresholds
2. :ref:`sphx_glr_auto_examples_basic_01_simulate_spike_trains_current_injection.py` - Generate spike trains
3. :ref:`sphx_glr_auto_examples_basic_02_simulate_muscle.py` - Build muscle model
4. :ref:`sphx_glr_auto_examples_basic_04_simulate_surface_emg.py` - Simulate EMG signals

**Want to see advanced features?** Explore paper reproductions:

- :ref:`sphx_glr_auto_examples_papers_watanabe` - Spinal network modeling with corticomuscular coherence

Gallery Organization
--------------------

**Basic Tutorials** (`basic/`)
    Self-contained examples covering individual features. Examples are numbered to suggest a learning path but can be run independently.

**Paper Reproductions** (`papers/`)
    Complete workflows reproducing published studies. Each reproduction is split into modular scripts for simulation, analysis, and visualization.

Running Examples
----------------

All examples can be run directly:

.. code-block:: bash

    python examples/basic/00_simulate_recruitment_thresholds.py

Or downloaded as Jupyter notebooks from the example pages in the documentation.

Getting Help
------------

- 📖 **Documentation**: Full API reference and guides
- 💬 **Issues**: Report bugs or request features on GitHub
- 🎓 **Tutorials**: Step-by-step learning materials in basic gallery

.. _examples-index:

==================
Examples
==================

This collection of examples demonstrates the complete MyoGen simulation pipeline, from basic motor unit recruitment to advanced EMG signal generation.

Workflow
-----------------------------

.. mermaid::
   :caption: Click any box to navigate to that example
   
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

        %% Column 4: Independent Utilities
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

        %% Modern Light Mode Styling
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

        %% Subgraph styling
        style neural_drive fill:#f3f0ff,stroke:#8b5cf6,stroke-width:2px,stroke-dasharray: 5 5
        style utilities fill:#fef2f2,stroke:#ef4444,stroke-width:2px,stroke-dasharray: 5 5
        style physiology fill:#fffbeb,stroke:#f59e0b,stroke-width:2px,stroke-dasharray: 5 5
        style emg fill:#ecfdf5,stroke:#10b981,stroke-width:2px,stroke-dasharray: 5 5

        %% Clickable links
        click A "00_simulate_recruitment_thresholds.html"
        click B "01_simulate_spike_trains.html"
        click C "07_simulate_cortical_input.html"
        click D "02_simulate_muscle.html"
        click E "03_simulate_surface_muaps.html"
        click F "04_simulate_surface_emg.html"
        click J "08_simulate_intramuscular_emg.html"
        click G "08_simulate_intramuscular_emg.html"
        click H "05_simulate_currents.html"
        click I "06_simulate_force.html"


Recommended Entry Path
---------------------------

1. :ref:`sphx_glr_auto_examples_00_simulate_recruitment_thresholds.py` - Create recruitment thresholds
2. :ref:`sphx_glr_auto_examples_01_simulate_spike_trains.py` - Neural drive 
3. :ref:`sphx_glr_auto_examples_02_simulate_muscle.py` - Muscle structure
4. :ref:`sphx_glr_auto_examples_04_simulate_surface_emg.py` - Surface EMG

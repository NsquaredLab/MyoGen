# MyoGen Examples — Backends

MyoGen's default and recommended simulation backend is **Jaxley** (JAX-based,
GPU-capable, differentiable). NEURON is an **optional** backend kept as a validation
reference.

- The **canonical** example for each topic (`XX_<name>.py`) is the **Jaxley** version and
  runs on a plain `pip install myogen` — no NEURON required.
- The NEURON version of a topic, where one exists, is the `XX_<name>_neuron.py` sibling
  and needs the optional extra: `pip install "myogen[neuron]"`.

## Default examples (Jaxley / backend-agnostic) — no NEURON needed

These run on a plain `pip install myogen`.

| Example | What it does |
|---|---|
| `01_basic/01_simulate_recruitment_thresholds.py` | Recruitment threshold distribution |
| `01_basic/02_simulate_spike_trains_current_injection.py` | Spike trains from current injection (Jaxley) |
| `01_basic/03_simulate_spike_trains_descending_drive.py` | Spike trains from descending drive (Jaxley) |
| `01_basic/04_simulate_muscle.py` | Muscle geometry / fiber distribution (`core/`) |
| `01_basic/05_simulate_surface_muaps.py` | Surface MUAP templates (`core/`) |
| `01_basic/06_simulate_surface_emg.py` | Surface EMG (`core/`) |
| `01_basic/07_simulate_currents.py` | Fiber currents (`core/`) |
| `01_basic/08_simulate_force.py` | Force from spikes (Jaxley) |
| `01_basic/09_simulate_intramuscular_emg.py` | Intramuscular EMG (`core/`) |
| `01_basic/10_extract_neuron_parameters.py` | Per-cell electrophysiology (Jaxley) |
| `01_basic/11_simulate_spinal_network.py` | Full closed-loop spinal network (Jaxley) |
| `01_basic/12_extract_data_from_neo_blocks.py` | Neo block post-processing |
| `01_basic/13_load_and_inspect_nwb_data.py` | NWB I/O |
| `02_finetune/05_plot_isi_cv_multi_muscle_comparison.py` | Plot saved ISI/CV results |
| `03_papers/watanabe/04_load_and_analyze_results.py` | Analyse saved Watanabe results |
| `03_papers/watanabe/05_compute_force_from_spinal_network.py` | Force from saved network |
| `03_papers/watanabe/06_visualize.py` | Visualise saved results |

New differentiable pipeline (used by the Jaxley examples and available directly):
`from myogen.simulator.jaxley import run_jax, value_and_grad_run, compile_run,
surface_emg_jax`. See `docs/DIFFERENTIABILITY_STATUS.md`.

## NEURON examples — require `pip install "myogen[neuron]"`

These import the NEURON runtime directly. The `_neuron.py` files are the NEURON
counterparts of the canonical Jaxley examples above.

| NEURON example | Canonical (Jaxley) counterpart |
|---|---|
| `01_basic/02_simulate_spike_trains_current_injection_neuron.py` | `02_simulate_spike_trains_current_injection.py` |
| `01_basic/03_simulate_spike_trains_descending_drive_neuron.py` | `03_simulate_spike_trains_descending_drive.py` |
| `01_basic/08_simulate_force_neuron.py` | `08_simulate_force.py` |
| `01_basic/10_extract_neuron_parameters_neuron.py` | `10_extract_neuron_parameters.py` |
| `01_basic/11_simulate_spinal_network_neuron.py` | `11_simulate_spinal_network.py` |
| `02_finetune/01_optimize_dd_for_target_firing_rate.py` | — (not yet ported) |
| `02_finetune/02_compute_force_from_optimized_dd.py` | — (not yet ported) |
| `02_finetune/03_optimize_dd_for_target_force.py` | — (not yet ported) |
| `02_finetune/04_extract_isi_and_cv_per_ramps.py` | — (not yet ported) |
| `03_papers/watanabe/01_compute_baseline_force.py` | — (not yet ported) |
| `03_papers/watanabe/02_optimize_oscillating_dc.py` | — (not yet ported) |
| `03_papers/watanabe/03_10pct_mvc_simulation.py` | — (not yet ported) |

Running a NEURON example without the extra raises `ModuleNotFoundError: No module
named 'neuron'` (or an equivalent import error) — install `myogen[neuron]` to run it.

## Migration status

The `02_finetune` and `03_papers` NEURON simulations are not yet ported to Jaxley.
Porting them (or building Jaxley equivalents) is tracked as follow-up work; until then
they remain runnable via the NEURON extra.

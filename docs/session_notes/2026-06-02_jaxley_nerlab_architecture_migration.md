# Jaxley Migration to the NERLab Motor-Neuron Architecture

**Date:** 2026-06-02

**One-line summary:** Switched the Jaxley motor-neuron pool from the Powers2017
cable cell to the NERLab single-isopotential-dendrite cell so it now matches
the NEURON production model. All five paired Jaxley examples (02, 03, 08, 10,
11) have been migrated and validated against NEURON.

---

## Why the change happened

Earlier project memory described the Jaxley motor neuron as "Powers2017 +
cable dendrite (ncomp=4)". The original MyoGen writer confirmed during this
session that **production NEURON does not use the 4-dendrite Powers2017
cell**. The production NEURON path is:

- `AlphaMN__Pool` defaults to `model="NERLab"` (set at
  [config/alpha_mn_default.yaml:5](../../myogen/config/alpha_mn_default.yaml#L5)).
- `_create_nerlab_cells` builds `cells.AlphaMN(... dendrites__count=1, model="NERLab")`
  ([simulator/neuron/populations/motor_neurons.py:517-524](../../myogen/simulator/neuron/populations/motor_neurons.py#L517-L524)).
- Soma uses the `napp` mechanism (Na fast + Na persistent + K fast + K slow +
  leak); dendrite uses `caL` (L-type Ca with no inactivation + leak).
- Voltage convention: **original 1952 HH frame**, not modern absolute
  voltages. V_rest ≈ 0 mV, ENa = +120 mV, EK = -10 mV, spike peaks ≈ +90 mV.

Every production NEURON example (`02_*`, `03_*`, `08_*`, `10_*`, `11_*`) uses
this default. The previous Jaxley "Powers2017 cable arch" was therefore
**not comparing the same cell** to NEURON, which explains a large fraction of
the historical mismatch the project tracked as "Jaxley needs further tuning".

The full re-verification trail is in this session's Bash output and in
`scripts/test_nerlab_*.py`.

---

## Architectural change in one diagram

```
BEFORE (Jaxley path, use_jaxley_mech=True):
            ┌──── soma ─────┐         Na3rp, Naps, KdrRL,
            │   ncomp=1     │         MAHP, Gh, Leak
            └───────┬───────┘
                    │ Ra (cable filtering)
            ┌───────┴───────┐
            │  dend comp 1  │
            │  dend comp 2  │         LCaInact, Gh, Leak
            │  dend comp 3  │         (same density on each)
            │  dend comp 4  │
            └───────────────┘
            ── Powers2017-style channels, 1952 HH absolute V frame? No, modern.
            ── Voltage frame: modern (V_rest ≈ -67 mV)


AFTER (Jaxley path, default model="NERLab"):
            ┌──── soma ─────┐         napp
            │   ncomp=1     │         (Na + Na_persistent + Kfast + Kslow + leak,
            │  L = diam     │          5-gate Hodgkin-Huxley)
            └───────┬───────┘
                    │
            ┌───────┴───────┐         caL
            │  isopotential │         (L-type Ca, no inactivation + leak)
            │  ncomp=1      │
            └───────────────┘
            ── Channel ports of myogen/simulator/nmodl_files/{napp,caL}.mod
            ── Voltage frame: original 1952 HH (V_rest ≈ 0 mV, peaks ≈ +90 mV)
```

This is now identical to the NEURON production cell's morphology and channel
set.

---

## What landed in the codebase

### New channel ports

[`myogen/simulator/jaxley/channels/nerlab_channels.py`](../../myogen/simulator/jaxley/channels/nerlab_channels.py)
— port of the `napp` and `caL` NMODL files. ~250 lines.

- `napp.Channel`: 5 gates (m, h, p, n, r). `vtrap`-style rate calc with a
  JAX-safe L'Hôpital expansion near the singularity. `init_state` evaluates
  steady-state gates at the holding V so the cell starts in equilibrium.
- `caL.Channel`: 1 gate (L), sigmoid Boltzmann activation with a fixed
  `Ltau`. Includes the bundled leak from `caL.mod`.

Exports added to
[`channels/__init__.py`](../../myogen/simulator/jaxley/channels/__init__.py).

### Pool and cell-builder dispatch

[`populations/motor_neurons.py`](../../myogen/simulator/jaxley/populations/motor_neurons.py):

- New `NERLAB_PARAMS` dict (lines 102-160) holding all per-cell parameter
  ranges as `(small_cell_value, large_cell_value)` tuples derived from
  [config/alpha_mn_default.yaml](../../myogen/config/alpha_mn_default.yaml).
- `_generate_parameter_arrays` dispatches on `self.model` to either
  `_generate_nerlab_params` or the legacy `_generate_powers2017_params`.
- `__init__` auto-sets `initial_voltage__mV = 0.0` and
  `spike_threshold__mV = 50.0` when `model="NERLab"` and the caller didn't
  explicitly override.

[`cells_api.py`](../../myogen/simulator/jaxley/cells_api.py):

- `_create_sections`, `_define_geometry`, and `_define_biophysics` dispatch
  on `self.model`. NERLab path builds `soma(ncomp=1) + dend(ncomp=1)` with
  parents `[-1, 0]`.
- New `_define_geometry_nerlab` and `_insert_nerlab_channels` methods.
- The Powers2017 path is preserved as a sibling for backward compatibility.
- `AlphaMN` constructor now accepts `gamma` and threads it through to
  `caL_gama` (was previously hard-coded to 1.0, 5× the NEURON value of 0.2).

### Pool V-init fix

[`populations/base.py:117-167`](../../myogen/simulator/jaxley/populations/base.py#L117-L167)
now actually applies `initial_voltage_values__mV` to each underlying `jx.Cell`
via `cell.set("v", ...)` + `init_states()`. Before this fix the field was
stored but never used, so direct `jx.integrate(cell_wrapper.cell, ...)` paths
silently ran at Jaxley's default -70 mV regardless of what the pool was told.

### Scan-step per-cell-type parameters (for mixed-frame networks)

[`jax_models.py`](../../myogen/simulator/jaxley/jax_models.py) now takes two
new kwargs on `make_scan_step`:

- `e_exc_mn: float | None` — separate excitatory AMPA reversal for the
  motor-neuron section of the combined network. Defaults to `e_exc` for
  backward compat. ex11 passes 70 mV (NERLab) while keeping `e_exc=0` for
  the modern-frame interneurons.
- `mn_spike_threshold_mV: float` — separate spike-detection threshold for
  motor neurons. Defaults to 0 mV (modern frame); ex11 passes 50 mV.

The internal scan body now uses `_e_exc_mn` to compute `df_mn` and `_mn_v_th`
for `mn_spikes_now`, so the existing modern-frame interneuron logic is
unaffected.

---

## Voltage convention — important detail

NERLab cells live in the **original 1952 Hodgkin-Huxley convention** with all
voltages shifted by ~+65 mV relative to modern absolute values:

| Quantity | NERLab (1952 HH) | Modern |
|---|---:|---:|
| V_rest | 0 mV | −65 to −70 mV |
| ENa | +120 mV | +50 to +55 mV |
| EK | −10 mV | −75 to −80 mV |
| AP peak | +80 to +110 mV | +20 to +40 mV |
| Spike-detect threshold | +50 mV | 0 mV |

When you reinitialise a NERLab cell's voltage manually (in the simulation
loop, not via the pool), use `cell.set("v", 0.0)` not `-65.0`. When you
detect spikes, look for `V > 50 mV` crossings.

Mixing NERLab cells with modern-frame interneurons in one `jx.Network` works
**only if** the synapse reversal and detection thresholds are set per
postsynaptic / presynaptic cell type — see the ex11 notes below.

---

## Per-example results

All five paired examples were migrated and tuned. Quantitative parity to
NEURON now:

| Example | What it tests | Jaxley NERLab vs NEURON |
|---|---|---|
| **02** — current injection | 15 nA step → spike trains | 94/100 active, 6078 vs 6061 spikes (**0.3 %**) |
| **03** — trapezoid DD | descending drive → spike trains | 85/100 plateau-active, 17.6 Hz mean vs ~18 Hz (**within 10 %**) |
| **08** — force from spikes | 15 nA trapezoid → Fuglevand force | 50/50 active, 14490 vs 14517 spikes (**0.19 %**) |
| **10** — parameter extraction | Vhold / Rin / τ / Ir / AP / AHP / F-I per cell | Rin and Ir within **3-5 %** per cell; AP, AHP, F-I within **20 %**. All five confirmable params inside the literature physiological range (Bui et al., Carp et al., etc.) |
| **11** — full spinal network | mixed populations, joint dynamics, force | 82/100 MN active, 0.56 F0 peak force, physiologically plausible recruitment |

`ex10` is the most precise — five-protocol per-cell extraction, apples-to-
apples against the NEURON CSV. This is the validation table that had been
missing.

---

## ex03 and ex11 — the IIR / Exp2Syn equivalence story

`ex03` and `ex11` both use an IIR + fixed-driving-force approximation in
place of NEURON's `Exp2Syn`. Naively matching NEURON's NetCon weights (e.g.
0.05 µS) produces 5-10× too much current because:

- NEURON's `Exp2Syn` has a shaped onset (`tau1 ≈ 0.5 ms` rise, `tau2 ≈
  2-5 ms` decay). Peak conductance per spike is only `weight × peak_factor`
  with `peak_factor ≈ 0.1-0.2`.
- The IIR adds the full weight per spike without shaping; effective per-
  spike charge is therefore ~5-10× higher.

The empirically-tuned equivalence factors:

- `ex03` DD → MN: NEURON `weight = 0.15 µS` → IIR `weight = 0.009 µS`
  (~0.06×).
- `ex11` DD → MN and Ia → MN: NEURON `weight = 0.05 µS` → IIR
  `weight = 0.01 µS` (~0.20×).

The two factors differ because the steady-state input rate and number of
connections per MN differ; the per-spike scaling math is the same. Both
weights are documented inline in their respective examples.

A proper fix would be to use a real Jaxley `IonotropicSynapse` (or build an
Exp2Syn-equivalent mechanism) for the excitatory paths too. ex11 already
uses `IonotropicSynapse` for inhibition.

---

## ex11 specifics — mixed-frame spinal network

ex11 wires NERLab MNs together with modern-frame `gII / gIb` interneurons
into one `jx.Network`. This required three additions:

1. `e_exc_mn = 70.0` and `e_exc = 0.0` set separately so DD/Ia synaptic
   drive computes its driving force from the appropriate reversal for each
   target.
2. `mn_spike_threshold_mV = 50.0` for MN spike detection; the interneuron
   sections still use the 0 mV crossing.
3. `IonotropicSynapse_e_syn = -15.0` for the gII / gIb → MN inhibitory
   reversal. NEURON ex11 uses `-75 mV` literally on NERLab targets, which
   gives a 75 mV hyperpolarising driving force and over-inhibits in our
   live-V Jaxley synapse model. The frame-shifted `-15 mV` matches the
   *physical* inhibitory driving force the original Powers2017 setup
   delivered.

The interneurons themselves still use Na3rp-style channels in the modern V
frame. Porting them to NERLab is left as future work (see "Remaining
caveats" below).

---

## Bugs caught and fixed along the way

Codex catches that materially changed the outcome — recording these because
each one would have produced silently-wrong recruitment:

1. **NERLab parameter interpolation was direction-flipped** on six entries.
   NEURON's [`special_interp`](../../myogen/simulator/neuron/populations/motor_neurons.py#L390)
   applies a `negative` flag from the YAML which swaps the (small, large)
   assignment. The first pass of `NERLAB_PARAMS` encoded the YAML literally
   — `gnapbar`, `gkfbar`, `gksbar`, `gcaLbar`, `vtraub_caL`, `ltau_caL` all
   had small and large cells swapped. Fix: re-encode in (small_cell,
   large_cell) semantics with explicit comments.
2. **`gamma` was ignored** in `_insert_nerlab_channels` — `caL_gama` was
   hard-coded to 1.0 instead of the NEURON YAML default of 0.2 (5× too
   strong dendritic Ca). Wired the pool's `gamma` argument through to the
   channel via the new `AlphaMN.gamma` field.
3. **`_apply_initial_voltages` was effectively a no-op.** `_Pool.__init__`
   stored `initial_voltage_values__mV` but never called `.set("v", ...)`
   on cells. So any code reaching in via `cell_wrapper.cell` ran at
   Jaxley's default -70 mV regardless of the pool's intended frame. Fix:
   actually apply the voltage in `_Pool.__init__`.
4. **IIR / Exp2Syn equivalence factor** — see above, separate section.
5. **ex11 used a single `e_exc = 0` for all populations** — collapsed the
   excitatory driving force to zero on NERLab MNs (`0 - 0 = 0`). Fix: per-
   cell-type reversals via the new `e_exc_mn` parameter in `jax_models.py`.

The original codex review reports from these catches are preserved in the
commit history if a future session wants to trace the reasoning.

---

## Remaining caveats

1. **F-I gain in ex10 is 10-20 % shallower** in Jaxley vs NEURON across
   cells. Could be protocol/extraction differences (settling window, step
   length) or genuine model-dynamics differences. Worth a follow-up.
2. **The IIR / Exp2Syn equivalence factors (0.06 and 0.20)** are empirical
   per example. A proper Jaxley Exp2Syn-equivalent synapse model would
   eliminate this knob.
3. **ex11 is mixed-frame** by design. Works only because all synapses in
   that example target NERLab MNs and the interneurons are sources. Any new
   synapse type targeting an interneuron would need its own per-cell-type
   `e_exc_X`. Cleaner long-term fix: port `gII / gIb` populations to NERLab.
4. **AHP duration in ex10 is broader** than the literature lower bound when
   estimated via half-decay. The script measures full duration (`V → vhold
   − 0.15 mV`); a proper half-decay measurement would be a useful addition.

---

## What "use this from now on" looks like

```python
# Build a NERLab motor-neuron pool — the default now matches NEURON production.
from myogen.simulator.jaxley.populations import AlphaMN__Pool
pool = AlphaMN__Pool(recruitment_thresholds__array=rt)   # model="NERLab" is default

# The pool sets V_rest = 0 mV and spike_threshold = +50 mV automatically.
# If you reach in via cell_wrapper.cell and manually re-init, use:
#     cell.set("v", 0.0)
#     cell.init_states()

# To opt into the legacy Powers2017 cable cell (e.g. for the
# pre-NERLab spinal-network setup):
pool = AlphaMN__Pool(
    recruitment_thresholds__array=rt,
    model="Powers2017",
    use_jaxley_mech=True,
)
```

---

## Pointers

- Updated `MEMORY.md` summary file in the user's auto-memory directory.
- Validation scripts: `scripts/test_nerlab_single_cell.py`,
  `scripts/test_nerlab_pool.py`.
- This document supersedes the cable-architecture sections of
  `2026-02-10_parameter_tuning_final.md` and the relevant parts of
  `2026-05-01_jaxley_neuron_replacement_status.md`. The earlier files are
  preserved unchanged for historical reference.

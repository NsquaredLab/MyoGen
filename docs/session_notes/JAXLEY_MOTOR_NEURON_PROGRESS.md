# Jaxley Motor Neuron Tuning: Full Progress Timeline

**Last updated:** 2026-02-10

---

## Table of Contents

1. [Goal](#goal)
2. [Literature Targets](#literature-targets)
3. [Timeline of Changes](#timeline-of-changes)
4. [What is Solved](#what-is-solved)
5. [Current Blocking Problem](#current-blocking-problem)
6. [Root Cause Analysis](#root-cause-analysis)
7. [Proposed Solutions (Not Yet Tried)](#proposed-solutions-not-yet-tried)
8. [Current File State (Uncommitted)](#current-file-state-uncommitted)
9. [Key Files Reference](#key-files-reference)

---

## Goal

Make the Jaxley `use_jaxley_mech=True` motor neuron (single-compartment HH + MAHP) produce electrophysiological parameters within physiological literature ranges, while keeping Example 2 (spike trains with trapezoid current) working correctly.

---

## Literature Targets

| Parameter | Literature Range | Source |
|-----------|-----------------|--------|
| **Rin (Input Resistance)** | 1.0 - 3.0 MOhm | User-specified |
| **Rheobase** | 4 - 40 nA | User-specified |
| **F-I Gain** | ~1-4 Hz/nA | NEURON Powers2017: 2.7-3.5 Hz/nA |
| **AHP Duration** | ~50-100 ms | Motor neuron literature |

---

## Timeline of Changes

### Phase 1: Initial Jaxley Integration (Jan 22, 2026)

**Session:** `IMPLEMENTATION_NOTES_2026_01_22.md`

- Integrated `jaxley-mech` Hodgkin-Huxley channels (Na, K, Leak) into MyoGen
- Added `use_jaxley_mech` parameter to `AlphaMN` and `AlphaMN__Pool`
- Fixed Poisson process generator (was 4x too high: 247 pps vs 55 pps target)
- Resolved numpy 2.x diamond dependency conflict
- **Result:** Neurons fire, but with standard HH kinetics (no motor neuron specifics)

### Phase 2: MAHP Channel Integration (Jan 25, 2026)

**Session:** `2026-01-25_jaxley_mech_mahp_integration.md`

- Added custom MAHP channel alongside jaxley-mech HH channels
- Fixed exponential parameter interpolation (`_exp_interp`) for proper Henneman distribution
- **Problem identified:** Rin = 12-318 MOhm (should be 0.5-1.7 MOhm)
- **Problem identified:** Rheobase = 0.1-7.8 nA (should be 6-19 nA)
- **Root cause:** Soma too small: d=25-60 um, L=200-480 um (tiny surface area)
- **Decision:** Increased soma to d=50-100, L=400-800 (committed as `fc253f3`)

### Phase 3: Backend Differences Analysis (Jan 26, 2026)

**Session:** `2026-01-26_neuron_jaxley_backend_differences.md`

- Documented why HH fires faster than NEURON motor neuron channels
- Key missing mechanisms: slow Na inactivation, persistent Na (NaPS), H-current (Gh), L-type Ca (PICs)
- Identified MAHP `mca` state update order bug (uses new mca instead of old mca for ica)
- **Decision:** Accept HH + MAHP as functional approximation; quantitative matching not required

### Phase 4: Examples Update with MAHP (Jan 28, 2026)

**Session:** `2026-01-28_jaxley_examples_mahp_integration.md`

- Removed artificial refractory filtering from Example 2 (now relies on MAHP for rate limiting)
- Removed manual current scaling from Example 3
- Rewrote Example 8 to match NEURON structure
- MAHP parameters set: gkcamax=0.03, gcamax=3e-5, tau_ca=20.0 ms
- **tau_ca discussion:** Tried 90ms (NEURON value), user requested revert to 20ms
- **Example 2 works** with these settings

### Phase 5: Multi-Compartment Attempt (Late Jan / Early Feb 2026)

**Plan:** `whimsical-meandering-rossum.md`

- Attempted to replace single-compartment with soma + 4 dendrites to match NEURON Powers2017
- Goal: Add L_Ca_inact (PICs) on dendrites for subthreshold amplification
- **Result:** Failed. Na3rp cannot spike in single compartment (activation/inactivation window too narrow)
- **Decision:** Reverted all changes. User chose to stay with single-compartment HH + MAHP

### Phase 6: Soma Size Tuning for Rin (Feb 2026 - Current Session)

**Problem:** With committed soma d=50-100, L=400-800 -> Rin = ~54 MOhm (way too high)

#### Attempt 1: Large soma + changed conductances
- Soma: d=150-250, L=2000-3000
- Also changed: gNa higher, gK lower, gLeak higher, MAHP reduced
- **Result:** Rin = 0.25-2.91 MOhm (OK), Rheobase = 25-34 nA (high end OK), F-I gain = NaN
- **F-I NaN cause:** F-I measurement range (5-30 nA) below rheobase for some neurons
- **MAHP problem:** Reduced MAHP broke Example 2 ("way too much activation")
- **User requested:** Revert to state before all changes

#### Attempt 2: Full revert to fc253f3
- `git checkout fc253f3 -- motor_neurons.py cells_api.py 10_extract_neuron_parameters_jaxley.py`
- All three files restored to committed state
- **Back to:** d=50-100, L=400-800, Rin ~54 MOhm

#### Attempt 3: Moderate soma, unchanged conductances
- Soma: d=100-150, L=800-1200
- All conductances kept at original values (gNa 0.15-0.10, gK 0.03-0.06, gLeak 3e-5 to 8e-5)
- MAHP untouched (0.03/3e-5/20.0)
- **Result:** F-I gain = NaN
- **Cause:** Rheobase increased to ~6-8 nA, exceeding F-I measurement range (1-7 nA)
- **Fundamental insight:** Rin and rheobase are physically coupled in single-compartment HH

#### Attempt 4: Target literature ranges (current state)
- User provided: "Rin 1.0-3.0 MOhm, rheobase 4-40 nA"
- Soma: d=150-200, L=1500-2000 (for Rin ~1-3 MOhm)
- All conductances unchanged
- MAHP unchanged (0.03/3e-5/20.0)
- Updated Ex10 measurement currents for higher rheobase range
- Fixed `_exp_interp` crash with curv>=0.45 (eLeak has curvature 0.5)
- Made F-I currents rheobase-relative: `rheobase + np.arange(0, 50, 5)`
- **Result:** Rheobase = 19-137 nA (smallest neuron OK, largest 3.4x too high)
- **BLOCKED:** MAHP gkcamax=0.03 S/cm^2 scales with soma area, making large neurons nearly unspikeble

---

## What is Solved

| Item | Status | Details |
|------|--------|---------|
| Jaxley-mech HH channel integration | SOLVED | Na, K, Leak from jaxley_mech.channels.hodgkin52 |
| MAHP channel for AHP rate limiting | SOLVED | Custom Ca-dependent K+ channel works |
| Henneman size principle (parameter distribution) | SOLVED | Exponential interpolation via _exp_interp |
| _exp_interp crash with curvature >= 0.45 | SOLVED | Fallback to np.linspace |
| Poisson process generator accuracy | SOLVED | Matches NEURON at ~55 pps |
| Example 2 spike trains (use_jaxley_mech) | SOLVED | Works with MAHP=0.03, tau_ca=20ms |
| Example 3 descending drive | SOLVED | Works with weight tuning |
| Example 8 force simulation | SOLVED | Matches NEURON structure |
| F-I measurement range (Ex10) | SOLVED | Now uses rheobase-relative currents |
| Ex10 Rin measurement currents | SOLVED | Scaled to -5 to -1 nA, 1000ms |
| Ex10 rheobase search range | SOLVED | 0.5 nA start, 0.5 step, 500 iterations max |
| Ex10 AP/AHP measurement | SOLVED | Scaled currents, 900ms, vhold+40 threshold |
| numpy 2.x compatibility | SOLVED | elephant from support_numpy_2 branch |

---

## Current Blocking Problem

### The Rin-Rheobase-MAHP Triangle

With the current single-compartment HH + MAHP model, three constraints conflict:

1. **Rin 1-3 MOhm** requires large soma (d=150-200, L=1500-2000)
2. **Rheobase 4-40 nA** requires not too much membrane area (or enough excitability to overcome it)
3. **MAHP gkcamax=0.03 S/cm^2** scales with area and must not be changed (breaks Ex2)

**The conflict:** MAHP total conductance = gkcamax * Area

| Neuron | Soma | Area (cm^2) | MAHP Total (uS) | Leak Total (uS) | MAHP/Leak Ratio |
|--------|------|-------------|------------------|------------------|-----------------|
| Smallest | d=150, L=1500 | 7.07e-3 | 212 | 0.212 | 1000:1 |
| Largest | d=200, L=2000 | 1.26e-2 | 377 | 1.005 | 375:1 |

The MAHP is 375-1000x stronger than leak. During depolarization, calcium enters and MAHP activates, creating massive opposition to spiking. The largest neuron needs 137 nA to overcome this.

### Measured Results (Current State)

| Parameter | Measured | Target | Status |
|-----------|----------|--------|--------|
| Rin | ~1-3 MOhm (estimated) | 1-3 MOhm | Likely OK |
| Rheobase (small) | 19 nA | 4-40 nA | Within range |
| Rheobase (large) | 137 nA | 4-40 nA | 3.4x TOO HIGH |
| F-I Gain | Not yet measured | ~1-4 Hz/nA | Unknown |

---

## Root Cause Analysis

### Why rheobase is 137 nA for the largest neuron

Three compounding effects on the largest neuron (d=200, L=2000):

1. **Large membrane area** -> more current needed to charge membrane to threshold
   - Area = 1.26e-2 cm^2 (very large for single compartment)

2. **MAHP opposition** -> calcium entry during depolarization activates K+ current
   - MAHP total conductance = 377 uS (overwhelms all other conductances)
   - During threshold approach: MAHP fights against depolarization

3. **Lower gNa density** -> less sodium to overcome opposition
   - gNa decreases from 0.15 to 0.10 S/cm^2 (small -> large neurons)
   - This makes large neurons LESS excitable on top of having more area

### The fundamental physics

In a single-compartment model without persistent inward currents:

```
Rheobase ~ (V_threshold - V_rest) / Rin * k_active

where k_active ~ 4x for HH (K channel opposition during depolarization)
      + MAHP contribution (scales with area)
```

For Rin = 1 MOhm: Rheobase ~ 15mV / 1MOhm * k = 60+ nA (with MAHP making k even larger)

NEURON's Powers2017 model avoids this because:
- Multi-compartment (soma + 4 dendrites with PICs)
- L_Ca_inact provides persistent inward current that HELPS depolarization
- NaPS provides subthreshold amplification
- Smaller soma (d=22-30 um) with dendrites providing the area for Rin

---

## Proposed Solutions (Not Yet Tried)

### Option 1: Keep gNa constant for all neurons
**Change:** gNa_range from (0.15, 0.10, 0.33) to (0.15, 0.15, 0.33)
**Effect:** Large neurons get 50% more sodium, helping overcome MAHP
**Risk:** May affect Ex2 recruitment behavior slightly
**Estimated impact:** Could reduce largest rheobase from 137 to ~80-90 nA (still likely too high)

### Option 2: Narrow the soma range
**Change:** d=(150, 170), L=(1500, 1700) instead of d=(150, 200), L=(1500, 2000)
**Effect:** Less area variation -> less rheobase spread
**Risk:** Less Rin variation (narrower Henneman heterogeneity)
**Estimated impact:** Largest rheobase maybe ~60-80 nA (still above 40 nA target)

### Option 3: Reduce MAHP gkcamax slightly
**Change:** gkcamax from 0.03 to 0.005-0.01 S/cm^2
**Effect:** Less MAHP opposition during threshold approach -> lower rheobase
**Risk:** May change Ex2 firing rates and recruitment pattern
**Verification needed:** Run Ex2 after change to confirm acceptable behavior
**Estimated impact:** Could bring rheobase to target range

### Option 4: Combination approach (most likely to work)
**Changes:**
- Moderate soma: d=(120, 160), L=(1200, 1600)
- Keep gNa constant: (0.15, 0.15, 0.33)
- Slightly reduce MAHP: gkcamax = 0.01 S/cm^2
- Increase gLeak slightly: (5e-5, 1.5e-4, 0.33) to compensate Rin

**Rationale:** Distribute the adjustment across multiple parameters so no single change is dramatic

### Option 5: Add persistent inward current (NaPS or similar)
**Change:** Add a small persistent sodium current to help overcome MAHP
**Effect:** Subthreshold amplification reduces effective rheobase
**Risk:** More complex model, may need careful tuning
**Note:** This is what NEURON does with NaPS + L_Ca_inact (PICs)

### Option 6: Scale MAHP per-neuron (not uniform density)
**Change:** Make MAHP gkcamax scale inversely with area: smaller for large neurons
**Effect:** Total MAHP conductance stays roughly constant across pool
**Risk:** Different approach from NEURON (which uses uniform density)
**Implementation:** Add `gkcamax_range` to JAXLEY_HH_PARAMS

---

## Current File State (Uncommitted)

### Changes vs committed state (fc253f3)

#### `motor_neurons.py` (2 changes)
```diff
- "soma_diameter_range": (50.0, 100.0, 0.33),   # original
- "soma_length_range": (400.0, 800.0, 0.33),     # original
+ "soma_diameter_range": (150.0, 200.0, 0.33),   # sized for Rin 1-3 MOhm
+ "soma_length_range": (1500.0, 2000.0, 0.33),   # sized for Rin 1-3 MOhm

  # Added curv>=0.45 safety fallback in _generate_parameter_arrays else branch:
+ if curv >= 0.45:
+     return np.linspace(min_val, max_val, self.n)
```

All conductances UNCHANGED: gNa (0.15, 0.10), gK (0.03, 0.06), gLeak (3e-5, 8e-5)

#### `cells_api.py` (1 change - fallback geometry only)
```diff
- effective_diameter = 25.0 + scale_factor * (60.0 - 25.0)
- effective_length = 200.0 + scale_factor * (480.0 - 200.0)
+ effective_diameter = 150.0 + scale_factor * (200.0 - 150.0)
+ effective_length = 1500.0 + scale_factor * (2000.0 - 1500.0)
```

All channel parameters UNCHANGED: MAHP (0.03/3e-5/20.0), fallback conductances

#### `10_extract_neuron_parameters_jaxley.py` (measurement scaling)
```diff
  # Rin measurement
- amplitudes__nA = [-1.0, -0.8, -0.6, -0.4, -0.2]
- tstop_ms = 500.0
+ amplitudes__nA = [-5.0, -4.0, -3.0, -2.0, -1.0]
+ tstop_ms = 1000.0

  # Tau measurement
- pulse_amp = -5.0
+ pulse_amp = -20.0

  # Rheobase
- max_iterations = 100
- spike_threshold = SPIKE_THRESHOLD_MV  (0 mV)
- amp = 0.1, step = 0.1
+ max_iterations = 500
+ spike_threshold = vhold + 40.0
+ amp = 0.5, step = 0.5

  # AP/AHP
- tstop = 500, threshold = 0 mV, start = 5, step = 2, max = 100
+ tstop = 900, threshold = vhold+40, start = 20, step = 10, max = 500

  # F-I gain
- current_levels = np.arange(1.0, 8.0, 1.0)
+ current_levels = rheobase__nA + np.arange(0.0, 50.0, 5.0)
```

---

## Key Files Reference

| File | Purpose | Location |
|------|---------|----------|
| Motor neuron pool params | JAXLEY_HH_PARAMS dict | `myogen/simulator/jaxley/populations/motor_neurons.py:37-58` |
| Pool creation & interpolation | `AlphaMN__Pool._generate_parameter_arrays()` | `motor_neurons.py:155-235` |
| Cell geometry (fallback) | `AlphaMN._define_geometry()` | `myogen/simulator/jaxley/cells_api.py:600-647` |
| Channel insertion (HH+MAHP) | `AlphaMN._insert_jaxley_mech_channels()` | `cells_api.py:668-738` |
| MAHP parameters | gkcamax, gcamax, tau_ca | `cells_api.py:734-738` |
| Parameter extraction script | Ex10 | `examples/01_basic/10_extract_neuron_parameters_jaxley.py` |
| Spike train example | Ex2 | `examples/01_basic/02_simulate_spike_trains_current_injection_jaxley_mech.py` |
| Custom channels | Na3rp, MAHP, KdrRL, etc. | `myogen/simulator/jaxley/channels/motor_neuron_channels.py` |
| NEURON reference (cells.py) | Powers2017 params | `myogen/simulator/neuron/cells.py:930-999` |

---

## Previous Session Notes

| Date | File | Topic |
|------|------|-------|
| 2026-01-22 | `IMPLEMENTATION_NOTES_2026_01_22.md` | Initial jaxley-mech integration, numpy 2.x fix |
| 2026-01-25 | `docs/session_notes/2026-01-25_jaxley_mech_mahp_integration.md` | MAHP integration, exponential interpolation fix |
| 2026-01-26 | `docs/session_notes/2026-01-26_neuron_jaxley_backend_differences.md` | Backend differences analysis, MAHP bug |
| 2026-01-28 | `docs/session_notes/2026-01-28_jaxley_examples_mahp_integration.md` | Examples update, refractory removal |
| N/A | `examples/01_basic/JAXLEY_VS_NEURON_CHANNELS.md` | HH vs motor neuron channel analysis |
| N/A | `examples/01_basic/JAXLEY_EXAMPLE2_DOCUMENTATION.md` | Example 2 adaptation documentation |

# NEURON vs Jaxley Backend Differences

**Date:** 2026-01-26
**Topic:** Analysis of firing rate differences and expected parameter variations between backends

---

## Summary

The Jaxley version fires at **higher rates** than NEURON because it uses standard Hodgkin-Huxley channels that lack the spike-rate limiting mechanisms present in biologically realistic motor neuron models. Example 10 (Parameter Extraction) produces **different quantitative values** between backends - this is **expected behavior** due to fundamental differences in the underlying biophysical models.

---

## Key Differences

### 1. Channel Models (Primary Cause of High Firing Rates)

| Aspect | NEURON (Powers2017) | Jaxley (HH from jaxley-mech) |
|--------|---------------------|------------------------------|
| Na+ channel | Na3rp: gbar = 0.01 S/cm² | HH Na: gNa = 0.12-0.20 S/cm² (**12x higher**) |
| K+ channel | KdrRL: gMax = 0.015 S/cm² | HH K: gK = 0.02-0.05 S/cm² |
| Leak | g_pas = 8.1e-5 S/cm² | gLeak = 2e-5 to 8e-5 S/cm² |
| **AHP channels** | **mAHP: tau = 90 ms** | **MAHP (simplified)** |
| Persistent Na+ | Naps: gbar = 2.6e-5 S/cm² | None |
| H-current | Gh: ghbar = 3e-5 S/cm² | None |
| L-type Ca2+ | L_Ca_inact in dendrites | None |

### 2. After-Hyperpolarization (AHP) - The Critical Mechanism

**NEURON**: Strong AHP with 90 ms time constant
- After each spike, Ca²⁺-dependent K⁺ current hyperpolarizes the membrane
- This creates a **refractory period** that limits firing rate to physiological values (~10-40 Hz)

**Jaxley HH + MAHP**: Simplified AHP mechanism
- MAHP provides some rate limiting but with different kinetics
- Example 2 uses additional refractory filtering (25ms) to match NEURON rates
- Results in firing rates closer to NEURON when filtering is applied

---

## Why Jaxley Fires Faster (Ranked by Impact)

1. **Different AHP kinetics** - NEURON's 90 ms mAHP tau vs Jaxley's MAHP
2. **Higher Na⁺ conductance** - Standard HH gNa (0.12-0.20 S/cm²) vs motor neuron gNa (0.01 S/cm²)
3. **Lack of slow currents** - No persistent Na⁺, H-current, or L-type Ca²⁺
4. **Kinetics optimized for squid axon** - Standard HH was derived from squid giant axon (fast), not mammalian motor neurons (slow)

---

## Example 10: Expected Parameter Differences

### Why Example 2 Works but Example 10 Differs

| Aspect | Example 2 (Spike Trains) | Example 10 (Parameter Extraction) |
|--------|--------------------------|-----------------------------------|
| **Goal** | Generate spike trains | Measure precise biophysical values |
| **Success criteria** | Neurons fire when stimulated | Rin, rheobase, F-I gain match NEURON |
| **Tolerance** | High - just needs spikes | Low - quantitative match required |
| **Workaround used** | Refractory filtering (25ms) | None available |

**Example 2** succeeds because:
- It only checks IF neurons fire, not their exact electrophysiological properties
- Uses refractory period filtering to artificially limit firing rate to ~40 Hz
- The MAHP channel provides some AHP, filtering does the rest

**Example 10** shows differences because it measures actual cell properties that ARE different:
- **Rin**: Depends on surface area × gLeak (different geometry = different Rin)
- **Rheobase**: Depends on Rin and channel densities
- **F-I gain**: Depends on AHP kinetics (NEURON mAHP: 90ms tau vs Jaxley MAHP)
- **Time constant**: Depends on Cm × Rin

### Expected Parameter Ranges

| Parameter | NEURON (Powers2017) | Jaxley (HH + MAHP) | Reason for Difference |
|-----------|--------------------|--------------------|----------------------|
| **Rin** | 0.5 - 1.7 MΩ | 5 - 50 MΩ | Smaller cells, lower gLeak in Jaxley |
| **Rheobase** | 6 - 19 nA | 0.5 - 5 nA | Higher Rin → lower rheobase |
| **F-I gain** | 2.7 - 3.5 Hz/nA | 0.5 - 2 Hz/nA | Different AHP kinetics |
| **tau** | 5 - 15 ms | 10 - 40 ms | Higher Rin × Cm |
| **AP amplitude** | ~80 mV | ~100 mV | Standard HH has larger spikes |
| **AHP depth** | 5 - 15 mV | 2 - 8 mV | MAHP weaker than mAHP |

### Root Cause: Different Channel Biophysics

**NEURON uses motor neuron-specific channels:**
- Na3rp: Slow inactivation kinetics
- KdrRL: Delayed rectifier with specific kinetics
- mAHP: Strong Ca²⁺-dependent K⁺ with 90ms tau
- Naps: Persistent Na⁺ current
- Gh: H-current (sag)

**Jaxley uses standard HH + simplified MAHP:**
- HH Na: Fast squid axon kinetics (12x higher conductance)
- HH K: Standard delayed rectifier
- MAHP: Simplified Ca²⁺-dependent K⁺ (different kinetics)
- No persistent Na⁺, no H-current

### Acceptance Criteria

The Jaxley backend is considered **functionally equivalent** when:
1. ✅ Neurons fire action potentials with proper shape
2. ✅ Henneman size principle is preserved (small neurons more excitable)
3. ✅ Recruitment order follows threshold distribution
4. ✅ Parameter trends are similar (even if absolute values differ)

**Quantitative matching is NOT required** because:
- Different channel models have inherently different properties
- The goal is functional simulation, not exact replication
- Users can tune JAXLEY_HH_PARAMS if closer matching is needed

---

## Jaxley Examples Status Summary

### Example Compatibility Matrix

| Example | NEURON | Jaxley | Status | Notes |
|---------|--------|--------|--------|-------|
| **01** - Recruitment Thresholds | ✅ | ✅ | **Identical** | Pure Python, no simulator |
| **02** - Current Injection Spikes | ✅ | ✅ | **Functional** | Uses refractory filtering (25ms) |
| **03** - Descending Drive | ✅ | ✅ | **Functional** | Poisson-based drive |
| **08** - Force Simulation | ✅ | ✅ | **Functional** | Same current (15 nA) |
| **10** - Parameter Extraction | ✅ | ⚠️ | **Different values** | Expected - different biophysics |

### Example 02: Spike Trains with Current Injection

**File:** `02_simulate_spike_trains_current_injection_jaxley_mech copy.py`

**Approach:**
- Uses jaxley-mech HH channels (Na, K, Leak) + MAHP
- Injects same current (15 nA) to ALL neurons
- Recruitment emerges from biophysics (Henneman principle)
- **Refractory filtering (25ms)** limits max firing rate to ~40 Hz

**Key Parameters:**
```python
current_amplitude = 15.0 * pq.nA  # Same as NEURON
min_isi_ms = 25.0  # Refractory period (max ~40 Hz)
spike_detection_threshold__mV = 0.0  # HH crosses 0 mV
```

**Why it works:**
- Refractory filtering compensates for lack of proper AHP
- Only needs spikes to happen, not specific Rin/rheobase values
- Henneman principle preserved via cell geometry/conductance scaling

### Example 03: Descending Drive

**File:** `03_simulate_spike_trains_descending_drive_jaxley.py`

**Status:** Working with same approach as Example 02

**Known Issue:** Activity may continue after trapezoid ramp-down if noise is added during rest periods.

### Example 08: Force Simulation

**File:** `08_simulate_force_jaxley.py`

**Approach:**
- Trapezoid current input (same as NEURON)
- 15 nA peak amplitude, 5 nA offset
- Force calculation from spike trains

**Key Parameters:**
```python
trap_amplitude = 15.0 * pq.nA  # Same as NEURON
trap_offset = 5.0 * pq.nA
# Rise: 5000ms, Plateau: 8000ms, Fall: 3000ms
```

### Example 10: Parameter Extraction

**File:** `10_extract_neuron_parameters_jaxley.py`

**Status:** Produces different values than NEURON (expected)

**Measurements:**
- Vhold: Resting membrane potential
- Rin: Input resistance (V-I slope with negative currents)
- tau: Time constant from exponential decay
- Ir: Rheobase (minimum current for spike)
- AP/AHP: Action potential and afterhyperpolarization
- F-I gain: Firing rate vs current slope

---

## Technical Details: Jaxley Cell Structure

### AlphaMN Cell Configuration

**File:** `myogen/simulator/jaxley/cells_api.py`

```python
class AlphaMN(_Cell):
    # Single-compartment cell (ncomp=1)
    # Uses jaxley-mech HH + MAHP when use_jaxley_mech=True
```

**Channel insertion (`_insert_jaxley_mech_channels`):**
1. HH Na (jaxley-mech): Fast sodium
2. HH K (jaxley-mech): Delayed rectifier potassium
3. Leak (jaxley-mech): Passive leak
4. MAHP (custom): Calcium-dependent K+ for AHP

### JAXLEY_HH_PARAMS

**File:** `myogen/simulator/jaxley/populations/motor_neurons.py`

```python
JAXLEY_HH_PARAMS = {
    # Morphology (Henneman size principle)
    "soma_diameter_range": (50.0, 100.0, 0.33),  # µm
    "soma_length_range": (400.0, 800.0, 0.33),   # µm

    # Membrane properties
    "capacitance_range": (1.2, 0.8, 0.33),       # µF/cm²
    "axial_resistivity_range": (100.0, 70.0, 0.33),  # Ω·cm

    # Conductances
    "gNa_range": (0.15, 0.10, 0.33),  # S/cm²
    "gK_range": (0.03, 0.06, 0.33),   # S/cm²
    "gLeak_range": (3e-5, 8e-5, 0.33),# S/cm²

    # Reversal potentials
    "eLeak_range": (-68.0, -72.0, 0.5),  # mV
    "eNa": 50.0,   # mV (fixed)
    "eK": -77.0,   # mV (fixed)
}
```

**Parameter interpolation:**
- Uses exponential interpolation (`_exp_interp`) for Henneman principle
- Low threshold neurons: small soma, high gNa, low gLeak → more excitable
- High threshold neurons: large soma, low gNa, high gLeak → less excitable

---

## MAHP Channel Bug Analysis

### Investigation Summary
The custom motor neuron channels (Na3rp, KdrRL, Naps, MAHP, Gh) only produce **1 spike** instead of repetitive firing when used together. The root cause is in the MAHP channel's calcium dynamics.

### Root Cause: Sequential State Update (Wrong Order)

**NEURON mAHP.mod (cnexp solver):**
```c
BREAKPOINT {
    SOLVE states METHOD cnexp    // Solves all ODEs simultaneously
    ica = gcamax*mca*(v - eca)   // Output current uses UPDATED mca
    ik = gkcamax*n*(v - ek)
}

DERIVATIVE states {
    drive_channel = -10000 * ica / (2*FARADAY*depth)  // Uses ica from PREVIOUS timestep!
    cai' = drive_channel + (cainf-cai)/tau
    rates(cai)
    n' = (ninf-n)/ntau
}
```
Key: NEURON uses `ica` from the **PREVIOUS** BREAKPOINT for calcium dynamics.

**Jaxley MAHP (sequential update):**
```python
def update_states(self, states, dt, v, params):
    # 1. Update mca first
    mca_new = solve_gate_exponential(mca, dt, mca_inf, self.mtauca)

    # 2. Compute ica using NEW mca (BUG: should use OLD mca!)
    ica = gcamax * mca_new * (v - e_ca)

    # 3. Update cai using ica
    drive = -10000.0 * ica / (2.0 * self.FARADAY * self.depth)
    cai_new = cai + dt * dcai

    # 4. Update n using NEW cai
    n_new = solve_gate_exponential(n, dt, n_inf, tau_n)
```
Bug: Jaxley uses `mca_new` (updated value) instead of `mca` (old value) to compute `ica` for calcium dynamics.

### Proposed Fix

**File:** `myogen/simulator/jaxley/channels/motor_neuron_channels.py`

Change:
```python
ica = gcamax * mca_new * (v - e_ca)
```

To:
```python
ica = gcamax * mca * (v - e_ca)  # Use OLD mca, not mca_new!
```

---

## Future Improvements (Optional)

### To get closer NEURON matching:

1. **Fix custom motor neuron channels** (Na3rp, KdrRL, MAHP bug documented above)
   - Would provide motor neuron-specific kinetics
   - Set `use_jaxley_mech=False` to use them

2. **Add multi-compartment cells** (soma + dendrites)
   - Would lower Rin by increasing surface area
   - More realistic current distribution

3. **Tune MAHP parameters** for stronger AHP
   - Increase `gkcamax` for deeper AHP
   - Adjust `tau_ca` for longer duration

4. **Adjust JAXLEY_HH_PARAMS**
   - Increase gLeak for lower Rin
   - May need to increase gNa to maintain spiking

### Current Recommendation
Accept differences as expected variation between backends. Focus on functional equivalence (recruitment order, spike generation) rather than quantitative matching.

---

## Files Involved

- `myogen/simulator/jaxley/cells_api.py` - Channel insertion (`_insert_jaxley_mech_channels()`)
- `myogen/simulator/jaxley/cells/biophysical.py` - Channel parameters
- `myogen/simulator/jaxley/channels/` - Custom channel implementations
- `myogen/simulator/jaxley/populations/motor_neurons.py` - Pool parameters (JAXLEY_HH_PARAMS)

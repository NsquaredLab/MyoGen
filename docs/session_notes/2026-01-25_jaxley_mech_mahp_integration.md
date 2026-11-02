# Jaxley-mech + MAHP Integration Session Notes

**Date:** 2026-01-25
**Topic:** Working jaxley-mech + MAHP channel combination for Jaxley backend

---

## Summary

Successfully integrated jaxley-mech Hodgkin-Huxley channels (Na, K, Leak) with the custom MAHP channel to provide biophysical after-hyperpolarization (AHP) rate limiting in Jaxley simulations.

---

## Working Channel Combination

### Channels Used
| Channel | Source | Purpose |
|---------|--------|---------|
| Na (Sodium) | jaxley-mech | Fast sodium for action potential upstroke |
| K (Potassium) | jaxley-mech | Delayed rectifier for repolarization |
| Leak | jaxley-mech | Passive leak conductance |
| MAHP | Custom (motor_neuron_channels.py) | Calcium-dependent K+ for AHP rate limiting |

### How to Enable
```python
from myogen.simulator.jaxley.populations import AlphaMN__Pool

pool = AlphaMN__Pool(
    recruitment_thresholds__array=recruitment_thresholds,
    mode="active",
    use_jaxley_mech=True,  # Enables jaxley-mech HH + MAHP
)
```

### Channel Insertion (cells_api.py)
When `use_jaxley_mech=True`:
1. jaxley-mech Na channel inserted with scaled gNa
2. jaxley-mech K channel inserted with scaled gK
3. Leak conductance updated with scaled gLeak
4. MAHP channel inserted for calcium-dependent AHP

---

## Key Fix: Exponential Parameter Interpolation

### Problem
Motor neuron pool parameters were distributed **linearly** instead of **exponentially**, causing poor recruitment behavior (step-like instead of gradual).

### Location
`myogen/simulator/jaxley/populations/motor_neurons.py` lines 169-199

### Before (Broken)
```python
def interp_by_threshold(min_val, max_val, _curv):
    """Linear interpolation - curvature ignored!"""
    return rt_norm * (max_val - min_val) + min_val
```

### After (Fixed)
```python
def interp_by_threshold(min_val, max_val, curv):
    """Exponential interpolation using _exp_interp reference curve."""
    if curv >= 0.45:
        return rt_norm * (max_val - min_val) + min_val

    try:
        n_ref = 100
        ref_curve = _exp_interp(min_val, max_val, n_ref, curv=curv)
        ref_x = np.linspace(0, 1, n_ref)
        return np.interp(rt_norm, ref_x, ref_curve)
    except RuntimeError:
        return rt_norm * (max_val - min_val) + min_val
```

### Impact
- Parameters with `curv=0.33` now follow exponential distribution
- Creates proper Henneman size principle heterogeneity
- Results in gradual recruitment instead of step-like activation

---

## Example 10: Parameter Extraction Comparison

### Extracted Parameter Ranges

| Parameter | NEURON | Jaxley (HH+MAHP) | Ratio |
|-----------|--------|------------------|-------|
| **Rheobase (nA)** | 6.00 - 18.90 | 0.10 - 7.80 | ~10-60x lower |
| **Input Resistance (MΩ)** | 0.53 - 1.70 | 12.84 - 318.20 | ~100-200x higher |
| **Time Constant (ms)** | N/A | 4.59 - 26.44 | - |
| **AP Amplitude (mV)** | N/A | 111.69 - 118.41 | - |
| **AHP Depth (mV)** | N/A | 3.01 - 6.58 | - |
| **F-I Gain (Hz/nA)** | 2.74 - 3.48 | 1.54 - 1.86 | ~2x lower |

### Root Cause: Cell Geometry Differences

| Property | NEURON | Jaxley | Impact |
|----------|--------|--------|--------|
| **Soma diameter** | ~77 µm | 25-60 µm | Jaxley cells 2-3x smaller |
| **Cell structure** | Multi-compartment (soma + dendrites) | Single compartment (soma only) | Much less surface area |
| **Surface area** | ~6000+ µm² | ~2000-11000 µm² | Variable |

### Why Parameters Differ

**1. Input Resistance (Rin)**
```
Rin = 1 / (surface_area × gLeak)
```
- Smaller cells → less surface area → higher Rin
- NEURON large cells: ~1 MΩ
- Jaxley small cells: ~100 MΩ

**2. Rheobase (Minimum Current)**
```
V = I × R  →  I = V / R
```
- To reach threshold (~15 mV depolarization):
  - NEURON: 15 mV / 1 MΩ = **15 nA needed**
  - Jaxley: 15 mV / 100 MΩ = **0.15 nA needed**

**3. F-I Gain**
- Depends on membrane capacitance (C = area × specific capacitance)
- Smaller cells charge faster → different dynamics
- AHP characteristics also differ

---

## Would Matching NEURON Geometry Affect Example 2?

### Current Jaxley Parameters (JAXLEY_HH_PARAMS)
```python
JAXLEY_HH_PARAMS = {
    "soma_diameter_range": (25.0, 60.0, 0.33),    # µm
    "soma_length_range": (200.0, 480.0, 0.33),    # µm
    "gNa_range": (0.20, 0.10, 0.33),              # S/cm²
    "gK_range": (0.02, 0.05, 0.33),               # S/cm²
    "gLeak_range": (2e-5, 8e-5, 0.33),            # S/cm²
}
```

### To Match NEURON Geometry
```python
# Proposed changes to match NEURON
JAXLEY_HH_PARAMS_NEURON_MATCHED = {
    "soma_diameter_range": (50.0, 100.0, 0.33),   # Larger: ~77 µm average
    "soma_length_range": (400.0, 800.0, 0.33),    # Proportionally larger
    "gLeak_range": (8e-5, 2e-4, 0.33),            # Higher leak for lower Rin
    # gNa, gK may need adjustment too
}
```

### Impact on Example 2

| Aspect | Current (Small Cells) | After Change (Large Cells) |
|--------|----------------------|---------------------------|
| **Current needed** | ~6 nA | ~15 nA (like NEURON) |
| **Rin** | ~100 MΩ | ~1-2 MΩ |
| **Simulation speed** | Same | Same (single compartment) |
| **Recruitment behavior** | Works with 6 nA | Would need 15 nA |
| **MAHP effectiveness** | Working | Should still work |

### Recommendation

**Option A: Keep Current Parameters (Recommended for now)**
- Examples 2, 8, 10 all work with current scaling
- Just document that Jaxley uses different current scales
- Simpler to maintain

**Option B: Match NEURON Geometry**
- Would require updating:
  - `JAXLEY_HH_PARAMS` in motor_neurons.py
  - Current amplitudes in all examples (6 nA → 15 nA)
  - Possibly MAHP parameters (tau, conductances)
- Risk: May break current working setup
- Benefit: More direct comparison with NEURON

**Option C: Add Configurable Geometry Profiles**
```python
# In motor_neurons.py
GEOMETRY_PROFILES = {
    "default": JAXLEY_HH_PARAMS,  # Current small cells
    "neuron_matched": JAXLEY_HH_PARAMS_NEURON_MATCHED,
}

# Usage
pool = AlphaMN__Pool(
    ...,
    geometry_profile="neuron_matched",  # Optional parameter
)
```

---

## Files Modified Today

| File | Changes |
|------|---------|
| `motor_neurons.py` | Fixed exponential interpolation for parameter distribution |
| `08_simulate_force_jaxley.py` | Rewritten to match NEURON logic with jaxley-mech + MAHP |
| `10_extract_neuron_parameters_jaxley.py` | Updated documentation for HH + MAHP |
| `cells_api.py` | Already had jaxley-mech + MAHP insertion (from previous session) |

---

## Pending Issues

### Na3rp Channel (Custom Sodium)
- Still produces 0 spikes
- Reverted to jaxley-mech Na for now
- Root cause: Unknown (tried trap0 fix, gbar scaling, sh adjustment)

### MAHP with Custom Na3rp
- When Na3rp is fixed, the full custom channel set would be:
  - Na3rp (fast sodium with slow inactivation)
  - KdrRL (delayed rectifier)
  - Naps (persistent sodium)
  - MAHP (calcium-dependent K+)
  - Gh (H-current)
- This would more closely match NEURON's biophysics

---

## Quick Reference: Current Scaling

| Simulator | Example 2 Current | Example 8 Current |
|-----------|-------------------|-------------------|
| NEURON | 15 nA | 15 nA peak, 5 nA baseline |
| Jaxley | 6 nA | 6 nA peak, 2 nA baseline |
| **Ratio** | ~40% | ~40% |

The ~40% scaling accounts for Jaxley's smaller cell geometry and higher input resistance.

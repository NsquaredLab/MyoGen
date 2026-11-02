# Jaxley vs NEURON Channel Implementation Analysis

## Overview

This document summarizes the differences between the Jaxley-mech Hodgkin-Huxley (HH) channels and the original NEURON custom motor neuron channels, explaining why we observe different spike counts and firing rates.

## The Two Approaches

### 1. Jaxley-mech HH Channels (`use_jaxley_mech=True`)
- **Source**: `jaxley_mech.channels.hodgkin52` (Na, K, Leak)
- **Status**: Works correctly with repetitive firing
- **Firing rate**: ~60-80 Hz (too fast compared to biological motor neurons)

### 2. NEURON Custom Channels (`use_jaxley_mech=False`)
- **Channels**: Na3rp, KdrRL, MAHP, Gh, Naps
- **Status**: Jaxley implementation has bugs - only produces 1 spike (depolarization block)
- **NEURON firing rate**: ~25 Hz (biologically accurate)

---

## Why HH Channels Fire Faster Than Motor Neuron Channels

### 1. No Slow Sodium Inactivation
**HH channels**: Standard m³h gating with only fast inactivation (h gate recovers in ~1-5 ms)

**Motor neuron Na3rp**: Has m³hs gating with:
- Fast inactivation (h): recovers in ~1-5 ms
- Slow inactivation (s): recovers in ~100-1000 ms
- This causes **spike frequency adaptation** - firing slows over time

### 2. No Medium After-Hyperpolarization (mAHP)
**HH channels**: No calcium-dependent potassium current

**Motor neuron MAHP channel**:
- Calcium entry during spike activates KCa channels
- Produces 50-100 ms hyperpolarization after each spike
- Forces longer inter-spike intervals
- Key mechanism for controlling firing rate

### 3. Different Sodium Channel Kinetics
**HH channels**: Based on squid giant axon at 6.3°C
- Fast activation/inactivation designed for rapid conduction

**Motor neuron Na3rp**: Based on mammalian neurons at 37°C
- Slower kinetics tuned for ~10-50 Hz firing rates
- Threshold shift parameters (sh) adjust activation voltage

### 4. No Persistent Sodium (Naps)
**HH channels**: Only transient sodium

**Motor neuron Naps**:
- Small persistent sodium current
- Contributes to subthreshold membrane dynamics
- Affects excitability but not directly firing rate

### 5. No H-current (Gh)
**HH channels**: No hyperpolarization-activated current

**Motor neuron Gh**:
- Activated by hyperpolarization
- Produces "sag" in response to inhibition
- Affects recovery and membrane stability

---

## Quantitative Comparison

| Property | HH (Jaxley-mech) | Motor Neuron (NEURON) | Difference |
|----------|------------------|----------------------|------------|
| Firing rate | 60-80 Hz | 20-30 Hz | HH 2-3x faster |
| Spike adaptation | None | Strong | HH lacks adaptation |
| AHP duration | ~2-5 ms | ~50-100 ms | HH recovers faster |
| Na inactivation | Fast only | Fast + Slow | HH fully available |
| Current threshold | ~5-6 nA | ~15 nA | HH more excitable |

---

## Known Issues with Jaxley Custom Channel Implementation

### Critical Bug: Slow Inactivation Collapse
The Na3rp slow inactivation gate (s) collapses to near-zero incorrectly:

```
Expected behavior (ar=1.0, no slow inactivation):
  t=0 ms:   s = 1.0
  t=50 ms:  s = 1.0  (should stay at 1.0)
  t=100 ms: s = 1.0

Actual behavior:
  t=0 ms:   s = 1.0
  t=50 ms:  s = 0.0006  (collapsed!)
  t=100 ms: s = 0.0006
```

This causes:
1. First spike fires normally
2. Slow inactivation immediately inactivates Na channels
3. Cell cannot repolarize properly (depolarization block)
4. No further spikes possible

### Parameter Mismatches (Partially Fixed)
| Parameter | NEURON Value | Original Jaxley | Fixed |
|-----------|-------------|-----------------|-------|
| Na3rp_gbar | 0.01 S/cm² | 0.12 S/cm² | Yes |
| Na3rp_qinf | 8.0 mV | 4.0 mV | Yes |
| KdrRL_gbar | 0.015 S/cm² | 0.036 S/cm² | Yes |
| KdrRL_tmin | 0.8 ms | 1.4 ms | Yes |
| KdrRL_taumax | 20.0 ms | 11.9 ms | Yes |
| Cell geometry | L=2952, d=22 µm | L=320, d=40 µm | Yes |

### Remaining Issues
1. **Slow inactivation kinetics**: The `ar` parameter isn't preventing slow inactivation as it should
2. **Gate equations**: May have sign or formula errors in the Jaxley implementation
3. **State initialization**: States may not be properly initialized before simulation

---

## Recommendations

### For Immediate Use: Use Jaxley-mech HH Channels
```python
pool = AlphaMN__Pool(
    recruitment_thresholds__array=thresholds,
    mode="active",
    use_jaxley_mech=True,  # Use HH channels
)
```

**Adjust for firing rate differences**:
- Reduce current amplitude (~6 nA vs 15 nA for NEURON)
- Accept that firing rates will be 2-3x higher than biological
- Results are still useful for:
  - Testing pipeline functionality
  - Relative comparisons between conditions
  - Demonstrating spike-based computations

### For Accurate Motor Neuron Simulation
Until the custom channels are debugged:
1. Use NEURON backend for biologically accurate motor neuron simulations
2. Or accept HH channel approximations with caveats documented

### To Fix Custom Channels (Future Work)
1. Debug the slow inactivation (s gate) kinetics in Na3rp
2. Verify all gate equations match NMODL exactly
3. Add comprehensive unit tests comparing Jaxley vs NEURON channel outputs
4. Consider line-by-line comparison of NMODL vs Python implementations

---

## Code Locations

| Component | File |
|-----------|------|
| HH Channels | `jaxley_mech.channels.hodgkin52` (external package) |
| Custom Channels | `myogen/simulator/jaxley/channels/motor_neuron_channels.py` |
| Cell Configuration | `myogen/simulator/jaxley/cells_api.py` |
| NEURON NMODL Files | `myogen/simulator/nmodl_files/*.mod` |
| Example Script | `examples/01_basic/02_simulate_spike_trains_current_injection_jaxley_mech.py` |

---

## Summary

**Why HH fires faster**: Missing slow inactivation, mAHP, and motor-neuron-specific kinetics that normally limit firing rate to ~25 Hz.

**Why custom channels fail**: Bug in slow inactivation gate causes immediate sodium channel inactivation after first spike, preventing repetitive firing.

**Current solution**: Use HH channels with adjusted parameters, accepting ~2-3x higher firing rates than biological motor neurons.

# Jaxley Examples: MAHP + jaxley-mech Integration

**Date:** 2026-01-28
**Topic:** Updating Jaxley examples to use MAHP + jaxley-mech HH channels consistently

---

## Summary

Updated Examples 2, 3, and 8 (Jaxley versions) to use the **MAHP + jaxley-mech combo** for consistent biophysical simulation. Removed artificial workarounds (refractory filtering, manual current scaling) in favor of biophysics-driven behavior where possible.

---

## Changes Made

### Example 2: Current Injection (`02_simulate_spike_trains_current_injection_jaxley_mech copy.py`)

**Before:**
- Had refractory filtering (`min_isi_ms = 25.0`) to artificially limit firing rate

**After:**
- Removed `apply_refractory_filter` function
- Removed `min_isi_ms` parameter
- Relies solely on **MAHP channel** for AHP rate limiting
- Same current (15 nA) injected to ALL neurons
- Recruitment emerges from biophysics (Henneman principle)

**Key code removed:**
```python
# REMOVED: Refractory filtering
def apply_refractory_filter(spike_times_ms, min_isi_ms=25.0):
    ...
```

---

### Example 8: Force Simulation (`08_simulate_force_jaxley.py`)

**Before:**
- Different structure from NEURON version
- Missing firing rate statistics

**After:**
- Rewrote to match NEURON Example 8 structure closely
- Uses `use_jaxley_mech=True` for jaxley-mech HH + MAHP
- Same parameters as NEURON:
  - `timestep__ms = 0.05 * pq.ms`
  - `trap_amplitude = 15.0 * pq.nA`
  - `trap_offset = 5.0 * pq.nA`
- Added detailed firing rate statistics with warning for high rates (>50 Hz)
- No refractory filtering - relies on MAHP for rate limiting

**Known Issue:** Force plot may still be jagged due to MAHP tau_ca (20 ms) being faster than NEURON mAHP tau (90 ms). User chose to keep tau_ca at 20 ms.

---

### Example 3: Descending Drive (`03_simulate_spike_trains_descending_drive_jaxley.py`)

**Before:**
- Manual current scaling: `current_amplification = np.exp(-4.5 * normalized_threshold)`
- This artificially amplified current for low-threshold neurons

**After:**
- Removed manual current scaling
- Same synaptic current injected to ALL neurons (NEURON approach)
- Recruitment should emerge from biophysics (Henneman principle)

**Key code removed:**
```python
# REMOVED: Manual current scaling
normalized_threshold = (threshold_nA - threshold_min) / (threshold_max - threshold_min)
current_amplification = np.exp(-4.5 * normalized_threshold)
i_exc_nA = i_syn_nA * current_amplification
```

---

## Recruitment Challenge Discovered

### Problem
With manual current scaling removed, **100/100 MNs fired** instead of ~74/100 (NEURON target).

### Root Cause
The JAXLEY_HH_PARAMS Henneman scaling doesn't provide enough differentiation:
- Rin ratio: ~10x between lowest and highest threshold neurons
- But with high synaptic input, ALL neurons exceed rheobase

### Synaptic Current Analysis

With 65 pps DD drive, 50% connectivity (50 DD neurons), tau=5ms:
```
g_steady = rate × tau × weight
I_steady = g_steady × 70mV (driving force)
```

| Weight (µS) | g_steady (µS) | I_steady (nA) | Result |
|-------------|---------------|---------------|--------|
| 0.15 | 2.4 | 170 | 100/100 active |
| 0.025 | 0.4 | 28 | 100/100 active |
| 0.003 | 0.05 | 3.5 | Testing... |

### Current Status
Set `base_synaptic_weight = 0.003 µS` to get I_steady ≈ 3.5 nA, which should be within the Jaxley rheobase range (~0.5-5 nA) for partial recruitment.

---

## MAHP vs Recruitment: Key Distinction

| Mechanism | What it affects | How |
|-----------|-----------------|-----|
| **MAHP (AHP)** | Firing rate | After spike, hyperpolarizes → limits max rate to ~30-50 Hz |
| **Henneman scaling** | Recruitment | Cell geometry & conductances → whether neuron reaches threshold |

**MAHP does NOT affect recruitment** - it only limits firing rate after the neuron is already spiking. Recruitment depends on whether the neuron can reach threshold at all.

---

## Channel Configuration

### jaxley-mech + MAHP (use_jaxley_mech=True)

When `AlphaMN__Pool` is created with `use_jaxley_mech=True`:

1. **HH Na** (jaxley-mech): Fast sodium from `jaxley_mech.channels.hodgkin52`
2. **HH K** (jaxley-mech): Delayed rectifier potassium
3. **Leak** (jaxley-mech): Passive leak conductance
4. **MAHP** (custom): Calcium-dependent K+ for after-hyperpolarization

### MAHP Parameters (tau_ca discussion)

| Parameter | Current Value | NEURON mAHP | Notes |
|-----------|---------------|-------------|-------|
| tau_ca | 20.0 ms | 90 ms | User chose to keep at 20 ms |
| gkcamax | 0.03 S/cm² | Similar | Controls AHP depth |
| gcamax | 3e-5 S/cm² | Similar | Calcium current |

**Note:** Attempted to change tau_ca to 90 ms to match NEURON but user requested revert.

---

## Files Modified

### Example Files
- `examples/01_basic/02_simulate_spike_trains_current_injection_jaxley_mech copy.py`
  - Removed refractory filtering
  - Updated comments to mention MAHP

- `examples/01_basic/03_simulate_spike_trains_descending_drive_jaxley.py`
  - Removed manual current scaling (`exp(-4.5 * normalized_threshold)`)
  - Reduced synaptic weight from 0.15 µS to 0.003 µS
  - Updated docstring to describe MAHP + jaxley-mech approach

- `examples/01_basic/08_simulate_force_jaxley.py`
  - Rewrote to match NEURON structure
  - Added firing rate statistics
  - Uses MAHP + jaxley-mech

### Core Files (tau_ca reverted)
- `myogen/simulator/jaxley/cells_api.py` (line 736): tau_ca = 20.0 ms
- `myogen/simulator/jaxley/cells/biophysical.py` (lines 88, 344, 440): tau_ca = 20.0 ms
- `myogen/simulator/jaxley/channels/motor_neuron_channels.py` (line 480): tau_ca = 20.0 ms

---

## Recommendations

### For Better Recruitment (Future Work)

1. **Increase JAXLEY_HH_PARAMS scaling range**
   - Current: 2x soma diameter, 2.67x gLeak → ~10x Rin
   - Need: Wider range for more differentiation

2. **Adjust gLeak range more aggressively**
   - Low threshold: very low gLeak → very high Rin
   - High threshold: very high gLeak → very low Rin

3. **Consider hybrid approach**
   - Use mild threshold-based scaling if pure biophysics insufficient
   - Less aggressive than `exp(-4.5 * threshold)`

### Current Approach
Accept that synaptic weight tuning (0.003 µS) is needed to achieve proper recruitment with current JAXLEY_HH_PARAMS scaling.

---

## Testing Notes

Run examples in order:
1. Example 1 (generates thresholds.pkl)
2. Example 2 (current injection - validates MAHP rate limiting)
3. Example 3 (descending drive - validates recruitment)
4. Example 8 (force - validates end-to-end pipeline)

Expected results:
- Example 2: ~77-84/100 active neurons, firing rates ~10-40 Hz
- Example 3: ~74/100 active neurons (target)
- Example 8: Smooth force output (may still be slightly jagged due to tau_ca=20ms)

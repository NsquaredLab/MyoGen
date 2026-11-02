# Jaxley Implementation of Example 2: Spike Train Generation with Current Injection

## Overview

Example 2 demonstrates how to simulate spike trains in a population of alpha motor neurons using time-varying (trapezoid) current injection. This document compares the **Jaxley implementation** (using jaxley-mech channels) with the **NEURON implementation** and highlights critical factors for adaptation.

---

## What the Jaxley Implementation Does

### Workflow

1. **Create Motor Neuron Pools**: Uses `AlphaMN__Pool` with `use_jaxley_mech=True` to create 2 pools of 100 motor neurons each, using Hodgkin-Huxley channels from jaxley-mech.

2. **Generate Trapezoid Current**: Creates a trapezoid-shaped input current with:
   - 500 ms delay (no current)
   - 100-500 ms rise time (ramp up)
   - 1000-2000 ms plateau (constant peak)
   - 1000-2000 ms fall time (ramp down)

3. **Simulate Each Neuron**: For each neuron below the recruitment threshold:
   - Reset voltage to -65 mV
   - Initialize channel states to steady-state (`cell.init_states()`)
   - Scale current based on recruitment threshold
   - Run `jx.integrate()` to solve HH equations
   - Detect spikes via threshold crossing at 0 mV

4. **Output**: Neo `Block` with `SpikeTrain` objects, firing rate statistics, and raster plot visualization.

### Ion Channels Used

The Jaxley implementation uses **standard Hodgkin-Huxley 1952 channels** from `jaxley_mech.channels.hodgkin52`:

| Channel | Gating | Function |
|---------|--------|----------|
| **Na** | m³h | Fast sodium - action potential upstroke |
| **K** | n⁴ | Delayed rectifier potassium - repolarization |
| **Leak** | - | Passive leak conductance - resting potential |

---

## Key Differences from NEURON Implementation

### Comparison Table

| Aspect | NEURON | Jaxley (jaxley-mech) |
|--------|--------|---------------------|
| **Simulator** | NEURON 8.2+ | Jaxley (JAX-based) |
| **Backend** | C/Python bindings | Pure JAX (GPU-compatible) |
| **Ion Channels** | Custom motor neuron model | Standard Hodgkin-Huxley |
| **Channel Set** | Na3rp, NaPs, KdrRL, mAHP, Gh, L_Ca_inact | Na, K, Leak |
| **Current Amplitude** | 15 nA | 2 nA |
| **Timestep** | 0.05 ms | 0.025 ms |
| **Spike Threshold** | 50 mV | 0 mV |
| **Action Potential Peak** | ~80-100 mV | ~40 mV |
| **Integration Method** | NEURON's implicit solver | `jx.integrate()` (explicit) |
| **Compilation** | NMODL mechanisms | JAX JIT compilation |

### Channel Differences Explained

**NEURON Motor Neuron Channels:**
- **Na3rp**: Fast sodium with slow inactivation (bistability)
- **NaPs**: Persistent sodium (plateau potentials, PICs)
- **KdrRL**: Delayed rectifier potassium
- **mAHP**: Calcium-dependent potassium (medium AHP, firing rate adaptation)
- **Gh**: Hyperpolarization-activated cation (sag potential)
- **L_Ca_inact**: L-type calcium with inactivation (dendritic PICs)

**Jaxley-mech HH Channels:**
- Simple m³h sodium and n⁴ potassium gating
- No slow inactivation, no persistent currents
- No calcium dynamics or AHP
- No sag potential

### Physiological Implications

The jaxley-mech `hodgkin52` module implements the **classic Hodgkin-Huxley 1952 squid giant axon model** with only three channel types:
- **Na**: m³h gating (fast activation and fast inactivation)
- **K**: n⁴ gating (delayed rectifier)
- **Leak**: passive conductance

**Source**: [jaxley-mech documentation](https://github.com/jaxleyverse/jaxley-mech) - *"Classic Hodgkin-Huxley ion channels from the 1952 squid giant axon model"*

The following features are **absent by design** in the classic HH model (these are not jaxley limitations, but fundamental properties of the 1952 model):

| Feature | NEURON Motor Neuron | Jaxley HH (hodgkin52) | Why Missing in HH |
|---------|--------------------|-----------------------|-------------------|
| Firing rate adaptation | Yes (mAHP) | No | No Ca²⁺ channels or Ca²⁺-dependent K⁺ |
| Plateau potentials | Yes (NaPs, L_Ca) | No | No persistent Na⁺ or L-type Ca²⁺ |
| Bistability | Yes (slow inactivation) | No | Only fast h-gate inactivation |
| Sag response | Yes (Gh) | No | No hyperpolarization-activated (Ih) channels |
| PICs | Yes | No | No persistent inward currents |

**Note**: jaxley-mech provides other channel types beyond HH (e.g., Ca²⁺ channels, KCa channels from other modules) that could be used to add these features if needed.

---

## Critical Adaptation Factors

### 1. Current Amplitude

**Problem**: HH channels have different current-frequency relationships than motor neuron channels. Specifically, HH channels **lack firing rate adaptation** (no mAHP), so they fire faster for the same current.

**Solution**:
- NEURON uses **15 nA**
- Jaxley uses **2 nA**

The motor neuron model has adaptation mechanisms (mAHP, slow Na⁺ inactivation) that limit firing rate. The standard HH model fires ~3-4x faster at the same current level because it lacks these adaptation mechanisms. To achieve similar spike counts (~6000 spikes/pool at ~25 Hz), we must use significantly lower current.

### 2. Spike Detection Threshold

**Problem**: Action potential peaks differ significantly between models.

**Solution**:
- NEURON: **50 mV** threshold (motor neuron APs reach 80-100 mV)
- Jaxley: **0 mV** threshold (HH APs reach ~30-40 mV)

**Why can't we use 50 mV for Jaxley?**

The standard Hodgkin-Huxley model has a sodium reversal potential of `eNa = 50 mV`. The action potential peak **cannot exceed** the sodium reversal potential. Typical HH action potentials peak at only **30-40 mV**.

If you set the threshold to 50 mV in Jaxley, you would detect **zero spikes** because the voltage never reaches 50 mV!

| Model | Na Reversal (eNa) | AP Peak | Appropriate Threshold |
|-------|-------------------|---------|----------------------|
| NEURON motor neuron | 60 mV | 80-100 mV | 50 mV |
| Jaxley HH | 50 mV | 30-40 mV | 0 mV |

**Important**: The threshold difference does NOT cause different spike counts. Both thresholds are set appropriately to detect the same physiological events (action potentials) - they're just calibrated for each model's AP amplitude. A 0 mV threshold is well above the resting potential (-65 mV) and is only crossed during actual action potentials.

### 3. Cell State Initialization

**Critical**: Before each simulation, you MUST call:
```python
cell.set("v", -65.0)      # Reset voltage to resting potential
cell.init_states()        # Initialize gating variables to steady-state
```

**Why**: Without `init_states()`, the HH gating variables (m, h, n) may not be at their equilibrium values, causing transient firing at simulation start.

### 4. Recruitment Threshold Handling

**Implementation**:
```python
max_recruitment = 0.80  # Neurons above this threshold don't fire

# Scale current by recruitment threshold
scale_factor = 1.0 - (neuron_threshold / max_recruitment) * 0.3
scaled_current = pool_current * scale_factor
```

- Neurons with threshold > 0.80 are not recruited (no simulation)
- Lower threshold neurons receive more current (scale 1.0 → 0.7)
- This implements the size principle: small motor units fire first

### 5. Timestep Requirements

**Jaxley HH** requires finer timestep (0.025 ms) than NEURON (0.05 ms) for numerical stability of the explicit integration method.

### 6. Clearing Cell State Between Simulations

**Required** for each neuron:
```python
cell.delete_recordings()
cell.delete_stimuli()
```

This clears previous simulation state before setting up new recordings and stimuli.

---

## Expected Output Comparison

### Typical Results (with current_amplitude = 2.0 nA)

| Metric | NEURON | Jaxley |
|--------|--------|--------|
| Active neurons per pool | ~94/100 | ~94/100 |
| Total spikes per pool | ~5500-6500 | ~5000-6500 |
| Mean firing rate | ~20-30 Hz | ~20-30 Hz |
| Max firing rate | ~40-50 Hz | ~35-45 Hz |

### Qualitative Differences

1. **Firing Rate Adaptation**: NEURON model shows adaptation (decreasing rate during plateau) due to mAHP. HH model fires at constant rate.

2. **Recruitment Pattern**: Both show graded recruitment following the trapezoid current, with low-threshold units firing first.

3. **Action Potential Shape**: HH has narrower, smaller APs. Motor neuron model has larger, more complex APs.

### Why Spike Counts May Differ

The spike count differences between NEURON and Jaxley are **NOT** due to the spike detection threshold (see section above). The actual causes are:

| Factor | Effect on Jaxley Spike Count |
|--------|------------------------------|
| **No firing rate adaptation** | Higher - HH lacks mAHP (Ca²⁺-dependent K⁺), so neurons fire at constant rate instead of slowing down during sustained input |
| **No slow inactivation** | Higher - HH Na⁺ channels recover faster between spikes, allowing higher sustained firing rates |
| **Current amplitude tuning** | Variable - 6 nA may produce slightly different rates than intended |
| **No persistent currents** | Lower - HH lacks NaPs and L-type Ca²⁺, reducing subthreshold excitability |

The net effect depends on the balance of these factors. To match NEURON spike counts (~6000 spikes/pool), use `current_amplitude = 2.0 nA`. Higher values (5-7 nA) result in excessive firing rates (~80+ Hz) due to the lack of adaptation.

---

## File Locations

| File | Description |
|------|-------------|
| `02_simulate_spike_trains_current_injection.py` | NEURON implementation |
| `02_simulate_spike_trains_current_injection_jaxley_mech.py` | Jaxley implementation |
| `results/spike_train__Block_utility.pkl` | NEURON spike train output |
| `results/spike_train__Block_utility_jaxley.pkl` | Jaxley spike train output |
| `results/spike_trains_jaxley_mech.png` | Jaxley raster plot |

---

## Summary

The Jaxley implementation successfully replicates the core functionality of Example 2 using standard Hodgkin-Huxley channels from jaxley-mech. While the biophysics differ (no adaptation, no PICs, no bistability), the implementation produces comparable spike train patterns that follow the trapezoid input current.

**Key adaptations required**:
1. Reduced current amplitude (2 nA vs 15 nA) due to lack of adaptation
2. Lower spike detection threshold (0 mV vs 50 mV)
3. Finer timestep (0.025 ms vs 0.05 ms)
4. Proper cell initialization (`init_states()`)
5. Recruitment threshold handling for size principle

For applications requiring motor neuron-specific features (adaptation, PICs, bistability), the custom MyoGen channels should be used instead of jaxley-mech HH channels.

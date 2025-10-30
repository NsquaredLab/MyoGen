# Direct Current Injection - CoV Control Guide

## Overview

This script (`extract_isi_cv_direct_current.py`) drives motor neurons with **direct current injection** instead of using a descending drive network. This gives you **precise, independent control** over:

- **Firing Rate**: Controlled by current amplitude
- **CoV (firing variability)**: Controlled by current noise
- **Recruitment**: Based on recruitment thresholds

### Key Advantage
Eliminates gamma process variability from DD neurons, giving the cleanest control over CoV.

---

## Quick Start

### Single Run (Test)
```bash
cd examples/finetune

python extract_isi_cv_direct_current.py \
    --muscle VLVM \
    --mvc-level 30 \
    --current-noise-std 0.1 \
    --base-current 5.0 \
    --current-range 10.0
```

### Parameter Sweep (Find Optimal Noise)
```bash
cd /home/oj98yqyk/code/simulators/MyoGen
bash bash_scripts/run_direct_current_cv_sweep.sh
```

This will test noise levels: 0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0 nA

---

## Parameters Explained

### `--current-noise-std` (Controls CoV) ⭐
- **What it does**: Standard deviation of Gaussian noise added to current
- **Effect on CoV**:
  - Lower → More regular firing → **Lower CoV**
  - Higher → More irregular firing → **Higher CoV**
- **Typical values**:
  - `0.0` - Minimal CoV (~0.05-0.10, very regular)
  - `0.05` - Low CoV (~0.10-0.15)
  - `0.1` - Moderate CoV (~0.15-0.25)
  - `0.5` - High CoV (~0.30-0.40)
  - `1.0` - Very high CoV (~0.40+)

### `--base-current` (Controls Firing Rate)
- **What it does**: Baseline current amplitude (nA)
- **Effect**: Higher → Higher firing rates
- **Typical range**: 3.0 - 10.0 nA
- **Default**: 5.0 nA

### `--current-range` (Controls Recruitment Gradient)
- **What it does**: Current difference between early and late recruited MUs
- **Effect**: Larger → More spread in firing rates across pool
- **Typical range**: 5.0 - 20.0 nA
- **Default**: 10.0 nA

Example:
- `base_current=5.0, current_range=10.0`
- Early MUs get: 5.0 + 10.0 = 15.0 nA
- Late MUs get: 5.0 nA

---

## Expected Results

### Output Files
All saved to `./results/`:

1. **Spike trains**: `*_direct_current_spike_trains_*.pkl`
2. **ISI/CV data**: `*_isi_cv_direct_*.csv`
3. **Verification plot**: `*_direct_current_verification_*.png`

### Typical CoV Values by Noise Level

| Noise Std (nA) | Expected CoV | Regularity |
|----------------|--------------|------------|
| 0.0            | 0.05-0.10    | Clock-like |
| 0.05           | 0.10-0.15    | Very regular |
| 0.1            | 0.15-0.25    | Regular |
| 0.2            | 0.25-0.35    | Moderate |
| 0.5            | 0.35-0.45    | Irregular |
| 1.0            | 0.45+        | Very irregular |

---

## Optimization Strategy

### Goal: Match Target FR and CV

1. **Step 1**: Run sweep to find noise level that gives desired CoV
   ```bash
   bash bash_scripts/run_direct_current_cv_sweep.sh
   ```

2. **Step 2**: Check `results/direct_current_sweep/noise_sweep_results.csv`

3. **Step 3**: If firing rate is off, adjust `--base-current`:
   - Too low: Increase base current
   - Too high: Decrease base current

4. **Step 4**: If firing rate spread is wrong, adjust `--current-range`:
   - Too narrow: Increase range
   - Too wide: Decrease range

### Example Workflow

**Target**: 16.8 ± 2.5 Hz with CoV ≈ 0.20

```bash
# Initial test
python extract_isi_cv_direct_current.py --current-noise-std 0.1

# Check results → FR = 14.2 Hz, CoV = 0.18
# Need higher FR, similar CoV

# Adjust base current
python extract_isi_cv_direct_current.py \
    --current-noise-std 0.1 \
    --base-current 6.0  # Increased from 5.0

# Check results → FR = 16.5 Hz, CoV = 0.19
# Perfect! Save this configuration
```

---

## Comparison with DD Network Approach

| Feature | Direct Current | DD Network |
|---------|---------------|------------|
| **CoV control** | Direct (noise std) | Indirect (gamma shape) |
| **Min achievable CoV** | ~0.05 | ~0.18 (gamma=25) |
| **Simulation speed** | Fast | Slower |
| **Complexity** | Simple | Complex |
| **Biological realism** | Lower | Higher |
| **Parameter count** | 3 | 6+ |

**Use direct current when**:
- You need very low CoV (<0.20)
- You want independent control of FR and CoV
- Speed matters
- You're testing/debugging

**Use DD network when**:
- Biological realism is critical
- Studying network dynamics
- CoV > 0.20 is acceptable

---

## Troubleshooting

### Problem: CoV too high even with noise=0
**Cause**: Intrinsic biophysical noise (ion channels)
**Solution**: This is the minimum CoV for your neuron model (~0.05-0.10)

### Problem: Some neurons not firing
**Cause**: Current too low or current range too large
**Solutions**:
- Increase `--base-current`
- Decrease `--current-range`
- Check that base_current > recruitment threshold

### Problem: Firing rates too variable across pool
**Cause**: Current range too large
**Solution**: Decrease `--current-range`

### Problem: All neurons fire at same rate
**Cause**: Current range too small
**Solution**: Increase `--current-range`

---

## Advanced: Custom Current Waveforms

To create custom temporal patterns, modify line 125 in `extract_isi_cv_direct_current.py`:

```python
# Example: Sinusoidal modulation
trapezoid_current[i] = mu_current_amplitude * (
    1.0 + 0.2 * np.sin(2 * np.pi * t / 1000.0)  # 1 Hz modulation
)

# Example: Step function
trapezoid_current[i] = mu_current_amplitude if t > 500 else 0

# Example: Exponential ramp
progress = (t - trapezoid_start) / RAMP_UP_DURATION__ms
trapezoid_current[i] = mu_current_amplitude * (1 - np.exp(-3 * progress))
```

---

## References

- Current injection: NEURON `IClamp` mechanism
- Recruitment thresholds: `myogen.simulator.RecruitmentThresholds`
- Motor neuron model: `AlphaMN` (Powers et al. 2017)

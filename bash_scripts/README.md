# Bash Scripts for MyoGen

This directory contains bash scripts for running common MyoGen tasks and optimizations.

## 🚀 Quick Start: Full Pipeline

The **recommended approach** is to use `run_full_pipeline.sh` which automatically runs all 5 steps and correctly propagates gamma shape values:

```bash
# High CV input (gamma shape 0.5-0.75, CV ≈ 1.15-1.41)
./bash_scripts/run_full_pipeline.sh --gamma-shape-min 0.5 --gamma-shape-max 0.75

# Regular CV input (gamma shape 2.0-3.0, CV ≈ 0.58-0.71)
./bash_scripts/run_full_pipeline.sh --gamma-shape-min 2.0 --gamma-shape-max 3.0

# With Gfluctdv membrane noise enabled
./bash_scripts/run_full_pipeline.sh --gamma-shape-min 0.5 --gamma-shape-max 0.75 --enable-gfluctdv

# Skip already-completed steps (faster reruns)
./bash_scripts/run_full_pipeline.sh --gamma-shape-min 0.5 --gamma-shape-max 0.75 --skip-dd-optimization
```

**Pipeline steps:**
1. DD optimization for all firing rates (THIRTY, TWENTYFIVE, TWENTY, FIFTEEN, TEN, FIVE)
2. Force computation at MVC for each muscle
3. Force optimization at multiple force levels (5%, 15%, 30%, 50% MVC)
4. ISI and CV extraction
5. Multi-muscle comparison plots

**Options:**
- `--gamma-shape-min VALUE` - Gamma shape minimum (required)
- `--gamma-shape-max VALUE` - Gamma shape maximum (required)
- `--enable-gfluctdv` - Enable Gfluctdv noise mechanism for motor neurons
- `--skip-dd-optimization` - Skip DD optimization (use existing results)
- `--skip-force-computation` - Skip force computation
- `--skip-force-optimization` - Skip force optimization
- `--skip-isi-extraction` - Skip ISI/CV extraction
- `--skip-plotting` - Skip final plotting
- `--output-format FORMAT` - Plot format (jpg, png, pdf, svg; default: jpg)

---

## Available Scripts

### `run_dd_optimization.sh`

Multi-objective optimization script for descending drive parameters to match target motor neuron firing rates.

**Purpose**: Automatically tunes descending drive network parameters (number of neurons, connection probability, drive frequency, gamma shape) to achieve physiologically realistic motor neuron firing rates for different muscles.

**Usage**:
```bash
# Run from project root
./bash_scripts/run_dd_optimization.sh [muscle_type]
```

**Available muscle types**:
- `VLVM` - Vastus Lateralis/Vastus Medialis (default, 16.8±2.5 Hz)
- `FDI` - First Dorsal Interosseous (12.0±3.0 Hz)
- `TA` - Tibialis Anterior (14.0±2.8 Hz)
- `CUSTOM` - Custom parameters (edit script to modify)
- `TEST` - Quick test with 10 trials

**Examples**:
```bash
# Optimize for VLVM muscle (default)
./bash_scripts/run_dd_optimization.sh VLVM

# Optimize for FDI muscle
./bash_scripts/run_dd_optimization.sh FDI

# Quick test run (10 trials)
./bash_scripts/run_dd_optimization.sh TEST

# Custom optimization (edit script first)
./bash_scripts/run_dd_optimization.sh CUSTOM
```

**Requirements**:
- MyoGen virtual environment activated (`.venv`)
- NEURON mechanisms compiled (run `python -c "from myogen.utils.nmodl import load_nmodl_mechanisms; load_nmodl_mechanisms()"` first)
- Sufficient computational resources (each trial runs a 5-second NEURON simulation)

**Output**:
Results are saved to `results/dd_optimization/`:
- `{MUSCLE}_dd_optimized_params.json` - Optimized parameters (best FR, best balanced, Pareto front)
- `{MUSCLE}_optuna_dd_optimization.db` - SQLite database with full optimization history
- `{MUSCLE}_study.pkl` - Pickled Optuna study object for visualization

**Optimization Parameters**:
The script optimizes these parameters simultaneously:
- `dd_neurons` - Number of descending drive neurons (100-1000)
- `conn_prob` - Connection probability to motor neurons (0.1-1.0)
- `dd_drive` - Drive frequency in Hz (5.0-250.0)
- `mvc_shape_value` - Gamma distribution shape parameter (7.0-20.0)

**Objective Functions**:
1. **Firing rate error** - Weighted error between simulated and target mean/std
2. **Connection probability deviation** - Distance from target connection probability
3. **Neuron count deviation** - Distance from target number of DD neurons

**Typical Runtime**:
- 100 trials: ~30-60 minutes (depending on hardware)
- 10 trials (TEST): ~3-6 minutes

**Customization**:
To add your own muscle configuration, edit the configuration section at the top of the script with your target parameters:
```bash
--target-fr-mean 20.0        # Target mean firing rate (Hz)
--target-fr-std 3.5          # Target firing rate std dev (Hz)
--target-conn-prob 0.35      # Target connection probability
--target-n-dd-neurons 500    # Target number of DD neurons
--n-trials 100               # Number of optimization trials
--n-motor-units 100          # Number of motor units to simulate
```

---

### `run_force_validation.sh`

Validates force output using optimized descending drive parameters.

**Purpose**: Runs a 10-second NEURON simulation with optimized DD parameters and computes resulting muscle force using the Fuglevand force model. Validates that optimized parameters produce physiologically realistic force output.

**Usage**:
```bash
# Run from project root
./bash_scripts/run_force_validation.sh [muscle_type]
```

**Available muscle types**:
- `VLVM` - Vastus Lateralis/Vastus Medialis (default)
- `FDI` - First Dorsal Interosseous
- `TA` - Tibialis Anterior
- Any muscle type for which you have run optimization

**Examples**:
```bash
# Validate force for VLVM muscle
./bash_scripts/run_force_validation.sh VLVM

# Run in parallel after optimization completes
./bash_scripts/run_dd_optimization.sh VLVM && ./bash_scripts/run_force_validation.sh VLVM &
./bash_scripts/run_dd_optimization.sh FDI && ./bash_scripts/run_force_validation.sh FDI &
wait
```

**Requirements**:
- MyoGen virtual environment activated (`.venv`)
- NEURON mechanisms compiled
- **Optimized DD parameters must exist** (run `run_dd_optimization.sh` first)

**Input**:
Reads from `results/dd_optimization/`:
- `{MUSCLE}_dd_optimized_params.json` - Optimized DD parameters from optimization

**Output**:
Results saved to `results/force_validation/`:
- `{MUSCLE}_force_results.json` - Force statistics and firing rate validation

**Typical Runtime**:
- ~2-5 minutes per muscle (10-second simulation)

**Output Format**:
```json
{
  "dd_parameters": {...},
  "firing_rate": {
    "mean__Hz": 16.5,
    "std__Hz": 2.3,
    "n_active": 98
  },
  "force": {
    "mean__au": 0.1234,
    "std__au": 0.0056,
    "cov": 0.045
  }
}
```

---

## Script Development Guidelines

When adding new bash scripts to this directory:

1. **Follow bash best practices**:
   - Use `set -euo pipefail` for error handling
   - Add proper shebang: `#!/usr/bin/env bash`
   - Include comprehensive comments and documentation
   - Use color-coded logging functions

2. **Environment checks**:
   - Verify virtual environment is activated
   - Check for required dependencies
   - Validate working directory

3. **Make scripts executable**:
   ```bash
   chmod +x bash_scripts/your_script.sh
   ```

4. **Document in this README**:
   - Add script description
   - Provide usage examples
   - List requirements and outputs

5. **Use descriptive naming**:
   - Prefix with action verb: `run_`, `generate_`, `analyze_`
   - Use snake_case: `run_dd_optimization.sh`

## Quick Reference

### Activate Environment
```bash
source .venv/bin/activate
```

### Compile NEURON Mechanisms
```bash
python -c "from myogen.utils.nmodl import load_nmodl_mechanisms; load_nmodl_mechanisms()"
```

### Check Results
```bash
# View optimization results
cat results/dd_optimization/VLVM_dd_optimized_params.json | python -m json.tool

# View force validation results
cat results/force_validation/VLVM_force_results.json | python -m json.tool

# List all optimization databases
ls -lh results/dd_optimization/*.db
```

### Run Full Pipeline (Optimize + Validate)
```bash
# Sequential: optimize then validate
./bash_scripts/run_dd_optimization.sh VLVM
./bash_scripts/run_force_validation.sh VLVM

# Multiple muscles in parallel
for muscle in VLVM FDI TA; do
    ./bash_scripts/run_dd_optimization.sh "$muscle" && \
    ./bash_scripts/run_force_validation.sh "$muscle" &
done
wait
```

## Troubleshooting

### Virtual environment not found
```bash
# Create environment with UV
uv sync
source .venv/bin/activate
```

### NEURON mechanisms not compiled
```bash
# Compile mechanisms
python -c "from myogen.utils.nmodl import load_nmodl_mechanisms; load_nmodl_mechanisms()"
```

### Script permission denied
```bash
# Make script executable
chmod +x bash_scripts/run_dd_optimization.sh
```

### Optimization fails with errors
- Check that NEURON is properly installed: `python -c "import neuron; print(neuron.h)"`
- Verify all dependencies: `uv sync`
- Try a quick test first: `./bash_scripts/run_dd_optimization.sh TEST`

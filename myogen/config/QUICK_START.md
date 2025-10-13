# Quick Start: Configuration Files

## TL;DR

```python
from myogen.simulator.neuron.populations import AlphaMN__Pool

# Use default config (automatic)
pool = AlphaMN__Pool(n=100)

# Use custom config
pool = AlphaMN__Pool(n=100, config_file="my_config.yaml")

# Override specific parameters
pool = AlphaMN__Pool(n=100, config_file="my_config.yaml", gamma=0.5)
```

## Common Use Cases

### 1. Default Configuration

```python
# Just specify the number of neurons
pool = AlphaMN__Pool(n=100)
```

### 2. Custom Configuration

```python
# Create my_config.yaml with your parameters
pool = AlphaMN__Pool(n=100, config_file="my_config.yaml")
```

### 3. Quick Parameter Override

```python
# Use default config but change gamma
pool = AlphaMN__Pool(n=100, gamma=0.4)
```

### 4. Mix Config and Overrides

```python
# Load config, then override specific values
pool = AlphaMN__Pool(
    n=100,
    config_file="base_config.yaml",
    gamma=0.5,  # Override
    axon_length=0.8  # Override
)
```

## Creating a Custom Config

### Step 1: Copy the example

```bash
cp myogen/config/alpha_mn_custom_example.yaml myogen/config/my_experiment.yaml
```

### Step 2: Edit your parameters

```yaml
# my_experiment.yaml
model: "Powers2017"
gamma: 0.4
axon_velocities: [45, 70]
```

### Step 3: Use it

```python
pool = AlphaMN__Pool(n=100, config_file="my_experiment.yaml")
```

## Available Configs

- `alpha_mn_default.yaml` - Default parameters (NERLab model)
- `alpha_mn_custom_example.yaml` - Example custom config (Powers2017 model)

## Key Parameters

```yaml
model: "NERLab"  # or "Powers2017"
mode: "active"  # or "passive"
gamma: 0.2  # neuromodulation level
axon_velocities: [50, 65]  # [min, max] m/s
axon_length: 0.6  # mm
lambda_factor: 1.0  # Powers2017 only
initial_voltage__mV: -67
spike_threshold__mV: 50.0
```

## Tips

✓ Only specify parameters you want to change
✓ Add comments to document your choices
✓ Keep configs in version control
✓ Use descriptive filenames

## Need More Help?

- Full guide: `myogen/config/README.md`
- Examples: `examples/config_usage_example.py`
- All parameters: `myogen/config/alpha_mn_default.yaml`


# MyoGen Configuration Files

This directory contains YAML configuration files for various MyoGen components.

## Overview

Configuration files allow you to define model parameters in a structured, reusable way instead of hardcoding them in your scripts. This makes it easier to:

- Share parameter sets across different simulations
- Version control your experimental configurations
- Document parameter choices
- Quickly switch between different parameter sets

## Available Configurations

### Alpha Motor Neuron Pool

- **`alpha_mn_default.yaml`**: Default parameters for alpha motor neuron populations
- **`alpha_mn_custom_example.yaml`**: Example showing how to create custom configurations

## Usage

### Using Default Configuration

By default, `AlphaMN__Pool` uses the parameters from `alpha_mn_default.yaml`:

```python
from myogen.simulator.neuron.populations import AlphaMN__Pool

# Uses default configuration automatically
pool = AlphaMN__Pool(n=100)
```

### Using a Custom Configuration File

You can specify a custom configuration file in three ways:

#### 1. By filename (searches in myogen/config/)

```python
pool = AlphaMN__Pool(
    n=100,
    config_file="alpha_mn_custom_example.yaml"
)
```

#### 2. By relative path

```python
pool = AlphaMN__Pool(
    n=100,
    config_file="./my_configs/experiment1.yaml"
)
```

#### 3. By absolute path

```python
from pathlib import Path

config_path = Path("/path/to/my/config.yaml")
pool = AlphaMN__Pool(
    n=100,
    config_file=config_path
)
```

### Overriding Specific Parameters

You can override any parameter from the config file by passing it explicitly:

```python
pool = AlphaMN__Pool(
    n=100,
    config_file="alpha_mn_default.yaml",
    gamma=0.5,  # Override gamma from config
    axon_length=0.8,  # Override axon_length from config
)
```

Parameters passed explicitly always take precedence over config file values.

### Creating Custom Configuration Files

1. Copy `alpha_mn_custom_example.yaml` as a starting point
2. Modify the parameters you want to change
3. Save with a descriptive name (e.g., `alpha_mn_high_gamma.yaml`)
4. Use in your simulations

You only need to specify the parameters you want to change - all others will use default values.

## Configuration File Structure

### Alpha Motor Neuron Configuration

```yaml
# General parameters
model: "NERLab"  # or "Powers2017"
mode: "active"  # or "passive"
axon_velocities: [50, 65]  # [min, max] in m/s
axon_length: 0.6  # in mm
gamma: 0.2  # neuromodulation level
lambda_factor: 1.0  # for Powers2017 model
initial_voltage__mV: -67
spike_threshold__mV: 50.0

# Powers2017 specific parameters
powers2017:
  soma:
    length_range: [min, max, curve]
    diameter_range: [min, max, curve]
    # ... other soma parameters
    
  dendrite:
    length_range: [min, max, curve]
    diameter_range: [min, max, curve]
    ca_conductance_ranges:
      - [min, max, curve]  # Dendrite 1
      - [min, max, curve]  # Dendrite 2
      - [min, max, curve]  # Dendrite 3
      - [min, max, curve]  # Dendrite 4
    # ... other dendrite parameters
```

## Parameter Precedence

Parameters are resolved in the following order (highest to lowest priority):

1. **Explicitly passed parameters** to `AlphaMN__Pool.__init__()`
2. **Custom config file** (if specified)
3. **Default config file** (`alpha_mn_default.yaml`)
4. **Hardcoded defaults** in the code (fallback if config loading fails)

## Requirements

Configuration file loading requires PyYAML:

```bash
pip install pyyaml
```

This is automatically installed as a dependency of MyoGen.

## Tips

- Use descriptive filenames for your configs (e.g., `alpha_mn_low_threshold.yaml`)
- Add comments in your YAML files to document parameter choices
- Keep custom configs in version control alongside your analysis scripts
- Create separate configs for different experimental conditions
- Use the example file as a template for new configurations

## Troubleshooting

### FileNotFoundError

If you get a `FileNotFoundError`, check:
- The filename is spelled correctly
- The file exists in the expected location
- You're using the correct path (relative vs absolute)

### ImportError: PyYAML not installed

Install PyYAML:
```bash
pip install pyyaml
```

### Invalid YAML syntax

Use a YAML validator or linter to check your configuration file syntax.
Common issues:
- Incorrect indentation (use spaces, not tabs)
- Missing colons after keys
- Incorrect list syntax


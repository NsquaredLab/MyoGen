"""
Configuration file loader for MyoGen.

This module provides utilities for loading YAML configuration files
for various components of the MyoGen simulator.
"""

from pathlib import Path
from typing import Any, Union

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file. Can be:
        - An absolute path
        - A relative path (relative to current working directory)
        - A filename (will search in myogen/config directory)

    Returns
    -------
    dict
        Dictionary containing the configuration parameters.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the configuration file cannot be found.
    ValueError
        If the YAML file is invalid or empty.

    Examples
    --------
    >>> # Load default configuration
    >>> config = load_yaml_config("alpha_mn_default.yaml")
    >>>
    >>> # Load custom configuration with absolute path
    >>> config = load_yaml_config("/path/to/my_config.yaml")
    >>>
    >>> # Load custom configuration with relative path
    >>> config = load_yaml_config("./configs/custom_alpha_mn.yaml")
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load configuration files. Install it with: pip install pyyaml"
        )

    config_path = Path(config_path)

    # If path is not absolute and doesn't exist, try to find it in myogen/config
    if not config_path.is_absolute() and not config_path.exists():
        # Get the myogen package directory
        myogen_dir = Path(__file__).parent.parent
        config_dir = myogen_dir / "config"

        # Try to find the file in the config directory
        potential_path = config_dir / config_path
        if potential_path.exists():
            config_path = potential_path
        else:
            # If still not found, raise an error with helpful message
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Searched in:\n"
                f"  - {config_path.absolute()}\n"
                f"  - {potential_path}\n"
                f"Available configs in {config_dir}:\n"
                f"  {list(config_dir.glob('*.yaml')) if config_dir.exists() else 'None'}"
            )

    # Load the YAML file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {config_path}\nError: {e}")

    if config is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    return config


def get_default_config_path(component: str) -> Path:
    """
    Get the path to a default configuration file.

    Parameters
    ----------
    component : str
        Name of the component (e.g., "alpha_mn", "muscle", etc.)

    Returns
    -------
    Path
        Path to the default configuration file.

    Examples
    --------
    >>> path = get_default_config_path("alpha_mn")
    >>> config = load_yaml_config(path)
    """
    myogen_dir = Path(__file__).parent.parent
    config_dir = myogen_dir / "config"
    return config_dir / f"{component}_default.yaml"


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """
    Merge two configuration dictionaries, with override_config taking precedence.

    Parameters
    ----------
    base_config : dict
        Base configuration dictionary.
    override_config : dict
        Configuration dictionary with values to override.

    Returns
    -------
    dict
        Merged configuration dictionary.

    Examples
    --------
    >>> base = {"a": 1, "b": {"c": 2, "d": 3}}
    >>> override = {"b": {"c": 10}, "e": 4}
    >>> merged = merge_configs(base, override)
    >>> merged
    {'a': 1, 'b': {'c': 10, 'd': 3}, 'e': 4}
    """
    result = base_config.copy()

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result

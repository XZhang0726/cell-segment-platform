"""
Configuration management.

Provides configuration file loading and management.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from .paths import get_configs_dir


class Config:
    """Configuration container."""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize the configuration.

        Args:
            config_dict: Configuration dictionary.
        """
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key; dotted keys such as 'model.name' access nested values.
            default: Default value.

        Returns:
            Configuration value.
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value.

        Args:
            key: Configuration key.
            value: Configuration value.
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return self._config.copy()

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            yaml_file: Path to the YAML file.

        Returns:
            Config instance.
        """
        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")

        with open(yaml_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def save_yaml(self, yaml_file: Path):
        """
        Save configuration to a YAML file.

        Args:
            yaml_file: Path to the YAML file.
        """
        yaml_file = Path(yaml_file)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)


def load_config(config_name: str) -> Config:
    """
    Load a configuration file.

    Args:
        config_name: Configuration filename without an extension, or a full path.

    Returns:
        Config instance.
    """
    config_path = Path(config_name)

    # Try the supplied path first.
    if config_path.exists():
        return Config.from_yaml(config_path)

    # For a configuration name, search the configs directory.
    config_path = get_configs_dir() / f"{config_name}.yaml"
    if config_path.exists():
        return Config.from_yaml(config_path)

    # Return an empty configuration if no file is found.
    return Config()

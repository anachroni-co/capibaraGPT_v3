import os
import sys
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Import TOML library
try:
    import tomli
except ImportError:
    try:
        import tomllib as tomli
    except ImportError:
        # Fallback for systems without tomli
        import json
        class TomliFallback:
            @staticmethod
            def load(f):
                # Simple fallback - try JSON if there is no TOML
                return json.load(f)
        tomli = TomliFallback()

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Dict[str, Any]] = {}
        
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load a configuration from a TOML file."""
        try:
            config_path = self.config_dir / f"{name}.toml"
            with open(config_path, "rb") as f:
                config = tomli.load(f)
            self.configs[name] = config
            logger.info(f"Configuration {name} loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration {name}: {str(e)}")
            raise

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets a loaded configuration."""
        return self.configs.get(name)

    def get_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Gets a specific value from a configuration."""
        config = self.get_config(config_name)
        if not config:
            return default

        try:
            value = config
            for key in key_path.split("."):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def reload_all(self) -> None:
        """Reloads all configurations."""
        for name in self.configs:
            self.load_config(name)

    def validate_config(self, name: str, schema: Dict[str, Any]) -> bool:
        """Validates a configuration against a schema."""
        config = self.get_config(name)
        if not config:
            return False

        try:
            self._validate_schema(config, schema)
            return True
        except ValueError as e:
            logger.error(f"Validation error in {name}: {str(e)}")
            return False

    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Recursively validates a configuration schema."""
        for key, expected_type in schema.items():
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

            if isinstance(expected_type, dict):
                if not isinstance(config[key], dict):
                    raise ValueError(f"Incorrect type for {key}")
                self._validate_schema(config[key], expected_type)
            elif not isinstance(config[key], expected_type):
                raise ValueError(f"Incorrect type for {key}") 
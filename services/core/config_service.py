"""
Configuration Service for FLUX-Pro-Finetuning-UI.

Provides centralized configuration management with environment variable support,
validation, and secure handling of sensitive information.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigService:
    """
    Service for managing application configuration.
    
    Handles loading configuration from files and environment variables,
    with validation and secure handling of sensitive information.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the configuration service.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.example_path = Path("config/config.example.json")
        self.config: Dict[str, Any] = {}
        
        # Ensure config directory exists
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Create example config if it doesn't exist
        if not self.example_path.exists():
            self._create_example_config()
            
        # Load configuration
        self._load_config()
    
    def _create_example_config(self) -> None:
        """Create the example configuration file."""
        example_config = {
            "api_key": "your-api-key-here",
            "api_host": "api.us1.bfl.ai",
            "storage": {
                "models_dir": "data",
                "images_dir": "generated_images"
            }
        }
        
        with open(self.example_path, 'w') as f:
            json.dump(example_config, f, indent=4)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration file.
        
        Returns:
            The loaded configuration
            
        Raises:
            FileNotFoundError: If configuration file is not found
            ValueError: If configuration is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}. "
                f"Please copy {self.example_path} to {self.config_path} "
                "and update with your settings."
            )
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self._validate_config()
        return self.config
    
    def _validate_config(self) -> None:
        """
        Validate the loaded configuration.
        
        Raises:
            ValueError: If required fields are missing or invalid
            TypeError: If fields have incorrect types
        """
        required_fields = {
            "api_key": str,
            "api_host": str,
            "storage": dict
        }
        
        for field, field_type in required_fields.items():
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(self.config[field], field_type):
                raise TypeError(
                    f"Field {field} must be of type {field_type.__name__}"
                )
        
        # Check if API key is the default value
        if self.config["api_key"] == "your-api-key-here":
            # Check environment variable as fallback
            env_api_key = os.environ.get("FLUX_API_KEY")
            if env_api_key:
                self.config["api_key"] = env_api_key
            else:
                raise ValueError(
                    "Please update config.json with your actual API key "
                    "or set the FLUX_API_KEY environment variable."
                )
        
        # Ensure storage paths exist
        storage = self.config["storage"]
        for dir_name in storage.values():
            # Create parent directory first if needed
            Path(dir_name).mkdir(exist_ok=True)
    
    def get_api_key(self) -> str:
        """
        Get the API key from config or environment variable.
        
        Returns:
            The API key
        """
        # Check environment variable first
        env_api_key = os.environ.get("FLUX_API_KEY")
        if env_api_key:
            return env_api_key
            
        return self.config.get("api_key", "")
    
    def get_api_host(self) -> str:
        """
        Get the API host from config or environment variable.
        
        Returns:
            The API host
        """
        # Check environment variable first
        env_host = os.environ.get("FLUX_API_HOST")
        if env_host:
            return env_host
            
        return self.config.get("api_host", "api.us1.bfl.ai")
    
    def get_storage_path(self, key: str) -> Path:
        """
        Get a storage path from config.
        
        Args:
            key: The storage key to retrieve
            
        Returns:
            The storage path as a Path object
        """
        storage = self.config.get("storage", {})
        return Path(storage.get(key, key))
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        # Check for environment variable with FLUX_ prefix
        env_key = f"FLUX_{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
            
        # Navigate nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
            
        return self.config.get(key, default)
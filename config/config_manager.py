import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    def __init__(self):
        self.config_dir = Path("config")
        self.config_file = self.config_dir / "config.json"
        self.example_file = self.config_dir / "config.example.json"
        self.config: Dict[str, Any] = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        # Create example config if it doesn't exist
        if not self.example_file.exists():
            self._create_example_config()
    
    def _create_example_config(self):
        """Create the example configuration file."""
        example_config = {
            "api_key": "your-api-key-here",
            "api_host": "api.us1.bfl.ai",
            "storage": {
                "models_dir": "data",
                "images_dir": "generated_images"
            }
        }
        
        with open(self.example_file, 'w') as f:
            json.dump(example_config, f, indent=4)
    
    def load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_file}. "
                f"Please copy {self.example_file} to {self.config_file} "
                "and update with your settings."
            )
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        self._validate_config()
        return self.config
    
    def _validate_config(self):
        """Validate the loaded configuration."""
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
        
        if self.config["api_key"] == "your-api-key-here":
            raise ValueError(
                "Please update config.json with your actual API key"
            )
        
        # Ensure storage paths exist
        storage = self.config["storage"]
        for dir_name in storage.values():
            # Create parent directory first if needed
            Path(dir_name).mkdir(exist_ok=True)
    
    def get_api_key(self) -> str:
        """Get the API key from config."""
        return self.config.get("api_key", "")
    
    def get_api_host(self) -> str:
        """Get the API host from config."""
        return self.config.get("api_host", "api.us1.bfl.ai")
    
    def get_storage_path(self, key: str) -> Path:
        """Get a storage path from config."""
        storage = self.config.get("storage", {})
        return Path(storage.get(key, key))
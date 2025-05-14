"""Tests for configuration manager utility."""

import unittest
import json
import os
from pathlib import Path
import shutil
from utils.config.config_manager import ConfigManager, ConfigError

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config_dir = "test_config"
        self.config_path = Path(self.test_config_dir) / "config.json"
        self.env_config_path = Path(self.test_config_dir) / "config.test.json"

        # Create test directory
        Path(self.test_config_dir).mkdir(exist_ok=True)

        # Sample valid configuration
        self.valid_config = {
            "api_key": "test_api_key",
            "api_endpoint": "https://api.test.com",
            "model_defaults": {
                "steps": 40,
                "guidance": 2.5,
                "strength": 1.0
            },
            "logging": {
                "level": "INFO",
                "file": "app.log",
                "format": "json"
            },
            "performance": {
                "cache_size": 1024,
                "request_timeout": 30,
                "max_retries": 3
            },
            "security": {
                "rate_limit": 60,
                "allowed_origins": ["localhost"],
                "token_expiry": 3600
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.test_config_dir).exists():
            shutil.rmtree(self.test_config_dir)

        # Clear any environment variables we set
        for key in list(os.environ.keys()):
            if key.startswith("FLUX_"):
                del os.environ[key]

    def _write_config(self, config: dict, path: Path) -> None:
        """Helper to write config file."""
        with open(path, 'w') as f:
            json.dump(config, f)

    def test_load_valid_config(self):
        """Test loading valid configuration."""
        self._write_config(self.valid_config, self.config_path)

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        self.assertEqual(
            config_manager.get("api_key"),
            self.valid_config["api_key"]
        )
        self.assertEqual(
            config_manager.get("api_endpoint"),
            self.valid_config["api_endpoint"]
        )

    def test_load_with_env_override(self):
        """Test loading config with environment override."""
        base_config = self.valid_config.copy()
        env_config = self.valid_config.copy()
        env_config["api_key"] = "test_env_api_key"

        self._write_config(base_config, self.config_path)
        self._write_config(env_config, self.env_config_path)

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        self.assertEqual(config_manager.get("api_key"), "test_env_api_key")

    def test_environment_variable_override(self):
        """Test environment variable override."""
        self._write_config(self.valid_config, self.config_path)

        # Set environment variable
        os.environ["FLUX_API_KEY"] = "env_var_api_key"

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        self.assertEqual(config_manager.get("api_key"), "env_var_api_key")

    def test_invalid_config_validation(self):
        """Test configuration validation."""
        invalid_configs = [
            # Missing required field
            {k: v for k, v in self.valid_config.items() if k != "api_key"},

            # Invalid API endpoint
            {
                **self.valid_config,
                "api_endpoint": "invalid-url"
            },

            # Invalid type for model_defaults
            {
                **self.valid_config,
                "model_defaults": "invalid"
            }
        ]

        for invalid_config in invalid_configs:
            self._write_config(invalid_config, self.config_path)

            config_manager = ConfigManager(
                config_path=str(self.config_path),
                env="test"
            )

            with self.assertRaises(ConfigError):
                config_manager.load_config()

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        config_manager = ConfigManager(
            config_path="nonexistent.json",
            env="test"
        )

        with self.assertRaises(ConfigError):
            config_manager.load_config()

    def test_get_with_default(self):
        """Test get with default value."""
        self._write_config(self.valid_config, self.config_path)

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        self.assertEqual(
            config_manager.get("nonexistent", "default"),
            "default"
        )

    def test_nested_config_override(self):
        """Test nested configuration override."""
        base_config = self.valid_config.copy()
        env_config = {
            "model_defaults": {
                "steps": 50
            }
        }

        self._write_config(base_config, self.config_path)
        self._write_config(env_config, self.env_config_path)

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        self.assertEqual(
            config_manager.get("model_defaults")["steps"],
            50
        )
        self.assertEqual(
            config_manager.get("model_defaults")["guidance"],
            base_config["model_defaults"]["guidance"]
        )

    def test_json_environment_variable(self):
        """Test JSON parsing in environment variables."""
        self._write_config(self.valid_config, self.config_path)

        # Set JSON environment variable
        os.environ["FLUX_MODEL_DEFAULTS"] = json.dumps({
            "steps": 60,
            "guidance": 3.0
        })

        config_manager = ConfigManager(
            config_path=str(self.config_path),
            env="test"
        )
        config_manager.load_config()

        model_defaults = config_manager.get("model_defaults")
        self.assertEqual(model_defaults["steps"], 60)
        self.assertEqual(model_defaults["guidance"], 3.0)

    def test_get_schema(self):
        """Test getting configuration schema."""
        config_manager = ConfigManager()
        schema = config_manager.get_schema()

        required_keys = [
            "api_key",
            "api_endpoint",
            "model_defaults",
            "logging",
            "performance",
            "security"
        ]

        for key in required_keys:
            self.assertIn(key, schema)

if __name__ == "__main__":
    unittest.main()

"""
Enhanced Configuration Service for FLUX-Pro-Finetuning-UI.

Provides robust configuration management with support for:
- Chunked access to configuration
- Environment-specific configurations
- Configuration validation
- Default values for missing configuration
- Configuration reloading without application restart
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize configuration error.

        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigService:
    """
    Enhanced service for configuration management.

    Provides robust configuration management with support for:
    - Chunked access to configuration
    - Environment-specific configurations
    - Configuration validation
    - Default values for missing configuration
    - Configuration reloading without application restart
    """

    # Error codes
    ERROR_FILE_NOT_FOUND = "file_not_found"
    ERROR_INVALID_JSON = "invalid_json"
    ERROR_INVALID_CONFIG = "invalid_config"
    ERROR_ACCESS_ERROR = "access_error"
    ERROR_UNKNOWN = "unknown_error"

    def __init__(
        self,
        config_path: Optional[str] = None,
        env: Optional[str] = None,
        auto_reload: bool = False,
        reload_interval: int = 60
    ):
        """
        Initialize the enhanced configuration service.

        Args:
            config_path: Path to the configuration file
            env: Environment (dev, test, prod)
            auto_reload: Whether to automatically reload configuration
            reload_interval: Interval in seconds for auto-reload
        """
        self.logger = logging.getLogger(__name__)

        # Configuration path
        self.config_path = config_path or os.environ.get(
            "CONFIG_PATH",
            os.path.join("config", "config.json")
        )

        # Environment
        self.env = env or os.environ.get("ENV", "dev")
        self.logger.info(f"Using environment: {self.env}")

        # Auto-reload settings
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval
        self.last_reload_time = 0

        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        self._config_chunks: Dict[str, Dict[str, Any]] = {}
        self._last_modified_time = 0

        # Load initial configuration
        self._load_config()

        # Set up auto-reload if enabled
        if self.auto_reload:
            self._setup_auto_reload()

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key is not found

        Returns:
            Configuration value or default if not found
        """
        # Check if auto-reload is enabled and configuration needs to be reloaded
        if self.auto_reload and self._should_reload():
            self._reload_config()

        # Split key into parts for nested access
        parts = key.split(".")

        # Try to get value from cache
        try:
            # Start with the full configuration
            value = self._config_cache

            # Traverse the configuration tree
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default

            return value

        except Exception as e:
            self.logger.error(f"Error getting configuration value for key '{key}': {e}")
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section as dictionary
        """
        # Check if auto-reload is enabled and configuration needs to be reloaded
        if self.auto_reload and self._should_reload():
            self._reload_config()

        # Check if section is already cached
        if section in self._config_chunks:
            return self._config_chunks[section]

        # Get section from configuration
        value = self.get_value(section, {})

        # Cache section if it's a dictionary
        if isinstance(value, dict):
            self._config_chunks[section] = value

        return value if isinstance(value, dict) else {}

    def get_api_key(self) -> str:
        """
        Get API key from configuration.

        Returns:
            API key or empty string if not found
        """
        return str(self.get_value("api_key", ""))

    def get_api_host(self) -> str:
        """
        Get API host from configuration.

        Returns:
            API host or default if not found
        """
        return str(self.get_value("api_host", "api.us1.bfl.ai"))

    def get_storage_path(self, storage_type: str) -> str:
        """
        Get storage path from configuration.

        Args:
            storage_type: Type of storage (models_dir, images_dir, temp_dir, logs_dir)

        Returns:
            Storage path or default if not found
        """
        # Get storage section
        storage = self.get_section("storage")

        # Get storage path
        path = storage.get(storage_type, "")

        # Use default if not found
        if not path:
            defaults = {
                "models_dir": "data",
                "images_dir": "generated_images",
                "temp_dir": "temp",
                "logs_dir": "logs"
            }
            path = defaults.get(storage_type, "")

        return path

    def get_feature_flags(self) -> Dict[str, bool]:
        """
        Get feature flags from configuration.

        Returns:
            Feature flags as dictionary
        """
        return self.get_section("features")

    def is_feature_enabled(self, feature: str, default: bool = False) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature: Feature name
            default: Default value if feature is not found

        Returns:
            True if feature is enabled, False otherwise
        """
        features = self.get_feature_flags()
        return bool(features.get(feature, default))

    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Get UI settings from configuration.

        Returns:
            UI settings as dictionary
        """
        return self.get_section("ui")

    def get_logging_settings(self) -> Dict[str, Any]:
        """
        Get logging settings from configuration.

        Returns:
            Logging settings as dictionary
        """
        return self.get_section("logging")

    def get_api_settings(self) -> Dict[str, Any]:
        """
        Get API settings from configuration.

        Returns:
            API settings as dictionary
        """
        return self.get_section("api")

    def get_defaults(self, category: str) -> Dict[str, Any]:
        """
        Get default settings for a category.

        Args:
            category: Category name (ultra, standard, finetune)

        Returns:
            Default settings as dictionary
        """
        defaults = self.get_section("defaults")
        return defaults.get(category, {})

    def reload(self) -> bool:
        """
        Reload configuration from file.

        Returns:
            True if configuration was reloaded successfully, False otherwise
        """
        try:
            self._reload_config()
            return True
        except Exception as e:
            self.logger.error(f"Error reloading configuration: {e}")
            return False

    def _load_config(self) -> None:
        """
        Load configuration from file.

        Raises:
            ConfigError: If configuration file is not found or invalid
        """
        try:
            # Check if configuration file exists
            if not os.path.exists(self.config_path):
                # Try to find configuration file in common locations
                common_locations = [
                    os.path.join("config", "config.json"),
                    os.path.join("config", f"config.{self.env}.json"),
                    os.path.join(".", "config.json"),
                    os.path.join(".", f"config.{self.env}.json")
                ]

                for location in common_locations:
                    if os.path.exists(location):
                        self.config_path = location
                        self.logger.info(f"Found configuration file at {location}")
                        break
                else:
                    # No configuration file found
                    raise ConfigError(
                        f"Configuration file not found at {self.config_path}",
                        self.ERROR_FILE_NOT_FOUND,
                        {"path": self.config_path}
                    )

            # Get last modified time
            self._last_modified_time = os.path.getmtime(self.config_path)

            # Load configuration from file
            with open(self.config_path, "r") as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError as e:
                    raise ConfigError(
                        f"Invalid JSON in configuration file: {e}",
                        self.ERROR_INVALID_JSON,
                        {"path": self.config_path, "error": str(e)}
                    )

            # Validate configuration
            if not isinstance(config, dict):
                raise ConfigError(
                    "Invalid configuration format (must be a dictionary)",
                    self.ERROR_INVALID_CONFIG,
                    {"path": self.config_path}
                )

            # Store configuration in cache
            self._config_cache = config

            # Clear chunk cache
            self._config_chunks = {}

            self.logger.info(f"Configuration loaded from {self.config_path}")

        except ConfigError:
            # Re-raise ConfigError
            raise
        except Exception as e:
            # Wrap other exceptions in ConfigError
            raise ConfigError(
                f"Error loading configuration: {e}",
                self.ERROR_UNKNOWN,
                {"path": self.config_path, "error": str(e)}
            )

    def _reload_config(self) -> None:
        """
        Reload configuration from file if it has changed.

        Raises:
            ConfigError: If configuration file is not found or invalid
        """
        try:
            # Check if configuration file exists
            if not os.path.exists(self.config_path):
                raise ConfigError(
                    f"Configuration file not found at {self.config_path}",
                    self.ERROR_FILE_NOT_FOUND,
                    {"path": self.config_path}
                )

            # Check if configuration file has changed
            current_modified_time = os.path.getmtime(self.config_path)
            if current_modified_time <= self._last_modified_time:
                # Configuration file has not changed
                return

            # Load configuration from file
            self._load_config()

            # Update last reload time
            self.last_reload_time = current_modified_time

        except ConfigError:
            # Re-raise ConfigError
            raise
        except Exception as e:
            # Wrap other exceptions in ConfigError
            raise ConfigError(
                f"Error reloading configuration: {e}",
                self.ERROR_UNKNOWN,
                {"path": self.config_path, "error": str(e)}
            )

    def _should_reload(self) -> bool:
        """
        Check if configuration should be reloaded.

        Returns:
            True if configuration should be reloaded, False otherwise
        """
        # Check if auto-reload is enabled
        if not self.auto_reload:
            return False

        # Check if configuration file exists
        if not os.path.exists(self.config_path):
            return False

        # Check if configuration file has changed
        try:
            current_modified_time = os.path.getmtime(self.config_path)
            return current_modified_time > self._last_modified_time
        except Exception as e:
            self.logger.error(f"Error checking if configuration should be reloaded: {e}")
            return False

    def _setup_auto_reload(self) -> None:
        """
        Set up auto-reload for configuration.

        This method is a placeholder for future implementation.
        In a real implementation, you would set up a background thread
        or use a file watcher to reload configuration when it changes.
        """
        self.logger.info(f"Auto-reload enabled with interval {self.reload_interval} seconds")
        # In a real implementation, you would set up a background thread
        # or use a file watcher to reload configuration when it changes

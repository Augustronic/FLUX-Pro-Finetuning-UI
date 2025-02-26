"""Configuration management for FLUX Pro Finetuning UI."""

import json
import os
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from utils.error_handling.error_handler import ConfigError, ErrorContext
from utils.logging.logger import get_logger
from utils.validation.validator import Validator, ValidationRule

@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    api_key: str
    api_endpoint: str
    model_defaults: Dict[str, Any]
    logging: Dict[str, Any]
    performance: Dict[str, Any]
    security: Dict[str, Any]

class ConfigManager:
    """Configuration management utility."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        env: Optional[str] = None
    ):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config file (optional)
            env: Environment name (optional)
        """
        self.logger = get_logger(__name__)
        self.validator = Validator()
        self.env = env or os.getenv("FLUX_ENV", "development")
        self.config_path = config_path or "config.json"
        self.config: Dict[str, Any] = {}
        
        # Define validation rules
        self.validation_rules = [
            ValidationRule(
                field="api_key",
                rule_type="required"
            ),
            ValidationRule(
                field="api_endpoint",
                rule_type="required"
            ),
            ValidationRule(
                field="api_endpoint",
                rule_type="pattern",
                value=r"^https?://.*$",
                message="API endpoint must be a valid URL"
            ),
            ValidationRule(
                field="model_defaults",
                rule_type="type",
                value=dict
            ),
            ValidationRule(
                field="logging",
                rule_type="type",
                value=dict
            ),
            ValidationRule(
                field="performance",
                rule_type="type",
                value=dict
            ),
            ValidationRule(
                field="security",
                rule_type="type",
                value=dict
            )
        ]

    def load_config(self) -> None:
        """Load and validate configuration.
        
        Raises:
            ConfigError: If configuration is invalid or missing
        """
        try:
            # Load base config
            base_config = self._load_config_file(self.config_path)
            
            # Load environment-specific config
            env_config_path = f"config.{self.env}.json"
            env_config = self._load_config_file(env_config_path) if Path(env_config_path).exists() else {}
            
            # Merge configs
            self.config = self._merge_configs(base_config, env_config)
            
            # Validate config
            self._validate_config()
            
            # Apply environment variables overrides
            self._apply_env_overrides()
            
            self.logger.info(
                f"Configuration loaded successfully for environment: {self.env}"
            )
            
        except Exception as e:
            raise ConfigError(
                f"Failed to load configuration: {str(e)}",
                context=ErrorContext(
                    component="ConfigManager",
                    operation="load_config",
                    details={"env": self.env}
                )
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            path: Path to config file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigError: If file cannot be loaded
        """
        try:
            if not Path(path).exists():
                return {}
                
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(
                f"Failed to load config file {path}: {str(e)}",
                context=ErrorContext(
                    component="ConfigManager",
                    operation="_load_config_file",
                    details={"path": path}
                )
            )

    def _merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result

    def _validate_config(self) -> None:
        """Validate configuration against schema.
        
        Raises:
            ConfigError: If configuration is invalid
        """
        try:
            self.validator.validate(
                self.config,
                self.validation_rules,
                "ConfigManager"
            )
        except Exception as e:
            raise ConfigError(
                f"Configuration validation failed: {str(e)}",
                context=ErrorContext(
                    component="ConfigManager",
                    operation="_validate_config"
                )
            )

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        env_prefix = "FLUX_"
        
        for key in self.config.keys():
            env_key = f"{env_prefix}{key.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                try:
                    # Try to parse as JSON for complex values
                    self.config[key] = json.loads(env_value)
                except json.JSONDecodeError:
                    # Use raw string if not valid JSON
                    self.config[key] = env_value
                    
                self.logger.info(f"Applied environment override for {key}")

    def get_schema(self) -> Dict[str, Any]:
        """Get configuration schema.
        
        Returns:
            Configuration schema
        """
        return {
            "api_key": "API key for authentication",
            "api_endpoint": "Base URL for API endpoints",
            "model_defaults": {
                "steps": "Default number of steps for generation",
                "guidance": "Default guidance scale",
                "strength": "Default model strength"
            },
            "logging": {
                "level": "Logging level (INFO, DEBUG, etc)",
                "file": "Log file path",
                "format": "Log format specification"
            },
            "performance": {
                "cache_size": "Maximum cache size in MB",
                "request_timeout": "API request timeout in seconds",
                "max_retries": "Maximum number of API retry attempts"
            },
            "security": {
                "rate_limit": "Maximum requests per minute",
                "allowed_origins": "List of allowed CORS origins",
                "token_expiry": "JWT token expiry time in seconds"
            }
        }
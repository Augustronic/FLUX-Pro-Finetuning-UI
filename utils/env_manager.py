"""
Environment Variable Manager

This module provides utilities for loading environment variables
from .env files and accessing them in a consistent manner.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Mapping
from dotenv import load_dotenv, find_dotenv, dotenv_values

logger = logging.getLogger(__name__)


class EnvManager:
    """
    Manages environment variables for the application.
    Loads variables from .env files and provides methods to access them.
    """

    def __init__(self, env_file: Optional[str] = None,
                 env_prefix: str = "FLUX_"):
        """
        Initialize the EnvManager.

        Args:
            env_file: Path to the .env file. If None, automatically
                searches for .env files.
            env_prefix: Prefix for application-specific environment variables.
        """
        self.env_prefix = env_prefix
        self.env_file = env_file

        # Auto-detect .env file if not specified
        if not env_file:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                self.env_file = env_path
                logger.info(f"Found .env file at: {env_path}")
            else:
                logger.info(
                    "No .env file found, using system environment variables"
                )

        # Load environment variables
        if self.env_file and Path(self.env_file).exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")

            # Store dotenv values for later access
            self._dotenv_values = dotenv_values(self.env_file)
        else:
            self._dotenv_values = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an environment variable.

        Args:
            key: The environment variable name (without prefix)
            default: Default value if the environment variable is not set

        Returns:
            The value of the environment variable or the default value
        """
        env_key = f"{self.env_prefix}{key.upper()}"
        return os.environ.get(env_key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean environment variable.

        Args:
            key: The environment variable name (without prefix)
            default: Default value if the environment variable is not set

        Returns:
            Boolean value of the environment variable
        """
        value = self.get(key, "").lower()
        if value in ("1", "true", "yes", "y", "on"):
            return True
        if value in ("0", "false", "no", "n", "off"):
            return False
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get an integer environment variable.

        Args:
            key: The environment variable name (without prefix)
            default: Default value if the environment variable is not set
                or invalid

        Returns:
            Integer value of the environment variable
        """
        value = self.get(key, "")
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get a float environment variable.

        Args:
            key: The environment variable name (without prefix)
            default: Default value if the environment variable is not set
                or invalid

        Returns:
            Float value of the environment variable
        """
        value = self.get(key, "")
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_list(self, key: str, default: Optional[list] = None,
                 delimiter: str = ",") -> list:
        """
        Get a list environment variable by splitting a string.

        Args:
            key: The environment variable name (without prefix)
            default: Default value if the environment variable is not set
            delimiter: String delimiter to split the value

        Returns:
            List of values
        """
        if default is None:
            default = []

        value = self.get(key, "")
        if not value:
            return default

        return [item.strip() for item in value.split(delimiter)]

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        """
        Get all environment variables with a specific prefix as a dictionary.

        Args:
            key: The prefix for environment variables (without the main prefix)
            default: Default value if no matching variables are found

        Returns:
            Dictionary of matching environment variables
        """
        if default is None:
            default = {}

        prefix = f"{self.env_prefix}{key.upper()}_"
        result = {}

        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Extract the key part after the prefix
                dict_key = env_key[len(prefix):].lower()
                result[dict_key] = env_value

        return result if result else default

    def get_all(self) -> Mapping[str, Optional[str]]:
        """
        Get all environment variables loaded from the .env file.

        Returns:
            Dictionary of all environment variables from the .env file
        """
        return dict(self._dotenv_values)

    def get_env(self) -> str:
        """
        Get the current environment name.

        Returns:
            Environment name (development, production, etc.)
        """
        return os.environ.get(f"{self.env_prefix}ENV", "development")


# Create a default instance for easy imports
env_manager = EnvManager()


def get_env(key: str, default: Any = None) -> Any:
    """Get an environment variable."""
    return env_manager.get(key, default)


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable."""
    return env_manager.get_bool(key, default)


def get_int_env(key: str, default: int = 0) -> int:
    """Get an integer environment variable."""
    return env_manager.get_int(key, default)


def get_float_env(key: str, default: float = 0.0) -> float:
    """Get a float environment variable."""
    return env_manager.get_float(key, default)


def get_list_env(key: str, default: Optional[list] = None,
                 delimiter: str = ",") -> list:
    """Get a list environment variable."""
    return env_manager.get_list(key, default, delimiter)


def get_env_name() -> str:
    """Get the current environment name."""
    return env_manager.get_env()

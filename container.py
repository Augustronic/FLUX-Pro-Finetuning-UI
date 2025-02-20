"""Dependency injection container for the FLUX Pro Finetuning UI."""

from typing import Dict, Optional, Any
from config_manager import ConfigManager
from model_manager import ModelManager


class Container:
    """Dependency injection container for managing application dependencies."""
    
    _instance: Optional['Container'] = None
    _config_manager: Optional[ConfigManager] = None
    _model_manager: Optional[ModelManager] = None
    
    def __new__(cls) -> 'Container':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super(Container, cls).__new__(cls)
        return cls._instance
    
    @property
    def config_manager(self) -> ConfigManager:
        """Get or create ConfigManager instance."""
        if self._config_manager is None:
            self._config_manager = ConfigManager()
            self._config_manager.load_config()
        return self._config_manager
    
    @property
    def model_manager(self) -> ModelManager:
        """Get or create ModelManager instance."""
        if self._model_manager is None:
            api_key = self.config_manager.get_api_key()
            self._model_manager = ModelManager(api_key=api_key)
        return self._model_manager
    
    def reset(self) -> None:
        """Reset all dependencies."""
        self._config_manager = None
        self._model_manager = None


# Global container instance
container = Container()
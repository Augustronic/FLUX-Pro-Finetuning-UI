"""
Service Container with Lazy Loading for FLUX-Pro-Finetuning-UI.

Provides dependency injection for services and components with lazy initialization.
"""

import logging
from typing import Dict, Any, Set

# Core services
from services.core.config_service_enhanced import ConfigService
from services.core.api_service import APIService
from services.core.storage_service import StorageService
from services.core.validation_service_refactored import ValidationService
from services.core.feature_flag_service import FeatureFlagService
from services.core.simple_api_client import SimpleAPIClient

# Business services
from services.business.model_service import ModelService
from services.business.model_service_enhanced import ModelService as ModelServiceEnhanced
from services.business.finetuning_service_enhanced import FinetuningService
from services.business.inference_service_enhanced import InferenceService
from services.business.simple_model_interface import SimpleModelInterface

# UI components
from ui.base import BaseUI
from ui.finetune_ui_enhanced import FineTuneUI
from ui.model_browser_ui_enhanced import ModelBrowserUI
from ui.inference_ui_enhanced import InferenceUI


class ServiceContainer:
    """
    Container for services with dependency injection and lazy loading.
    
    Initializes services on-demand to reduce startup time and memory usage.
    """
    
    def __init__(self):
        """Initialize the service container."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing service container")
        
        # Service registry
        self._services: Dict[str, Any] = {}
        self._initialized: Set[str] = set()
        
        self.logger.info("Service container initialized")
    
    def _initialize_service(self, service_name: str) -> None:
        """
        Initialize a service by name.
        
        Args:
            service_name: Name of the service to initialize
            
        Raises:
            ValueError: If service initialization fails
        """
        try:
            if service_name == 'config':
                # Initialize enhanced config service
                self._services[service_name] = ConfigService()  
            elif service_name == 'feature_flags':
                # Initialize feature flag service
                self._services[service_name] = FeatureFlagService(self.get_service('config'))
            elif service_name == 'api':
                # Initialize API service
                self._services[service_name] = APIService(self.get_service('config'))
            elif service_name == 'simple_api':
                # Initialize simplified API client
                config = self.get_service('config')
                self._services[service_name] = SimpleAPIClient(
                    api_key=config.get_api_key(),
                    host=config.get_api_host()
                )
            elif service_name == 'storage':
                # Initialize storage service
                self._services[service_name] = StorageService(self.get_service('config'))
            elif service_name == 'validation':
                # Initialize validation service
                self._services[service_name] = ValidationService()
            elif service_name == 'model':
                # Initialize model service
                self._services[service_name] = ModelServiceEnhanced(
                    self.get_service('api'),
                    self.get_service('storage'),
                    self.get_service('validation')
,
                    self.get_service('feature_flags')
                )
            elif service_name == 'simple_model':
                # Initialize simplified model interface
                self._services[service_name] = SimpleModelInterface(
                    self.get_service('model')
                )
            elif service_name == 'finetuning':
                # Initialize finetuning service
                self._services[service_name] = FinetuningService(
                    self.get_service('api'),
                    self.get_service('model'),
                    self.get_service('storage'),
                    self.get_service('validation')
,
                    self.get_service('feature_flags')
                )
            elif service_name == 'inference':
                # Initialize inference service
                self._services[service_name] = InferenceService(
                    self.get_service('api'),
                    self.get_service('model'),
                    self.get_service('storage'),
                    self.get_service('validation')
,
                    self.get_service('feature_flags')
                )
            elif service_name == 'finetune_ui':
                # Initialize finetune UI
                self._services[service_name] = FineTuneUI(
                    self.get_service('finetuning'),
                    self.get_service('model')
,
                    self.get_service('feature_flags')
                )
            elif service_name == 'model_browser_ui':
                # Initialize model browser UI
                self._services[service_name] = ModelBrowserUI(
                    self.get_service('model')
,
                    self.get_service('feature_flags')
                )
            elif service_name == 'inference_ui':
                # Initialize inference UI
                self._services[service_name] = InferenceUI(
                    self.get_service('inference'),
                    self.get_service('model')
,
                    self.get_service('feature_flags')
                )
            else:
                raise ValueError(f"Unknown service: {service_name}")
                
            self._initialized.add(service_name)
            self.logger.info(f"Service initialized: {service_name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing service {service_name}: {e}")
            raise ValueError(f"Failed to initialize service {service_name}: {e}")
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name, initializing it if necessary.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not found or initialization fails
        """
        if service_name not in self._initialized:
            self._initialize_service(service_name)
            
        return self._services[service_name]
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        config_service = self.get_service('config')
        return config_service.get_value(key, default)
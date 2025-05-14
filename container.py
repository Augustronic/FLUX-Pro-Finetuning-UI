"""
Service Container for FLUX-Pro-Finetuning-UI.

Provides dependency injection for services and components.
"""

import logging
from typing import Dict, Any

# Core services
from services.core.config_service import ConfigService
from services.core.api_service import APIService
from services.core.storage_service import StorageService
from services.core.validation_service import ValidationService

# Business services
from services.business.model_service import ModelService
from services.business.finetuning_service import FinetuningService
from services.business.inference_service import InferenceService


class ServiceContainer:
    """
    Container for services with dependency injection.

    Initializes and provides access to all services with proper dependencies.
    """

    def __init__(self):
        """Initialize the service container with all services."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing service container")

        # Initialize core services
        self.config_service = ConfigService()
        self.api_service = APIService(self.config_service)
        self.storage_service = StorageService(self.config_service)
        self.validation_service = ValidationService()

        # Initialize business services
        self.model_service = ModelService(
            self.api_service,
            self.storage_service,
            self.validation_service
        )

        self.finetuning_service = FinetuningService(
            self.api_service,
            self.model_service,
            self.storage_service,
            self.validation_service
        )

        self.inference_service = InferenceService(
            self.api_service,
            self.model_service,
            self.storage_service,
            self.validation_service
        )

        self.logger.info("Service container initialized")

    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.

        Args:
            service_name: Name of the service to retrieve

        Returns:
            Service instance

        Raises:
            ValueError: If service is not found
        """
        service_map = {
            # Core services
            'config': self.config_service,
            'api': self.api_service,
            'storage': self.storage_service,
            'validation': self.validation_service,

            # Business services
            'model': self.model_service,
            'finetuning': self.finetuning_service,
            'inference': self.inference_service
        }

        if service_name not in service_map:
            raise ValueError(f"Service not found: {service_name}")

        return service_map[service_name]

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        return self.config_service.get_value(key, default)

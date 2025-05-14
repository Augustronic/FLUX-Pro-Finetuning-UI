"""
Model Service for FLUX-Pro-Finetuning-UI.

Provides functionality for managing fine-tuned models, including
listing, retrieving, and refreshing model data.
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import time

from services.core.api_service import APIService, APIError
from services.core.storage_service import StorageService, StorageError
from services.core.validation_service import ValidationService, ValidationError


@dataclass
class ModelMetadata:
    """Model metadata for fine-tuned models."""

    finetune_id: str
    model_name: str
    trigger_word: str
    mode: str
    type: str
    rank: Optional[int] = None
    iterations: Optional[int] = None
    timestamp: Optional[str] = None
    learning_rate: Optional[float] = None
    priority: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, excluding None values.

        Returns:
            Dictionary representation of the model metadata
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelService:
    """
    Service for managing fine-tuned models.

    Provides methods for listing, retrieving, and refreshing model data.
    """

    def __init__(
        self,
        api_service: APIService,
        storage_service: StorageService,
        validation_service: ValidationService
    ):
        """
        Initialize the model service.

        Args:
            api_service: API service for communicating with the API
            storage_service: Storage service for file operations
            validation_service: Validation service for input validation
        """
        self.api = api_service
        self.storage = storage_service
        self.validation = validation_service
        self.logger = logging.getLogger(__name__)

        # Load models from storage
        self.models: Dict[str, ModelMetadata] = {}
        self._load_models()

    def _load_models(self) -> None:
        """
        Load models from storage.

        Loads model metadata from storage and populates the models dictionary.
        """
        try:
            # Load models from storage
            model_data = self.storage.load_model_metadata()

            # Convert to ModelMetadata objects
            for finetune_id, data in model_data.items():
                try:
                    # Validate model data
                    if self.validation.validate_model_metadata(data):
                        self.models[finetune_id] = ModelMetadata(**data)
                except (ValidationError, TypeError) as e:
                    self.logger.warning(f"Skipping invalid model data: {e}")
                    continue

            self.logger.info(f"Loaded {len(self.models)} models from storage")

        except StorageError as e:
            self.logger.error(f"Error loading models: {e}")
            self.models = {}

    def _save_models(self) -> None:
        """
        Save models to storage.

        Converts model metadata to dictionaries and saves to storage.
        """
        try:
            # Convert models to dictionaries
            model_data = {
                model.finetune_id: model.to_dict()
                for model in self.models.values()
            }

            # Save to storage
            self.storage.save_model_metadata(model_data)

            self.logger.info(f"Saved {len(self.models)} models to storage")

        except StorageError as e:
            self.logger.error(f"Error saving models: {e}")

    def list_models(self) -> List[ModelMetadata]:
        """
        List all models.

        Returns:
            List of model metadata objects
        """
        return list(self.models.values())

    def get_model(self, finetune_id: str) -> Optional[ModelMetadata]:
        """
        Get a model by ID.

        Args:
            finetune_id: ID of the model to retrieve

        Returns:
            Model metadata or None if not found

        Raises:
            ValidationError: If finetune_id format is invalid
        """
        # Validate finetune_id format
        try:
            if not finetune_id or not isinstance(finetune_id, str):
                raise ValidationError("Invalid finetune ID", "finetune_id", finetune_id)

            # Check if model exists in cache
            return self.models.get(finetune_id)

        except ValidationError as e:
            self.logger.error(f"Error getting model: {e}")
            raise

    def add_model(self, metadata: ModelMetadata) -> None:
        """
        Add or update a model.

        Args:
            metadata: Model metadata to add or update

        Raises:
            ValidationError: If metadata is invalid
        """
        try:
            # Validate metadata
            if self.validation.validate_model_metadata(metadata.to_dict()):
                # Add or update model
                self.models[metadata.finetune_id] = metadata

                # Save models to storage
                self._save_models()

                self.logger.info(f"Added/updated model: {metadata.model_name}")

        except ValidationError as e:
            self.logger.error(f"Error adding model: {e}")
            raise

    def delete_model(self, finetune_id: str) -> bool:
        """
        Delete a model.

        Args:
            finetune_id: ID of the model to delete

        Returns:
            True if model was deleted, False otherwise

        Raises:
            ValidationError: If finetune_id format is invalid
        """
        try:
            # Validate finetune_id format
            if not finetune_id or not isinstance(finetune_id, str):
                raise ValidationError("Invalid finetune ID", "finetune_id", finetune_id)

            # Check if model exists
            if finetune_id not in self.models:
                return False

            # Delete model
            model_name = self.models[finetune_id].model_name
            del self.models[finetune_id]

            # Save models to storage
            self._save_models()

            self.logger.info(f"Deleted model: {model_name} ({finetune_id})")
            return True

        except ValidationError as e:
            self.logger.error(f"Error deleting model: {e}")
            raise

    def get_model_details(self, finetune_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model from the API.

        Args:
            finetune_id: ID of the model to retrieve details for

        Returns:
            Model details or None if not found

        Raises:
            ValidationError: If finetune_id format is invalid
        """
        try:
            # Validate finetune_id format
            if not finetune_id or not isinstance(finetune_id, str):
                raise ValidationError("Invalid finetune ID", "finetune_id", finetune_id)

            # Get model details from API
            details = self.api.get_model_details(finetune_id)

            if details:
                self.logger.info(f"Retrieved details for model: {finetune_id}")
                return details

            return None

        except (ValidationError, APIError) as e:
            self.logger.error(f"Error getting model details: {e}")
            return None

    def update_model_from_api(self, finetune_id: str) -> bool:
        """
        Update model metadata from API.

        Args:
            finetune_id: ID of the model to update

        Returns:
            True if model was updated, False otherwise
        """
        try:
            # Get model details from API
            details = self.get_model_details(finetune_id)

            if not details:
                self.logger.warning(f"No details found for model: {finetune_id}")
                return False

            # Create model metadata
            metadata = ModelMetadata(
                finetune_id=finetune_id,
                model_name=details.get("finetune_comment", ""),
                trigger_word=details.get("trigger_word", ""),
                mode=details.get("mode", ""),
                type=details.get("finetune_type", "lora"),
                rank=details.get("lora_rank"),
                iterations=details.get("iterations"),
                timestamp=details.get("timestamp") or time.strftime("%Y-%m-%d %H:%M:%S"),
                learning_rate=details.get("learning_rate"),
                priority=details.get("priority")
            )

            # Add or update model
            self.add_model(metadata)

            return True

        except (ValidationError, APIError) as e:
            self.logger.error(f"Error updating model from API: {e}")
            return False

    def refresh_models(self) -> int:
        """
        Refresh all models from API.

        Returns:
            Number of models refreshed
        """
        try:
            # Get list of models from API
            finetune_ids = self.api.list_finetunes()

            if not finetune_ids:
                self.logger.warning("No models found in API")
                return 0

            self.logger.info(f"Found {len(finetune_ids)} models in API")

            # Update each model
            updated_count = 0
            for finetune_id in finetune_ids:
                if self.update_model_from_api(finetune_id):
                    updated_count += 1

            # Save models to storage
            self._save_models()

            return updated_count

        except APIError as e:
            self.logger.error(f"Error refreshing models: {e}")
            return 0

    def format_model_choice(self, model: ModelMetadata) -> str:
        """
        Format model metadata for dropdown display.

        Args:
            model: Model metadata

        Returns:
            Formatted string for display
        """
        if not model:
            return ""

        # Sanitize display values
        parts = [
            f"{self.validation.sanitize_display_text(model.model_name)}",
            f"({self.validation.sanitize_display_text(model.trigger_word)})",
            f"{self.validation.sanitize_display_text(model.type).upper()}",
            f"{self.validation.sanitize_display_text(model.mode).capitalize()}",
        ]

        # Add rank if present
        if model.rank is not None:
            parts.append(f"Rank {int(model.rank)}")

        return " - ".join(parts)

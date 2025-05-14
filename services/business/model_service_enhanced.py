"""
Enhanced Model Service for FLUX-Pro-Finetuning-UI.

Provides functionality for managing fine-tuned models, including
listing, retrieving, and refreshing model data. Uses feature flags
for conditional functionality and improved error handling.
"""

from typing import Dict, List, Optional, Any, Set
import logging
from dataclasses import dataclass, asdict
import time
import json
import os

from services.core.api_service import APIService, APIError
from services.core.storage_service import StorageService, StorageError
from services.core.validation_service import ValidationService, ValidationError
from services.core.feature_flag_service import FeatureFlagService


class ModelError(Exception):
    """Exception raised for model service errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model error.

        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


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

    # Additional fields for enhanced functionality
    tags: Optional[List[str]] = None
    favorite: bool = False
    last_used: Optional[str] = None
    usage_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary, excluding None values.

        Returns:
            Dictionary representation of the model metadata
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelService:
    """
    Enhanced service for managing fine-tuned models.

    Provides methods for listing, retrieving, and refreshing model data.
    Uses feature flags for conditional functionality and improved error handling.
    """

    # Error codes
    ERROR_VALIDATION = "validation_error"
    ERROR_API = "api_error"
    ERROR_STORAGE = "storage_error"
    ERROR_NOT_FOUND = "not_found"
    ERROR_UNKNOWN = "unknown_error"

    def __init__(
        self,
        api_service: APIService,
        storage_service: StorageService,
        validation_service: ValidationService,
        feature_flag_service: FeatureFlagService
    ):
        """
        Initialize the enhanced model service.

        Args:
            api_service: API service for communicating with the API
            storage_service: Storage service for file operations
            validation_service: Validation service for input validation
            feature_flag_service: Feature flag service for conditional functionality
        """
        self.api = api_service
        self.storage = storage_service
        self.validation = validation_service
        self.feature_flags = feature_flag_service
        self.logger = logging.getLogger(__name__)

        # Load models from storage
        self.models: Dict[str, ModelMetadata] = {}

        # Cache for model details
        self.details_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_timestamp: Dict[str, float] = {}

        # Load models
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

            # Load cache if feature is enabled
            if self.feature_flags.is_enabled("model_caching", True):
                self._load_cache()

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

            # Save cache if feature is enabled
            if self.feature_flags.is_enabled("model_caching", True):
                self._save_cache()

        except StorageError as e:
            self.logger.error(f"Error saving models: {e}")
            raise ModelError(
                f"Error saving models: {e}",
                self.ERROR_STORAGE,
                {"error": str(e)}
            )

    def _load_cache(self) -> None:
        """
        Load model details cache from storage.

        Only used if model_caching feature is enabled.
        """
        try:
            cache_path = os.path.join(
                self.storage.models_dir,
                "model_details_cache.json"
            )

            if not os.path.exists(cache_path):
                self.logger.info("No model details cache found")
                return

            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            self.details_cache = cache_data.get("details", {})
            self.cache_timestamp = cache_data.get("timestamps", {})

            self.logger.info(f"Loaded cache for {len(self.details_cache)} models")

        except (IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading model details cache: {e}")
            self.details_cache = {}
            self.cache_timestamp = {}

    def _save_cache(self) -> None:
        """
        Save model details cache to storage.

        Only used if model_caching feature is enabled.
        """
        try:
            cache_path = os.path.join(
                self.storage.models_dir,
                "model_details_cache.json"
            )

            cache_data = {
                "details": self.details_cache,
                "timestamps": self.cache_timestamp
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)

            self.logger.info(f"Saved cache for {len(self.details_cache)} models")

        except IOError as e:
            self.logger.error(f"Error saving model details cache: {e}")

    def list_models(self, filter_type: Optional[str] = None, filter_mode: Optional[str] = None) -> List[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            filter_type: Optional filter by model type (e.g., "lora", "full")
            filter_mode: Optional filter by model mode (e.g., "general", "character")

        Returns:
            List of model metadata objects
        """
        # Get all models
        models = list(self.models.values())

        # Apply filters if provided
        if filter_type:
            models = [m for m in models if m.type == filter_type]

        if filter_mode:
            models = [m for m in models if m.mode == filter_mode]

        # Filter experimental model types if feature is disabled
        if not self.feature_flags.is_enabled("experimental_model_types", False):
            models = [m for m in models if m.type in ["full", "lora"]]

        # Sort models by timestamp (newest first) if available
        models.sort(
            key=lambda m: m.timestamp if m.timestamp else "",
            reverse=True
        )

        return models

    def get_model(self, finetune_id: str) -> Optional[ModelMetadata]:
        """
        Get a model by ID.

        Args:
            finetune_id: ID of the model to retrieve

        Returns:
            Model metadata or None if not found

        Raises:
            ModelError: If finetune_id format is invalid
        """
        # Validate finetune_id format
        try:
            if not finetune_id or not isinstance(finetune_id, str):
                raise ValidationError("Invalid finetune ID", "finetune_id", finetune_id)

            # Check if model exists in cache
            return self.models.get(finetune_id)

        except ValidationError as e:
            self.logger.error(f"Error getting model: {e}")
            raise ModelError(
                f"Invalid finetune ID: {finetune_id}",
                self.ERROR_VALIDATION,
                {"finetune_id": finetune_id}
            )

    def add_model(self, metadata: ModelMetadata) -> None:
        """
        Add or update a model.

        Args:
            metadata: Model metadata to add or update

        Raises:
            ModelError: If metadata is invalid
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
            raise ModelError(
                f"Invalid model metadata: {e}",
                self.ERROR_VALIDATION,
                {"metadata": metadata.to_dict()}
            )

    def delete_model(self, finetune_id: str) -> bool:
        """
        Delete a model.

        Args:
            finetune_id: ID of the model to delete

        Returns:
            True if model was deleted, False otherwise

        Raises:
            ModelError: If finetune_id format is invalid
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

            # Remove from cache if present
            if finetune_id in self.details_cache:
                del self.details_cache[finetune_id]

            if finetune_id in self.cache_timestamp:
                del self.cache_timestamp[finetune_id]

            # Save models to storage
            self._save_models()

            self.logger.info(f"Deleted model: {model_name} ({finetune_id})")
            return True

        except ValidationError as e:
            self.logger.error(f"Error deleting model: {e}")
            raise ModelError(
                f"Invalid finetune ID: {finetune_id}",
                self.ERROR_VALIDATION,
                {"finetune_id": finetune_id}
            )

    def get_model_details(self, finetune_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a model from the API.

        Args:
            finetune_id: ID of the model to retrieve details for
            force_refresh: Whether to force a refresh from the API

        Returns:
            Model details or None if not found

        Raises:
            ModelError: If finetune_id format is invalid
        """
        try:
            # Validate finetune_id format
            if not finetune_id or not isinstance(finetune_id, str):
                raise ValidationError("Invalid finetune ID", "finetune_id", finetune_id)

            # Check cache if enabled and not forcing refresh
            if (
                self.feature_flags.is_enabled("model_caching", True) and

                not force_refresh and
                finetune_id in self.details_cache
            ):
                # Check if cache is still valid
                cache_ttl = 3600  # Default: 1 hour (3600 seconds)
                # Try to get cache TTL from config if available
                if hasattr(self.feature_flags, 'config'):
                    cache_ttl = self.feature_flags.config.get_value("model_cache_ttl", 3600)

                if time.time() - self.cache_timestamp.get(finetune_id, 0) < cache_ttl:
                    self.logger.info(f"Using cached details for model: {finetune_id}")
                    return self.details_cache[finetune_id]

            # Get model details from API
            details = self.api.get_model_details(finetune_id)

            if details:
                self.logger.info(f"Retrieved details for model: {finetune_id}")

                # Update cache if enabled
                if self.feature_flags.is_enabled("model_caching", True):
                    self.details_cache[finetune_id] = details
                    self.cache_timestamp[finetune_id] = time.time()

                    # Save cache if auto-save is enabled
                    if self.feature_flags.is_enabled("auto_save_cache", True):
                        self._save_cache()

                return details

            return None

        except ValidationError as e:
            self.logger.error(f"Error getting model details: {e}")
            raise ModelError(
                f"Invalid finetune ID: {finetune_id}",
                self.ERROR_VALIDATION,
                {"finetune_id": finetune_id}
            )
        except APIError as e:
            self.logger.error(f"API error getting model details: {e}")
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
            details = self.get_model_details(finetune_id, force_refresh=True)

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

            # Add tags if feature is enabled
            if self.feature_flags.is_enabled("model_tagging", False):
                # Extract tags from model name or details
                tags = []
                if "tags" in details:
                    tags = details.get("tags", [])
                elif metadata.model_name:
                    # Extract hashtags from model name
                    import re
                    hashtags = re.findall(r'#(\w+)', metadata.model_name)
                    if hashtags:
                        tags = hashtags

                metadata.tags = tags

            # Preserve favorite status and usage count if model exists
            if finetune_id in self.models:
                existing_model = self.models[finetune_id]
                metadata.favorite = existing_model.favorite
                metadata.usage_count = existing_model.usage_count
                metadata.last_used = existing_model.last_used

            # Add or update model
            self.add_model(metadata)

            return True

        except (ValidationError, APIError, ModelError) as e:
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

        # Add favorite indicator if feature is enabled
        if self.feature_flags.is_enabled("model_favorites", False) and model.favorite:
            parts.insert(0, "⭐")

        return " - ".join(parts)

    def set_favorite(self, finetune_id: str, favorite: bool) -> bool:
        """
        Set favorite status for a model.

        Args:
            finetune_id: ID of the model
            favorite: Whether the model is a favorite

        Returns:
            True if successful, False otherwise
        """
        # Check if feature is enabled
        if not self.feature_flags.is_enabled("model_favorites", False):
            self.logger.warning("Model favorites feature is disabled")
            return False

        # Get model
        model = self.get_model(finetune_id)
        if not model:
            self.logger.warning(f"Model not found: {finetune_id}")
            return False

        # Update favorite status
        model.favorite = favorite

        # Save models
        self._save_models()

        self.logger.info(f"Set favorite status for model {model.model_name} to {favorite}")
        return True

    def record_model_usage(self, finetune_id: str) -> bool:
        """
        Record usage of a model.

        Args:
            finetune_id: ID of the model

        Returns:
            True if successful, False otherwise
        """
        # Check if feature is enabled
        if not self.feature_flags.is_enabled("model_usage_tracking", False):
            return False

        # Get model
        model = self.get_model(finetune_id)
        if not model:
            self.logger.warning(f"Model not found: {finetune_id}")
            return False

        # Update usage count and timestamp
        model.usage_count += 1
        model.last_used = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save models
        self._save_models()

        self.logger.info(f"Recorded usage for model {model.model_name}")
        return True

    def get_favorite_models(self) -> List[ModelMetadata]:
        """
        Get list of favorite models.

        Returns:
            List of favorite models
        """
        # Check if feature is enabled
        if not self.feature_flags.is_enabled("model_favorites", False):
            return []

        # Filter models by favorite status
        favorites = [model for model in self.models.values() if model.favorite]

        # Sort by usage count if tracking is enabled
        if self.feature_flags.is_enabled("model_usage_tracking", False):
            favorites.sort(key=lambda m: m.usage_count, reverse=True)
        else:
            # Sort by timestamp
            favorites.sort(
                key=lambda m: m.timestamp if m.timestamp else "",
                reverse=True
            )

        return favorites

    def get_recent_models(self, limit: int = 5) -> List[ModelMetadata]:
        """
        Get list of recently used models.

        Args:
            limit: Maximum number of models to return

        Returns:
            List of recently used models
        """
        # Check if feature is enabled
        if not self.feature_flags.is_enabled("model_usage_tracking", False):
            # Fall back to timestamp-based sorting
            models = list(self.models.values())
            models.sort(
                key=lambda m: m.timestamp if m.timestamp else "",
                reverse=True
            )
            return models[:limit]

        # Filter models with usage data
        used_models = [model for model in self.models.values() if model.last_used]

        # Sort by last used timestamp
        used_models.sort(
            key=lambda m: m.last_used if m.last_used else "",
            reverse=True
        )

        return used_models[:limit]

    def search_models(self, query: str) -> List[ModelMetadata]:
        """
        Search for models by name, trigger word, or tags.

        Args:
            query: Search query

        Returns:
            List of matching models
        """
        if not query:
            return []

        query = query.lower()
        results = []

        for model in self.models.values():
            # Search in model name
            if query in model.model_name.lower():
                results.append(model)
                continue

            # Search in trigger word
            if query in model.trigger_word.lower():
                results.append(model)
                continue

            # Search in tags if feature is enabled
            if (
                self.feature_flags.is_enabled("model_tagging", False) and
                model.tags and
                any(query in tag.lower() for tag in model.tags)
            ):
                results.append(model)
                continue

        return results

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get statistics about models.

        Returns:
            Dictionary of model statistics
        """
        # Count models by type
        type_counts = {}
        for model in self.models.values():
            model_type = model.type
            type_counts[model_type] = type_counts.get(model_type, 0) + 1

        # Count models by mode
        mode_counts = {}
        for model in self.models.values():
            mode = model.mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        # Get most used models if tracking is enabled
        most_used = []
        if self.feature_flags.is_enabled("model_usage_tracking", False):
            used_models = [model for model in self.models.values() if model.usage_count > 0]
            used_models.sort(key=lambda m: m.usage_count, reverse=True)
            most_used = used_models[:5]

        return {
            "total": len(self.models),
            "by_type": type_counts,
            "by_mode": mode_counts,
            "most_used": [
                {
                    "name": model.model_name,
                    "id": model.finetune_id,
                    "usage_count": model.usage_count
                }
                for model in most_used
            ]
        }

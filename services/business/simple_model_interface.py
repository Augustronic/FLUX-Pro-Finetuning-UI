"""
Simple Model Interface for FLUX-Pro-Finetuning-UI.

Provides a simplified interface for model operations with minimal dependencies.
"""

from typing import Dict, List, Any, Optional
import logging
from services.business.model_service import ModelService, ModelMetadata


class SimpleModelInterface:
    """
    Simplified interface to ModelService with essential operations only.

    Provides a simplified interface for common model operations.
    """

    def __init__(self, model_service: ModelService):
        """
        Initialize the simple model interface.

        Args:
            model_service: Model service
        """
        self.service = model_service
        self.logger = logging.getLogger(__name__)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models in simplified format.

        Returns:
            List of model dictionaries
        """
        try:
            models = self.service.list_models()
            return [
                {
                    'id': model.finetune_id,
                    'name': model.model_name,
                    'trigger_word': model.trigger_word,
                    'type': model.type,
                    'mode': model.mode
                }
                for model in models
            ]
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []

    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by ID in simplified format.

        Args:
            model_id: Model ID

        Returns:
            Model dictionary or None if not found
        """
        try:
            model = self.service.get_model(model_id)
            if not model:
                return None

            return {
                'id': model.finetune_id,
                'name': model.model_name,
                'trigger_word': model.trigger_word,
                'type': model.type,
                'mode': model.mode,
                'rank': model.rank,
                'iterations': model.iterations,
                'timestamp': model.timestamp
            }
        except Exception as e:
            self.logger.error(f"Error getting model by ID: {e}")
            return None

    def refresh_models(self) -> int:
        """
        Refresh models from API.

        Returns:
            Number of models refreshed
        """
        try:
            return self.service.refresh_models()
        except Exception as e:
            self.logger.error(f"Error refreshing models: {e}")
            return 0

    def get_model_choices(self) -> List[str]:
        """
        Get formatted model choices for dropdown.

        Returns:
            List of formatted model choices
        """
        try:
            models = self.service.list_models()
            return [
                self.service.format_model_choice(model)
                for model in models
                if model is not None
            ]
        except Exception as e:
            self.logger.error(f"Error getting model choices: {e}")
            return []

    def get_model_id_from_choice(self, choice: str) -> Optional[str]:
        """
        Get model ID from formatted choice string.

        Args:
            choice: Formatted choice string from dropdown

        Returns:
            Model ID or None if not found
        """
        try:
            if not choice or not isinstance(choice, str):
                return None

            for model in self.service.list_models():
                if model and self.service.format_model_choice(model) == choice:
                    return model.finetune_id
            return None
        except Exception as e:
            self.logger.error(f"Error getting model ID from choice: {e}")
            return None

    def add_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Add or update a model.

        Args:
            model_data: Model data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create ModelMetadata from dictionary
            metadata = ModelMetadata(
                finetune_id=model_data.get('id', ''),
                model_name=model_data.get('name', ''),
                trigger_word=model_data.get('trigger_word', ''),
                mode=model_data.get('mode', ''),
                type=model_data.get('type', 'lora'),
                rank=model_data.get('rank'),
                iterations=model_data.get('iterations'),
                timestamp=model_data.get('timestamp'),
                learning_rate=model_data.get('learning_rate'),
                priority=model_data.get('priority')
            )

            # Add or update model
            self.service.add_model(metadata)
            return True
        except Exception as e:
            self.logger.error(f"Error adding model: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.

        Args:
            model_id: Model ID

        Returns:
            True if successful, False otherwise
        """
        try:
            return self.service.delete_model(model_id)
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False

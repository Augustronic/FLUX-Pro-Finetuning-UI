"""
Model Validation Service for FLUX-Pro-Finetuning-UI.

Provides validation functionality for model-related data.
"""

from typing import Dict, Any, Optional, List
import logging
from .base_validation import BaseValidationService, ValidationError


class ModelValidationService(BaseValidationService):
    """
    Validation service for model-related data.
    
    Provides validation methods for model metadata, parameters, etc.
    """
    
    def __init__(self):
        """Initialize the model validation service."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    def validate_model_metadata(self, data: Dict[str, Any]) -> bool:
        """
        Validate model metadata format.
        
        Args:
            data: Model metadata to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError("Model metadata must be a dictionary", "metadata", data)
            
        # Check required fields
        required_fields = ['finetune_id', 'model_name', 'trigger_word', 'mode', 'type']
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field, None)
                
        # Validate field types
        self.validate_string(data.get('finetune_id'), 'finetune_id')
        self.validate_string(data.get('model_name'), 'model_name')
        self.validate_string(data.get('trigger_word'), 'trigger_word')
        self.validate_string(data.get('mode'), 'mode')
        self.validate_string(data.get('type'), 'type')
        
        # Validate optional fields if present
        if 'rank' in data and data['rank'] is not None:
            self.validate_numeric(data['rank'], 'rank', min_val=1)
            
        if 'iterations' in data and data['iterations'] is not None:
            self.validate_numeric(data['iterations'], 'iterations', min_val=1)
            
        if 'learning_rate' in data and data['learning_rate'] is not None:
            self.validate_numeric(data['learning_rate'], 'learning_rate', min_val=0)
            
        if 'timestamp' in data and data['timestamp'] is not None:
            self.validate_string(data['timestamp'], 'timestamp')
            
        if 'priority' in data and data['priority'] is not None:
            self.validate_string(data['priority'], 'priority')
            
        return True
    
    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate model ID format.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not model_id or not isinstance(model_id, str):
            raise ValidationError("Invalid model ID", "model_id", model_id)
            
        # Additional validation for model ID format could be added here
        # For example, checking if it matches a specific pattern
            
        return True
    
    def validate_model_choice(self, choice: str) -> bool:
        """
        Validate model choice format.
        
        Args:
            choice: Model choice to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not choice or not isinstance(choice, str):
            raise ValidationError("Invalid model choice", "model_choice", choice)
            
        return True
    
    def validate_model_type(self, model_type: str, allowed_types: Optional[List[str]] = None) -> bool:
        """
        Validate model type.
        
        Args:
            model_type: Model type to validate
            allowed_types: List of allowed model types
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not model_type or not isinstance(model_type, str):
            raise ValidationError("Invalid model type", "model_type", model_type)
            
        if allowed_types and model_type not in allowed_types:
            raise ValidationError(
                f"Model type must be one of: {', '.join(allowed_types)}",
                "model_type",
                model_type
            )
            
        return True
    
    def validate_model_mode(self, mode: str, allowed_modes: Optional[List[str]] = None) -> bool:
        """
        Validate model mode.
        
        Args:
            mode: Model mode to validate
            allowed_modes: List of allowed model modes
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not mode or not isinstance(mode, str):
            raise ValidationError("Invalid model mode", "mode", mode)
            
        if allowed_modes and mode not in allowed_modes:
            raise ValidationError(
                f"Model mode must be one of: {', '.join(allowed_modes)}",
                "mode",
                mode
            )
            
        return True
    
    def validate_trigger_word(self, trigger_word: str) -> bool:
        """
        Validate trigger word.
        
        Args:
            trigger_word: Trigger word to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if not trigger_word or not isinstance(trigger_word, str):
            raise ValidationError("Invalid trigger word", "trigger_word", trigger_word)
            
        # Additional validation for trigger word format could be added here
        # For example, checking if it contains only valid characters
            
        return True
    
    def validate_rank(self, rank: Any) -> bool:
        """
        Validate model rank.
        
        Args:
            rank: Model rank to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if rank is None:
            return True
            
        try:
            rank_value = int(rank)
            if rank_value < 1:
                raise ValidationError("Rank must be a positive integer", "rank", rank)
        except (ValueError, TypeError):
            raise ValidationError("Rank must be an integer", "rank", rank)
            
        return True
    
    def validate_iterations(self, iterations: Any) -> bool:
        """
        Validate model iterations.
        
        Args:
            iterations: Model iterations to validate
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if iterations is None:
            return True
            
        try:
            iterations_value = int(iterations)
            if iterations_value < 1:
                raise ValidationError("Iterations must be a positive integer", "iterations", iterations)
        except (ValueError, TypeError):
            raise ValidationError("Iterations must be an integer", "iterations", iterations)
            
        return True
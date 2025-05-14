"""
Validation Service for FLUX-Pro-Finetuning-UI.

Provides centralized validation logic for all input data.
"""

from typing import Dict, Any, Optional, List, Union
import logging
from services.core.validation.composite_validation import CompositeValidationService, ValidationError


class ValidationService:
    """
    Service for validating input data.

    Delegates to the composite validation service for actual validation.
    """

    def __init__(self):
        """Initialize the validation service."""
        self.logger = logging.getLogger(__name__)
        self.validator = CompositeValidationService()

    # Model validation methods

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
        return self.validator.validate_model_metadata(data)

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
        return self.validator.validate_model_id(model_id)

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
        return self.validator.validate_model_choice(choice)

    # Input validation methods

    def validate_prompt(self, prompt: str, max_length: int = 1000, allow_empty: bool = False) -> bool:
        """
        Validate text prompt format and content.

        Args:
            prompt: Prompt to validate
            max_length: Maximum allowed length
            allow_empty: Whether empty prompts are allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_prompt(prompt, max_length, allow_empty)

    def validate_negative_prompt(self, prompt: str, max_length: int = 1000) -> bool:
        """
        Validate negative prompt format and content.

        Args:
            prompt: Negative prompt to validate
            max_length: Maximum allowed length

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_negative_prompt(prompt, max_length)

    def validate_aspect_ratio(self, aspect_ratio: str, allowed_ratios: Optional[List[str]] = None) -> bool:
        """
        Validate aspect ratio.

        Args:
            aspect_ratio: Aspect ratio to validate
            allowed_ratios: List of allowed aspect ratios

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_aspect_ratio(aspect_ratio, allowed_ratios)

    def validate_numeric_param(
        self,
        value: Any,
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_none: bool = True
    ) -> bool:
        """
        Validate numeric parameter within range.

        Args:
            value: Value to validate
            field_name: Name of the field for error message
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_numeric_param(value, field_name, min_val, max_val, allow_none)

    def validate_dimensions(
        self,
        width: Any,
        height: Any,
        min_dim: int = 256,
        max_dim: int = 1440,
        step: int = 32,
        allow_none: bool = False
    ) -> bool:
        """
        Validate image dimensions.

        Args:
            width: Image width to validate
            height: Image height to validate
            min_dim: Minimum allowed dimension
            max_dim: Maximum allowed dimension
            step: Dimension step size
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_dimensions(width, height, min_dim, max_dim, step, allow_none)

    def validate_output_format(
        self,
        format: str,
        allowed_formats: Optional[List[str]] = None
    ) -> bool:
        """
        Validate output format.

        Args:
            format: Output format to validate
            allowed_formats: List of allowed formats

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_output_format(format, allowed_formats)

    # Common methods

    def sanitize_display_text(self, text: Any) -> str:
        """
        Sanitize text for display in UI.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        return self.validator.sanitize_display_text(text)

    # Batch validation methods

    def validate_generation_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate image generation parameters.

        Args:
            params: Generation parameters to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate required parameters
            self.validate_prompt(params.get('prompt', ''))

            # Validate optional parameters
            if 'negative_prompt' in params:
                self.validate_negative_prompt(params['negative_prompt'])

            if 'aspect_ratio' in params:
                allowed_ratios = [
                    "21:9", "16:9", "3:2", "4:3", "1:1",
                    "3:4", "2:3", "9:16", "9:21"
                ]
                self.validate_aspect_ratio(params['aspect_ratio'], allowed_ratios)

            if 'num_steps' in params:
                self.validator.validate_num_steps(params['num_steps'])

            if 'guidance_scale' in params:
                self.validator.validate_guidance_scale(params['guidance_scale'])

            if 'strength' in params:
                self.validator.validate_strength(params['strength'])

            if 'seed' in params:
                self.validator.validate_seed(params['seed'])

            if 'width' in params and 'height' in params:
                self.validate_dimensions(params['width'], params['height'])

            if 'safety_tolerance' in params:
                self.validator.validate_safety_tolerance(params['safety_tolerance'])

            if 'output_format' in params:
                self.validate_output_format(params['output_format'], ['jpeg', 'png'])

            return True

        except ValidationError as e:
            self.logger.error(f"Validation error in generation parameters: {e}")
            raise

    def validate_finetune_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate finetuning parameters.

        Args:
            params: Finetuning parameters to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate required parameters
            if 'model_name' not in params:
                raise ValidationError("Model name is required", "model_name", None)
            self.validator.validate_string(params['model_name'], 'model_name', min_length=1)

            if 'trigger_word' not in params:
                raise ValidationError("Trigger word is required", "trigger_word", None)
            self.validator.validate_trigger_word(params['trigger_word'])

            if 'mode' not in params:
                raise ValidationError("Mode is required", "mode", None)
            self.validator.validate_model_mode(params['mode'])

            # Validate optional parameters
            if 'rank' in params:
                self.validator.validate_rank(params['rank'])

            if 'iterations' in params:
                self.validator.validate_iterations(params['iterations'])

            if 'learning_rate' in params:
                self.validator.validate_numeric_param(
                    params['learning_rate'],
                    'learning_rate',
                    min_val=0.0001,
                    max_val=0.1
                )

            return True

        except ValidationError as e:
            self.logger.error(f"Validation error in finetuning parameters: {e}")
            raise

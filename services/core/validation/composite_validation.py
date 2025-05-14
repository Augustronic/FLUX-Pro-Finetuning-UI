"""
Composite Validation Service for FLUX-Pro-Finetuning-UI.

Provides a unified interface for validation by delegating to specialized validators.
"""

from typing import Dict, Any, Optional, List, Union
import logging
from .base_validation import BaseValidationService, ValidationError
from .model_validation import ModelValidationService
from .input_validation import InputValidationService


class CompositeValidationService(BaseValidationService):
    """
    Composite validation service that delegates to specialized validators.

    Provides a unified interface for validation while maintaining separation of concerns.
    """

    def __init__(self):
        """Initialize the composite validation service."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize specialized validators
        self.model_validator = ModelValidationService()
        self.input_validator = InputValidationService()

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
        return self.model_validator.validate_model_metadata(data)

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
        return self.model_validator.validate_model_id(model_id)

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
        return self.model_validator.validate_model_choice(choice)

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
        return self.model_validator.validate_model_type(model_type, allowed_types)

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
        return self.model_validator.validate_model_mode(mode, allowed_modes)

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
        return self.model_validator.validate_trigger_word(trigger_word)

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
        return self.model_validator.validate_rank(rank)

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
        return self.model_validator.validate_iterations(iterations)

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
        return self.input_validator.validate_prompt(prompt, max_length, allow_empty)

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
        return self.input_validator.validate_negative_prompt(prompt, max_length)

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
        return self.input_validator.validate_aspect_ratio(aspect_ratio, allowed_ratios)

    def validate_num_steps(
        self,
        num_steps: Any,
        min_steps: int = 1,
        max_steps: int = 50,
        allow_none: bool = True
    ) -> bool:
        """
        Validate number of steps.

        Args:
            num_steps: Number of steps to validate
            min_steps: Minimum allowed steps
            max_steps: Maximum allowed steps
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.input_validator.validate_num_steps(num_steps, min_steps, max_steps, allow_none)

    def validate_guidance_scale(
        self,
        guidance_scale: Any,
        min_scale: float = 1.0,
        max_scale: float = 20.0,
        allow_none: bool = True
    ) -> bool:
        """
        Validate guidance scale.

        Args:
            guidance_scale: Guidance scale to validate
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.input_validator.validate_guidance_scale(guidance_scale, min_scale, max_scale, allow_none)

    def validate_strength(
        self,
        strength: Any,
        min_strength: float = 0.1,
        max_strength: float = 2.0,
        allow_none: bool = False
    ) -> bool:
        """
        Validate finetune strength.

        Args:
            strength: Finetune strength to validate
            min_strength: Minimum allowed strength
            max_strength: Maximum allowed strength
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.input_validator.validate_strength(strength, min_strength, max_strength, allow_none)

    def validate_seed(
        self,
        seed: Any,
        min_seed: int = 0,
        max_seed: int = 9999999999,
        allow_none: bool = True
    ) -> bool:
        """
        Validate random seed.

        Args:
            seed: Random seed to validate
            min_seed: Minimum allowed seed
            max_seed: Maximum allowed seed
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.input_validator.validate_seed(seed, min_seed, max_seed, allow_none)

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
        return self.input_validator.validate_dimensions(width, height, min_dim, max_dim, step, allow_none)

    def validate_safety_tolerance(
        self,
        tolerance: Any,
        min_tolerance: int = 0,
        max_tolerance: int = 6,
        allow_none: bool = False
    ) -> bool:
        """
        Validate safety tolerance.

        Args:
            tolerance: Safety tolerance to validate
            min_tolerance: Minimum allowed tolerance
            max_tolerance: Maximum allowed tolerance
            allow_none: Whether None is allowed

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        return self.input_validator.validate_safety_tolerance(tolerance, min_tolerance, max_tolerance, allow_none)

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
        return self.input_validator.validate_output_format(format, allowed_formats)

    # Common methods

    def sanitize_display_text(self, text: Any) -> str:
        """
        Sanitize text for display in UI.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        return self.input_validator.sanitize_display_text(text)

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
        return self.validate_numeric(value, field_name, min_val, max_val, allow_none)

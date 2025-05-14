"""
Input Validation Service for FLUX-Pro-Finetuning-UI.

Provides validation functionality for user input data.
"""

import re
import os
from typing import Any, Optional, List, Dict, Union
import logging
from .base_validation import BaseValidationService, ValidationError


class InputValidationService(BaseValidationService):
    """
    Validation service for user input.

    Provides validation methods for prompts, parameters, etc.
    """

    def __init__(self):
        """Initialize the input validation service."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

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
        if not prompt or not isinstance(prompt, str):
            if allow_empty:
                return True
            raise ValidationError("Prompt must be a non-empty string", "prompt", prompt)

        if len(prompt) > max_length:
            raise ValidationError(
                f"Prompt is too long (max {max_length} characters)",
                "prompt",
                prompt
            )

        return True

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
        # Negative prompt can be empty
        if not prompt:
            return True

        if not isinstance(prompt, str):
            raise ValidationError("Negative prompt must be a string", "negative_prompt", prompt)

        if len(prompt) > max_length:
            raise ValidationError(
                f"Negative prompt is too long (max {max_length} characters)",
                "negative_prompt",
                prompt
            )

        return True

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
        if not aspect_ratio or not isinstance(aspect_ratio, str):
            raise ValidationError("Invalid aspect ratio", "aspect_ratio", aspect_ratio)

        if allowed_ratios and aspect_ratio not in allowed_ratios:
            raise ValidationError(
                f"Aspect ratio must be one of: {', '.join(allowed_ratios)}",
                "aspect_ratio",
                aspect_ratio
            )

        return True

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
        if num_steps is None:
            if allow_none:
                return True
            raise ValidationError("Number of steps is required", "num_steps", num_steps)

        try:
            steps = int(num_steps)
            if steps < min_steps:
                raise ValidationError(
                    f"Number of steps must be at least {min_steps}",
                    "num_steps",
                    num_steps
                )
            if steps > max_steps:
                raise ValidationError(
                    f"Number of steps cannot exceed {max_steps}",
                    "num_steps",
                    num_steps
                )
        except (ValueError, TypeError):
            raise ValidationError("Number of steps must be an integer", "num_steps", num_steps)

        return True

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
        if guidance_scale is None:
            if allow_none:
                return True
            raise ValidationError("Guidance scale is required", "guidance_scale", guidance_scale)

        try:
            scale = float(guidance_scale)
            if scale < min_scale:
                raise ValidationError(
                    f"Guidance scale must be at least {min_scale}",
                    "guidance_scale",
                    guidance_scale
                )
            if scale > max_scale:
                raise ValidationError(
                    f"Guidance scale cannot exceed {max_scale}",
                    "guidance_scale",
                    guidance_scale
                )
        except (ValueError, TypeError):
            raise ValidationError("Guidance scale must be a number", "guidance_scale", guidance_scale)

        return True

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
        if strength is None:
            if allow_none:
                return True
            raise ValidationError("Finetune strength is required", "strength", strength)

        try:
            strength_value = float(strength)
            if strength_value < min_strength:
                raise ValidationError(
                    f"Finetune strength must be at least {min_strength}",
                    "strength",
                    strength
                )
            if strength_value > max_strength:
                raise ValidationError(
                    f"Finetune strength cannot exceed {max_strength}",
                    "strength",
                    strength
                )
        except (ValueError, TypeError):
            raise ValidationError("Finetune strength must be a number", "strength", strength)

        return True

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
        if seed is None or seed == "":
            if allow_none:
                return True
            raise ValidationError("Random seed is required", "seed", seed)

        try:
            seed_value = int(seed)
            if seed_value < min_seed:
                raise ValidationError(
                    f"Random seed must be at least {min_seed}",
                    "seed",
                    seed
                )
            if seed_value > max_seed:
                raise ValidationError(
                    f"Random seed cannot exceed {max_seed}",
                    "seed",
                    seed
                )
        except (ValueError, TypeError):
            raise ValidationError("Random seed must be an integer", "seed", seed)

        return True

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
        # Validate width
        if width is None:
            if allow_none:
                return True
            raise ValidationError("Image width is required", "width", width)

        try:
            width_value = int(width)
            if width_value < min_dim:
                raise ValidationError(
                    f"Image width must be at least {min_dim}",
                    "width",
                    width
                )
            if width_value > max_dim:
                raise ValidationError(
                    f"Image width cannot exceed {max_dim}",
                    "width",
                    width
                )
            if width_value % step != 0:
                raise ValidationError(
                    f"Image width must be a multiple of {step}",
                    "width",
                    width
                )
        except (ValueError, TypeError):
            raise ValidationError("Image width must be an integer", "width", width)

        # Validate height
        if height is None:
            if allow_none:
                return True
            raise ValidationError("Image height is required", "height", height)

        try:
            height_value = int(height)
            if height_value < min_dim:
                raise ValidationError(
                    f"Image height must be at least {min_dim}",
                    "height",
                    height
                )
            if height_value > max_dim:
                raise ValidationError(
                    f"Image height cannot exceed {max_dim}",
                    "height",
                    height
                )
            if height_value % step != 0:
                raise ValidationError(
                    f"Image height must be a multiple of {step}",
                    "height",
                    height
                )
        except (ValueError, TypeError):
            raise ValidationError("Image height must be an integer", "height", height)

        return True

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
        if tolerance is None:
            if allow_none:
                return True
            raise ValidationError("Safety tolerance is required", "safety_tolerance", tolerance)

        try:
            tolerance_value = int(tolerance)
            if tolerance_value < min_tolerance:
                raise ValidationError(
                    f"Safety tolerance must be at least {min_tolerance}",
                    "safety_tolerance",
                    tolerance
                )
            if tolerance_value > max_tolerance:
                raise ValidationError(
                    f"Safety tolerance cannot exceed {max_tolerance}",
                    "safety_tolerance",
                    tolerance
                )
        except (ValueError, TypeError):
            raise ValidationError("Safety tolerance must be an integer", "safety_tolerance", tolerance)

        return True

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
        if not format or not isinstance(format, str):
            raise ValidationError("Invalid output format", "output_format", format)

        if allowed_formats and format.lower() not in [f.lower() for f in allowed_formats]:
            raise ValidationError(
                f"Output format must be one of: {', '.join(allowed_formats)}",
                "output_format",
                format
            )

        return True

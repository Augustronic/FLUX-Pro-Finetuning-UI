"""
Validation Utilities for FLUX-Pro-Finetuning-UI.

Provides centralized validation functions for the application.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple


class ValidationError(Exception):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.field = field
        self.value = value
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


# Error codes
ERROR_REQUIRED = "required"
ERROR_TYPE = "type"
ERROR_FORMAT = "format"
ERROR_RANGE = "range"
ERROR_LENGTH = "length"
ERROR_PATTERN = "pattern"
ERROR_ENUM = "enum"
ERROR_UNIQUE = "unique"
ERROR_CUSTOM = "custom"


def validate_required(value: Any, field: str) -> None:
    """
    Validate that a value is not None or empty.

    Args:
        value: Value to validate
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError(
            f"{field} is required",
            field,
            value,
            ERROR_REQUIRED
        )

    if isinstance(value, str) and not value.strip():
        raise ValidationError(
            f"{field} cannot be empty",
            field,
            value,
            ERROR_REQUIRED
        )

    if isinstance(value, (list, dict)) and not value:
        raise ValidationError(
            f"{field} cannot be empty",
            field,
            value,
            ERROR_REQUIRED
        )


def validate_type(value: Any, expected_type: Union[type, Tuple[type, ...]], field: str) -> None:
    """
    Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is not None and not isinstance(value, expected_type):
        type_name = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValidationError(
            f"{field} must be of type {type_name}",
            field,
            value,
            ERROR_TYPE
        )


def validate_numeric_range(
    value: Union[int, float],
    field: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> None:
    """
    Validate that a numeric value is within the specified range.

    Args:
        value: Value to validate
        field: Field name for error message
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field} must be a number",
            field,
            value,
            ERROR_TYPE
        )

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{field} must be at least {min_value}",
            field,
            value,
            ERROR_RANGE,
            {"min_value": min_value}
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field} must be at most {max_value}",
            field,
            value,
            ERROR_RANGE,
            {"max_value": max_value}
        )


def validate_string_length(
    value: str,
    field: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
) -> None:
    """
    Validate that a string value has the specified length.

    Args:
        value: Value to validate
        field: Field name for error message
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"{field} must be at least {min_length} characters",
            field,
            value,
            ERROR_LENGTH,
            {"min_length": min_length}
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field} must be at most {max_length} characters",
            field,
            value,
            ERROR_LENGTH,
            {"max_length": max_length}
        )


def validate_pattern(value: str, pattern: str, field: str) -> None:
    """
    Validate that a string value matches the specified pattern.

    Args:
        value: Value to validate
        pattern: Regular expression pattern
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    if not re.match(pattern, value):
        raise ValidationError(
            f"{field} must match pattern {pattern}",
            field,
            value,
            ERROR_PATTERN,
            {"pattern": pattern}
        )


def validate_enum(value: Any, allowed_values: List[Any], field: str) -> None:
    """
    Validate that a value is one of the allowed values.

    Args:
        value: Value to validate
        allowed_values: List of allowed values
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if value not in allowed_values:
        raise ValidationError(
            f"{field} must be one of: {', '.join(map(str, allowed_values))}",
            field,
            value,
            ERROR_ENUM,
            {"allowed_values": allowed_values}
        )


def validate_url(value: str, field: str) -> None:
    """
    Validate that a string value is a valid URL.

    Args:
        value: Value to validate
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    # Simple URL validation
    url_pattern = r"^(https?|ftp)://[^\s/$.?#].[^\s]*$|^data:[^;]+;base64,[a-zA-Z0-9+/]+=*$"
    if not re.match(url_pattern, value):
        raise ValidationError(
            f"{field} must be a valid URL",
            field,
            value,
            ERROR_FORMAT
        )


def validate_email(value: str, field: str) -> None:
    """
    Validate that a string value is a valid email address.

    Args:
        value: Value to validate
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    # Simple email validation
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(email_pattern, value):
        raise ValidationError(
            f"{field} must be a valid email address",
            field,
            value,
            ERROR_FORMAT
        )


def validate_file_extension(value: str, allowed_extensions: List[str], field: str) -> None:
    """
    Validate that a file has one of the allowed extensions.

    Args:
        value: File path or name
        allowed_extensions: List of allowed extensions (without dot)
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    # Get file extension
    extension = os.path.splitext(value)[1].lstrip(".")

    if extension.lower() not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(
            f"{field} must have one of these extensions: {', '.join(allowed_extensions)}",
            field,
            value,
            ERROR_FORMAT,
            {"allowed_extensions": allowed_extensions}
        )


def validate_json(value: str, field: str) -> None:
    """
    Validate that a string value is valid JSON.

    Args:
        value: Value to validate
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return

    if not isinstance(value, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            value,
            ERROR_TYPE
        )

    try:
        json.loads(value)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"{field} must be valid JSON: {str(e)}",
            field,
            value,
            ERROR_FORMAT
        )


def validate_dimensions(
    width: int,
    height: int,
    min_dim: int = 256,
    max_dim: int = 1440,
    step: int = 32,
    field: str = "dimensions"
) -> None:
    """
    Validate image dimensions.

    Args:
        width: Image width
        height: Image height
        min_dim: Minimum allowed dimension
        max_dim: Maximum allowed dimension
        step: Dimension step size
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    # Validate width
    if not isinstance(width, int):
        raise ValidationError(
            "Width must be an integer",
            "width",
            width,
            ERROR_TYPE
        )

    if width < min_dim:
        raise ValidationError(
            f"Width must be at least {min_dim} pixels",
            "width",
            width,
            ERROR_RANGE,
            {"min_dim": min_dim}
        )

    if width > max_dim:
        raise ValidationError(
            f"Width must be at most {max_dim} pixels",
            "width",
            width,
            ERROR_RANGE,
            {"max_dim": max_dim}
        )

    if width % step != 0:
        raise ValidationError(
            f"Width must be a multiple of {step}",
            "width",
            width,
            ERROR_CUSTOM,
            {"step": step}
        )

    # Validate height
    if not isinstance(height, int):
        raise ValidationError(
            "Height must be an integer",
            "height",
            height,
            ERROR_TYPE
        )

    if height < min_dim:
        raise ValidationError(
            f"Height must be at least {min_dim} pixels",
            "height",
            height,
            ERROR_RANGE,
            {"min_dim": min_dim}
        )

    if height > max_dim:
        raise ValidationError(
            f"Height must be at most {max_dim} pixels",
            "height",
            height,
            ERROR_RANGE,
            {"max_dim": max_dim}
        )

    if height % step != 0:
        raise ValidationError(
            f"Height must be a multiple of {step}",
            "height",
            height,
            ERROR_CUSTOM,
            {"step": step}
        )


def validate_aspect_ratio(aspect_ratio: str, allowed_ratios: Optional[List[str]] = None, field: str = "aspect_ratio") -> None:
    """
    Validate aspect ratio.

    Args:
        aspect_ratio: Aspect ratio to validate (e.g., "16:9")
        allowed_ratios: List of allowed aspect ratios
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if aspect_ratio is None:
        return

    if not isinstance(aspect_ratio, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            aspect_ratio,
            ERROR_TYPE
        )

    # Check if aspect ratio is in allowed ratios
    if allowed_ratios and aspect_ratio not in allowed_ratios:
        raise ValidationError(
            f"{field} must be one of: {', '.join(allowed_ratios)}",
            field,
            aspect_ratio,
            ERROR_ENUM,
            {"allowed_ratios": allowed_ratios}
        )

    # Validate aspect ratio format
    if ":" not in aspect_ratio:
        raise ValidationError(
            f"{field} must be in format 'width:height'",
            field,
            aspect_ratio,
            ERROR_FORMAT
        )

    # Validate aspect ratio values
    try:
        width, height = map(int, aspect_ratio.split(":"))
        if width <= 0 or height <= 0:
            raise ValidationError(
                f"{field} must have positive values",
                field,
                aspect_ratio,
                ERROR_RANGE
            )
    except ValueError:
        raise ValidationError(
            f"{field} must contain numeric values",
            field,
            aspect_ratio,
            ERROR_FORMAT
        )


def validate_prompt(prompt: str, max_length: int = 1000, allow_empty: bool = False, field: str = "prompt") -> None:
    """
    Validate text prompt.

    Args:
        prompt: Prompt to validate
        max_length: Maximum allowed length
        allow_empty: Whether empty prompts are allowed
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if prompt is None:
        if not allow_empty:
            raise ValidationError(
                f"{field} is required",
                field,
                prompt,
                ERROR_REQUIRED
            )
        return

    if not isinstance(prompt, str):
        raise ValidationError(
            f"{field} must be a string",
            field,
            prompt,
            ERROR_TYPE
        )

    if not allow_empty and not prompt.strip():
        raise ValidationError(
            f"{field} cannot be empty",
            field,
            prompt,
            ERROR_REQUIRED
        )

    if len(prompt) > max_length:
        raise ValidationError(
            f"{field} must be at most {max_length} characters",
            field,
            prompt,
            ERROR_LENGTH,
            {"max_length": max_length}
        )


def validate_seed(seed: Any, min_seed: int = 0, max_seed: int = 9999999999, allow_none: bool = True, field: str = "seed") -> None:
    """
    Validate random seed.

    Args:
        seed: Seed to validate
        min_seed: Minimum allowed seed
        max_seed: Maximum allowed seed
        allow_none: Whether None is allowed
        field: Field name for error message

    Raises:
        ValidationError: If validation fails
    """
    if seed is None:
        if not allow_none:
            raise ValidationError(
                f"{field} is required",
                field,
                seed,
                ERROR_REQUIRED
            )
        return

    if not isinstance(seed, int):
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            raise ValidationError(
                f"{field} must be an integer",
                field,
                seed,
                ERROR_TYPE
            )

    if seed < min_seed:
        raise ValidationError(
            f"{field} must be at least {min_seed}",
            field,
            seed,
            ERROR_RANGE,
            {"min_seed": min_seed}
        )

    if seed > max_seed:
        raise ValidationError(
            f"{field} must be at most {max_seed}",
            field,
            seed,
            ERROR_RANGE,
            {"max_seed": max_seed}
        )


def sanitize_display_text(text: Any) -> str:
    """
    Sanitize text for display in UI.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Replace potentially harmful characters
    text = re.sub(r"[<>\"\'&]", "", text)

    return text

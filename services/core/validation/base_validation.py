"""
Base Validation Service for FLUX-Pro-Finetuning-UI.

Provides base functionality for validation services.
"""

from typing import Any, Optional
import logging


class ValidationError(Exception):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
        """
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class BaseValidationService:
    """
    Base class for validation services.
    
    Provides common validation functionality.
    """
    
    def __init__(self):
        """Initialize the base validation service."""
        self.logger = logging.getLogger(__name__)
    
    def sanitize_display_text(self, text: Any) -> str:
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
            try:
                text = str(text)
            except Exception:
                return ""
                
        # Remove potentially harmful characters
        return text.replace("<", "&lt;").replace(">", "&gt;")
    
    def validate_required(self, value: Any, field_name: str) -> bool:
        """
        Validate that a value is not None or empty.
        
        Args:
            value: Value to validate
            field_name: Name of the field for error message
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            raise ValidationError(f"{field_name} is required", field_name, value)
            
        if isinstance(value, str) and not value.strip():
            raise ValidationError(f"{field_name} cannot be empty", field_name, value)
            
        if isinstance(value, (list, dict)) and not value:
            raise ValidationError(f"{field_name} cannot be empty", field_name, value)
            
        return True
    
    def validate_string(
        self,
        value: Any,
        field_name: str,
        min_length: int = 0,
        max_length: Optional[int] = None,
        allow_none: bool = False
    ) -> bool:
        """
        Validate that a value is a string with valid length.
        
        Args:
            value: Value to validate
            field_name: Name of the field for error message
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_none: Whether None is allowed
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            if allow_none:
                return True
            raise ValidationError(f"{field_name} is required", field_name, value)
            
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field_name, value)
            
        if min_length > 0 and len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field_name,
                value
            )
            
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{field_name} cannot exceed {max_length} characters",
                field_name,
                value
            )
            
        return True
    
    def validate_numeric(
        self,
        value: Any,
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        allow_none: bool = False
    ) -> bool:
        """
        Validate that a value is a number within a valid range.
        
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
        if value is None:
            if allow_none:
                return True
            raise ValidationError(f"{field_name} is required", field_name, value)
            
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a number", field_name, value)
            
        if min_val is not None and numeric_value < min_val:
            raise ValidationError(
                f"{field_name} must be at least {min_val}",
                field_name,
                value
            )
            
        if max_val is not None and numeric_value > max_val:
            raise ValidationError(
                f"{field_name} cannot exceed {max_val}",
                field_name,
                value
            )
            
        return True
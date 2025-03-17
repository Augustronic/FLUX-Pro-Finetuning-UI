"""
Validation Service for FLUX-Pro-Finetuning-UI.

Provides centralized validation logic for all input data with consistent
error handling and sanitization.
"""

import re
from typing import Any, Dict, Optional, Union, List, Tuple
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


class ValidationService:
    """
    Service for validating input data.
    
    Provides methods for validating different types of input data with
    consistent error handling and sanitization.
    """
    
    def __init__(self):
        """Initialize the validation service."""
        self.logger = logging.getLogger(__name__)
    
    def validate_model_metadata(self, data: Dict[str, Any]) -> bool:
        """
        Validate model metadata format.
        
        Args:
            data: Model metadata to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        required_fields = {
            'finetune_id': str,
            'model_name': str,
            'trigger_word': str,
            'mode': str,
            'type': str
        }
        
        # Check required fields
        for field, field_type in required_fields.items():
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)
            
            if not isinstance(data[field], field_type):
                raise ValidationError(
                    f"Field {field} must be of type {field_type.__name__}",
                    field,
                    data[field]
                )
        
        # Validate finetune_id format (alphanumeric with hyphens)
        if not re.match(r'^[a-zA-Z0-9-]+$', data['finetune_id']):
            raise ValidationError(
                "Invalid finetune_id format (must be alphanumeric with hyphens)",
                'finetune_id',
                data['finetune_id']
            )
        
        # Validate model_name (alphanumeric with basic punctuation)
        if not re.match(r'^[\w\s\-_.]+$', data['model_name']):
            raise ValidationError(
                "Invalid model_name format (must be alphanumeric with basic punctuation)",
                'model_name',
                data['model_name']
            )
        
        # Validate trigger_word (alphanumeric with basic punctuation)
        if not re.match(r'^[\w\s\-_.]+$', data['trigger_word']):
            raise ValidationError(
                "Invalid trigger_word format (must be alphanumeric with basic punctuation)",
                'trigger_word',
                data['trigger_word']
            )
        
        # Validate mode is one of the allowed values
        if data['mode'] not in ['general', 'character', 'style', 'product']:
            raise ValidationError(
                "Invalid mode (must be one of: general, character, style, product)",
                'mode',
                data['mode']
            )
        
        # Validate type is either 'lora' or 'full'
        if data['type'] not in ['lora', 'full']:
            raise ValidationError(
                "Invalid type (must be one of: lora, full)",
                'type',
                data['type']
            )
        
        return True
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        Validate text prompt format and content.
        
        Args:
            prompt: Text prompt to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt cannot be empty", "prompt", prompt)
        
        # Remove excessive whitespace
        prompt = prompt.strip()
        if len(prompt) == 0:
            raise ValidationError("Prompt cannot be empty", "prompt", prompt)
        
        # Check for valid characters
        if not re.match(r'^[\w\s\-_.,!?()[\]{}@#$%^&*+=<>:/\\|\'\"]+$', prompt):
            raise ValidationError(
                "Prompt contains invalid characters",
                "prompt",
                prompt
            )
        
        # Check length
        if len(prompt) > 1000:  # Maximum prompt length
            raise ValidationError(
                f"Prompt is too long (max 1000 characters, got {len(prompt)})",
                "prompt",
                prompt
            )
        
        return True
    
    def validate_numeric_param(
        self,
        value: Optional[Union[int, float]],
        min_val: Union[int, float],
        max_val: Union[int, float],
        allow_none: bool = True,
        field_name: str = "parameter"
    ) -> bool:
        """
        Validate numeric parameter within range.
        
        Args:
            value: Value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_none: Whether None is allowed
            field_name: Name of the field for error messages
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        if value is None:
            if allow_none:
                return True
            else:
                raise ValidationError(
                    f"{field_name} cannot be None",
                    field_name,
                    value
                )
        
        try:
            numeric_value = float(value)
            
            if numeric_value < min_val:
                raise ValidationError(
                    f"{field_name} must be at least {min_val} (got {numeric_value})",
                    field_name,
                    value
                )
                
            if numeric_value > max_val:
                raise ValidationError(
                    f"{field_name} must be at most {max_val} (got {numeric_value})",
                    field_name,
                    value
                )
                
            return True
            
        except (TypeError, ValueError):
            raise ValidationError(
                f"{field_name} must be a number",
                field_name,
                value
            )
    
    def sanitize_display_text(self, text: str, max_length: int = 100) -> str:
        """
        Sanitize text for display in UI.
        
        Args:
            text: Text to sanitize
            max_length: Maximum length of sanitized text
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove control characters and limit length
        text = "".join(char for char in text if char.isprintable())
        text = text[:max_length]  # Limit length
        
        # Only allow alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-_.,()]', '', text)
        return text.strip()
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("API key cannot be empty", "api_key")
        
        # Remove whitespace
        api_key = api_key.strip()
        if len(api_key) == 0:
            raise ValidationError("API key cannot be empty", "api_key")
        
        # Check for valid format (UUID-like or API key format)
        if not re.match(r'^[a-zA-Z0-9-]+$', api_key):
            raise ValidationError(
                "API key contains invalid characters",
                "api_key"
            )
        
        return True
    
    def validate_url(self, url: str, field_name: str = "url") -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            field_name: Name of the field for error messages
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        if not url or not isinstance(url, str):
            raise ValidationError(f"{field_name} cannot be empty", field_name, url)
        
        # Remove whitespace
        url = url.strip()
        if len(url) == 0:
            raise ValidationError(f"{field_name} cannot be empty", field_name, url)
        
        # Check for valid format
        if url.startswith("data:"):
            # Validate base64 data URL format
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:"):
                    raise ValidationError(
                        f"Invalid {field_name} format (invalid data URL)",
                        field_name,
                        url
                    )
                return True
            except Exception:
                raise ValidationError(
                    f"Invalid {field_name} format (invalid data URL)",
                    field_name,
                    url
                )
        else:
            # Validate HTTP(S) URL
            if not re.match(r'^https?://[\w\-.]+(?::\d+)?(?:/[^?]*)?(?:\?[^#]*)?$', url):
                raise ValidationError(
                    f"Invalid {field_name} format (must be a valid HTTP/HTTPS URL)",
                    field_name,
                    url
                )
        
        return True
    
    def validate_file_extension(
        self,
        filename: str,
        allowed_extensions: List[str],
        field_name: str = "file"
    ) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: Filename to validate
            allowed_extensions: List of allowed extensions (without dot)
            field_name: Name of the field for error messages
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        if not filename or not isinstance(filename, str):
            raise ValidationError(f"{field_name} name cannot be empty", field_name, filename)
        
        # Get extension
        extension = filename.split(".")[-1].lower() if "." in filename else ""
        
        if not extension:
            raise ValidationError(
                f"{field_name} must have an extension",
                field_name,
                filename
            )
        
        if extension not in allowed_extensions:
            raise ValidationError(
                f"{field_name} must have one of these extensions: {', '.join(allowed_extensions)}",
                field_name,
                filename
            )
        
        return True
    
    def validate_dimensions(
        self,
        width: int,
        height: int,
        min_dim: int = 256,
        max_dim: int = 1440,
        multiple_of: int = 32
    ) -> bool:
        """
        Validate image dimensions.
        
        Args:
            width: Image width
            height: Image height
            min_dim: Minimum dimension
            max_dim: Maximum dimension
            multiple_of: Dimensions must be a multiple of this value
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValidationError: If validation fails with details
        """
        # Validate width
        if not isinstance(width, int):
            raise ValidationError("Width must be an integer", "width", width)
        
        if width < min_dim:
            raise ValidationError(
                f"Width must be at least {min_dim} pixels",
                "width",
                width
            )
        
        if width > max_dim:
            raise ValidationError(
                f"Width must be at most {max_dim} pixels",
                "width",
                width
            )
        
        if width % multiple_of != 0:
            raise ValidationError(
                f"Width must be a multiple of {multiple_of}",
                "width",
                width
            )
        
        # Validate height
        if not isinstance(height, int):
            raise ValidationError("Height must be an integer", "height", height)
        
        if height < min_dim:
            raise ValidationError(
                f"Height must be at least {min_dim} pixels",
                "height",
                height
            )
        
        if height > max_dim:
            raise ValidationError(
                f"Height must be at most {max_dim} pixels",
                "height",
                height
            )
        
        if height % multiple_of != 0:
            raise ValidationError(
                f"Height must be a multiple of {multiple_of}",
                "height",
                height
            )
        
        return True
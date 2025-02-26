"""Input validation utilities for FLUX Pro Finetuning UI."""

from typing import Any, Dict, List, Optional, Type, Union
import re
from dataclasses import dataclass
from utils.error_handling.error_handler import ValidationError, ErrorContext

@dataclass
class ValidationRule:
    """Validation rule configuration."""
    field: str
    rule_type: str
    value: Any = None
    message: Optional[str] = None

class Validator:
    """Input validation utility."""

    def __init__(self):
        """Initialize validator."""
        self._rules = {
            "required": self._validate_required,
            "min_length": self._validate_min_length,
            "max_length": self._validate_max_length,
            "pattern": self._validate_pattern,
            "type": self._validate_type,
            "range": self._validate_range,
            "enum": self._validate_enum,
            "custom": self._validate_custom
        }

    def validate(
        self,
        data: Dict[str, Any],
        rules: List[ValidationRule],
        component: str
    ) -> None:
        """Validate input data against rules.
        
        Args:
            data: Input data to validate
            rules: List of validation rules
            component: Component name for error context
            
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        
        for rule in rules:
            try:
                if rule.rule_type in self._rules:
                    self._rules[rule.rule_type](data.get(rule.field), rule)
            except ValidationError as e:
                errors.append({
                    "field": rule.field,
                    "message": str(e)
                })

        if errors:
            raise ValidationError(
                "Validation failed",
                context=ErrorContext(
                    component=component,
                    operation="validate",
                    details={"errors": errors}
                )
            )

    def _validate_required(self, value: Any, rule: ValidationRule) -> None:
        """Validate required field.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValidationError(
                rule.message or f"Field '{rule.field}' is required"
            )

    def _validate_min_length(self, value: Union[str, List], rule: ValidationRule) -> None:
        """Validate minimum length.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None and len(value) < rule.value:
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' must be at least {rule.value} characters long"
            )

    def _validate_max_length(self, value: Union[str, List], rule: ValidationRule) -> None:
        """Validate maximum length.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None and len(value) > rule.value:
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' must not exceed {rule.value} characters"
            )

    def _validate_pattern(self, value: str, rule: ValidationRule) -> None:
        """Validate regex pattern.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None and not re.match(rule.value, value):
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' does not match required pattern"
            )

    def _validate_type(self, value: Any, rule: ValidationRule) -> None:
        """Validate value type.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None and not isinstance(value, rule.value):
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' must be of type {rule.value.__name__}"
            )

    def _validate_range(self, value: Union[int, float], rule: ValidationRule) -> None:
        """Validate numeric range.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None:
            min_val, max_val = rule.value
            if value < min_val or value > max_val:
                raise ValidationError(
                    rule.message or 
                    f"Field '{rule.field}' must be between {min_val} and {max_val}"
                )

    def _validate_enum(self, value: Any, rule: ValidationRule) -> None:
        """Validate enum values.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if value is not None and value not in rule.value:
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' must be one of {rule.value}"
            )

    def _validate_custom(self, value: Any, rule: ValidationRule) -> None:
        """Run custom validation function.
        
        Args:
            value: Field value
            rule: Validation rule
            
        Raises:
            ValidationError: If validation fails
        """
        if not rule.value(value):
            raise ValidationError(
                rule.message or 
                f"Field '{rule.field}' failed custom validation"
            )

# Common validation rules
MODEL_NAME_RULES = [
    ValidationRule(
        field="model_name",
        rule_type="required"
    ),
    ValidationRule(
        field="model_name",
        rule_type="pattern",
        value=r"^[a-zA-Z0-9_-]+$",
        message="Model name can only contain letters, numbers, underscores and hyphens"
    ),
    ValidationRule(
        field="model_name",
        rule_type="max_length",
        value=100
    )
]

PROMPT_RULES = [
    ValidationRule(
        field="prompt",
        rule_type="required"
    ),
    ValidationRule(
        field="prompt",
        rule_type="max_length",
        value=1000
    )
]

API_KEY_RULES = [
    ValidationRule(
        field="api_key",
        rule_type="required"
    ),
    ValidationRule(
        field="api_key",
        rule_type="pattern",
        value=r"^[A-Za-z0-9-_]+$",
        message="Invalid API key format"
    )
]
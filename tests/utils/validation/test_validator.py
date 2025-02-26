"""Tests for validation utility."""

import unittest
from utils.validation.validator import (
    Validator,
    ValidationRule,
    ValidationError
)
from utils.error_handling.error_handler import ErrorContext

class TestValidator(unittest.TestCase):
    """Test cases for Validator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = Validator()
        self.test_component = "TestComponent"

    def test_required_validation(self):
        """Test required field validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="required"
            )
        ]

        # Test valid case
        valid_data = {"test_field": "value"}
        self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid cases
        invalid_cases = [
            {},
            {"test_field": None},
            {"test_field": ""},
            {"test_field": "   "}
        ]

        for invalid_data in invalid_cases:
            with self.assertRaises(ValidationError) as context:
                self.validator.validate(invalid_data, rules, self.test_component)
            self.assertIn("required", str(context.exception))

    def test_min_length_validation(self):
        """Test minimum length validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="min_length",
                value=3
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "abc"},
            {"test_field": "abcd"},
            {"test_field": [1, 2, 3]},
            {"test_field": None}  # None should pass as min_length isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": "ab"}, rules, self.test_component)
        self.assertIn("at least 3", str(context.exception))

    def test_max_length_validation(self):
        """Test maximum length validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="max_length",
                value=3
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "abc"},
            {"test_field": "ab"},
            {"test_field": [1, 2]},
            {"test_field": None}  # None should pass as max_length isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": "abcd"}, rules, self.test_component)
        self.assertIn("must not exceed", str(context.exception))

    def test_pattern_validation(self):
        """Test pattern validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="pattern",
                value=r"^[A-Za-z0-9_-]+$"
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "abc123"},
            {"test_field": "ABC_123"},
            {"test_field": "test-field"},
            {"test_field": None}  # None should pass as pattern isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": "invalid@pattern"}, rules, self.test_component)
        self.assertIn("pattern", str(context.exception))

    def test_type_validation(self):
        """Test type validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="type",
                value=str
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "string"},
            {"test_field": ""},
            {"test_field": None}  # None should pass as type isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": 123}, rules, self.test_component)
        self.assertIn("type", str(context.exception))

    def test_range_validation(self):
        """Test range validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="range",
                value=(1, 10)
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": 1},
            {"test_field": 5},
            {"test_field": 10},
            {"test_field": None}  # None should pass as range isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid cases
        invalid_cases = [
            {"test_field": 0},
            {"test_field": 11},
            {"test_field": -1}
        ]

        for invalid_data in invalid_cases:
            with self.assertRaises(ValidationError) as context:
                self.validator.validate(invalid_data, rules, self.test_component)
            self.assertIn("between", str(context.exception))

    def test_enum_validation(self):
        """Test enum validation."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="enum",
                value=["option1", "option2", "option3"]
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "option1"},
            {"test_field": "option2"},
            {"test_field": "option3"},
            {"test_field": None}  # None should pass as enum isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": "invalid_option"}, rules, self.test_component)
        self.assertIn("must be one of", str(context.exception))

    def test_custom_validation(self):
        """Test custom validation."""
        def custom_validator(value):
            return value and str(value).startswith("test_")

        rules = [
            ValidationRule(
                field="test_field",
                rule_type="custom",
                value=custom_validator,
                message="Value must start with 'test_'"
            )
        ]

        # Test valid cases
        valid_cases = [
            {"test_field": "test_value"},
            {"test_field": None}  # None should pass as custom isn't required
        ]

        for valid_data in valid_cases:
            self.validator.validate(valid_data, rules, self.test_component)

        # Test invalid case
        with self.assertRaises(ValidationError) as context:
            self.validator.validate({"test_field": "invalid_value"}, rules, self.test_component)
        self.assertIn("start with 'test_'", str(context.exception))

    def test_multiple_rules(self):
        """Test multiple validation rules."""
        rules = [
            ValidationRule(
                field="test_field",
                rule_type="required"
            ),
            ValidationRule(
                field="test_field",
                rule_type="min_length",
                value=3
            ),
            ValidationRule(
                field="test_field",
                rule_type="pattern",
                value=r"^[A-Za-z]+$"
            )
        ]

        # Test valid case
        self.validator.validate({"test_field": "valid"}, rules, self.test_component)

        # Test invalid cases
        invalid_cases = [
            {},  # Missing required field
            {"test_field": "ab"},  # Too short
            {"test_field": "123"}  # Invalid pattern
        ]

        for invalid_data in invalid_cases:
            with self.assertRaises(ValidationError):
                self.validator.validate(invalid_data, rules, self.test_component)

if __name__ == "__main__":
    unittest.main()
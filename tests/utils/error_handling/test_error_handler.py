"""Tests for error handling utility."""

import unittest
from unittest.mock import Mock, patch
import logging
from utils.error_handling.error_handler import (
    ErrorHandler,
    ErrorContext,
    ErrorSeverity,
    AppError,
    ValidationError,
    APIError,
    ConfigError
)

class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = Mock(spec=logging.Logger)
        self.handler = ErrorHandler(self.logger)

    def test_handle_error_with_app_error(self):
        """Test handling AppError."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            details={"test": "details"}
        )
        error = AppError(
            "Test error",
            severity=ErrorSeverity.ERROR,
            context=context
        )

        result = self.handler.handle_error(error)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["message"], "Test error")
        self.assertEqual(result["error"]["severity"], "ERROR")
        self.assertEqual(
            result["error"]["context"]["component"],
            "TestComponent"
        )
        self.logger.error.assert_called_once()

    def test_handle_error_with_standard_exception(self):
        """Test handling standard Python exception."""
        error = ValueError("Invalid value")
        result = self.handler.handle_error(error)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["message"], "Invalid value")
        self.assertEqual(result["error"]["severity"], "ERROR")
        self.logger.error.assert_called_once()

    def test_handle_error_with_validation_error(self):
        """Test handling ValidationError."""
        context = ErrorContext(
            component="TestComponent",
            operation="validate",
            details={"field": "test_field"}
        )
        error = ValidationError(
            "Validation failed",
            severity=ErrorSeverity.WARNING,
            context=context
        )

        result = self.handler.handle_error(error)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["message"], "Validation failed")
        self.assertEqual(result["error"]["severity"], "WARNING")
        self.logger.warning.assert_called_once()

    def test_handle_error_with_api_error(self):
        """Test handling APIError."""
        context = ErrorContext(
            component="TestComponent",
            operation="api_call",
            details={"endpoint": "/test"}
        )
        error = APIError(
            "API request failed",
            severity=ErrorSeverity.CRITICAL,
            context=context
        )

        result = self.handler.handle_error(error)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["message"], "API request failed")
        self.assertEqual(result["error"]["severity"], "CRITICAL")
        self.logger.critical.assert_called_once()

    def test_handle_error_with_config_error(self):
        """Test handling ConfigError."""
        context = ErrorContext(
            component="TestComponent",
            operation="load_config",
            details={"config_file": "test.json"}
        )
        error = ConfigError(
            "Configuration invalid",
            severity=ErrorSeverity.ERROR,
            context=context
        )

        result = self.handler.handle_error(error)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["message"], "Configuration invalid")
        self.assertEqual(result["error"]["severity"], "ERROR")
        self.logger.error.assert_called_once()

    def test_error_formatting(self):
        """Test error message formatting."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            details={"key": "value"}
        )
        error = AppError("Test error", context=context)

        formatted = self.handler._format_error_message(error)

        self.assertIn("Test error", formatted)
        self.assertIn("TestComponent", formatted)
        self.assertIn("test_operation", formatted)
        self.assertIn("key", formatted)
        self.assertIn("value", formatted)

    @patch("logging.getLogger")
    def test_default_logger_creation(self, mock_get_logger):
        """Test default logger creation."""
        mock_logger = Mock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger

        handler = ErrorHandler()
        self.assertEqual(handler.logger, mock_logger)
        mock_get_logger.assert_called_once_with("utils.error_handling.error_handler")

if __name__ == "__main__":
    unittest.main()
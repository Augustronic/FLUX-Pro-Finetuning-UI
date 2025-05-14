"""Centralized error handling for FLUX Pro Finetuning UI."""

import logging
from typing import Any, Optional, Dict, Type
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    operation: str
    details: Optional[Dict[str, Any]] = None

class AppError(Exception):
    """Base exception class for application errors."""
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Optional[ErrorContext] = None
    ):
        self.message = message
        self.severity = severity
        self.context = context
        super().__init__(self.message)

class ValidationError(AppError):
    """Validation related errors."""
    pass

class APIError(AppError):
    """API related errors."""
    pass

class ConfigError(AppError):
    """Configuration related errors."""
    pass

class ErrorHandler:
    """Centralized error handling utility."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        error_type: Optional[Type[AppError]] = None
    ) -> Dict[str, Any]:
        """Handle an error and return appropriate response.

        Args:
            error: The exception to handle
            context: Optional error context
            error_type: Optional specific error type to use

        Returns:
            Dict containing error details
        """
        # Convert to AppError if needed
        if not isinstance(error, AppError):
            if error_type:
                app_error = error_type(str(error), context=context)
            else:
                app_error = AppError(str(error), context=context)
        else:
            app_error = error

        # Log the error
        self._log_error(app_error)

        # Return error response
        return {
            "success": False,
            "error": {
                "message": app_error.message,
                "severity": app_error.severity.value,
                "context": {
                    "component": app_error.context.component if app_error.context else None,
                    "operation": app_error.context.operation if app_error.context else None,
                    "details": app_error.context.details if app_error.context else None
                }
            }
        }

    def _log_error(self, error: AppError) -> None:
        """Log error with appropriate severity.

        Args:
            error: The error to log
        """
        log_message = self._format_error_message(error)

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message, exc_info=True)
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _format_error_message(self, error: AppError) -> str:
        """Format error message with context.

        Args:
            error: The error to format

        Returns:
            Formatted error message
        """
        message_parts = [f"Error: {error.message}"]

        if error.context:
            message_parts.extend([
                f"Component: {error.context.component}",
                f"Operation: {error.context.operation}"
            ])

            if error.context.details:
                message_parts.append(f"Details: {error.context.details}")

        return " | ".join(message_parts)

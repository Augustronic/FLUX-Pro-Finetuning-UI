"""Logging configuration for FLUX Pro Finetuning UI."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class AppLogger:
    """Application logger configuration."""

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
            max_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(exist_ok=True)

        # File handler with rotation
        log_file = Path(log_dir) / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatters and add them to the handlers
        file_formatter = self._create_json_formatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging.

        Returns:
            Logging formatter for JSON output
        """
        class JSONFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                """Format log record as JSON.

                Args:
                    record: Log record to format

                Returns:
                    JSON formatted log string
                """
                # Extract basic log data
                log_data: Dict[str, Any] = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }

                # Add exception info if present
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)

                # Add any extra attributes from the record
                extra_attrs = {
                    key: value
                    for key, value in record.__dict__.items()
                    if key not in {
                        'args', 'asctime', 'created', 'exc_info', 'exc_text',
                        'filename', 'funcName', 'levelname', 'levelno',
                        'lineno', 'module', 'msecs', 'msg', 'name', 'pathname',
                        'process', 'processName', 'relativeCreated', 'stack_info',
                        'thread', 'threadName'
                    }
                }

                if extra_attrs:
                    log_data["extra"] = extra_attrs

                return json.dumps(log_data)

        return JSONFormatter()

    def get_logger(self) -> logging.Logger:
        """Get configured logger instance.

        Returns:
            Logger instance
        """
        return self.logger

def get_logger(
    name: Optional[str] = None,
    log_dir: str = "logs",
    log_level: int = logging.INFO
) -> logging.Logger:
    """Get or create a logger instance.

    Args:
        name: Logger name (optional)
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        Logger instance
    """
    logger_name = name or "flux_pro"
    return AppLogger(logger_name, log_dir, log_level).get_logger()

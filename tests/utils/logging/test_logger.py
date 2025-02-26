"""Tests for logging utility."""

import unittest
import logging
import json
import os
import shutil
from pathlib import Path
from utils.logging.logger import AppLogger, get_logger

class TestLogger(unittest.TestCase):
    """Test cases for logging utility."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_log_dir = "test_logs"
        self.test_logger_name = "test_logger"
        Path(self.test_log_dir).mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.test_log_dir).exists():
            shutil.rmtree(self.test_log_dir)

    def test_logger_creation(self):
        """Test logger initialization."""
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir
        ).get_logger()

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, self.test_logger_name)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(len(logger.handlers), 2)  # File and console handlers

    def test_log_file_creation(self):
        """Test log file is created."""
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir
        ).get_logger()

        log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log"
        self.assertTrue(log_file.exists())

    def test_json_formatting(self):
        """Test JSON formatting of log messages."""
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir
        ).get_logger()

        test_message = "Test log message"
        extra_data = {"test_key": "test_value"}
        
        logger.info(test_message, extra=extra_data)

        log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log"
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.readline())

        self.assertEqual(log_entry["message"], test_message)
        self.assertEqual(log_entry["level"], "INFO")
        self.assertEqual(log_entry["logger"], self.test_logger_name)
        self.assertEqual(log_entry["extra"]["test_key"], "test_value")

    def test_exception_logging(self):
        """Test exception logging."""
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir
        ).get_logger()

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Error occurred")

        log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log"
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.readline())

        self.assertIn("exception", log_entry)
        self.assertIn("ValueError: Test exception", log_entry["exception"])

    def test_log_rotation(self):
        """Test log file rotation."""
        max_size = 1024  # 1KB
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir,
            max_size=max_size
        ).get_logger()

        # Write enough data to trigger rotation
        large_message = "x" * (max_size // 2)
        for _ in range(4):
            logger.info(large_message)

        base_log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log"
        rotated_log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log.1"

        self.assertTrue(base_log_file.exists())
        self.assertTrue(rotated_log_file.exists())

    def test_get_logger_helper(self):
        """Test get_logger helper function."""
        logger1 = get_logger("test_logger", self.test_log_dir)
        logger2 = get_logger("test_logger", self.test_log_dir)

        # Should return the same logger instance
        self.assertEqual(logger1, logger2)

    def test_custom_log_level(self):
        """Test custom log level setting."""
        logger = AppLogger(
            self.test_logger_name,
            log_dir=self.test_log_dir,
            log_level=logging.DEBUG
        ).get_logger()

        debug_message = "Debug test message"
        logger.debug(debug_message)

        log_file = Path(self.test_log_dir) / f"{self.test_logger_name}.log"
        with open(log_file, 'r') as f:
            log_entry = json.loads(f.readline())

        self.assertEqual(log_entry["message"], debug_message)
        self.assertEqual(log_entry["level"], "DEBUG")

if __name__ == "__main__":
    unittest.main()
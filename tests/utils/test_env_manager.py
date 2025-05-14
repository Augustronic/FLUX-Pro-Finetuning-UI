"""
Test suite for the environment variable manager.
Tests loading and accessing environment variables from .env files.
"""

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from utils.env_manager import (
    EnvManager,
    get_env,
    get_bool_env,
    get_int_env,
    get_float_env,
    get_list_env
)


class TestEnvManager(unittest.TestCase):
    """Test the EnvManager class and utility functions."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary .env file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file_path = Path(self.temp_dir.name) / ".env"

        # Sample env variables to write to the file
        env_content = """
        # Test environment file
        TEST_STRING=Hello World
        TEST_INT=42
        TEST_FLOAT=3.14
        TEST_BOOL_TRUE=true
        TEST_BOOL_FALSE=false
        TEST_LIST=item1,item2,item3
        TEST_WITH_EXPANSION=${TEST_STRING} Extended
        TEST_EMPTY=
        TEST_MULTILINE="Line 1
        Line 2"
        """

        with open(self.env_file_path, "w") as f:
            f.write(env_content)

        # Create a test EnvManager instance
        self.env_manager = EnvManager(
            env_file=str(self.env_file_path),
            env_prefix="TEST_"
        )

    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and .env file
        self.temp_dir.cleanup()

        # Clear any environment variables we set
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]

    def test_get_string_value(self):
        """Test retrieving a string value."""
        self.assertEqual(self.env_manager.get("STRING"), "Hello World")

    def test_get_with_default(self):
        """Test retrieving a non-existent value with a default."""
        self.assertEqual(
            self.env_manager.get("NONEXISTENT", "default"),
            "default"
        )

    def test_get_bool_true(self):
        """Test retrieving a boolean true value."""
        self.assertTrue(self.env_manager.get_bool("BOOL_TRUE"))

    def test_get_bool_false(self):
        """Test retrieving a boolean false value."""
        self.assertFalse(self.env_manager.get_bool("BOOL_FALSE"))

    def test_get_bool_invalid(self):
        """Test retrieving an invalid boolean returns the default."""
        self.assertFalse(self.env_manager.get_bool("STRING", False))
        self.assertTrue(self.env_manager.get_bool("STRING", True))

    def test_get_int_valid(self):
        """Test retrieving a valid integer."""
        self.assertEqual(self.env_manager.get_int("INT"), 42)

    def test_get_int_invalid(self):
        """Test retrieving an invalid integer returns the default."""
        self.assertEqual(self.env_manager.get_int("STRING", 100), 100)

    def test_get_float_valid(self):
        """Test retrieving a valid float."""
        self.assertEqual(self.env_manager.get_float("FLOAT"), 3.14)

    def test_get_float_invalid(self):
        """Test retrieving an invalid float returns the default."""
        self.assertEqual(self.env_manager.get_float("STRING", 2.71), 2.71)

    def test_get_list(self):
        """Test retrieving a list of values."""
        expected = ["item1", "item2", "item3"]
        self.assertEqual(self.env_manager.get_list("LIST"), expected)

    def test_get_empty_list(self):
        """Test retrieving an empty list."""
        self.assertEqual(self.env_manager.get_list("EMPTY"), [])

    def test_get_custom_delimiter_list(self):
        """Test retrieving a list with custom delimiter."""
        # Set environment variable with semicolon delimiter
        os.environ["TEST_SEMICOLON_LIST"] = "a;b;c"
        self.assertEqual(
            self.env_manager.get_list("SEMICOLON_LIST", delimiter=";"),
            ["a", "b", "c"]
        )

    def test_get_all(self):
        """Test retrieving all environment variables."""
        all_vars = self.env_manager.get_all()
        self.assertIn("TEST_STRING", all_vars)
        self.assertIn("TEST_INT", all_vars)
        self.assertIn("TEST_FLOAT", all_vars)

    def test_get_dict(self):
        """Test retrieving variables with a specific prefix as dict."""
        # Set environment variables with prefixes
        os.environ["TEST_CONFIG_DEBUG"] = "true"
        os.environ["TEST_CONFIG_PORT"] = "8080"

        config = self.env_manager.get_dict("CONFIG")
        self.assertEqual(config["debug"], "true")
        self.assertEqual(config["port"], "8080")

    def test_variable_expansion(self):
        """Test that variable expansion works."""
        self.assertEqual(
            self.env_manager.get("WITH_EXPANSION"),
            "Hello World Extended"
        )

    def test_multiline_value(self):
        """Test retrieving a multiline value."""
        # The actual value will include the indentation from the .env file
        value = self.env_manager.get("MULTILINE")
        self.assertTrue(value.startswith("Line 1\n"))
        self.assertTrue("Line 2" in value)

    @patch.dict(os.environ, {"TEST_ENV": "production"})
    def test_get_env(self):
        """Test getting the current environment."""
        self.assertEqual(self.env_manager.get_env(), "production")


class TestEnvManagerFunctions(unittest.TestCase):
    """Test the convenience functions that use the default EnvManager."""

    def setUp(self):
        """Set up test environment."""
        # Set some environment variables for testing
        os.environ["FLUX_TEST_STRING"] = "value"
        os.environ["FLUX_TEST_BOOL"] = "true"
        os.environ["FLUX_TEST_INT"] = "123"
        os.environ["FLUX_TEST_FLOAT"] = "4.56"
        os.environ["FLUX_TEST_LIST"] = "a,b,c"

    def tearDown(self):
        """Clean up after tests."""
        # Clear any environment variables we set
        for key in list(os.environ.keys()):
            if key.startswith("FLUX_TEST_"):
                del os.environ[key]

    def test_get_env(self):
        """Test the get_env convenience function."""
        self.assertEqual(get_env("TEST_STRING"), "value")
        self.assertEqual(get_env("NONEXISTENT", "default"), "default")

    def test_get_bool_env(self):
        """Test the get_bool_env convenience function."""
        self.assertTrue(get_bool_env("TEST_BOOL"))
        self.assertFalse(get_bool_env("NONEXISTENT"))

    def test_get_int_env(self):
        """Test the get_int_env convenience function."""
        self.assertEqual(get_int_env("TEST_INT"), 123)
        self.assertEqual(get_int_env("NONEXISTENT", 42), 42)

    def test_get_float_env(self):
        """Test the get_float_env convenience function."""
        self.assertEqual(get_float_env("TEST_FLOAT"), 4.56)
        self.assertEqual(get_float_env("NONEXISTENT", 1.23), 1.23)

    def test_get_list_env(self):
        """Test the get_list_env convenience function."""
        self.assertEqual(get_list_env("TEST_LIST"), ["a", "b", "c"])
        self.assertEqual(get_list_env("NONEXISTENT"), [])


if __name__ == "__main__":
    unittest.main()

"""
Storage Service for FLUX-Pro-Finetuning-UI.

Provides centralized file operations with proper error handling,
security, and consistent directory management.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, BinaryIO
import base64
from datetime import datetime
import logging


class StorageError(Exception):
    """Exception raised for storage-related errors."""

    def __init__(self, message: str, path: Optional[str] = None):
        """
        Initialize storage error.

        Args:
            message: Error message
            path: Path related to the error
        """
        self.message = message
        self.path = path
        super().__init__(self.message)


class StorageService:
    """
    Service for handling file storage operations.

    Provides methods for file operations with proper error handling,
    security, and consistent directory management.
    """

    def __init__(self, config_service):
        """
        Initialize the storage service.

        Args:
            config_service: Configuration service for retrieving storage settings
        """
        self.config = config_service
        self.logger = logging.getLogger(__name__)

        # Get storage paths from config
        self.models_dir = self.config.get_storage_path("models_dir")
        self.images_dir = self.config.get_storage_path("images_dir")

        # Ensure directories exist with proper permissions
        self.setup_directories()

    def setup_directories(self) -> None:
        """
        Ensure all required directories exist with proper permissions.

        Creates directories if they don't exist and sets appropriate permissions.
        """
        directories = [
            self.models_dir,
            self.images_dir
        ]

        for directory in directories:
            try:
                # Create directory if it doesn't exist
                directory.mkdir(mode=0o700, exist_ok=True)

                # Update permissions if directory exists
                os.chmod(directory, 0o700)  # Only owner can read/write

                self.logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                self.logger.error(f"Error setting up directory {directory}: {e}")
                raise StorageError(f"Failed to set up directory: {e}", str(directory))

    def _sanitize_path(self, path: Union[str, Path]) -> Path:
        """
        Sanitize file path to prevent directory traversal.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path as Path object
        """
        if isinstance(path, str):
            path = Path(path)

        # Resolve to absolute path and ensure it doesn't escape base directories
        resolved = path.resolve()

        # Check if path is within allowed directories
        allowed_dirs = [
            self.models_dir.resolve(),
            self.images_dir.resolve()
        ]

        # Also allow paths within the current directory
        allowed_dirs.append(Path.cwd().resolve())

        for allowed_dir in allowed_dirs:
            if str(resolved).startswith(str(allowed_dir)):
                return resolved

        raise StorageError(
            f"Path {path} is outside of allowed directories",
            str(path)
        )

    def save_file(self, content: Union[str, bytes], path: Union[str, Path], mode: str = "wb") -> Path:
        """
        Save content to a file with proper error handling.

        Args:
            content: Content to save (string or bytes)
            path: Path to save to
            mode: File open mode

        Returns:
            Path to the saved file

        Raises:
            StorageError: If file cannot be saved
        """
        path = self._sanitize_path(path)

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create temporary file with secure permissions
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = Path(temp_file.name)

                # Handle different content types based on mode
                binary_content = None

                if isinstance(content, str):
                    # String content
                    binary_content = content.encode("utf-8")
                elif isinstance(content, bytes):
                    # Bytes content
                    binary_content = content
                else:
                    # Try to convert to bytes
                    try:
                        binary_content = bytes(content)
                    except (TypeError, ValueError):
                        # Last resort
                        binary_content = str(content).encode("utf-8")

                # Write content to file
                temp_file.write(binary_content)

            # Set secure permissions
            os.chmod(temp_path, 0o600)

            # Atomic rename for safer file writing
            shutil.move(temp_path, path)

            self.logger.debug(f"Saved file to {path}")
            return path

        except Exception as e:
            self.logger.error(f"Error saving file to {path}: {e}")
            # Clean up temporary file if it exists
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink()

            raise StorageError(f"Failed to save file: {e}", str(path))

    def read_file(self, path: Union[str, Path], mode: str = "rb") -> Union[str, bytes]:
        """
        Read content from a file with proper error handling.

        Args:
            path: Path to read from
            mode: File open mode

        Returns:
            File content as string or bytes

        Raises:
            StorageError: If file cannot be read
        """
        path = self._sanitize_path(path)

        if not path.exists():
            raise StorageError(f"File not found: {path}", str(path))

        try:
            with open(path, mode) as f:
                content = f.read()

            self.logger.debug(f"Read file from {path}")
            return content

        except Exception as e:
            self.logger.error(f"Error reading file from {path}: {e}")
            raise StorageError(f"Failed to read file: {e}", str(path))

    def save_model_metadata(self, models: Dict[str, Any]) -> None:
        """
        Save model metadata to storage.

        Args:
            models: Dictionary of model metadata

        Raises:
            StorageError: If metadata cannot be saved
        """
        models_file = self.models_dir / "models.json"

        try:
            # Convert models to list format
            data = [model for model in models.values()]

            # Save to file
            self.save_file(json.dumps(data, indent=2), models_file, "w")

            self.logger.info(f"Saved {len(models)} models to storage")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise StorageError(f"Failed to save model metadata: {e}", str(models_file))

    def load_model_metadata(self) -> Dict[str, Any]:
        """
        Load model metadata from storage.

        Returns:
            Dictionary of model metadata

        Raises:
            StorageError: If metadata cannot be loaded
        """
        models_file = self.models_dir / "models.json"
        models = {}

        try:
            if models_file.exists():
                # Ensure file permissions are secure
                os.chmod(models_file, 0o600)

                # Read file content
                content = self.read_file(models_file, "r")

                # Parse JSON
                data = json.loads(content)

                self.logger.info(f"Loading {len(data)} models from {models_file}")

                # Convert to dictionary
                for item in data:
                    if "finetune_id" in item:
                        models[item["finetune_id"]] = item

            return models

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return {}

    def save_generated_image(self, image_data: Union[str, bytes], format: str = "jpeg") -> Path:
        """
        Save a generated image to storage.

        Args:
            image_data: Image data (base64 string or bytes)
            format: Image format (jpeg or png)

        Returns:
            Path to the saved image

        Raises:
            StorageError: If image cannot be saved
        """
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"generated_image_{timestamp}.{format.lower()}"
        image_path = self.images_dir / filename

        try:
            # Convert base64 string to bytes if needed
            if isinstance(image_data, str) and image_data.startswith("data:"):
                # Extract base64 data from data URL
                header, encoded = image_data.split(",", 1)
                image_data = base64.b64decode(encoded)
            elif isinstance(image_data, str):
                # Assume it's already base64 encoded
                image_data = base64.b64decode(image_data)

            # Save image
            self.save_file(image_data, image_path)

            # Set secure permissions
            os.chmod(image_path, 0o600)

            self.logger.info(f"Saved generated image to {image_path}")
            return image_path

        except Exception as e:
            self.logger.error(f"Error saving generated image: {e}")
            raise StorageError(f"Failed to save generated image: {e}", str(image_path))

    def process_upload(self, file: BinaryIO, original_filename: str) -> Path:
        """
        Process uploaded file and save to storage.

        Args:
            file: File-like object
            original_filename: Original filename

        Returns:
            Path to the saved file

        Raises:
            StorageError: If file cannot be processed
        """
        # Create a temporary directory for uploads
        upload_dir = Path(tempfile.mkdtemp())

        try:
            # Create a safe filename
            safe_filename = Path(original_filename).name
            save_path = upload_dir / safe_filename

            # Read from file and write to our location
            content = file.read()

            if len(content) == 0:
                raise StorageError("Uploaded file is empty")

            # Save file
            self.save_file(content, save_path)

            self.logger.info(f"Processed upload: {original_filename} -> {save_path}")
            return save_path

        except Exception as e:
            self.logger.error(f"Error processing upload: {e}")
            # Clean up temporary directory
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise StorageError(f"Failed to process upload: {e}")

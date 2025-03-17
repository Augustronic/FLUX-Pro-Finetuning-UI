"""
File Utilities for FLUX-Pro-Finetuning-UI.

Provides centralized file operations for the application.
"""

import os
import json
import shutil
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union, BinaryIO
from pathlib import Path


class FileError(Exception):
    """Exception raised for file operation errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize file error.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


# Error codes
ERROR_FILE_NOT_FOUND = "file_not_found"
ERROR_PERMISSION_DENIED = "permission_denied"
ERROR_INVALID_PATH = "invalid_path"
ERROR_INVALID_JSON = "invalid_json"
ERROR_IO_ERROR = "io_error"
ERROR_UNKNOWN = "unknown_error"


def ensure_directory(directory: str) -> str:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to the directory
        
    Raises:
        FileError: If directory creation fails
    """
    try:
        # Convert to absolute path
        directory = os.path.abspath(directory)
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        return directory
        
    except PermissionError as e:
        raise FileError(
            f"Permission denied when creating directory: {e}",
            ERROR_PERMISSION_DENIED,
            {"directory": directory}
        )
    except Exception as e:
        raise FileError(
            f"Error creating directory: {e}",
            ERROR_UNKNOWN,
            {"directory": directory}
        )


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        File contents as string
        
    Raises:
        FileError: If file reading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileError(
                f"File not found: {file_path}",
                ERROR_FILE_NOT_FOUND,
                {"file_path": file_path}
            )
            
        # Read file
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
            
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when reading file: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except UnicodeDecodeError as e:
        raise FileError(
            f"Error decoding file: {e}",
            ERROR_IO_ERROR,
            {"file_path": file_path, "encoding": encoding}
        )
    except Exception as e:
        raise FileError(
            f"Error reading file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def read_binary_file(file_path: str) -> bytes:
    """
    Read binary data from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File contents as bytes
        
    Raises:
        FileError: If file reading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileError(
                f"File not found: {file_path}",
                ERROR_FILE_NOT_FOUND,
                {"file_path": file_path}
            )
            
        # Read file
        with open(file_path, "rb") as f:
            return f.read()
            
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when reading file: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error reading file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def write_file(file_path: str, content: str, encoding: str = "utf-8") -> None:
    """
    Write text to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding
        
    Raises:
        FileError: If file writing fails
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)
            
        # Write file
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)
            
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when writing file: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error writing file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def write_binary_file(file_path: str, content: bytes) -> None:
    """
    Write binary data to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        
    Raises:
        FileError: If file writing fails
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory(directory)
            
        # Write file
        with open(file_path, "wb") as f:
            f.write(content)
            
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when writing file: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error writing file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def read_json_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read JSON from a file.
    
    Args:
        file_path: Path to the file
        encoding: File encoding
        
    Returns:
        JSON data as dictionary
        
    Raises:
        FileError: If file reading or JSON parsing fails
    """
    try:
        # Read file
        content = read_file(file_path, encoding)
        
        # Parse JSON
        return json.loads(content)
        
    except FileError:
        # Re-raise FileError
        raise
    except json.JSONDecodeError as e:
        raise FileError(
            f"Invalid JSON in file: {e}",
            ERROR_INVALID_JSON,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error reading JSON file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def write_json_file(file_path: str, data: Dict[str, Any], encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Write JSON to a file.
    
    Args:
        file_path: Path to the file
        data: JSON data to write
        encoding: File encoding
        indent: JSON indentation
        
    Raises:
        FileError: If file writing or JSON serialization fails
    """
    try:
        # Serialize JSON
        content = json.dumps(data, indent=indent)
        
        # Write file
        write_file(file_path, content, encoding)
        
    except FileError:
        # Re-raise FileError
        raise
    except TypeError as e:
        raise FileError(
            f"Error serializing JSON: {e}",
            ERROR_INVALID_JSON,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error writing JSON file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def copy_file(source_path: str, destination_path: str) -> None:
    """
    Copy a file from source to destination.
    
    Args:
        source_path: Path to the source file
        destination_path: Path to the destination file
        
    Raises:
        FileError: If file copying fails
    """
    try:
        # Check if source file exists
        if not os.path.exists(source_path):
            raise FileError(
                f"Source file not found: {source_path}",
                ERROR_FILE_NOT_FOUND,
                {"source_path": source_path}
            )
            
        # Ensure destination directory exists
        directory = os.path.dirname(destination_path)
        if directory:
            ensure_directory(directory)
            
        # Copy file
        shutil.copy2(source_path, destination_path)
        
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when copying file: {e}",
            ERROR_PERMISSION_DENIED,
            {"source_path": source_path, "destination_path": destination_path}
        )
    except Exception as e:
        raise FileError(
            f"Error copying file: {e}",
            ERROR_UNKNOWN,
            {"source_path": source_path, "destination_path": destination_path}
        )


def move_file(source_path: str, destination_path: str) -> None:
    """
    Move a file from source to destination.
    
    Args:
        source_path: Path to the source file
        destination_path: Path to the destination file
        
    Raises:
        FileError: If file moving fails
    """
    try:
        # Check if source file exists
        if not os.path.exists(source_path):
            raise FileError(
                f"Source file not found: {source_path}",
                ERROR_FILE_NOT_FOUND,
                {"source_path": source_path}
            )
            
        # Ensure destination directory exists
        directory = os.path.dirname(destination_path)
        if directory:
            ensure_directory(directory)
            
        # Move file
        shutil.move(source_path, destination_path)
        
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when moving file: {e}",
            ERROR_PERMISSION_DENIED,
            {"source_path": source_path, "destination_path": destination_path}
        )
    except Exception as e:
        raise FileError(
            f"Error moving file: {e}",
            ERROR_UNKNOWN,
            {"source_path": source_path, "destination_path": destination_path}
        )


def delete_file(file_path: str) -> None:
    """
    Delete a file.
    
    Args:
        file_path: Path to the file
        
    Raises:
        FileError: If file deletion fails
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            # File doesn't exist, nothing to delete
            return
            
        # Delete file
        os.remove(file_path)
        
    except PermissionError as e:
        raise FileError(
            f"Permission denied when deleting file: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error deleting file: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def list_files(directory: str, pattern: Optional[str] = None) -> List[str]:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (glob)
        
    Returns:
        List of file paths
        
    Raises:
        FileError: If directory listing fails
    """
    try:
        # Check if directory exists
        if not os.path.exists(directory):
            raise FileError(
                f"Directory not found: {directory}",
                ERROR_FILE_NOT_FOUND,
                {"directory": directory}
            )
            
        # List files
        if pattern:
            import glob
            return glob.glob(os.path.join(directory, pattern))
        else:
            return [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
            ]
            
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when listing directory: {e}",
            ERROR_PERMISSION_DENIED,
            {"directory": directory}
        )
    except Exception as e:
        raise FileError(
            f"Error listing directory: {e}",
            ERROR_UNKNOWN,
            {"directory": directory}
        )


def create_temp_file(prefix: str = "", suffix: str = "", content: Optional[Union[str, bytes]] = None) -> str:
    """
    Create a temporary file.
    
    Args:
        prefix: File name prefix
        suffix: File name suffix
        content: Optional content to write to the file
        
    Returns:
        Path to the temporary file
        
    Raises:
        FileError: If temporary file creation fails
    """
    try:
        # Create temporary file
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        
        # Write content if provided
        if content is not None:
            if isinstance(content, str):
                write_file(path, content)
            else:
                write_binary_file(path, content)
                
        return path
        
    except Exception as e:
        raise FileError(
            f"Error creating temporary file: {e}",
            ERROR_UNKNOWN,
            {"prefix": prefix, "suffix": suffix}
        )


def create_temp_directory(prefix: str = "") -> str:
    """
    Create a temporary directory.
    
    Args:
        prefix: Directory name prefix
        
    Returns:
        Path to the temporary directory
        
    Raises:
        FileError: If temporary directory creation fails
    """
    try:
        # Create temporary directory
        return tempfile.mkdtemp(prefix=prefix)
        
    except Exception as e:
        raise FileError(
            f"Error creating temporary directory: {e}",
            ERROR_UNKNOWN,
            {"prefix": prefix}
        )


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileError: If file size retrieval fails
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileError(
                f"File not found: {file_path}",
                ERROR_FILE_NOT_FOUND,
                {"file_path": file_path}
            )
            
        # Get file size
        return os.path.getsize(file_path)
        
    except FileError:
        # Re-raise FileError
        raise
    except PermissionError as e:
        raise FileError(
            f"Permission denied when getting file size: {e}",
            ERROR_PERMISSION_DENIED,
            {"file_path": file_path}
        )
    except Exception as e:
        raise FileError(
            f"Error getting file size: {e}",
            ERROR_UNKNOWN,
            {"file_path": file_path}
        )


def get_file_extension(file_path: str) -> str:
    """
    Get file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension (without dot)
    """
    return os.path.splitext(file_path)[1].lstrip(".")


def is_valid_file_path(file_path: str) -> bool:
    """
    Check if a file path is valid.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if path is valid, False otherwise
    """
    try:
        # Check if path is absolute
        if os.path.isabs(file_path):
            # Check if path exists
            return os.path.exists(file_path)
            
        # Check if path is valid
        Path(file_path)
        return True
        
    except Exception:
        return False


def save_uploaded_file(file: BinaryIO, directory: str, filename: Optional[str] = None) -> str:
    """
    Save an uploaded file.
    
    Args:
        file: File-like object
        directory: Directory to save the file
        filename: Optional filename (if not provided, use the original filename)
        
    Returns:
        Path to the saved file
        
    Raises:
        FileError: If file saving fails
    """
    try:
        # Ensure directory exists
        ensure_directory(directory)
        
        # Get filename
        if not filename:
            if hasattr(file, 'name'):
                filename = os.path.basename(file.name)
            else:
                # Generate a random filename
                import uuid
                filename = f"{uuid.uuid4()}.bin"
                
        # Save file
        file_path = os.path.join(directory, filename)
        
        # Read file content
        file.seek(0)
        content = file.read()
        
        # Write file
        write_binary_file(file_path, content)
        
        return file_path
        
    except FileError:
        # Re-raise FileError
        raise
    except Exception as e:
        raise FileError(
            f"Error saving uploaded file: {e}",
            ERROR_UNKNOWN,
            {"directory": directory, "filename": filename}
        )
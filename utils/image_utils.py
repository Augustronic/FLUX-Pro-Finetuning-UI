"""
Image Utilities for FLUX-Pro-Finetuning-UI.

Provides centralized image processing functions for the application.
"""

import os
import io
import base64
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image


class ImageError(Exception):
    """Exception raised for image processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize image error.
        
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
ERROR_INVALID_IMAGE = "invalid_image"
ERROR_INVALID_FORMAT = "invalid_format"
ERROR_INVALID_DIMENSIONS = "invalid_dimensions"
ERROR_PROCESSING_FAILED = "processing_failed"
ERROR_IO_ERROR = "io_error"
ERROR_UNKNOWN = "unknown_error"


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
        
    Raises:
        ImageError: If image loading fails
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise ImageError(
                f"Image file not found: {image_path}",
                ERROR_IO_ERROR,
                {"image_path": image_path}
            )
            
        # Load image
        return Image.open(image_path)
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error loading image: {e}",
            ERROR_UNKNOWN,
            {"image_path": image_path}
        )


def save_image(image: Image.Image, output_path: str, format: Optional[str] = None, quality: int = 95) -> None:
    """
    Save an image to a file.
    
    Args:
        image: PIL Image object
        output_path: Path to save the image
        format: Image format (if None, inferred from output_path)
        quality: JPEG quality (0-100)
        
    Raises:
        ImageError: If image saving fails
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Save image
        image.save(output_path, format=format, quality=quality)
        
    except Exception as e:
        raise ImageError(
            f"Error saving image: {e}",
            ERROR_IO_ERROR,
            {"output_path": output_path}
        )


def resize_image(
    image: Union[Image.Image, np.ndarray, str],
    width: Optional[int] = None,
    height: Optional[int] = None,
    maintain_aspect_ratio: bool = True
) -> Image.Image:
    """
    Resize an image.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        width: Target width (if None, calculated from height and aspect ratio)
        height: Target height (if None, calculated from width and aspect ratio)
        maintain_aspect_ratio: Whether to maintain aspect ratio
        
        
    Returns:
        Resized PIL Image object
        
    Raises:
        ImageError: If image resizing fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate target dimensions
        if width is None and height is None:
            # No resizing needed
            return image
            
        if width is None:
            # Calculate width from height and aspect ratio
            if maintain_aspect_ratio:
                width = int((height or 0) * original_width / original_height)
            else:
                width = original_width
                
        if height is None:
            # Calculate height from width and aspect ratio
            if maintain_aspect_ratio:
                height = int((width or 0) * original_height / original_width)
            else:
                height = original_height
                
        # Resize image
        return image.resize((width, height))
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error resizing image: {e}",
            ERROR_PROCESSING_FAILED,
            {"width": width, "height": height}
        )


def crop_image(
    image: Union[Image.Image, np.ndarray, str],
    left: int,
    top: int,
    right: int,
    bottom: int
) -> Image.Image:
    """
    Crop an image.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        left: Left coordinate
        top: Top coordinate
        right: Right coordinate
        bottom: Bottom coordinate
        
    Returns:
        Cropped PIL Image object
        
    Raises:
        ImageError: If image cropping fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Crop image
        return image.crop((left, top, right, bottom))
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error cropping image: {e}",
            ERROR_PROCESSING_FAILED,
            {"left": left, "top": top, "right": right, "bottom": bottom}
        )


def convert_image_format(
    image: Union[Image.Image, np.ndarray, str],
    format: str
) -> Image.Image:
    """
    Convert image format.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        format: Target format (e.g., "JPEG", "PNG")
        
    Returns:
        Converted PIL Image object
        
    Raises:
        ImageError: If image conversion fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Convert image format
        if image.format == format:
            # No conversion needed
            return image
            
        # Convert to RGB if format is JPEG
        if format.upper() == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
            
        # Create a new image with the target format
        output = io.BytesIO()
        image.save(output, format=format)
        output.seek(0)
        
        return Image.open(output)
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error converting image format: {e}",
            ERROR_PROCESSING_FAILED,
            {"format": format}
        )


def image_to_base64(
    image: Union[Image.Image, np.ndarray, str],
    format: str = "JPEG",
    quality: int = 95
) -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        format: Image format (e.g., "JPEG", "PNG")
        quality: JPEG quality (0-100)
        
    Returns:
        Base64-encoded image string
        
    Raises:
        ImageError: If image conversion fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Convert to RGB if format is JPEG
        if format.upper() == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
            
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=quality)
        buffer.seek(0)
        
        # Get base64 string
        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Add data URL prefix
        mime_type = f"image/{format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error converting image to base64: {e}",
            ERROR_PROCESSING_FAILED,
            {"format": format}
        )


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to image.
    
    Args:
        base64_string: Base64-encoded image string
        
    Returns:
        PIL Image object
        
    Raises:
        ImageError: If image conversion fails
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",", 1)[1]
            
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        return Image.open(io.BytesIO(image_data))
        
    except Exception as e:
        raise ImageError(
            f"Error converting base64 to image: {e}",
            ERROR_PROCESSING_FAILED,
            {}
        )


def image_to_numpy(image: Union[Image.Image, str]) -> np.ndarray:
    """
    Convert image to numpy array.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        Numpy array
        
    Raises:
        ImageError: If image conversion fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert to numpy array
        return np.array(image)
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error converting image to numpy array: {e}",
            ERROR_PROCESSING_FAILED,
            {}
        )


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to image.
    
    Args:
        array: Numpy array
        
    Returns:
        PIL Image object
        
    Raises:
        ImageError: If image conversion fails
    """
    try:
        # Convert to PIL Image
        return Image.fromarray(array)
        
    except Exception as e:
        raise ImageError(
            f"Error converting numpy array to image: {e}",
            ERROR_PROCESSING_FAILED,
            {}
        )


def get_image_dimensions(image: Union[Image.Image, np.ndarray, str]) -> Tuple[int, int]:
    """
    Get image dimensions.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ImageError: If image dimensions retrieval fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Get dimensions from numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                height, width = image.shape[:2]
                return width, height
            elif len(image.shape) == 2:
                height, width = image.shape
                return width, height
            else:
                raise ImageError(
                    f"Invalid numpy array shape: {image.shape}",
                    ERROR_INVALID_DIMENSIONS,
                    {"shape": image.shape}
                )
                
        # Get dimensions from PIL Image
        return image.size
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error getting image dimensions: {e}",
            ERROR_UNKNOWN,
            {}
        )


def is_valid_image(image: Union[Image.Image, np.ndarray, str]) -> bool:
    """
    Check if an image is valid.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            if not os.path.exists(image):
                return False
                
            image = load_image(image)
            
        # Check numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) not in [2, 3]:
                return False
                
            if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
                return False
                
            return True
            
        # Check PIL Image
        image.verify()
        return True
        
    except Exception:
        return False


def get_image_format(image: Union[Image.Image, str]) -> str:
    """
    Get image format.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        Image format (e.g., "JPEG", "PNG")
        
    Raises:
        ImageError: If image format retrieval fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Get format from PIL Image
        if image.format:
            return image.format
            
        # Try to determine format from mode
        if image.mode == "RGB":
            return "JPEG"
        elif image.mode == "RGBA":
            return "PNG"
        else:
            return "UNKNOWN"
            
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error getting image format: {e}",
            ERROR_UNKNOWN,
            {}
        )


def create_thumbnail(
    image: Union[Image.Image, np.ndarray, str],
    max_size: int = 128,
    format: str = "JPEG",
    quality: int = 85
) -> Image.Image:
    """
    Create a thumbnail from an image.
    
    Args:
        image: PIL Image object, numpy array, or path to image file
        max_size: Maximum size (width or height)
        format: Image format (e.g., "JPEG", "PNG")
        quality: JPEG quality (0-100)
        
    Returns:
        Thumbnail as PIL Image object
        
    Raises:
        ImageError: If thumbnail creation fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            image = load_image(image)
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Create a copy of the image
        thumbnail = image.copy()
        
        # Resize to thumbnail size
        thumbnail.thumbnail((max_size, max_size))
        
        # Convert to RGB if format is JPEG
        if format.upper() == "JPEG" and thumbnail.mode != "RGB":
            thumbnail = thumbnail.convert("RGB")
            
        return thumbnail
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error creating thumbnail: {e}",
            ERROR_PROCESSING_FAILED,
            {"max_size": max_size}
        )


def apply_aspect_ratio(
    width: int,
    height: int,
    aspect_ratio: str
) -> Tuple[int, int]:
    """
    Apply aspect ratio to dimensions.
    
    Args:
        width: Original width
        height: Original height
        aspect_ratio: Target aspect ratio (e.g., "16:9", "4:3", "1:1")
        
    Returns:
        Tuple of (new_width, new_height)
        
    Raises:
        ImageError: If aspect ratio application fails
    """
    try:
        # Parse aspect ratio
        if ":" not in aspect_ratio:
            raise ImageError(
                f"Invalid aspect ratio format: {aspect_ratio}",
                ERROR_INVALID_DIMENSIONS,
                {"aspect_ratio": aspect_ratio}
            )
            
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        
        # Calculate target aspect ratio
        target_ratio = w_ratio / h_ratio
        
        # Calculate current aspect ratio
        current_ratio = width / height
        
        # Apply aspect ratio
        if current_ratio > target_ratio:
            # Width is too large
            new_width = int(height * target_ratio)
            new_height = height
        else:
            # Height is too large
            new_width = width
            new_height = int(width / target_ratio)
            
        return new_width, new_height
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error applying aspect ratio: {e}",
            ERROR_PROCESSING_FAILED,
            {"width": width, "height": height, "aspect_ratio": aspect_ratio}
        )


def parse_aspect_ratio(aspect_ratio: str) -> Tuple[int, int]:
    """
    Parse aspect ratio string to width and height ratio.
    
    Args:
        aspect_ratio: Aspect ratio string (e.g., "16:9", "4:3", "1:1")
        
    Returns:
        Tuple of (width_ratio, height_ratio)
        
    Raises:
        ImageError: If aspect ratio parsing fails
    """
    try:
        # Parse aspect ratio
        if ":" not in aspect_ratio:
            raise ImageError(
                f"Invalid aspect ratio format: {aspect_ratio}",
                ERROR_INVALID_DIMENSIONS,
                {"aspect_ratio": aspect_ratio}
            )
            
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
        
        return w_ratio, h_ratio
        
    except ImageError:
        # Re-raise ImageError
        raise
    except Exception as e:
        raise ImageError(
            f"Error parsing aspect ratio: {e}",
            ERROR_PROCESSING_FAILED,
            {"aspect_ratio": aspect_ratio}
        )
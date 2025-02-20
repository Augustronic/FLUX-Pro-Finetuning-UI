"""Shared configuration values and constants for the FLUX Pro Finetuning UI."""

from pathlib import Path
from typing import Final, Dict, List

# API Configuration
API_HOST: Final[str] = "api.us1.bfl.ai"
API_VERSION: Final[str] = "v1"

# API Endpoints
class Endpoints:
    """API endpoint constants."""
    ULTRA: Final[str] = "flux-pro-1.1-ultra-finetuned"
    STANDARD: Final[str] = "flux-pro-finetuned"
    FINETUNE: Final[str] = "finetune"
    FINETUNE_DETAILS: Final[str] = "finetune_details"
    MY_FINETUNES: Final[str] = "my_finetunes"
    GENERATE: Final[str] = "generate"
    GET_RESULT: Final[str] = "get_result"

# File System Paths
class Paths:
    """File system path constants."""
    DATA_DIR: Final[Path] = Path("data")
    MODELS_FILE: Final[Path] = DATA_DIR / "models.json"
    IMAGES_DIR: Final[Path] = Path("generated_images")
    CONFIG_FILE: Final[Path] = Path("config.json")

# File Permissions
class Permissions:
    """File permission constants."""
    PRIVATE_DIR: Final[int] = 0o700  # Only owner can read/write/execute
    PRIVATE_FILE: Final[int] = 0o600  # Only owner can read/write

# Model Configuration
class ModelConfig:
    """Model-related configuration constants."""
    VALID_MODES: Final[List[str]] = ["general", "character", "style", "product"]
    VALID_TYPES: Final[List[str]] = ["lora", "full"]
    VALID_PRIORITIES: Final[List[str]] = ["speed", "quality", "high_res_only"]
    
    # Parameter ranges
    LORA_RANK_RANGE: Final[Dict[str, int]] = {"min": 4, "max": 128}
    ITERATIONS_RANGE: Final[Dict[str, int]] = {"min": 100, "max": 1000}
    LEARNING_RATE_RANGE: Final[Dict[str, float]] = {
        "min": 0.000001,
        "max": 0.005,
        "default_full": 0.00001,
        "default_lora": 0.0001
    }

# Generation Parameters
class GenerationConfig:
    """Image generation configuration constants."""
    VALID_FORMATS: Final[List[str]] = ["jpeg", "png"]
    ASPECT_RATIOS: Final[List[str]] = [
        "21:9", "16:9", "3:2", "4:3", "1:1",
        "3:4", "2:3", "9:16", "9:21"
    ]
    
    # Parameter ranges
    STEPS_RANGE: Final[Dict[str, int]] = {"min": 1, "max": 50, "default": 40}
    GUIDANCE_RANGE: Final[Dict[str, float]] = {
        "min": 1.5,
        "max": 5.0,
        "default": 2.5
    }
    STRENGTH_RANGE: Final[Dict[str, float]] = {
        "min": 0.1,
        "max": 2.0,
        "default": 1.1
    }
    SAFETY_RANGE: Final[Dict[str, int]] = {"min": 0, "max": 6, "default": 2}
    DIMENSION_RANGE: Final[Dict[str, int]] = {
        "min": 256,
        "max": 1440,
        "step": 32,
        "default_width": 1024,
        "default_height": 768
    }

# Request Configuration
class RequestConfig:
    """HTTP request configuration constants."""
    TIMEOUT: Final[int] = 30  # seconds
    MAX_RETRIES: Final[int] = 3
    MAX_IMAGE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: Final[int] = 8192  # 8KB chunks for streaming
    USER_AGENT: Final[str] = "FLUX-Pro-Finetuning-UI"

# Validation Configuration
class ValidationConfig:
    """Input validation configuration constants."""
    MAX_PROMPT_LENGTH: Final[int] = 1000
    VALID_FILENAME_CHARS: Final[str] = r"^[\w\s\-_.,]+$"
    VALID_PROMPT_CHARS: Final[str] = r"^[\w\s\-_.,!?()[\]{}@#$%^&*+=<>:/\\|\'\"]+$"
    VALID_ID_FORMAT: Final[str] = r"^[a-zA-Z0-9-]+$"
    VALID_HOST_FORMAT: Final[str] = r"^[\w.-]+\.[a-zA-Z]{2,}$"
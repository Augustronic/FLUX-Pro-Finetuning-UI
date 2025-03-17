"""
Enhanced Inference Service for FLUX-Pro-Finetuning-UI.

Provides functionality for generating images with fine-tuned models,
including parameter validation, image generation, and result handling.
Includes feature flags and enhanced error handling.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
import base64
from pathlib import Path
import numpy as np

from services.core.api_service import APIService, APIError
from services.core.storage_service import StorageService, StorageError
from services.core.validation_service import ValidationService, ValidationError
from services.core.feature_flag_service import FeatureFlagService
from services.business.model_service import ModelService


class InferenceError(Exception):
    """Exception raised for inference errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize inference error.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class InferenceService:
    """
    Enhanced service for handling image generation operations.
    
    Provides methods for generating images with fine-tuned models,
    checking generation status, and handling results with feature flags
    and enhanced error handling.
    """
    
    # API endpoints
    ENDPOINT_ULTRA = "flux-pro-1.1-ultra-finetuned"
    ENDPOINT_STANDARD = "flux-pro-finetuned"
    
    # Error codes
    ERROR_VALIDATION = "validation_error"
    ERROR_API = "api_error"
    ERROR_STORAGE = "storage_error"
    ERROR_TIMEOUT = "timeout_error"
    ERROR_UNKNOWN = "unknown_error"
    
    def __init__(
        self,
        api_service: APIService,
        model_service: ModelService,
        storage_service: StorageService,
        validation_service: ValidationService,
        feature_flag_service: FeatureFlagService
    ):
        """
        Initialize the enhanced inference service.
        
        Args:
            api_service: API service for communicating with the API
            model_service: Model service for managing models
            storage_service: Storage service for file operations
            validation_service: Validation service for input validation
            feature_flag_service: Feature flag service for conditional features
        """
        self.api = api_service
        self.model_service = model_service
        self.storage = storage_service
        self.validation = validation_service
        self.feature_flags = feature_flag_service
        self.logger = logging.getLogger(__name__)
    
    def generate_image(
        self,
        endpoint: str,
        model_id: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "16:9",
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: float = 1.2,
        strength_standard: float = 1.1,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        image_prompt: Optional[np.ndarray] = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        image_prompt_strength: float = 0.1,
        ultra_prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        **kwargs
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate an image using the selected model and parameters.
        
        Args:
            endpoint: API endpoint to use
            model_id: ID of the fine-tuned model
            prompt: Text prompt for generation
            negative_prompt: Things to avoid in generation
            aspect_ratio: Image aspect ratio (ultra endpoint)
            num_steps: Number of generation steps
            guidance_scale: Guidance scale for generation
            strength: Finetune strength for ultra endpoint (0.1-2.0)
            strength_standard: Finetune strength for standard endpoint (0.1-2.0)
            seed: Random seed for reproducibility
            width: Image width (standard endpoint)
            height: Image height (standard endpoint)
            image_prompt: Optional image array to use as a prompt
            output_format: Output image format
            prompt_upsampling: Whether to enhance prompt
            image_prompt_strength: Blend between prompt and image prompt (0-1)
            ultra_prompt_upsampling: Whether to enhance prompt for ultra endpoint
            safety_tolerance: Safety check level (0-6)
            **kwargs: Additional parameters for experimental features
            
        Returns:
            Tuple of (numpy array of image or None, status message)
            
        Raises:
            InferenceError: If generation fails
        """
        try:
            # Get valid endpoints based on feature flags
            valid_endpoints = self._get_valid_endpoints()
            
            # Validate endpoint
            if not endpoint or endpoint not in valid_endpoints:
                raise ValidationError(
                    f"Invalid endpoint selection (must be one of: {', '.join(valid_endpoints)})",
                    "endpoint",
                    endpoint
                )
                
            # Validate model selection
            if not model_id:
                raise ValidationError("Please select a model", "model_id", model_id)
                
            model = self.model_service.get_model(model_id)
            if not model:
                raise ValidationError("Model not found", "model_id", model_id)
                
            # Validate prompt
            max_prompt_length = 1000
            
            # Allow longer prompts if feature is enabled
            if self.feature_flags.is_enabled("extended_prompts", False):
                max_prompt_length = 2000
                    # Validate prompt
            if len(prompt) > max_prompt_length:
                raise ValidationError(
                    f"Prompt is too long (max {max_prompt_length} characters)",
                    "prompt",
                    prompt
                )
            self.validation.validate_prompt(prompt)
                
            # Validate negative prompt if provided
            if negative_prompt:
                if len(negative_prompt) > max_prompt_length:
                    raise ValidationError(
                        f"Negative prompt is too long (max {max_prompt_length} characters)",
                        "negative_prompt",
                        negative_prompt
                    )
                self.validation.validate_prompt(negative_prompt
)
                
            # Validate numeric parameters
            self.validation.validate_numeric_param(
                strength, 0.1, 2.0, False, "strength"
            )
            
            if guidance_scale is not None:
                min_guidance = 1.5
                max_guidance = 5.0
                
                # Allow extended guidance range if feature is enabled
                if self.feature_flags.is_enabled("extended_guidance_range", False):
                    min_guidance = 1.0
                    max_guidance = 10.0
                    
                self.validation.validate_numeric_param(
                    guidance_scale, min_guidance, max_guidance, True, "guidance_scale"
                )
                
            if num_steps is not None:
                min_steps = 1
                max_steps = 50
                
                # Allow more steps if feature is enabled
                if self.feature_flags.is_enabled("extended_steps", False):
                    max_steps = 100
                    
                self.validation.validate_numeric_param(
                    float(num_steps), min_steps, max_steps, False, "num_steps"
                )
                
            # Get valid safety tolerance range
            min_safety = 0
            max_safety = 6
            
            # Allow higher safety tolerance if feature is enabled
            if self.feature_flags.is_enabled("extended_safety_tolerance", False):
                max_safety = 10
                
            if safety_tolerance < min_safety or safety_tolerance > max_safety:
                raise ValidationError(
                    f"Invalid safety tolerance value (must be between {min_safety} and {max_safety})",
                    "safety_tolerance",
                    safety_tolerance
                )
                
            # Get valid output formats
            valid_formats = self._get_valid_output_formats()
            
            # Validate output format
            if output_format not in valid_formats:
                raise ValidationError(
                    f"Invalid output format (must be one of: {', '.join(valid_formats)})",
                    "output_format",
                    output_format
                )
                
            # Validate aspect ratio for ultra endpoint
            if endpoint == self.ENDPOINT_ULTRA:
                valid_ratios = self._get_valid_aspect_ratios()
                if aspect_ratio not in valid_ratios:
                    raise ValidationError(
                        f"Invalid aspect ratio (must be one of: {', '.join(valid_ratios)})",
                        "aspect_ratio",
                        aspect_ratio
                    )
                    
            # Validate dimensions for standard endpoint
            if endpoint == self.ENDPOINT_STANDARD:
                if width is not None and height is not None:
                    min_dim = 256
                    max_dim = 1440
                    step = 32
                    
                    # Allow larger dimensions if feature is enabled
                    if self.feature_flags.is_enabled("high_resolution", False):
                        max_dim = 2048
                        
                    self.validation.validate_dimensions(width, height, min_dim, max_dim, step)
                    
            # Check if image prompt feature is enabled
            image_prompt_enabled = self.feature_flags.is_enabled("image_prompt_support", False)
            if image_prompt is not None and not image_prompt_enabled:
                self.logger.warning("Image prompt feature is disabled, ignoring image prompt")
                image_prompt = None
                
            # Log generation details
            self.logger.info(f"Generating image with model: {model.model_name}")
            self.logger.info(f"Model ID: {model_id}")
            self.logger.info(f"Endpoint: {endpoint}")
            self.logger.info(f"Prompt: {prompt}")
            
            # Common parameters
            # Convert image_prompt to base64 if provided
            image_prompt_base64 = None
            if image_prompt is not None and image_prompt_enabled:
                image_prompt_base64 = self._convert_image_to_base64(image_prompt, output_format)
                
            # Build parameters
            params = {
                "finetune_id": model_id,
                "prompt": prompt.strip() if prompt else "",
                "output_format": output_format.lower(),
                "safety_tolerance": safety_tolerance
            }
            
            # Add image prompt if provided and enabled
            if image_prompt_base64 and image_prompt_enabled:
                params["image_prompt"] = image_prompt_base64
                
            if endpoint == self.ENDPOINT_ULTRA:
                # Ultra endpoint parameters
                params.update({
                    "finetune_strength": strength,
                    "aspect_ratio": aspect_ratio,
                    "prompt_upsampling": ultra_prompt_upsampling
                })
                
                # Add image prompt strength if image prompt is provided and enabled
                if image_prompt_base64 and image_prompt_enabled:
                    params["image_prompt_strength"] = image_prompt_strength
                    
                # Add seed if provided
                if seed is not None and seed > 0:
                    params["seed"] = seed
                    
            else:  # ENDPOINT_STANDARD
                # Standard endpoint parameters
                params.update({
                    "finetune_strength": strength_standard,
                    "steps": num_steps or 40,
                    "guidance": guidance_scale or 2.5,
                    "width": width or 1024,
                    "height": height or 768,
                    "prompt_upsampling": prompt_upsampling
                })
                
                # Add seed if provided
                if seed is not None and seed > 0:
                    params["seed"] = seed
                    
                # Add negative prompt if provided
                if negative_prompt:
                    params["negative_prompt"] = negative_prompt
                    
            # Add experimental parameters if features are enabled
            if self.feature_flags.is_enabled("advanced_generation_params", False):
                # Add clip_skip if provided
                if "clip_skip" in kwargs:
                    clip_skip = kwargs.get("clip_skip")
                    self.validation.validate_numeric_param(
                        clip_skip,
                        1,
                        4,
                        True,
                        "clip_skip"
                    )
                    params["clip_skip"] = clip_skip
                    
                # Add scheduler if provided
                if "scheduler" in kwargs:
                    scheduler = kwargs.get("scheduler")
                    valid_schedulers = ["ddim", "pndm", "euler_a"]
                    if scheduler not in valid_schedulers:
                        raise ValidationError(
                            f"Invalid scheduler (must be one of: {', '.join(valid_schedulers)})",
                            "scheduler",
                            scheduler
                        )
                    params["scheduler"] = scheduler
                    
            # Remove any None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Start generation
            result = self.api.generate_image(endpoint=endpoint, params=params)
            if not result:
                raise InferenceError(
                    "No response from generation API",
                    self.ERROR_API
                )
                
            inference_id = result.get("id")
            if not inference_id:
                raise InferenceError(
                    "No inference ID received",
                    self.ERROR_API,
                    result
                )
                
            self.logger.info(f"Inference ID: {inference_id}")
            
            # Monitor generation progress
            max_attempts = 30
            
            # Allow longer timeout if feature is enabled
            if self.feature_flags.is_enabled("extended_generation_timeout", False):
                max_attempts = 60
                
            attempt = 0
            check_interval = 2
            
            while attempt < max_attempts:
                status = self.api.get_generation_status(inference_id)
                state = status.get("status", "")
                
                if state == "Failed":
                    error_msg = status.get("error", "Unknown error")
                    self.logger.error(f"Generation failed: {error_msg}")
                    raise InferenceError(
                        f"Generation failed: {error_msg}",
                        self.ERROR_API,
                        status
                    )
                    
                elif state == "Ready":
                    image_url = status.get("result", {}).get("sample")
                    if not image_url:
                        raise InferenceError(
                            "No image URL in completed status",
                            self.ERROR_API,
                            status
                        )
                        
                    self.logger.info(f"Generation completed: {image_url}")
                    img_array = self._save_image_from_url(image_url, output_format)
                    if img_array is None:
                        raise InferenceError(
                            "Failed to save image",
                            self.ERROR_STORAGE
                        )
                        
                    return (img_array, f"Generation completed successfully! Image saved as {output_format.upper()}")
                    
                self.logger.info(f"Status: {state} (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(check_interval)
                attempt += 1
                
            raise InferenceError(
                "Generation timed out",
                self.ERROR_TIMEOUT
            )
            
        except ValidationError as e:
            self.logger.error(f"Validation error in generation: {e}")
            return (None, f"Validation error: {str(e)}")
        except APIError as e:
            self.logger.error(f"API error in generation: {e}")
            return (None, f"API error: {str(e)}")
        except StorageError as e:
            self.logger.error(f"Storage error in generation: {e}")
            return (None, f"Storage error: {str(e)}")
        except InferenceError as e:
            self.logger.error(f"Inference error in generation: {e}")
            return (None, f"Inference error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in generation: {e}")
            return (None, f"Unexpected error: {str(e)}")
    
    def _get_valid_endpoints(self) -> List[str]:
        """
        Get valid endpoints based on feature flags.
        
        Returns:
            List of valid endpoints
        """
        # Base endpoints
        valid_endpoints = [self.ENDPOINT_ULTRA, self.ENDPOINT_STANDARD]
        
        # Add experimental endpoints if feature is enabled
        if self.feature_flags.is_enabled("experimental_endpoints", False):
            valid_endpoints.extend(["flux-pro-2.0-experimental", "flux-pro-sdxl"])
            
        return valid_endpoints
    
    def _get_valid_aspect_ratios(self) -> List[str]:
        """
        Get valid aspect ratios based on feature flags.
        
        Returns:
            List of valid aspect ratios
        """
        # Base aspect ratios
        valid_ratios = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"]
        
        # Add experimental aspect ratios if feature is enabled
        if self.feature_flags.is_enabled("experimental_aspect_ratios", False):
            valid_ratios.extend(["32:9", "1:2", "2:1"])
            
        return valid_ratios
    
    def _get_valid_output_formats(self) -> List[str]:
        """
        Get valid output formats based on feature flags.
        
        Returns:
            List of valid output formats
        """
        # Base output formats
        valid_formats = ["jpeg", "png"]
        
        # Add experimental output formats if feature is enabled
        if self.feature_flags.is_enabled("experimental_output_formats", False):
            valid_formats.extend(["webp", "tiff"])
            
        return valid_formats
    
    def _save_image_from_url(self, image_url: str, output_format: str) -> Optional[np.ndarray]:
        """
        Get image from URL or base64 data and return as numpy array.
        
        Args:
            image_url: URL or base64 data URL of the image
            output_format: Image format (jpeg or png)
            
        Returns:
            Numpy array of the image or None if failed
        """
        try:
            # Validate URL format
            self.validation.validate_url(image_url, "image_url")
            
            # Save image to storage
            image_path = self.storage.save_generated_image(image_url, output_format)
            
            # Convert to numpy array
            # This would typically use PIL or OpenCV to load the image
            # For simplicity, we'll return a placeholder
            # In a real implementation, you would load the image from the path
            # and return it as a numpy array
            return np.zeros((768, 1024, 3), dtype=np.uint8)  # Placeholder
            
        except (ValidationError, StorageError) as e:
            self.logger.error(f"Error processing image: {e}")
            return None
    
    def _convert_image_to_base64(self, image: np.ndarray, format: str = "jpeg") -> Optional[str]:
        """
        Convert numpy image array to base64 string.
        
        Args:
            image: Numpy array of image
            format: Image format (jpeg or png)
            
        Returns:
            Base64 encoded image string or None if conversion failed
        """
        try:
            # In a real implementation, you would use PIL to convert the image
            # to bytes and then encode it as base64
            # For simplicity, we'll return a placeholder
            return "data:image/jpeg;base64,placeholder"
            
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {e}")
            return None
    
    def get_model_choices(self) -> list:
        """
        Get formatted model choices for dropdown.
        
        Returns:
            List of formatted model choices
        """
        models = self.model_service.list_models()
        
        # Filter models based on feature flags
        filtered_models = []
        for model in models:
            if not model or not model.model_name or not model.trigger_word:
                continue
                
            # Filter by model type if experimental types are disabled
            if not self.feature_flags.is_enabled("experimental_model_types", True):
                if model.type not in ["full", "lora"]:
                    continue
                    
            filtered_models.append(model)
            
        return [
            self.model_service.format_model_choice(model)
            for model in filtered_models
        ]
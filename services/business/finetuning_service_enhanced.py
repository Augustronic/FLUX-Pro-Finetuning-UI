"""
Enhanced Finetuning Service for FLUX-Pro-Finetuning-UI.

Provides functionality for managing fine-tuning operations, including
starting fine-tuning jobs, checking status, and processing uploads.
Includes feature flags and enhanced error handling.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, BinaryIO, Tuple, List
from pathlib import Path

from services.core.api_service import APIService, APIError
from services.core.storage_service import StorageService, StorageError
from services.core.validation_service import ValidationService, ValidationError
from services.core.feature_flag_service import FeatureFlagService
from services.business.model_service import ModelService, ModelMetadata


class FinetuningError(Exception):
    """Exception raised for fine-tuning errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize fine-tuning error.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class FinetuningService:
    """
    Enhanced service for managing fine-tuning operations.
    
    Provides methods for starting fine-tuning jobs, checking status,
    and processing uploads with feature flags and enhanced error handling.
    """
    
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
        Initialize the enhanced fine-tuning service.
        
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
        
        # Current job ID
        self.current_job_id: Optional[str] = None
    
    def process_upload(self, file: BinaryIO, original_filename: str) -> Tuple[str, str]:
        """
        Process uploaded file and return path and filename.
        
        Args:
            file: File-like object
            original_filename: Original filename
            
        Returns:
            Tuple of (file path, status message)
            
        Raises:
            FinetuningError: If file processing fails
        """
        try:
            # Get allowed file extensions
            allowed_extensions = self._get_allowed_extensions()
            
            # Validate file extension
            self.validation.validate_file_extension(
                original_filename,
                allowed_extensions,
                "Training dataset"
            )
            
            # Process upload
            save_path = self.storage.process_upload(file, original_filename)
            
            return str(save_path), f"File saved as {original_filename}"
            
        except ValidationError as e:
            self.logger.error(f"Validation error processing upload: {e}")
            raise FinetuningError(
                f"Validation error: {str(e)}",
                self.ERROR_VALIDATION,
                {"field": getattr(e, "field", None)}
            )
        except StorageError as e:
            self.logger.error(f"Storage error processing upload: {e}")
            raise FinetuningError(
                f"Storage error: {str(e)}",
                self.ERROR_STORAGE
            )
        except Exception as e:
            self.logger.error(f"Unexpected error processing upload: {e}")
            raise FinetuningError(
                f"Unexpected error: {str(e)}",
                self.ERROR_UNKNOWN
            )
    
    def _get_allowed_extensions(self) -> List[str]:
        """
        Get allowed file extensions based on feature flags.
        
        Returns:
            List of allowed file extensions
        """
        # Base allowed extensions
        allowed_extensions = ["zip"]
        
        # Add additional extensions if features are enabled
        if self.feature_flags.is_enabled("allow_tar_files", False):
            allowed_extensions.extend(["tar", "tar.gz"])
            
        if self.feature_flags.is_enabled("allow_image_folders", False):
            allowed_extensions.extend(["jpg", "jpeg", "png"])
            
        return allowed_extensions
    
    def start_finetune(
        self,
        file_path: str,
        model_name: str,
        trigger_word: str,
        mode: str = "general",
        finetune_type: str = "full",
        iterations: int = 300,
        lora_rank: Optional[int] = None,
        learning_rate: Optional[float] = None,
        priority: str = "quality",
        auto_caption: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start a fine-tuning job.
        
        Args:
            file_path: Path to the training data file
            model_name: Name for the fine-tuned model
            trigger_word: Trigger word for the model
            mode: Training mode (general, character, style, product)
            finetune_type: Type of fine-tuning (full, lora)
            iterations: Number of training iterations
            lora_rank: LoRA rank (16 or 32)
            learning_rate: Learning rate for training
            priority: Training priority (speed, quality, high_res_only)
            auto_caption: Whether to enable auto-captioning
            **kwargs: Additional parameters for experimental features
            
        Returns:
            API response with fine-tune ID
            
        Raises:
            FinetuningError: If fine-tuning fails
        """
        try:
            # Validate inputs
            if not os.path.exists(file_path):
                raise ValidationError(f"File not found: {file_path}", "file_path", file_path)
                
            if not all([model_name, trigger_word]):
                raise ValidationError("Model name and trigger word are required")
                
            # Get valid modes based on feature flags
            valid_modes = self._get_valid_modes()
            
            # Validate mode
            if mode not in valid_modes:
                raise ValidationError(
                    f"Invalid mode (must be one of: {', '.join(valid_modes)})",
                    "mode",
                    mode
                )
                
            # Get valid finetune types based on feature flags
            valid_finetune_types = self._get_valid_finetune_types()
            
            # Validate finetune_type
            if finetune_type not in valid_finetune_types:
                raise ValidationError(
                    f"Invalid finetune_type (must be one of: {', '.join(valid_finetune_types)})",
                    "finetune_type",
                    finetune_type
                )
                
            # Validate iterations
            min_iterations = 100
            max_iterations = 1000
            
            # Allow extended iterations if feature is enabled
            if self.feature_flags.is_enabled("extended_iterations", False):
                max_iterations = 2000
                
            self.validation.validate_numeric_param(
                iterations,
                min_iterations,
                max_iterations,
                False,
                "iterations"
            )
            
            # Validate lora_rank if provided
            if finetune_type == "lora" and lora_rank is not None:
                valid_ranks = [16, 32]
                
                # Add additional ranks if feature is enabled
                if self.feature_flags.is_enabled("advanced_lora_ranks", False):
                    valid_ranks.extend([4, 8, 64])
                    
                if lora_rank not in valid_ranks:
                    raise ValidationError(
                        f"Invalid lora_rank (must be one of: {', '.join(map(str, valid_ranks))})",
                        "lora_rank",
                        lora_rank
                    )
                    
            # Validate learning_rate if provided
            if learning_rate is not None:
                min_lr = 0.000001
                max_lr = 0.005
                
                # Allow extended learning rate range if feature is enabled
                if self.feature_flags.is_enabled("extended_learning_rates", False):
                    min_lr = 0.0000001
                    max_lr = 0.01
                    
                self.validation.validate_numeric_param(
                    learning_rate,
                    min_lr,
                    max_lr,
                    True,
                    "learning_rate"
                )
                
            # Get valid priorities based on feature flags
            valid_priorities = self._get_valid_priorities()
            
            # Validate priority
            if priority not in valid_priorities:
                raise ValidationError(
                    f"Invalid priority (must be one of: {', '.join(valid_priorities)})",
                    "priority",
                    priority
                )
                
            # Encode the file
            file_data = self.api.encode_file(file_path)
            
            # Prepare payload
            payload = {
                "file_data": file_data,
                "finetune_comment": model_name,
                "mode": mode,
                "trigger_word": trigger_word,
                "iterations": iterations,
                "captioning": auto_caption,
                "priority": priority,
                "finetune_type": finetune_type
            }
            
            # Add optional parameters
            if finetune_type == "lora" and lora_rank is not None:
                payload["lora_rank"] = lora_rank
                
            if learning_rate is not None:
                payload["learning_rate"] = learning_rate
                
            # Add experimental parameters if features are enabled
            if self.feature_flags.is_enabled("advanced_finetune_params", False):
                # Add dropout if provided
                if "dropout" in kwargs:
                    dropout = kwargs.get("dropout")
                    self.validation.validate_numeric_param(
                        dropout,
                        0.0,
                        0.5,
                        True,
                        "dropout"
                    )
                    payload["dropout"] = dropout
                    
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
                    payload["clip_skip"] = clip_skip
                    
            # Start fine-tuning
            self.logger.info(f"Starting fine-tuning for model: {model_name}")
            result = self.api.start_finetune(payload)
            
            if not result or "finetune_id" not in result:
                raise FinetuningError(
                    "Failed to start fine-tuning job",
                    self.ERROR_API,
                    result
                )
                
            finetune_id = result["finetune_id"]
            self.current_job_id = finetune_id
            
            # Add model to manager
            self._handle_finetune_completion(
                finetune_id=finetune_id,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=mode,
                finetune_type=finetune_type,
                rank=lora_rank if finetune_type == "lora" else None,
                iterations=iterations,
                learning_rate=learning_rate,
                priority=priority
            )
            
            return result
            
        except ValidationError as e:
            self.logger.error(f"Validation error starting fine-tuning: {e}")
            raise FinetuningError(
                f"Validation error: {str(e)}",
                self.ERROR_VALIDATION,
                {"field": getattr(e, "field", None)}
            )
        except APIError as e:
            self.logger.error(f"API error starting fine-tuning: {e}")
            raise FinetuningError(
                f"API error: {str(e)}",
                self.ERROR_API,
                {"status_code": getattr(e, "status_code", None)}
            )
        except StorageError as e:
            self.logger.error(f"Storage error starting fine-tuning: {e}")
            raise FinetuningError(
                f"Storage error: {str(e)}",
                self.ERROR_STORAGE
            )
        except Exception as e:
            self.logger.error(f"Unexpected error starting fine-tuning: {e}")
            raise FinetuningError(
                f"Unexpected error: {str(e)}",
                self.ERROR_UNKNOWN
            )
    
    def _get_valid_modes(self) -> List[str]:
        """
        Get valid training modes based on feature flags.
        
        Returns:
            List of valid training modes
        """
        # Base modes
        valid_modes = ["general", "character", "style", "product"]
        
        # Add experimental modes if feature is enabled
        if self.feature_flags.is_enabled("experimental_training_modes", False):
            valid_modes.extend(["concept", "environment"])
            
        return valid_modes
    
    def _get_valid_finetune_types(self) -> List[str]:
        """
        Get valid fine-tune types based on feature flags.
        
        Returns:
            List of valid fine-tune types
        """
        # Base types
        valid_types = ["full", "lora"]
        
        # Add experimental types if feature is enabled
        if self.feature_flags.is_enabled("experimental_finetune_types", False):
            valid_types.extend(["dreambooth", "textual_inversion"])
            
        return valid_types
    
    def _get_valid_priorities(self) -> List[str]:
        """
        Get valid priorities based on feature flags.
        
        Returns:
            List of valid priorities
        """
        # Base priorities
        valid_priorities = ["speed", "quality", "high_res_only"]
        
        # Add experimental priorities if feature is enabled
        if self.feature_flags.is_enabled("experimental_priorities", False):
            valid_priorities.extend(["balanced", "ultra_quality"])
            
        return valid_priorities
    
    def _handle_finetune_completion(
        self,
        finetune_id: str,
        model_name: str,
        trigger_word: str,
        mode: str,
        finetune_type: str,
        rank: Optional[int],
        iterations: int,
        learning_rate: Optional[float],
        priority: str
    ) -> None:
        """
        Handle successful fine-tune completion by adding model to manager.
        
        Args:
            finetune_id: ID of the fine-tuned model
            model_name: Name of the model
            trigger_word: Trigger word for the model
            mode: Training mode
            finetune_type: Type of fine-tuning
            rank: LoRA rank
            iterations: Number of training iterations
            learning_rate: Learning rate for training
            priority: Training priority
        """
        try:
            # Create model metadata
            metadata = ModelMetadata(
                finetune_id=finetune_id,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=mode,
                type=finetune_type,
                rank=rank,
                iterations=iterations,
                learning_rate=learning_rate,
                priority=priority,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Add to model manager
            self.model_service.add_model(metadata)
            self.logger.info(f"Added model {model_name} to model manager")
            
        except Exception as e:
            self.logger.error(f"Error adding model to manager: {e}")
    
    def check_status(self, finetune_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.
        
        Args:
            finetune_id: ID of the fine-tuning job
            
        Returns:
            Status information
            
        Raises:
            FinetuningError: If status check fails
        """
        try:
            # Validate finetune_id
            if not finetune_id:
                raise ValidationError("Please enter a finetune ID", "finetune_id", finetune_id)
                
            # Extract the last part of the finetune ID (after the last hyphen)
            if '-' in finetune_id:
                finetune_id = finetune_id.split('-')[-1]
                
            # Get model details
            details = self.model_service.get_model_details(finetune_id)
            
            # Get status from API
            status_result = self.api.get_generation_status(finetune_id)
            
            # Combine results
            result = {
                "status": status_result.get("status", "Unknown"),
                "progress": status_result.get("progress"),
                "error": status_result.get("error"),
                "details": details or {},
                "is_completed": status_result.get("status") in ["Ready", "Task not found"]
            }
            
            # Add estimated time if feature is enabled
            if self.feature_flags.is_enabled("estimated_completion_time", False):
                progress = result.get("progress", 0)
                if progress > 0 and progress < 100:
                    # Calculate estimated time based on progress
                    # This is a simplified example - in a real implementation,
                    # you would use more sophisticated estimation
                    estimated_minutes = int((100 - progress) * 0.5)
                    result["estimated_minutes"] = estimated_minutes
                    
            # If completed, update model in manager
            if result["is_completed"] and not result.get("error"):
                self.model_service.update_model_from_api(finetune_id)
                
            return result
            
        except ValidationError as e:
            self.logger.error(f"Validation error checking status: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "error_code": self.ERROR_VALIDATION
            }
        except APIError as e:
            self.logger.error(f"API error checking status: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "error_code": self.ERROR_API
            }
        except Exception as e:
            self.logger.error(f"Unexpected error checking status: {e}")
            return {
                "status": "Error",
                "error": str(e),
                "error_code": self.ERROR_UNKNOWN
            }
    
    def update_learning_rate(self, finetune_type: str) -> float:
        """
        Update learning rate based on finetune type.
        
        Args:
            finetune_type: Type of fine-tuning (full, lora)
            
        Returns:
            Recommended learning rate
        """
        # Base learning rates
        base_rates = {
            "full": 0.00001,
            "lora": 0.0001
        }
        
        # Get experimental types if feature is enabled
        if self.feature_flags.is_enabled("experimental_finetune_types", False):
            base_rates.update({
                "dreambooth": 0.00005,
                "textual_inversion": 0.00002
            })
            
        return base_rates.get(finetune_type, 0.0001)
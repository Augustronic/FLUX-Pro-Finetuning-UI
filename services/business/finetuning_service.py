"""
Finetuning Service for FLUX-Pro-Finetuning-UI.

Provides functionality for managing fine-tuning operations, including
starting fine-tuning jobs, checking status, and processing uploads.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, BinaryIO, Tuple
from pathlib import Path

from services.core.api_service import APIService, APIError
from services.core.storage_service import StorageService, StorageError
from services.core.validation_service import ValidationService, ValidationError
from services.business.model_service import ModelService, ModelMetadata


class FinetuningError(Exception):
    """Exception raised for fine-tuning errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize fine-tuning error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FinetuningService:
    """
    Service for managing fine-tuning operations.
    
    Provides methods for starting fine-tuning jobs, checking status,
    and processing uploads.
    """
    
    def __init__(
        self,
        api_service: APIService,
        model_service: ModelService,
        storage_service: StorageService,
        validation_service: ValidationService
    ):
        """
        Initialize the fine-tuning service.
        
        Args:
            api_service: API service for communicating with the API
            model_service: Model service for managing models
            storage_service: Storage service for file operations
            validation_service: Validation service for input validation
        """
        self.api = api_service
        self.model_service = model_service
        self.storage = storage_service
        self.validation = validation_service
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
            # Validate file extension
            self.validation.validate_file_extension(
                original_filename,
                ["zip"],
                "Training dataset"
            )
            
            # Process upload
            save_path = self.storage.process_upload(file, original_filename)
            
            return str(save_path), f"File saved as {original_filename}"
            
        except (ValidationError, StorageError) as e:
            self.logger.error(f"Error processing upload: {e}")
            raise FinetuningError(f"Error processing upload: {str(e)}")
    
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
        auto_caption: bool = True
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
                
            # Validate mode
            if mode not in ["general", "character", "style", "product"]:
                raise ValidationError(
                    "Invalid mode (must be one of: general, character, style, product)",
                    "mode",
                    mode
                )
                
            # Validate finetune_type
            if finetune_type not in ["full", "lora"]:
                raise ValidationError(
                    "Invalid finetune_type (must be one of: full, lora)",
                    "finetune_type",
                    finetune_type
                )
                
            # Validate iterations
            self.validation.validate_numeric_param(
                iterations,
                100,
                1000,
                False,
                "iterations"
            )
            
            # Validate lora_rank if provided
            if finetune_type == "lora" and lora_rank is not None:
                if lora_rank not in [16, 32]:
                    raise ValidationError(
                        "Invalid lora_rank (must be 16 or 32)",
                        "lora_rank",
                        lora_rank
                    )
                    
            # Validate learning_rate if provided
            if learning_rate is not None:
                self.validation.validate_numeric_param(
                    learning_rate,
                    0.000001,
                    0.005,
                    True,
                    "learning_rate"
                )
                
            # Validate priority
            if priority not in ["speed", "quality", "high_res_only"]:
                raise ValidationError(
                    "Invalid priority (must be one of: speed, quality, high_res_only)",
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
                
            # Start fine-tuning
            self.logger.info(f"Starting fine-tuning for model: {model_name}")
            result = self.api.start_finetune(payload)
            
            if not result or "finetune_id" not in result:
                raise FinetuningError("Failed to start fine-tuning job", result)
                
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
            
        except (ValidationError, APIError, StorageError) as e:
            self.logger.error(f"Error starting fine-tuning: {e}")
            raise FinetuningError(f"Error starting fine-tuning: {str(e)}")
    
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
            
            # If completed, update model in manager
            if result["is_completed"] and not result.get("error"):
                self.model_service.update_model_from_api(finetune_id)
                
            return result
            
        except (ValidationError, APIError) as e:
            self.logger.error(f"Error checking status: {e}")
            return {
                "status": "Error",
                "error": str(e)
            }
    
    def update_learning_rate(self, finetune_type: str) -> float:
        """
        Update learning rate based on finetune type.
        
        Args:
            finetune_type: Type of fine-tuning (full, lora)
            
        Returns:
            Recommended learning rate
        """
        return 0.00001 if finetune_type == "full" else 0.0001
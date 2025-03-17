"""
Enhanced Fine-tuning UI Component for FLUX-Pro-Finetuning-UI.

Provides UI for fine-tuning models, including file upload, parameter configuration,
and status checking. Uses feature flags and improved error handling.
"""

import os
import gradio as gr
import logging
from typing import Dict, Any, Optional, List, Tuple

from ui.base import BaseUI
from services.business.finetuning_service import FinetuningService, FinetuningError
from services.business.model_service import ModelService
from services.core.feature_flag_service import FeatureFlagService


class FineTuneUI(BaseUI):
    """
    Enhanced UI component for fine-tuning models.
    
    Provides UI for fine-tuning models, including file upload, parameter configuration,
    and status checking. Uses feature flags and improved error handling.
    """
    
    def __init__(
        self,
        finetuning_service: FinetuningService,
        model_service: ModelService,
        feature_flag_service: FeatureFlagService
    ):
        """
        Initialize the enhanced fine-tuning UI component.
        
        Args:
            finetuning_service: Service for fine-tuning operations
            model_service: Service for model management
            feature_flag_service: Service for feature flags
        """
        super().__init__(
            title="Model Finetuning",
            description="Upload your training dataset and configure finetuning parameters."
        )
        self.finetuning_service = finetuning_service
        self.model_service = model_service
        self.feature_flags = feature_flag_service
        self.logger = logging.getLogger(__name__)
    
    def start_finetuning(
        self,
        file,
        model_name: str,
        training_mode: str,
        finetune_type: str,
        trigger_word: str,
        iterations: int,
        lora_rank: int,
        learning_rate: float,
        priority: str,
        captioning: bool,
        **kwargs
    ) -> str:
        """
        Start the fine-tuning process.
        
        Args:
            file: Uploaded file
            model_name: Name for the fine-tuned model
            training_mode: Training mode (general, character, style, product)
            finetune_type: Type of fine-tuning (full, lora)
            trigger_word: Trigger word for the model
            iterations: Number of training iterations
            lora_rank: LoRA rank (16 or 32)
            learning_rate: Learning rate for training
            priority: Training priority (speed, quality, high_res_only)
            captioning: Whether to enable auto-captioning
            **kwargs: Additional parameters for experimental features
            
        Returns:
            Fine-tune ID or error message
        """
        try:
            # Process file upload
            file_path, msg = self.finetuning_service.process_upload(file, file.name)
            if not file_path:
                return msg
                
            # Validate inputs
            if not all([model_name, trigger_word]):
                return "Model name and trigger word are required."
                
            # Prepare additional parameters for experimental features
            additional_params = {}
            
            # Add dropout if feature is enabled
            if self.feature_flags.is_enabled("advanced_finetune_params", False) and "dropout" in kwargs:
                additional_params["dropout"] = kwargs.get("dropout")
                
            # Add clip_skip if feature is enabled
            if self.feature_flags.is_enabled("advanced_finetune_params", False) and "clip_skip" in kwargs:
                additional_params["clip_skip"] = kwargs.get("clip_skip")
                
            # Start fine-tuning
            result = self.finetuning_service.start_finetune(
                file_path=file_path,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=training_mode,
                finetune_type=finetune_type,
                iterations=iterations,
                lora_rank=lora_rank if finetune_type == "lora" else None,
                learning_rate=learning_rate,
                priority=priority,
                auto_caption=captioning,
                **additional_params
            )
            
            if not result or 'finetune_id' not in result:
                return "Failed to start finetuning job."
                
            finetune_id = result['finetune_id']
            return finetune_id
            
        except FinetuningError as e:
            error_message = f"Error starting finetuning: {e.message}"
            # Check if the error has details
            if hasattr(e, 'details') and e.details:
                error_message += f" (Details: {e.details})"
            self.logger.error(error_message)
            return error_message
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"
    
    def check_status(self, finetune_id: str) -> str:
        """
        Check the status of a fine-tuning job.
        
        Args:
            finetune_id: ID of the fine-tuning job
            
        Returns:
            Status message
        """
        if not finetune_id:
            return "Please enter a finetune ID."
            
        try:
            # Get status details
            result = self.finetuning_service.check_status(finetune_id)
            if not result:
                return "Error checking status"
                
            status = result.get('status', '')
            progress = result.get('progress', '')
            error = result.get('error', '')
            details = result.get('details', {})
            is_completed = result.get('is_completed', False)
            
            # Format status message
            if status == 'Failed':
                status_msg = f"Training failed: {error}"
            elif status == 'Not Found':
                status_msg = "Model not found. Please check the finetune ID."
            elif is_completed:
                status_msg = "✅ Training completed successfully!\n\n"
                # Add details
                if details:
                    status_msg += "Model details:\n"
                    name = details.get('finetune_comment', 'unknown')
                    mode = details.get('mode', 'unknown')
                    ftype = details.get('finetune_type', 'unknown')
                    word = details.get('trigger_word', 'unknown')
                    iters = details.get('iterations', 'unknown')
                    rate = details.get('learning_rate', 'unknown')
                    status_msg += (
                        f"- Name: {name}\n"
                        f"- Mode: {mode}\n"
                        f"- Type: {ftype}\n"
                        f"- Trigger word: {word}\n"
                        f"- Iterations: {iters}\n"
                        f"- Learning rate: {rate}\n"
                    )
                    if details.get('lora_rank'):
                        status_msg += f"- LoRA rank: {details['lora_rank']}\n"
                    status_msg += "\nModel is ready to use! ✨"
            else:
                status_msg = f"Status: {status}"
                if progress:
                    status_msg += f"\nProgress: {progress}"
                    
                # Add estimated time if feature is enabled
                if self.feature_flags.is_enabled("estimated_completion_time", False) and "estimated_minutes" in result:
                    estimated_minutes = result.get("estimated_minutes", 0)
                    if estimated_minutes > 0:
                        status_msg += f"\nEstimated time remaining: {estimated_minutes} minutes"
                        
                if details:
                    name = details.get('finetune_comment', '')
                    mode = details.get('mode', '')
                    ftype = details.get('finetune_type', '')
                    status_msg += (
                        f"\nModel: {name}"
                        f"\nMode: {mode}"
                        f"\nType: {ftype}"
                    )
                    
            return status_msg
            
        except Exception as e:
            self.logger.error(f"Error checking status: {e}")
            return f"Error checking status: {str(e)}"
    
    def update_learning_rate(self, finetune_type: str) -> float:
        """
        Update learning rate based on finetune type.
        
        Args:
            finetune_type: Type of fine-tuning (full, lora)
            
        Returns:
            Recommended learning rate
        """
        return self.finetuning_service.update_learning_rate(finetune_type)
    
    def create_ui(self) -> gr.Blocks:
        """
        Create the fine-tuning UI component.
        
        Returns:
            Gradio Blocks component
        """
        with gr.Blocks() as app:
            # Header
            title_text = self.title if self.title else "Model Finetuning"
            desc_text = self.description if self.description else "Upload your training dataset and configure finetuning parameters."
            self.create_section_header(title_text, desc_text)
            
            with gr.Row():
                with gr.Column():
                    # Get allowed file types based on feature flags
                    allowed_file_types = [".zip"]
                    if self.feature_flags.is_enabled("allow_tar_files", False):
                        allowed_file_types.extend([".tar", ".tar.gz"])
                    if self.feature_flags.is_enabled("allow_image_folders", False):
                        allowed_file_types.extend([".jpg", ".jpeg", ".png"])
                        
                    file_input = gr.File(
                        label=f"Upload training dataset ({', '.join(allowed_file_types)})",
                        file_types=allowed_file_types
                    )
                    model_name = gr.Textbox(
                        label="Model name",
                        placeholder="Enter a name for your finetuned model."
                    )
                    trigger_word = gr.Textbox(
                        label="Trigger word",
                        placeholder="Word to trigger your model (e.g., 'TOK')."
                    )
                    
                    # Get training modes based on feature flags
                    mode_choices = [
                        ("General", "general"),
                        ("Character", "character"),
                        ("Style", "style"),
                        ("Product", "product")
                    ]
                    
                    # Add experimental modes if feature is enabled
                    if self.feature_flags.is_enabled("experimental_training_modes", False):
                        mode_choices.extend([
                            ("Concept", "concept"),
                            ("Environment", "environment")
                        ])
                        
                    training_mode = gr.Radio(
                        choices=mode_choices,
                        value="general",
                        label="Training mode",
                        info=(
                            "Select the type of training that best matches "
                            "your dataset."
                        )
                    )
                    
                with gr.Column():
                    # Update learning rate when finetune type changes
                    def on_finetune_type_change(ft_type):
                        return self.update_learning_rate(ft_type)
                        
                    captioning = gr.Checkbox(
                        label="Enable auto-captioning",
                        value=True,
                        info="Auto-generate captions for training images."
                    )
                    
                    # Get priorities based on feature flags
                    priority_choices = [
                        ("Speed", "speed"),
                        ("Quality", "quality"),
                        ("High-res only", "high_res_only")
                    ]
                    
                    # Add experimental priorities if feature is enabled
                    if self.feature_flags.is_enabled("experimental_priorities", False):
                        priority_choices.extend([
                            ("Balanced", "balanced"),
                            ("Ultra Quality", "ultra_quality")
                        ])
                        
                    priority = gr.Radio(
                        choices=priority_choices,
                        value="quality",
                        label="Training priority"
                    )
                    
                    # Get iterations range based on feature flags
                    min_iterations = 100
                    max_iterations = 1000
                    
                    # Allow extended iterations if feature is enabled
                    if self.feature_flags.is_enabled("extended_iterations", False):
                        max_iterations = 2000
                        
                    iterations = gr.Slider(
                        minimum=min_iterations,
                        maximum=max_iterations,
                        value=300,
                        step=10,
                        label="Training iterations"
                    )
                    
                    # Get LoRA ranks based on feature flags
                    lora_ranks = [16, 32]
                    
                    # Add advanced ranks if feature is enabled
                    if self.feature_flags.is_enabled("advanced_lora_ranks", False):
                        lora_ranks = [4, 8, 16, 32, 64]
                        
                    lora_rank = gr.Radio(
                        choices=lora_ranks,
                        value=32,
                        label="LoRA rank",
                        info="Higher rank = more capacity but slower training."
                    )
                    
                    # Get finetune types based on feature flags
                    finetune_choices = [
                        ("Full", "full"),
                        ("LoRA", "lora")
                    ]
                    
                    # Add experimental finetune types if feature is enabled
                    if self.feature_flags.is_enabled("experimental_finetune_types", False):
                        finetune_choices.extend([
                            ("DreamBooth", "dreambooth"),
                            ("Textual Inversion", "textual_inversion")
                        ])
                        
                    finetune_type = gr.Radio(
                        choices=finetune_choices,
                        value="full",
                        label="Finetuning type",
                        info="LoRA is faster and uses less resources."
                    )
                    
                    # Get learning rate range based on feature flags
                    min_lr = 0.000001
                    max_lr = 0.005
                    
                    # Allow extended learning rates if feature is enabled
                    if self.feature_flags.is_enabled("extended_learning_rates", False):
                        min_lr = 0.0000001
                        max_lr = 0.01
                        
                    learning_rate = gr.Number(
                        label="Learning rate",
                        value=0.00001,
                        minimum=min_lr,
                        maximum=max_lr,
                        info="Automatically set based on finetune type."
                    )
                    
                    # Connect finetune_type change to learning_rate update
                    finetune_type.change(
                        fn=on_finetune_type_change,
                        inputs=[finetune_type],
                        outputs=[learning_rate]
                    )
                    
                    # Add advanced parameters if feature is enabled
                    advanced_params = {}
                    if self.feature_flags.is_enabled("advanced_finetune_params", False):
                        with gr.Accordion("Advanced Parameters", open=False):
                            dropout = gr.Slider(
                                minimum=0.0,
                                maximum=0.5,
                                value=0.0,
                                step=0.05,
                                label="Dropout",
                                info="Regularization to prevent overfitting."
                            )
                            clip_skip = gr.Slider(
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                                label="CLIP Skip",
                                info="Number of layers to skip in CLIP model."
                            )
                            advanced_params["dropout"] = dropout
                            advanced_params["clip_skip"] = clip_skip
                    
                with gr.Column():
                    with gr.Accordion(
                        "Getting Started: Step-by-Step Guide",
                        open=False
                    ):
                        gr.Markdown("""
1. Prepare Your Images
    - Create a local folder for training images.
    - Supported: JPG, JPEG, PNG, and WebP
    - Recommended: more than 5 images.
<br/><p style="color: #72a914;">High-quality datasets with clear subjects
improve results. Higher resolution helps but is capped at 1MP.</p>
2. Add Text Descriptions (Optional)
    - Create text files for image descriptions.
    - Files share names with their images.
    - Example: "sample.jpg" -> "sample.txt"
3. Package Your Data
    - Compress folder into ZIP.
4. Configure Parameters
    - Select appropriate settings.
5. Submit Task
    - Use script to submit.
6. Run Inference
    - Use model via endpoints.""")
                        
                    with gr.Accordion("Best Practices and Tips", open=False):
                        gr.Markdown("""
1. Concept Enhancement
    - Try strength >1 if concept is missing
    - Increase for better identity
    - Lower for generalization
2. Character Training
    - One character per image
    - Manual captions for complexity
    - Consider auto-caption settings
3. Quality Tips
    - Use high-quality images
    - Adjust learning rate
    - Monitor progress
4. Prompting
    - Use context in triggers
    - Prepend triggers to prompts
    - Add brief descriptions
    - Include style indicators""")
                        
                    with gr.Accordion("Note on training mode", open=False):
                        gr.Markdown("""
<p style="color: #72a914">General mode captions whole images without focus
areas. No subject improvements.</p>""")
                        
                    with gr.Accordion("Notes on learning rate", open=False):
                        gr.Markdown("""
<p style="color: #72a914;">Lower values: better results, more iterations.
Higher values: faster training, may reduce quality.</p>
<p style="color: #72a914;">LoRA: use 10x larger values than Full.</p>""")
                        
            # Start finetuning button
            start_btn = gr.Button(
                "▶️ Start Finetuning",
                variant="primary"
            )
            
            # Status checking section
            with gr.Row():
                with gr.Column():
                    finetune_id = gr.Textbox(
                        label="Finetune ID",
                        placeholder="Enter ID to check status.",
                        value=lambda: (
                            self.finetuning_service.current_job_id 
                            if self.finetuning_service.current_job_id 
                            else ""
                        )
                    )
                    check_status_btn = gr.Button(
                        "〽️ Check status"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
            # Handle finetuning start
            start_inputs = [
                file_input,
                model_name,
                training_mode,
                finetune_type,
                trigger_word,
                iterations,
                lora_rank,
                learning_rate,
                priority,
                captioning
            ]
            
            # Add advanced parameters if feature is enabled
            if self.feature_flags.is_enabled("advanced_finetune_params", False):
                start_inputs.extend([
                    advanced_params["dropout"],
                    advanced_params["clip_skip"]
                ])
                
            start_btn.click(
                fn=self.start_finetuning,
                inputs=start_inputs,
                outputs=finetune_id
            )
            
            # Handle status check
            check_status_btn.click(
                fn=self.check_status,
                inputs=[finetune_id],
                outputs=status_text
            )
            
        return app
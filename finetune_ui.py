import gradio as gr
import json
import os
from typing import Tuple, Optional
import threading
import time
from pathlib import Path

from model_manager import ModelManager, ModelMetadata
from finetune import FineTuneClient
from check_progress import FineTuneMonitor
from config_manager import ConfigManager

class FineTuneUI:
    def __init__(self):
        self.config = ConfigManager()
        self.config.load_config()
        self.api_key = self.config.get_api_key()
        self.client = FineTuneClient(api_key=self.api_key)
        self.model_manager = ModelManager(api_key=self.api_key)
        self.monitor = FineTuneMonitor(api_key=self.api_key)
        self.current_job_id = None
    
    def process_upload(self, file) -> Tuple[str, str]:
        """Process uploaded file and return path and filename."""
        if not file:
            return None, "No file uploaded"
            
        # Gradio provides a temp file path in file.name
        temp_path = file.name
        
        # Get the original filename from the uploaded file
        original_filename = os.path.basename(temp_path)
        
        # Ensure it's a ZIP file
        if not original_filename.lower().endswith('.zip'):
            return None, "Please upload a ZIP file"
            
        # Create a copy in our working directory
        save_path = os.path.join(os.getcwd(), original_filename)
        try:
            # Read from temp file and write to our location
            with open(temp_path, 'rb') as src, open(save_path, 'wb') as dst:
                dst.write(src.read())
                
            if os.path.getsize(save_path) == 0:
                return None, "Uploaded file is empty"
                
            return save_path, f"File saved as {original_filename}"
            
        except Exception as e:
            return None, f"Error processing upload: {str(e)}"
        
    def check_status(self, finetune_id: str) -> str:
        """Check the status of a fine-tuning job."""
        if not finetune_id:
            return "Please enter a fine-tune ID"
            
        try:
            # Get status details
            result = self.monitor.check_progress(finetune_id)
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
                status_msg = "Model not found. Please check the fine-tune ID."
            elif is_completed:
                status_msg = "✅ Training completed successfully!\n\n"
                # Add details
                if details:
                    status_msg += f"Model Details:\n"
                    status_msg += f"- Name: {details.get('finetune_comment', 'unknown')}\n"
                    status_msg += f"- Mode: {details.get('mode', 'unknown')}\n"
                    status_msg += f"- Type: {details.get('finetune_type', 'unknown')}\n"
                    status_msg += f"- Trigger Word: {details.get('trigger_word', 'unknown')}\n"
                    status_msg += f"- Iterations: {details.get('iterations', 'unknown')}\n"
                    status_msg += f"- Learning Rate: {details.get('learning_rate', 'unknown')}\n"
                    if details.get('lora_rank'):
                        status_msg += f"- LoRA Rank: {details.get('lora_rank')}\n"
                    status_msg += f"\nModel is ready to use! ✨"
                # Update model in manager
                self.model_manager.update_model_from_api(finetune_id)
            else:
                status_msg = f"Status: {status}"
                if progress:
                    status_msg += f"\nProgress: {progress}"
                if details:
                    status_msg += f"\nModel: {details.get('finetune_comment', '')}"
                    status_msg += f"\nMode: {details.get('mode', '')}"
                    status_msg += f"\nType: {details.get('finetune_type', '')}"
            
            return status_msg
            
        except Exception as e:
            return f"Error checking status: {str(e)}"
    
    def _handle_finetune_completion(self, finetune_id: str, model_name: str, trigger_word: str, 
                                  mode: str, finetune_type: str, rank: int, iterations: int,
                                  learning_rate: float, priority: str) -> None:
        """Handle successful fine-tune completion by adding model to manager."""
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
            self.model_manager.add_model(metadata)
            print(f"Added model {model_name} to model manager")
            
        except Exception as e:
            print(f"Error adding model to manager: {e}")
    
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
        captioning: bool
    ) -> str:
        """Start the fine-tuning process."""
        try:
            # Process file upload
            file_path, msg = self.process_upload(file)
            if not file_path:
                return msg
                
            # Validate inputs
            if not all([model_name, trigger_word]):
                return "Model name and trigger word are required"
                
            # Start fine-tuning
            result = self.client.start_finetune(
                file_path=file_path,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=training_mode,
                finetune_type=finetune_type,
                iterations=iterations,
                lora_rank=lora_rank if finetune_type == "lora" else None,
                learning_rate=learning_rate,
                priority=priority,
                auto_caption=captioning
            )
            
            if not result or 'finetune_id' not in result:
                return "Failed to start fine-tuning job"
                
            finetune_id = result['finetune_id']
            self.current_job_id = finetune_id
            
            # Add model to manager
            self._handle_finetune_completion(
                finetune_id=finetune_id,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=training_mode,
                finetune_type=finetune_type,
                rank=lora_rank if finetune_type == "lora" else None,
                iterations=iterations,
                learning_rate=learning_rate,
                priority=priority
            )
            
            return f"Fine-tuning started! Job ID: {finetune_id}"
            
        except Exception as e:
            return f"Error starting fine-tuning: {str(e)}"
    
    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="AI Model Fine-Tuning Interface") as app:
            gr.Markdown("""
            # AI Model Fine-Tuning Interface
            Upload your training data and configure fine-tuning parameters.
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Training Data (ZIP file)",
                        file_types=[".zip"]
                    )
                    model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="Enter a name for your fine-tuned model"
                    )
                    training_mode = gr.Radio(
                        choices=["general", "character", "style", "product"],
                        value="character",
                        label="Training Mode",
                        info="Select the type of training that best matches your data"
                    )
                    finetune_type = gr.Radio(
                        choices=["lora", "full"],
                        value="lora",
                        label="Fine-tuning Type",
                        info="LoRA is faster and uses less resources"
                    )
                    
                with gr.Column():
                    trigger_word = gr.Textbox(
                        label="Trigger Word",
                        placeholder="Word to trigger your model (e.g., 'mymodel')"
                    )
                    iterations = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=300,
                        step=100,
                        label="Training Iterations"
                    )
                    lora_rank = gr.Radio(
                        choices=[16, 32],
                        value=16,
                        label="LoRA Rank",
                        info="Higher rank = more capacity but slower training"
                    )
                    learning_rate = gr.Number(
                        label="Learning Rate (default: 0.0001)",
                        value=0.0001,
                        minimum=0.000001,
                        maximum=0.005,
                        info="You can modify this value or keep the default"
                    )
                    
                with gr.Column():
                    priority = gr.Radio(
                        choices=["quality", "speed"],
                        value="quality",
                        label="Training Priority"
                    )
                    captioning = gr.Checkbox(
                        label="Enable Auto-Captioning",
                        value=True,
                        info="Automatically generate captions for training images"
                    )
            
            # Status checking section
            with gr.Row():
                with gr.Column():
                    finetune_id = gr.Textbox(
                        label="Fine-tune ID",
                        placeholder="Enter your fine-tune ID to check status",
                        value=lambda: self.current_job_id if self.current_job_id else ""
                    )
                    check_status_btn = gr.Button("Check Status", variant="primary")
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            # Start fine-tuning button
            start_btn = gr.Button("Start Fine-tuning")
            
            # Handle fine-tuning start
            start_btn.click(
                fn=self.start_finetuning,
                inputs=[
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
                ],
                outputs=finetune_id
            )
            
            # Handle status check
            check_status_btn.click(
                fn=self.check_status,
                inputs=[finetune_id],
                outputs=status_text
            )
            
        return app

def create_ui():
    ui = FineTuneUI()
    return ui.create_ui()

if __name__ == "__main__":
    app = create_ui()
    app.launch() 
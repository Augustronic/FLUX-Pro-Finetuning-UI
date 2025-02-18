import gradio as gr
import os
from typing import Tuple
import time

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
            return "", "No file uploaded"

        # Gradio provides a temp file path in file.name
        temp_path = file.name

        # Get the original filename from the uploaded file
        original_filename = os.path.basename(temp_path)

        # Ensure it's a ZIP file
        if not original_filename.lower().endswith('.zip'):
            return "", "Please upload a ZIP file."

        # Create a copy in our working directory
        save_path = os.path.join(os.getcwd(), original_filename)
        try:
            # Read from temp file and write to our location
            with open(temp_path, 'rb') as src, open(save_path, 'wb') as dst:
                dst.write(src.read())

            if os.path.getsize(save_path) == 0:
                return "", "Uploaded file is empty."

            return save_path, f"File saved as {original_filename}"

        except Exception as e:
            return "", f"Error processing upload: {str(e)}"

    def check_status(self, finetune_id: str) -> str:
        """Check the status of a finetuning job."""
        if not finetune_id:
            return "Please enter a finetune ID."

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
                # Update model in manager
                self.model_manager.update_model_from_api(finetune_id)
            else:
                status_msg = f"Status: {status}"
                if progress:
                    status_msg += f"\nProgress: {progress}"
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
            return f"Error checking status: {str(e)}"

    def update_learning_rate(self, finetune_type: str) -> float:
        """Update learning rate based on finetune type."""
        return 0.00001 if finetune_type == "full" else 0.0001

    def _handle_finetune_completion(
        self,
        finetune_id: str,
        model_name: str,
        trigger_word: str,
        mode: str,
        finetune_type: str,
        rank: int,
        iterations: int,
        learning_rate: float,
        priority: str
    ) -> None:
        """Handle successful finetune completion by adding model to manager."""
        try:
            # Create model metadata
            metadata = ModelMetadata(
                finetune_id=finetune_id,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=mode,
                type=finetune_type,
                rank=rank or 0,  # Default to 0 if None
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
        """Start the finetuning process."""
        try:
            # Process file upload
            file_path, msg = self.process_upload(file)
            if not file_path:
                return msg

            # Validate inputs
            if not all([model_name, trigger_word]):
                return "Model name and trigger word are required."

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
                return "Failed to start finetuning job."

            finetune_id = result['finetune_id']
            self.current_job_id = finetune_id

            # Add model to manager
            self._handle_finetune_completion(
                finetune_id=finetune_id,
                model_name=model_name,
                trigger_word=trigger_word,
                mode=training_mode,
                finetune_type=finetune_type,
                rank=lora_rank if finetune_type == "lora" else 0,
                iterations=iterations,
                learning_rate=learning_rate,
                priority=priority
            )

            return f"Finetuning started! Job ID: {finetune_id}"

        except Exception as e:
            return f"Error starting finetuning: {str(e)}"

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Model Finetuning Interface") as app:
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload training dataset (ZIP file)",
                        file_types=[".zip"]
                    )
                    model_name = gr.Textbox(
                        label="Model name",
                        placeholder="Enter a name for your finetuned model."
                    )
                    trigger_word = gr.Textbox(
                        label="Trigger word",
                        placeholder="Word to trigger your model (e.g., 'TOK')."
                    )
                    mode_choices = [
                        ("General", "general"),
                        ("Character", "character"),
                        ("Style", "style"),
                        ("Product", "product")
                    ]
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
                    priority_choices = [
                        ("Speed", "speed"),
                        ("Quality", "quality"),
                        ("High-res only", "high_res_only")
                    ]
                    priority = gr.Radio(
                        choices=priority_choices,
                        value="quality",
                        label="Training priority"
                    )
                    iterations = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=300,
                        step=10,
                        label="Training iterations"
                    )
                    lora_rank = gr.Radio(
                        choices=[16, 32],
                        value=32,
                        label="LoRA rank",
                        info="Higher rank = more capacity but slower training."
                    )
                    finetune_choices = [
                        ("Full", "full"),
                        ("LoRA", "lora")
                    ]
                    finetune_type = gr.Radio(
                        choices=finetune_choices,
                        value="full",
                        label="Finetuning type",
                        info="LoRA is faster and uses less resources."
                    )
                    learning_rate = gr.Number(
                        label="Learning rate",
                        value=0.00001,
                        minimum=0.000001,
                        maximum=0.005,
                        info="Automatically set based on finetune type."
                    )
                    # Connect finetune_type change to learning_rate update
                    finetune_type.change(
                        fn=on_finetune_type_change,
                        inputs=[finetune_type],
                        outputs=[learning_rate]
                    )

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

            # Status checking section
            with gr.Row():
                with gr.Column():
                    finetune_id = gr.Textbox(
                        label="Finetune ID",
                        placeholder="Enter ID to check status.",
                        value=lambda: (
                            self.current_job_id if self.current_job_id else ""
                        )
                    )
                    check_status_btn = gr.Button(
                        "Check status",
                        variant="primary"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )

            # Start finetuning button
            start_btn = gr.Button("Start Finetuning")

            # Handle finetuning start
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
    """Create and return the UI instance."""
    ui = FineTuneUI()
    return ui.create_ui()


if __name__ == "__main__":
    app = create_ui()
    app.launch()

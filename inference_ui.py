import gradio as gr
import time
import os
import re
from typing import Any, Optional, Tuple
import base64
import requests
import json
from pathlib import Path
import tempfile

from model_manager import ModelManager


class ImageGenerationUI:
    """Handles the Gradio UI for image generation with fine-tuned models."""

    # API endpoints
    ENDPOINT_ULTRA = "flux-pro-1.1-ultra-finetuned"
    ENDPOINT_STANDARD = "flux-pro-finetuned"

    def __init__(self, model_manager: ModelManager) -> None:
        """Initialize the UI with a model manager."""
        if not isinstance(model_manager, ModelManager):
            raise ValueError("Invalid model manager instance")
            
        self.manager = model_manager
        
        # Create images directory in current working directory
        self.images_dir = Path("generated_images")
        self.images_dir.mkdir(exist_ok=True)

    def _format_model_choice(self, model: Any) -> str:
        """Format model metadata for dropdown display.
        
        Args:
            model: Model metadata object
            
        Returns:
            Formatted string for display
            
        Raises:
            ValueError: If model data is invalid
        """
        if not model or not hasattr(model, 'model_name') or not hasattr(model, 'trigger_word'):
            raise ValueError("Invalid model data")
            
        # Validate required attributes
        if not all([
            isinstance(getattr(model, attr, None), str)
            for attr in ['model_name', 'trigger_word', 'type', 'mode']
        ]):
            raise ValueError("Invalid model attributes")
            
        # Sanitize display values
        parts = [
            f"{self._sanitize_display_text(model.model_name)}",
            f"({self._sanitize_display_text(model.trigger_word)})",
            f"{self._sanitize_display_text(model.type).upper()}",
            f"{self._sanitize_display_text(model.mode).capitalize()}",
        ]
        
        # Add rank if present
        if hasattr(model, 'rank') and isinstance(model.rank, (int, float)):
            parts.append(f"Rank {int(model.rank)}")
            
        return " - ".join(parts)

    def _sanitize_display_text(self, text: str) -> str:
        """Sanitize text for display in UI.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove control characters and limit length
        text = "".join(char for char in text if char.isprintable())
        text = text[:100]  # Limit length
        
        # Only allow alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-_.,()]', '', text)
        return text.strip()

    def _get_model_id_from_choice(self, choice: str) -> str:
        """Extract model ID from formatted choice string.
        
        Args:
            choice: Formatted choice string from dropdown
            
        Returns:
            Model ID or empty string if not found
        """
        if not isinstance(choice, str) or not choice.strip():
            return ""
            
        try:
            for model in self.manager.list_models():
                if model and self._format_model_choice(model) == choice:
                    # Validate ID format
                    if re.match(r'^[a-zA-Z0-9-]+$', model.finetune_id):
                        return model.finetune_id
            return ""
        except Exception as e:
            print(f"Error extracting model ID: {e}")
            return ""

    def _validate_image_url(self, url: str) -> bool:
        """Validate image URL format and security."""
        if url.startswith("data:"):
            # Validate base64 data URL format
            try:
                header, encoded = url.split(",", 1)
                if not header.startswith("data:image/"):
                    return False
                base64.b64decode(encoded)  # Test if valid base64
                return True
            except Exception:
                return False
        else:
            # Validate HTTP(S) URL
            return bool(re.match(r'^https?://[\w\-.]+(:\d+)?(/[\w\-./]*)?$', url))

    def _save_image_from_url(self, image_url: str, output_format: str) -> str:
        """Save image from URL or base64 data to a file with secure handling."""
        if not isinstance(output_format, str) or output_format.lower() not in ["jpeg", "png"]:
            print("Invalid output format")
            return ""
            
        if not self._validate_image_url(image_url):
            print("Invalid image URL format")
            return ""

        try:
            # Create a temporary file first
            temp_file = Path(tempfile.mktemp(dir=self.images_dir))
            
            try:
                if image_url.startswith("data:"):
                    header, encoded = image_url.split(",", 1)
                    image_data = base64.b64decode(encoded)
                else:
                    # Use timeout and size limit for URL downloads
                    response = requests.get(
                        image_url,
                        timeout=30,
                        stream=True,
                        headers={"User-Agent": "FLUX-Pro-Finetuning-UI"}
                    )
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        raise ValueError("Invalid content type")
                    
                    # Read with size limit (50MB)
                    max_size = 50 * 1024 * 1024
                    content = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                        if len(content) > max_size:
                            raise ValueError("Image too large")
                    image_data = content

                # Write to temporary file
                with open(temp_file, "wb") as f:
                    f.write(image_data)

                # Set secure permissions
                os.chmod(temp_file, 0o600)

                # Create final filename
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"generated_image_{timestamp}.{output_format.lower()}"
                final_path = self.images_dir / filename

                # Atomic rename
                temp_file.replace(final_path)
                return str(final_path)

            finally:
                # Cleanup temp file if it still exists
                if temp_file.exists():
                    temp_file.unlink()

        except Exception as e:
            print(f"Error saving image: {e}")
            return ""

    def _validate_prompt(self, prompt: str) -> bool:
        """Validate prompt text for safety and format."""
        if not prompt or not isinstance(prompt, str):
            return False
        
        # Remove excessive whitespace
        prompt = prompt.strip()
        if len(prompt) == 0:
            return False
            
        # Check for valid characters
        if not re.match(r'^[\w\s\-_.,!?()[\]{}@#$%^&*+=<>:/\\|\'\"]+$', prompt):
            return False
            
        # Check length
        if len(prompt) > 1000:  # Maximum prompt length
            return False
            
        return True
        
    def _validate_numeric_param(
        self,
        value: Optional[float],
        min_val: float,
        max_val: float,
        allow_none: bool = True
    ) -> bool:
        """Validate numeric parameter within range."""
        if value is None:
            return allow_none
            
        try:
            value = float(value)
            return min_val <= value <= max_val
        except (TypeError, ValueError):
            return False

    def generate_image(
        self,
        endpoint: str,
        model_choice: str,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        num_steps: Optional[int],
        guidance_scale: Optional[float],
        strength: float,
        seed: Optional[int],
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        width: Optional[int] = None,
        height: Optional[int] = None,
        raw_mode: bool = False,
    ) -> Tuple[Any, str]:
        """
        Generate an image using the selected model and parameters.

        Args:
            endpoint: API endpoint to use
            model_choice: Selected model from dropdown
            prompt: Text prompt for generation
            negative_prompt: Things to avoid in generation
            aspect_ratio: Image aspect ratio (ultra endpoint)
            steps: Number of generation steps
            guidance: Guidance scale for generation
            strength: Model strength (0.1-2.0)
            seed: Random seed for reproducibility
            output_format: Output image format
            prompt_upsampling: Whether to enhance prompt
            safety_tolerance: Safety check level (0-6)
            width: Image width (standard endpoint)
            height: Image height (standard endpoint)
            raw_mode: Whether to use raw mode

        Returns:
            Tuple of (image path, status message)
            
        Raises:
            ValueError: If any input parameters are invalid
        """
        try:
            # Validate endpoint
            if not endpoint or endpoint not in [self.ENDPOINT_ULTRA, self.ENDPOINT_STANDARD]:
                return ("", "Error: Invalid endpoint selection")

            # Validate model selection
            if not model_choice:
                return ("", "Error: Please select a model")
                
            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return ("", "Error: Invalid model selection")

            model = self.manager.get_model(model_id)
            if not model:
                return ("", "Error: Model not found")
                
            # Validate prompt
            if not self._validate_prompt(prompt):
                return ("", "Error: Invalid prompt format or content")
                
            # Validate negative prompt if provided
            if negative_prompt and not self._validate_prompt(negative_prompt):
                return ("", "Error: Invalid negative prompt format")
                
            # Validate numeric parameters
            if not self._validate_numeric_param(strength, 0.1, 2.0, False):
                return ("", "Error: Invalid strength value (must be between 0.1 and 2.0)")
                
            if not self._validate_numeric_param(guidance_scale, 1.5, 5.0):
                return ("", "Error: Invalid guidance scale value (must be between 1.5 and 5.0)")
                
            if num_steps is not None and not self._validate_numeric_param(float(num_steps), 1, 50, False):
                return ("", "Error: Invalid number of steps value (must be between 1 and 50)")
                
            if safety_tolerance not in range(7):  # 0 to 6
                return ("", "Error: Invalid safety tolerance value")
                
            # Validate output format
            if output_format not in ["jpeg", "png"]:
                return ("", "Error: Invalid output format")
                
            # Validate aspect ratio for ultra endpoint
            if endpoint == self.ENDPOINT_ULTRA:
                valid_ratios = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"]
                if aspect_ratio not in valid_ratios:
                    return ("", "Error: Invalid aspect ratio")
                    
            # Validate dimensions for standard endpoint
            if endpoint == self.ENDPOINT_STANDARD:
                if width is not None and not self._validate_numeric_param(float(width), 256, 1440, False):
                    return ("", "Error: Invalid width value")
                if height is not None and not self._validate_numeric_param(float(height), 256, 1440, False):
                    return ("", "Error: Invalid height value")

            # Log generation details
            print(f"\nGenerating image with model: {model.model_name}")
            print(f"Model ID: {model_id}")
            print(f"Endpoint: {endpoint}")
            print(f"Prompt: {prompt}")

            # Prepare parameters based on example implementation
            params = {
                "finetune_id": model_id,
                "prompt": prompt.strip(),
                "finetune_strength": strength,
                "endpoint": endpoint,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "num_images": 1,
                "width": width or 1024,
                "height": height or 768,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "output_format": output_format.lower(),
                "safety_tolerance": safety_tolerance,
                "scheduler": "DPM++ 2M Karras"
            }

            # Remove None values and ensure negative_prompt is included if present
            if negative_prompt:
                params["negative_prompt"] = negative_prompt
            params = {k: v for k, v in params.items() if v is not None}

            print("Sending request to", endpoint)
            print("Parameters:", json.dumps(params, indent=2))

            # Start generation
            result = self.manager.generate_image(endpoint=endpoint, **params)
            if not result:
                return ("", "Error: No response from generation API")

            inference_id = result.get("id")
            if not inference_id:
                return ("", "Error: No inference ID received")

            print(f"Inference ID: {inference_id}")

            # Monitor generation progress
            max_attempts = 30
            attempt = 0
            check_interval = 2

            while attempt < max_attempts:
                status = self.manager.get_generation_status(inference_id)
                state = status.get("status", "")

                if state == "Failed":
                    error_msg = status.get("error", "Unknown error")
                    print(f"Generation failed: {error_msg}")
                    return ("", f"Generation failed: {error_msg}")

                elif state == "Ready":
                    image_url = status.get("result", {}).get("sample")
                    if not image_url:
                        return ("", "Error: No image URL in completed status")

                    print(f"Generation completed: {image_url}")
                    local_path = self._save_image_from_url(image_url, output_format)
                    if not local_path:
                        return None, "Error: Failed to save image"
                        
                    return local_path, f"Generation completed successfully! Image saved as {output_format.upper()}"

                print(
                    f"Status: {state} "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(check_interval)
                attempt += 1

            return ("", "Error: Generation timed out")

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return ("", f"Error generating image: {str(e)}")

    def create_ui(self) -> gr.Blocks:
        """Create the image generation interface."""
        with gr.Blocks(title="AI Image Generation") as interface:
            gr.Markdown(
                """
                **Important**: Include the model's trigger word in your prompt!
                """
            )

            with gr.Row():
                with gr.Column():
                    # Endpoint selection
                    endpoint = gr.Radio(
                        choices=[
                            ("FLUX 1.1 [pro] ultra Finetune",
                             self.ENDPOINT_ULTRA),
                            ("FLUX.1 [pro] Finetune",
                             self.ENDPOINT_STANDARD),
                        ],
                        value=self.ENDPOINT_ULTRA,
                        label="Generation endpoint",
                        info="Select the generation endpoint to use.",
                    )

                    # Model selection
                    models = self.manager.list_models()
                    if not models:
                        print("No models found, refreshing from API...")
                        self.manager.refresh_models()
                        models = self.manager.list_models()

                    # Create model choices list
                    model_choices = []
                    for model in models:
                        if model and model.model_name and model.trigger_word:
                            choice = self._format_model_choice(model)
                            model_choices.append(choice)
                            print(f"Added model choice: {choice}")

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0] if model_choices else None,
                            label="Select model",
                            info=(
                                "Model trigger word shown in parentheses. "
                                "Include in prompt."
                            ),
                        )
                        refresh_btn = gr.Button("🔄 Refresh models")

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder=(
                            "Enter prompt (include trigger word shown above)."
                        ),
                        lines=3,
                        info="Include model's trigger word in prompt.",
                    )

                    negative_prompt = gr.Textbox(
                        label="Negative prompt (optional)",
                        placeholder="Enter things to avoid in the image.",
                        lines=2,
                    )

                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### Image parameters")

                        # Ultra endpoint parameters
                        with gr.Column(visible=True) as ultra_params:
                            aspect_ratio = gr.Radio(
                                choices=[
                                    "21:9", "16:9", "3:2", "4:3", "1:1",
                                    "3:4", "2:3", "9:16", "9:21",
                                ],
                                value="16:9",
                                label="Aspect ratio",
                                info="Select image dimensions ratio.",
                            )

                            strength = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.1,
                                step=0.1,
                                label="Model strength",
                                info=(
                                    "How strongly to apply model's style "
                                    "(default: 1.1)."
                                ),
                            )

                        # Standard endpoint parameters
                        with gr.Column(visible=False) as standard_params:
                            with gr.Row():
                                width = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=1024,
                                    step=32,
                                    label="Width",
                                    info="Must be a multiple of 32",
                                )
                                height = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=768,
                                    step=32,
                                    label="Height",
                                    info="Must be a multiple of 32",
                                )

                            num_inference_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=30,
                                step=1,
                                label="Number of Steps",
                                info=(
                                    "Number of generation steps "
                                    "(quality vs speed)"
                                ),
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.5,
                                maximum=7.5,
                                value=7.5,
                                step=0.1,
                                label="Guidance Scale",
                                info="Controls prompt adherence strength",
                            )

                            prompt_upsampling = gr.Checkbox(
                                label="Enhance prompt",
                                info="Use AI to enhance the prompt",
                            )

                            raw_mode = gr.Checkbox(
                                label="Raw mode",
                                value=False,
                                info=(
                                    "Generate less processed, "
                                    "more natural images"
                                ),
                            )

                        # Common parameters
                        seed = gr.Number(
                            label="Seed",
                            value=0,
                            precision=0,
                            minimum=0,
                            maximum=9999999999,
                            info=(
                                "Leave empty for random seed, or set "
                                "value for reproducible results."
                            ),
                        )

                        safety_tolerance = gr.Slider(
                            minimum=0,
                            maximum=6,
                            value=2,
                            step=1,
                            label="Safety tolerance",
                            info="0 (most strict) to 6 (least strict).",
                        )

                        output_format = gr.Radio(
                            choices=["jpeg", "png"],
                            value="jpeg",
                            label="Output format",
                            info="Select image format.",
                        )

            # Output section
            with gr.Row():
                with gr.Column():
                    generate_btn = gr.Button(
                        "✨ Generate image", variant="primary"
                    )
                    status_text = gr.Textbox(
                        label="Status", interactive=False
                    )

                with gr.Column():
                    output_image = gr.Image(
                        label="Generated Image",
                        type="filepath",
                        interactive=False,
                        show_download_button=True
                    )

            # Event handlers
            def refresh_models():
                self.manager.refresh_models()
                models = self.manager.list_models()
                choices = []
                for model in models:
                    if model and model.model_name and model.trigger_word:
                        choice = self._format_model_choice(model)
                        choices.append(choice)
                        print(f"Refreshed model choice: {choice}")
                return choices

            refresh_btn.click(
                fn=refresh_models,
                inputs=[],
                outputs=[model_dropdown]
            )

            def toggle_endpoint_params(choice):
                is_ultra = choice == self.ENDPOINT_ULTRA
                is_standard = choice == self.ENDPOINT_STANDARD
                return [
                    gr.update(visible=is_ultra),
                    gr.update(visible=is_standard)
                ]

            endpoint.change(
                fn=toggle_endpoint_params,
                inputs=[endpoint],
                outputs=[ultra_params, standard_params]
            )

            # Generation inputs
            generate_inputs = [
                endpoint, model_dropdown, prompt, negative_prompt,
                aspect_ratio, num_inference_steps, guidance_scale, strength,
                seed, output_format, prompt_upsampling, safety_tolerance,
                width, height, raw_mode,
            ]

            generate_btn.click(
                fn=self.generate_image,
                inputs=generate_inputs,
                outputs=[output_image, status_text],
            )

            return interface

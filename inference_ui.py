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
import numpy as np
from PIL import Image
import io

from model_manager import ModelManager
from constants import Paths


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
        
        # Use images directory from constants
        self.images_dir = Paths.IMAGES_DIR
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
            # Validate HTTP(S) URL with query parameters and encoded characters
            return bool(re.match(r'^https?://[\w\-.]+(?::\d+)?(?:/[^?]*)?(?:\?[^#]*)?$', url))

    def _save_image_from_url(self, image_url: str, output_format: str) -> Optional[np.ndarray]:
        """Get image from URL or base64 data and return as numpy array or None if failed."""
        if not isinstance(output_format, str) or output_format.lower() not in ["jpeg", "png"]:
            print("Invalid output format")
            return None
            
        if not self._validate_image_url(image_url):
            print("Invalid image URL format")
            return None

        try:
            # Get image data
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

            # Convert to numpy array
            img = Image.open(io.BytesIO(image_data))
            img_array = np.array(img)

            # Also save to file as backup
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"generated_image_{timestamp}.{output_format.lower()}"
                final_path = self.images_dir / filename
                
                with open(final_path, "wb") as f:
                    f.write(image_data)
                os.chmod(final_path, 0o600)
                print(f"Backup saved to {final_path}")
            except Exception as save_error:
                print(f"Warning: Could not save backup file: {save_error}")

            return img_array

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def _convert_image_to_base64(self, image: Optional[np.ndarray], format: str = "jpeg") -> Optional[str]:
        """Convert numpy image array to base64 string.
        
        Args:
            image: Numpy array of image
            format: Image format (jpeg or png)
            
        Returns:
            Base64 encoded image string or None if conversion failed
        """
        if image is None:
            return None
            
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format.upper())
            
            # Convert to base64
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Format as data URL
            mime_type = f"image/{format.lower()}"
            data_url = f"data:{mime_type};base64,{img_str}"
            
            return data_url
            
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

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
        strength_standard: float,
        seed: Optional[int],
        width: Optional[int],
        height: Optional[int],
        image_prompt: Optional[np.ndarray] = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        image_prompt_strength: float = 0.1,
        ultra_prompt_upsampling: bool = False,
        safety_tolerance: int = 2
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate an image using the selected model and parameters.

        Args:
            endpoint: API endpoint to use
            model_choice: Selected model from dropdown
            prompt: Text prompt for generation
            image_prompt: Optional image array to use as a prompt
            negative_prompt: Things to avoid in generation
            aspect_ratio: Image aspect ratio (ultra endpoint)
            steps: Number of generation steps
            guidance: Guidance scale for generation
            strength: Finetune strength for ultra endpoint (0.1-2.0)
            strength_standard: Finetune strength for standard endpoint (0.1-2.0)
            seed: Random seed for reproducibility
            output_format: Output image format
            image_prompt_strength: Blend between prompt and image prompt (0-1)
 
            ultra_prompt_upsampling: Whether to enhance prompt for ultra endpoint
            prompt_upsampling: Whether to enhance prompt
            safety_tolerance: Safety check level (0-6)
            width: Image width (standard endpoint)
            height: Image height (standard endpoint)

        Returns:
            Tuple of (numpy array of image or None, status message)
            
        Raises:
            ValueError: If any input parameters are invalid
        """
        try:
            # Validate endpoint
            if not endpoint or endpoint not in [self.ENDPOINT_ULTRA, self.ENDPOINT_STANDARD]:
                return (None, "Error: Invalid endpoint selection")

            # Validate model selection
            if not model_choice:
                return (None, "Error: Please select a model")
                
            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return (None, "Error: Invalid model selection")

            model = self.manager.get_model(model_id)
            if not model:
                return (None, "Error: Model not found")
                
            # Validate prompt
            if not self._validate_prompt(prompt):
                return (None, "Error: Invalid prompt format or content")
                
            # Validate negative prompt if provided
            if negative_prompt and not self._validate_prompt(negative_prompt):
                return (None, "Error: Invalid negative prompt format")
                
            # Validate numeric parameters
            if not self._validate_numeric_param(strength, 0.1, 2.0, False):
                return (None, "Error: Invalid strength value (must be between 0.1 and 2.0)")
                
            if not self._validate_numeric_param(guidance_scale, 1.5, 5.0):
                return (None, "Error: Invalid guidance scale value (must be between 1.5 and 5.0)")
                
            if num_steps is not None and not self._validate_numeric_param(float(num_steps), 1, 50, False):
                return (None, "Error: Invalid number of steps value (must be between 1 and 50)")
                
            if safety_tolerance not in range(7):  # 0 to 6
                return (None, "Error: Invalid safety tolerance value")
                
            # Validate output format
            if output_format not in ["jpeg", "png"]:
                return (None, "Error: Invalid output format")
                
            # Validate aspect ratio for ultra endpoint
            if endpoint == self.ENDPOINT_ULTRA:
                valid_ratios = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"]
                if aspect_ratio not in valid_ratios:
                    return (None, "Error: Invalid aspect ratio")
                    
            # Validate dimensions for standard endpoint
            if endpoint == self.ENDPOINT_STANDARD:
                if width is not None and not self._validate_numeric_param(float(width), 256, 1440, False):
                    return (None, "Error: Invalid width value")
                if height is not None and not self._validate_numeric_param(float(height), 256, 1440, False):
                    return (None, "Error: Invalid height value")

            # Log generation details
            print(f"\nGenerating image with model: {model.model_name}")
            print(f"Model ID: {model_id}")
            print(f"Endpoint: {endpoint}")
            print(f"Prompt: {prompt}")
            print(f"Standard prompt_upsampling: {prompt_upsampling}")
            print(f"Ultra prompt_upsampling: {ultra_prompt_upsampling}")

            # Common parameters
            # Convert image_prompt to base64 if provided
            image_prompt_base64 = None
            # Temporarily disable image prompt feature due to validation issues
            # if image_prompt is not None:
            #     image_prompt_base64 = self._convert_image_to_base64(image_prompt, output_format)
            #     print(f"Image prompt converted to base64 (length: {len(image_prompt_base64) if image_prompt_base64 else 0})")
            
            # Build parameters
            params = {
                "finetune_id": model_id,
                "prompt": prompt.strip() if prompt else "",
                "output_format": output_format.lower(),
                "safety_tolerance": safety_tolerance
                # Temporarily disable image prompt feature
                # "image_prompt": image_prompt_base64,
            }
            
            print(f"DEBUG: Initial common params: {params}")
            
            if endpoint == self.ENDPOINT_ULTRA:
                # Ultra endpoint parameters
                params.update({
                    "finetune_strength": strength,
                    "aspect_ratio": aspect_ratio,
                    # Temporarily disable image prompt strength parameter
                    # "image_prompt_strength": image_prompt_strength
                    # Note: finetune_strength is already in common parameters
                })
                
                # Use the ultra prompt upsampling parameter
                params["prompt_upsampling"] = ultra_prompt_upsampling
                print(f"DEBUG: Setting ultra prompt_upsampling to {ultra_prompt_upsampling}")
                
                # Optional parameters for ultra endpoint
                if seed is not None and seed > 0:
                    params["seed"] = seed
                    
            else:  # ENDPOINT_STANDARD
                # Standard endpoint parameters
                params.update({
                    "finetune_strength": strength_standard,
                    "steps": num_steps,
                    # Note: finetune_strength is already in common parameters
                    "guidance": guidance_scale,
                    "width": width or 1024,
                    "height": height or 768,
                    "prompt_upsampling": prompt_upsampling
                })
                print(f"DEBUG: Setting standard prompt_upsampling to {prompt_upsampling}")
                
                # Optional parameters for standard endpoint
                if seed is not None and seed > 0:
                    params["seed"] = seed
                if negative_prompt:
                    params["negative_prompt"] = negative_prompt

            # Remove any None values
            params = {k: v for k, v in params.items() if v is not None}

            print("Sending request to", endpoint)
            print("Parameters:", json.dumps(params, indent=2))

            # Start generation
            result = self.manager.generate_image(endpoint=endpoint, **params)
            if not result:
                return (None, "Error: No response from generation API")

            inference_id = result.get("id")
            if not inference_id:
                return (None, "Error: No inference ID received")

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
                    return (None, f"Generation failed: {error_msg}")

                elif state == "Ready":
                    image_url = status.get("result", {}).get("sample")
                    if not image_url:
                        return (None, "Error: No image URL in completed status")

                    print(f"Generation completed: {image_url}")
                    img_array = self._save_image_from_url(image_url, output_format)
                    if img_array is None:
                        return (None, "Error: Failed to save image")
                        
                    return (img_array, f"Generation completed successfully! Image saved as {output_format.upper()}")

                print(
                    f"Status: {state} "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(check_interval)
                attempt += 1

            return (None, "Error: Generation timed out")

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return (None, f"Error generating image: {str(e)}")

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
                            value=None,  # Start with no selection
                            label="Select model",
                            info=(
                                "Model trigger word shown in parentheses. "
                                "Include in prompt."
                            )
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

                    image_prompt = gr.Image(
                        label="Image prompt (optional)",
                        type="numpy",
                        # Default type, converts to numpy array
                        sources=["upload", "clipboard"],
                        show_download_button=False,
                        height=200,
                        elem_id="image_prompt_upload"
,
                        # Temporarily disable the image prompt feature
                        interactive=False
                    )
                    gr.Markdown(
                        """
                        **Image Prompt**: Upload an image to use as a visual reference.
                        The model will blend this with your text prompt based on the
                        "Image prompt strength" slider.
                        """
                        """
                        **TEMPORARILY DISABLED**: This feature is currently unavailable due to technical issues.
                        """
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
                                value=1.2,
                                step=0.1,
                                label="Finetune strength",
                                info=(
                                    "How strongly to apply model's style "
                                    "(default: 1.2)."
                                ),
                            )

                            image_prompt_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.1,
                                step=0.05,
                                label="Image prompt strength",
                                info=(
                                    "Blend between the prompt and the image prompt (0-1)."
                                ),
                            )
                            
                            ultra_prompt_upsampling = gr.Checkbox(
                                label="Prompt upsampling",
                                value=False,
                                info="Use AI to enhance the prompt (may produce more creative results)",
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
                                value=40,
                                step=1,
                                label="Number of Steps",
                                info=(
                                    "Number of generation steps "
                                    "(quality vs speed)"
                                ),
                            )

                            strength_standard = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.1,
                                step=0.1,
                                label="Finetune strength",
                                info=(
                                    "How strongly to apply model's style (0.1-2.0)."
                                ),
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.5,
                                maximum=5.0,
                                value=2.5,
                                step=0.1,
                                label="Guidance Scale",
                                info="Controls prompt adherence strength (1.5 to 5.0)",
                            )

                            prompt_upsampling = gr.Checkbox(
                                label="Prompt upsampling",
                                value=False,
                                info="Use AI to enhance the prompt (may produce more creative results)",
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
                        type="numpy",
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
                return gr.update(choices=choices, value=choices[0] if choices else None)

            refresh_btn.click(
                fn=refresh_models,
                inputs=[],
                outputs=[model_dropdown]
            )

            def toggle_endpoint_params(choice):
                is_ultra = choice == self.ENDPOINT_ULTRA
                is_standard = choice == self.ENDPOINT_STANDARD
                # Update visibility of parameter sections based on endpoint
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
                aspect_ratio, num_inference_steps, guidance_scale, strength, strength_standard, seed,
                width, height, image_prompt, output_format, prompt_upsampling,
                image_prompt_strength, ultra_prompt_upsampling, safety_tolerance
            ]

            generate_btn.click(
                fn=self.generate_image,
                inputs=generate_inputs,
                outputs=[output_image, status_text],
            )

            return interface

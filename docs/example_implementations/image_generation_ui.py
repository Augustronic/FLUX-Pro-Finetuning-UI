import gradio as gr
import time
from model_manager import ModelManager
from typing import Tuple, Optional
import json
import base64
import requests
from pathlib import Path
import os

class ImageGenerationUI:
    # Constants for endpoints
    ENDPOINT_ULTRA = "flux-pro-1.1-ultra-finetuned"
    ENDPOINT_STANDARD = "flux-pro-finetuned"

    def __init__(self, model_manager: ModelManager):
        self.manager = model_manager
        # Create images directory if it doesn't exist
        self.images_dir = Path("generated_images")
        self.images_dir.mkdir(exist_ok=True)

    def _format_model_choice(self, model) -> str:
        """Format model metadata for dropdown display."""
        parts = [
            f"{model.model_name}",
            f"({model.trigger_word})",
            f"{model.type.upper()}",
            f"{model.mode.capitalize()}"
        ]
        if model.rank:
            parts.append(f"Rank {model.rank}")
        return " - ".join(parts)

    def _get_model_id_from_choice(self, choice: str) -> Optional[str]:
        """Extract model ID from formatted choice string."""
        for model in self.manager.list_models():
            if self._format_model_choice(model) == choice:
                return model.finetune_id
        return None

    def _save_image_from_url(self, image_url: str, output_format: str) -> str:
        """Save image from URL or base64 data to a file."""
        try:
            # Check if the URL is a data URL (base64)
            if image_url.startswith('data:'):
                # Extract the base64 data
                header, encoded = image_url.split(",", 1)
                image_data = base64.b64decode(encoded)
            else:
                # Download from regular URL
                response = requests.get(image_url)
                image_data = response.content

            # Generate unique filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"generated_image_{timestamp}.{output_format}"
            filepath = self.images_dir / filename

            # Save the image
            with open(filepath, 'wb') as f:
                f.write(image_data)

            return str(filepath)
        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def generate_image(
        self,
        endpoint: str,
        model_choice: str,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        num_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_prompt: str = None,
        image_prompt_strength: float = 0.1,
        width: int = None,
        height: int = None,
        raw_mode: bool = False
    ) -> Tuple[str, str]:
        """Generate image using selected model and parameters."""
        try:
            if not model_choice:
                return None, "Error: Please select a model"
            if not prompt or not prompt.strip():
                return None, "Error: Please enter a prompt"

            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return None, "Error: Invalid model selection"

            model = self.manager.get_model(model_id)
            if not model:
                return None, "Error: Model not found"

            print(f"\nGenerating image with model: {model.model_name}")
            print(f"Model ID: {model_id}")
            print(f"Endpoint: {endpoint}")
            print(f"Prompt: {prompt}")
            print(f"Raw Mode: {raw_mode}")

            # Common parameters
            params = {
                "finetune_id": model_id,
                "prompt": prompt,
                "output_format": output_format,
                "finetune_strength": strength,
                "safety_tolerance": safety_tolerance
            }

            if endpoint == self.ENDPOINT_ULTRA:
                # Ultra endpoint uses aspect_ratio parameter directly
                params.update({
                    "aspect_ratio": aspect_ratio,
                    "seed": seed if seed is not None else None,
                    "image_prompt": image_prompt if image_prompt else None,
                    "image_prompt_strength": image_prompt_strength if image_prompt else None
                })
            else:  # ENDPOINT_STANDARD
                # Standard endpoint uses width/height and supports raw mode
                params.update({
                    "steps": num_steps,
                    "guidance": guidance_scale,
                    "width": width or 1024,
                    "height": height or 768,
                    "prompt_upsampling": prompt_upsampling,
                    "seed": seed if seed is not None else None,
                    "image_prompt": None,  # Set to None for standard endpoint
                    "raw": raw_mode  # Add raw mode parameter
                })

            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}

            # Start generation
            result = self.manager.generate_image(endpoint=endpoint, **params)

            if not result:
                return None, "Error: No response from generation API"

            inference_id = result.get('id')
            if not inference_id:
                return None, "Error: No inference ID received"

            print(f"Inference ID: {inference_id}")

            max_attempts = 30
            attempt = 0

            while attempt < max_attempts:
                status = self.manager.get_generation_status(inference_id)
                state = status.get('status', '')

                if state == 'Failed':
                    error_msg = status.get('error', 'Unknown error')
                    print(f"Generation failed: {error_msg}")
                    return None, f"Generation failed: {error_msg}"

                elif state == 'Ready':
                    image_url = status.get('result', {}).get('sample')
                    if not image_url:
                        return None, "Error: No image URL in completed status"
                    print(f"Generation completed: {image_url}")

                    local_path = self._save_image_from_url(image_url, output_format)
                    if not local_path:
                        return None, "Error: Failed to save image"

                    return local_path, f"Generation completed successfully! Image saved as {output_format.upper()}"

                print(f"Status: {state} (Attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
                attempt += 1

            return None, "Error: Generation timed out"

        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return None, f"Error generating image: {str(e)}"

    def create_ui(self) -> gr.Blocks:
        """Create the image generation interface."""
        with gr.Blocks(title="AI Image Generation") as interface:
            gr.Markdown("""
            # AI Image Generation
            Generate images using your fine-tuned models.

            **Important**: Make sure to include the model's trigger word in your prompt!
            """)

            with gr.Row():
                with gr.Column():
                    # Endpoint selection
                    endpoint = gr.Radio(
                        choices=[
                            ("FLUX Pro 1.1 Ultra", self.ENDPOINT_ULTRA),
                            ("FLUX Pro Standard", self.ENDPOINT_STANDARD)
                        ],
                        value=self.ENDPOINT_ULTRA,
                        label="Generation Endpoint",
                        info="Select the generation endpoint to use"
                    )

                    # Model selection and basic parameters
                    model_choices = [
                        self._format_model_choice(model)
                        for model in self.manager.list_models()
                    ]

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            label="Select Model",
                            info="The model's trigger word is shown in parentheses"
                        )
                        refresh_btn = gr.Button("🔄", scale=0.1)

                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt (make sure to include the trigger word shown above)",
                        lines=3,
                        info="Important: Include the model's trigger word in your prompt"
                    )

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="Enter things to avoid in the image",
                        lines=2
                    )

                with gr.Column():
                    # Advanced parameters
                    with gr.Box():
                        gr.Markdown("### Image Parameters")

                        # Parameters for Ultra endpoint
                        with gr.Column(visible=True) as ultra_params:
                            aspect_ratio = gr.Radio(
                                choices=["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"],
                                value="16:9",
                                label="Aspect Ratio",
                                info="Select image dimensions ratio (between 21:9 and 9:21)"
                            )

                            strength = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.2,
                                step=0.1,
                                label="Model Strength",
                                info="How strongly to apply the model's style (default: 1.2)"
                            )

                        # Parameters for Standard endpoint
                        with gr.Column(visible=False) as standard_params:
                            with gr.Row():
                                width = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=1024,
                                    step=32,
                                    label="Width",
                                    info="Must be a multiple of 32"
                                )
                                height = gr.Slider(
                                    minimum=256,
                                    maximum=1440,
                                    value=768,
                                    step=32,
                                    label="Height",
                                    info="Must be a multiple of 32"
                                )

                            num_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=40,
                                step=1,
                                label="Number of Steps",
                                info="More steps = higher quality but slower"
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.5,
                                maximum=5.0,
                                value=2.5,
                                step=0.1,
                                label="Guidance Scale",
                                info="How closely to follow the prompt"
                            )

                            prompt_upsampling = gr.Checkbox(
                                label="Prompt Upsampling",
                                value=False,
                                info="Automatically modify prompt for more creative generation"
                            )

                            raw_mode = gr.Checkbox(
                                label="Raw Mode",
                                value=False,
                                info="Enable raw mode for more realistic results"
                            )

                        # Common parameters
                        seed = gr.Number(
                            label="Seed",
                            value=None,
                            precision=0,
                            minimum=0,
                            maximum=2147483647,
                            info="Leave empty for random seed, or set a specific value for reproducible results"
                        )

                        safety_tolerance = gr.Slider(
                            minimum=0,
                            maximum=6,
                            value=2,
                            step=1,
                            label="Safety Tolerance",
                            info="0 (most strict) to 6 (least strict)"
                        )

                        output_format = gr.Radio(
                            choices=["jpeg", "png"],
                            value="jpeg",
                            label="Output Format",
                            info="Select the image format"
                        )

            # Output section
            with gr.Row():
                with gr.Column():
                    generate_btn = gr.Button(
                        "Generate Image",
                        variant="primary"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )

                with gr.Column():
                    output_image = gr.Image(
                        label="Generated Image",
                        type="filepath",
                        interactive=False,
                        show_download_button=True
                    )

            # Handle model refresh
            def refresh_models():
                self.manager.refresh_models()
                return gr.Dropdown.update(
                    choices=[self._format_model_choice(m) for m in self.manager.list_models()]
                )

            refresh_btn.click(
                fn=refresh_models,
                inputs=[],
                outputs=[model_dropdown]
            )

            # Handle endpoint visibility
            def toggle_endpoint_params(choice):
                return (
                    gr.Column.update(visible=choice == self.ENDPOINT_ULTRA),
                    gr.Column.update(visible=choice == self.ENDPOINT_STANDARD)
                )

            endpoint.change(
                fn=toggle_endpoint_params,
                inputs=[endpoint],
                outputs=[ultra_params, standard_params]
            )

            # Handle generation
            generate_inputs = [
                endpoint,
                model_dropdown,
                prompt,
                negative_prompt,
                aspect_ratio,
                num_steps,
                guidance_scale,
                strength,
                seed,
                output_format,
                prompt_upsampling,
                safety_tolerance,
                gr.Textbox(visible=False, value=None),  # image_prompt
                gr.Number(visible=False, value=0.1),    # image_prompt_strength
                width,
                height,
                raw_mode
            ]

            generate_btn.click(
                fn=self.generate_image,
                inputs=generate_inputs,
                outputs=[output_image, status_text]
            )

        return interface 

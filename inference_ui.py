import gradio as gr
import time
from model_manager import ModelManager
from typing import Tuple, Union
import base64
import requests
from pathlib import Path


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
            f"{model.mode.capitalize()}",
        ]
        if model.rank:
            parts.append(f"Rank {model.rank}")
        return " - ".join(parts)

    def _get_model_id_from_choice(self, choice: str) -> str:
        """Extract model ID from formatted choice string."""
        for model in self.manager.list_models():
            if self._format_model_choice(model) == choice:
                return model.finetune_id
        return ""

    def _save_image_from_url(self, image_url: str, output_format: str) -> str:
        """Save image from URL or base64 data to a file."""
        try:
            # Check if the URL is a data URL (base64)
            if image_url.startswith("data:"):
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
            with open(filepath, "wb") as f:
                f.write(image_data)

            return str(filepath)
        except Exception as e:
            print(f"Error saving image: {e}")
            return ""

    def generate_image(
        self,
        endpoint: str,
        model_choice: str,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        num_steps: Union[int, None],
        guidance_scale: Union[float, None],
        strength: float,
        seed: Union[int, None],
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        image_prompt: Union[str, None] = None,
        image_prompt_strength: float = 0.1,
        width: Union[int, None] = None,
        height: Union[int, None] = None,
        raw_mode: bool = False,
    ) -> Tuple[str, str]:
        """Generate image using selected model and parameters."""
        try:
            if not model_choice:
                return ("", "Error: Please select a model")
            if not prompt or not prompt.strip():
                return ("", "Error: Please enter a prompt")

            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return ("", "Error: Invalid model selection")

            model = self.manager.get_model(model_id)
            if not model:
                return ("", "Error: Model not found")

            print(f"\nGenerating image with model: {model.model_name}")
            print(f"Model ID: {model_id}")
            print(f"Endpoint: {endpoint}")
            print(f"Prompt: {prompt}")
            print(f"Raw mode: {raw_mode}")

            # Common parameters
            params = {
                "finetune_id": model_id,
                "prompt": prompt,
                "output_format": output_format,
                "finetune_strength": strength,
                "safety_tolerance": safety_tolerance,
            }

            if endpoint == self.ENDPOINT_ULTRA:
                # Ultra endpoint uses aspect_ratio parameter directly
                params.update(
                    {
                        "aspect_ratio": aspect_ratio,
                        "seed": seed if seed is not None else 0,
                        "image_prompt": image_prompt if image_prompt else "",
                        "image_prompt_strength": (
                            image_prompt_strength if image_prompt else 0.1
                        ),
                    }
                )
            else:  # ENDPOINT_STANDARD
                # Standard endpoint uses width/height and supports raw mode
                params.update(
                    {
                        "steps": num_steps if num_steps is not None else 40,
                        # Default guidance scale is 2.5
                        "guidance": float(guidance_scale or 2.5),
                        "width": width if width is not None else 1024,
                        "height": height if height is not None else 768,
                        "prompt_upsampling": prompt_upsampling,
                        "seed": seed if seed is not None else 0,
                        # Empty string for standard endpoint
                        "image_prompt": "",
                        "raw": raw_mode,
                    }
                )

            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}

            # Start generation
            result = self.manager.generate_image(endpoint=endpoint, **params)

            if not result:
                return ("", "Error: No response from generation API")

            inference_id = result.get("id")
            if not inference_id:
                return ("", "Error: No inference ID received")

            print(f"Inference ID: {inference_id}")

            max_attempts = 30
            attempt = 0

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

                    local_path = self._save_image_from_url(
                        image_url, output_format
                    )
                    if not local_path:
                        return ("", "Error: Failed to save image")

                    msg = "Generation completed successfully!"
                    msg += f" Image saved as {output_format.upper()}"
                    return (local_path, msg)

                print(
                    f"Status: {state} "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(2)
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
                            ("FLUX Pro 1.1 Ultra", self.ENDPOINT_ULTRA),
                            ("FLUX Pro Standard", self.ENDPOINT_STANDARD),
                        ],
                        value=self.ENDPOINT_ULTRA,
                        label="Generation endpoint",
                        info="Select the generation endpoint to use.",
                    )

                    # Model selection and basic parameters
                    model_choices = [
                        self._format_model_choice(model)
                        for model in self.manager.list_models()
                    ]

                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            label="Select model",
                            info="Model trigger word in parentheses.",
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
                    # Advanced parameters
                    with gr.Group():
                        gr.Markdown("### Image parameters")

                        # Parameters for Ultra endpoint
                        with gr.Column(visible=True) as ultra_params:
                            aspect_ratio = gr.Radio(
                                choices=[
                                    "21:9", "16:9", "3:2", "4:3", "1:1",
                                    "3:4", "2:3", "9:16", "9:21",
                                ],
                                value="1:1",
                                label="Aspect ratio",
                                info="Select image dimensions ratio.",
                            )

                            strength = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.2,
                                step=0.1,
                                label="Model strength",
                                info=(
                                    "How strongly to apply model's style "
                                    "(default: 1.2)."
                                ),
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

                            num_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=40,
                                step=1,
                                label="Steps",
                                info="Quality vs speed",
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.5,
                                maximum=5.0,
                                value=2.5,
                                step=0.1,
                                label="CFG",
                                info="Prompt adherence strength",
                            )

                            prompt_upsampling = gr.Checkbox(
                                label="Enhance",
                                info="AI"
                            )

                            raw_mode = gr.Checkbox(
                                label="Raw mode", value=False, info="Raw mode"
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
                        "Generate image", variant="primary"
                    )
                    status_text = gr.Textbox(
                        label="Status", interactive=False
                    )

                with gr.Column():
                    output_image = gr.Image(
                        label="Generated image",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                    )

            # Handle model refresh
            def refresh_models():
                self.manager.refresh_models()
                return {
                    "choices": [
                        self._format_model_choice(m)
                        for m in self.manager.list_models()
                    ]
                }

            refresh_btn.click(
                fn=refresh_models,
                inputs=[],
                outputs=[model_dropdown]
            )

            # Handle endpoint visibility
            def toggle_endpoint_params(choice):
                return (
                    {"visible": choice == self.ENDPOINT_ULTRA},
                    {"visible": choice == self.ENDPOINT_STANDARD}
                )

            endpoint.change(
                fn=toggle_endpoint_params,
                inputs=[endpoint],
                outputs=[ultra_params, standard_params]
            )

            # Handle generation
            generate_inputs = [
                endpoint, model_dropdown, prompt, negative_prompt,
                aspect_ratio, num_steps, guidance_scale, strength,
                seed, output_format, prompt_upsampling, safety_tolerance,
                gr.Textbox(visible=False, value=""),  # image_prompt
                gr.Number(visible=False, value=0.1),  # image_prompt_strength
                width, height, raw_mode,
            ]

            generate_btn.click(
                fn=self.generate_image,
                inputs=generate_inputs,
                outputs=[output_image, status_text],
            )

        return interface

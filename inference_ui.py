import gradio as gr
import time
from typing import Dict, Any, Optional, Tuple
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
        self.manager = model_manager
        # Use system temp directory for generated images
        self.images_dir = Path(tempfile.gettempdir()) / "generated_images"
        self.images_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    def _format_model_choice(self, model: Any) -> str:
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
            if image_url.startswith("data:"):
                header, encoded = image_url.split(",", 1)
                image_data = base64.b64decode(encoded)
            else:
                response = requests.get(image_url)
                response.raise_for_status()
                image_data = response.content

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"generated_image_{timestamp}.{output_format}"
            filepath = self.images_dir / filename

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
        steps: Optional[int],
        guidance: Optional[float],
        strength: float,
        seed: Optional[int],
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        safety_tolerance: int = 2,
        width: Optional[int] = None,
        height: Optional[int] = None,
        raw_mode: bool = False,
    ) -> Tuple[str, str]:
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
        """
        try:
            # Validate inputs
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

            # Log generation details
            print(f"\nGenerating image with model: {model.model_name}")
            print(f"Model ID: {model_id}")
            print(f"Endpoint: {endpoint}")
            print(f"Prompt: {prompt}")

            # Common parameters
            params: Dict[str, Any] = {
                "finetune_id": model_id,
                "prompt": prompt.strip(),
                "output_format": output_format.lower(),
                "num_images": 1,
                "finetune_strength": strength,
                "safety_tolerance": safety_tolerance,
                "seed": seed if seed is not None else 0
            }

            # Add endpoint-specific parameters
            if endpoint == self.ENDPOINT_ULTRA:
                params.update({
                    "aspect_ratio": aspect_ratio,
                    "guidance_scale": guidance or 2.5
                })
            else:  # ENDPOINT_STANDARD
                # Ensure parameters are within valid ranges
                steps_value = min(max(steps or 40, 1), 50)
                guidance_value = min(max(guidance or 2.5, 1.5), 5.0)
                width_value = (width or 1024) // 32 * 32
                height_value = (height or 768) // 32 * 32

                params.update({
                    "steps": steps_value,
                    "guidance": float(guidance_value),
                    "width": width_value,
                    "height": height_value,
                    "raw": raw_mode,
                    "prompt_upsampling": prompt_upsampling
                })

            if negative_prompt:
                params["negative_prompt"] = negative_prompt

            # Remove None values
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

                    print(f"Generation completed: {image_url[:70]}...")
                    local_path = self._save_image_from_url(
                        image_url, output_format
                    )
                    if not local_path:
                        return ("", "Error: Failed to save image")

                    return (
                        local_path,
                        f"Generation completed successfully! "
                        f"Image saved as {output_format.upper()}"
                    )

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

                            steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=40,
                                step=1,
                                label="Steps",
                                info=(
                                    "Number of generation steps "
                                    "(quality vs speed)"
                                ),
                            )

                            guidance = gr.Slider(
                                minimum=1.5,
                                maximum=5.0,
                                value=2.5,
                                step=0.1,
                                label="Guidance scale",
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
                        label="Generated image",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
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
                aspect_ratio, steps, guidance, strength,
                seed, output_format, prompt_upsampling, safety_tolerance,
                width, height, raw_mode,
            ]

            generate_btn.click(
                fn=self.generate_image,
                inputs=generate_inputs,
                outputs=[output_image, status_text],
            )

            return interface

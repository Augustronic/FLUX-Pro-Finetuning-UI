"""Image generation component for FLUX Pro Finetuning UI."""

import os
from typing import Dict, Any, Optional, Tuple
import gradio as gr

from ui.base import UIComponent
from container import container
from constants import Endpoints, RequestConfig


class ImageGenerationComponent(UIComponent):
    """Handles image generation and display in the UI."""

    def __init__(self) -> None:
        """Initialize the image generation component."""
        super().__init__()
        self.manager = container.model_manager

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
        """Generate an image using the selected model and parameters.
        
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
            # Get model ID from choice
            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return ("", "Error: Invalid model selection")

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
            if endpoint == Endpoints.ULTRA:
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

            # Start generation
            result = self.manager.generate_image(endpoint=endpoint, **params)
            if not result:
                return ("", "Error: No response from generation API")

            inference_id = result.get("id")
            if not inference_id:
                return ("", "Error: No inference ID received")

            # Monitor generation progress
            max_attempts = 30
            attempt = 0
            check_interval = 2

            while attempt < max_attempts:
                status = self.manager.get_generation_status(inference_id)
                state = status.get("status", "")

                if state == "Failed":
                    error_msg = status.get("error", "Unknown error")
                    return ("", f"Generation failed: {error_msg}")

                elif state == "Ready":
                    image_url = status.get("result", {}).get("sample")
                    if not image_url:
                        return ("", "Error: No image URL in completed status")

                    return (
                        image_url,
                        f"Generation completed successfully! "
                        f"Image saved as {output_format.upper()}"
                    )

                attempt += 1

            return ("", "Error: Generation timed out")

        except Exception as e:
            return ("", f"Error generating image: {str(e)}")

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the image generation UI elements.
        
        Args:
            parent: Optional parent Blocks instance
            
        Returns:
            The created Gradio Blocks interface
        """
        blocks = parent or gr.Blocks()
        with blocks:
            with gr.Row():
                with gr.Column():
                    generate_btn = gr.Button(
                        "✨ Generate image",
                        variant="primary"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )

                with gr.Column():
                    output_image = gr.Image(
                        label="Generated image",
                        type="filepath",
                        interactive=False,
                        show_download_button=True,
                    )

            # Register elements
            self.register_element("generate_button", generate_btn)
            self.register_element("status_text", status_text)
            self.register_element("output_image", output_image)

        return blocks

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
                if model and hasattr(model, 'finetune_id'):
                    return model.finetune_id
            return ""
        except Exception as e:
            print(f"Error extracting model ID: {e}")
            return ""
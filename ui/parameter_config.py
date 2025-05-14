"""Parameter configuration component for FLUX Pro Finetuning UI."""

from typing import Dict, Any, Optional, Tuple
import re
import gradio as gr

from ui.base import UIComponent
from constants import (
    Endpoints,
    GenerationConfig as GC,
    ValidationConfig as VC
)

# Compile regex pattern
PROMPT_PATTERN = re.compile(VC.VALID_PROMPT_CHARS)


class ParameterConfigComponent(UIComponent):
    """Handles parameter configuration for image generation."""

    def __init__(self) -> None:
        """Initialize the parameter configuration component."""
        super().__init__()
        # Initialize as empty columns, will be set during create()
        self.ultra_params = gr.Column()
        self.standard_params = gr.Column()

    def _validate_prompt(self, prompt: str) -> bool:
        """Validate prompt text for safety and format.

        Args:
            prompt: Text to validate

        Returns:
            True if valid, False otherwise
        """
        if not prompt or not isinstance(prompt, str):
            return False

        # Remove excessive whitespace
        prompt = prompt.strip()
        if len(prompt) == 0 or len(prompt) > VC.MAX_PROMPT_LENGTH:
            return False

        # Check for valid characters
        return bool(PROMPT_PATTERN.match(prompt))

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the parameter configuration UI elements.

        Args:
            parent: Optional parent Blocks instance

        Returns:
            The created Gradio Blocks interface
        """
        blocks = parent or gr.Blocks()
        with blocks:
            with gr.Column():
                # Prompt inputs
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter prompt (include trigger word shown above).",
                    lines=3,
                    info="Include model's trigger word in prompt.",
                )

                negative_prompt = gr.Textbox(
                    label="Negative prompt (optional)",
                    placeholder="Enter things to avoid in the image.",
                    lines=2,
                )

                with gr.Group():
                    gr.Markdown("### Image parameters")

                    # Ultra endpoint parameters
                    with gr.Column(visible=True) as self.ultra_params:
                        aspect_ratio = gr.Radio(
                            choices=GC.ASPECT_RATIOS,
                            value="16:9",
                            label="Aspect ratio",
                            info="Select image dimensions ratio.",
                        )

                        strength = gr.Slider(
                            minimum=GC.STRENGTH_RANGE["min"],
                            maximum=GC.STRENGTH_RANGE["max"],
                            value=GC.STRENGTH_RANGE["default"],
                            step=0.1,
                            label="Model strength",
                            info="How strongly to apply model's style.",
                        )

                    # Standard endpoint parameters
                    with gr.Column(visible=False) as self.standard_params:
                        with gr.Row():
                            width = gr.Slider(
                                minimum=GC.DIMENSION_RANGE["min"],
                                maximum=GC.DIMENSION_RANGE["max"],
                                value=GC.DIMENSION_RANGE["default_width"],
                                step=GC.DIMENSION_RANGE["step"],
                                label="Width",
                                info="Must be a multiple of 32",
                            )
                            height = gr.Slider(
                                minimum=GC.DIMENSION_RANGE["min"],
                                maximum=GC.DIMENSION_RANGE["max"],
                                value=GC.DIMENSION_RANGE["default_height"],
                                step=GC.DIMENSION_RANGE["step"],
                                label="Height",
                                info="Must be a multiple of 32",
                            )

                        steps = gr.Slider(
                            minimum=GC.STEPS_RANGE["min"],
                            maximum=GC.STEPS_RANGE["max"],
                            value=GC.STEPS_RANGE["default"],
                            step=1,
                            label="Steps",
                            info="Number of generation steps (quality vs speed)",
                        )

                        guidance = gr.Slider(
                            minimum=GC.GUIDANCE_RANGE["min"],
                            maximum=GC.GUIDANCE_RANGE["max"],
                            value=GC.GUIDANCE_RANGE["default"],
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
                            info="Generate less processed, more natural images",
                        )

                    # Common parameters
                    seed = gr.Number(
                        label="Seed",
                        value=0,
                        precision=0,
                        minimum=0,
                        maximum=9999999999,
                        info="Leave empty for random seed, or set value for reproducible results.",
                    )

                    safety_tolerance = gr.Slider(
                        minimum=GC.SAFETY_RANGE["min"],
                        maximum=GC.SAFETY_RANGE["max"],
                        value=GC.SAFETY_RANGE["default"],
                        step=1,
                        label="Safety tolerance",
                        info="0 (most strict) to 6 (least strict).",
                    )

                    output_format = gr.Radio(
                        choices=GC.VALID_FORMATS,
                        value=GC.VALID_FORMATS[0],
                        label="Output format",
                        info="Select image format.",
                    )

                # Register elements
                self.register_element("prompt", prompt)
                self.register_element("negative_prompt", negative_prompt)
                self.register_element("aspect_ratio", aspect_ratio)
                self.register_element("strength", strength)
                self.register_element("width", width)
                self.register_element("height", height)
                self.register_element("steps", steps)
                self.register_element("guidance", guidance)
                self.register_element("prompt_upsampling", prompt_upsampling)
                self.register_element("raw_mode", raw_mode)
                self.register_element("seed", seed)
                self.register_element("safety_tolerance", safety_tolerance)
                self.register_element("output_format", output_format)

        return blocks

    def toggle_endpoint_params(self, endpoint: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Toggle visibility of endpoint-specific parameters.

        Args:
            endpoint: Selected endpoint

        Returns:
            Updates for ultra and standard parameter visibility
        """
        is_ultra = endpoint == Endpoints.ULTRA
        return (
            {"visible": is_ultra},
            {"visible": not is_ultra}
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values.

        Returns:
            Dictionary of parameter values
        """
        return {
            name: element.value
            for name, element in self._elements.items()
        }

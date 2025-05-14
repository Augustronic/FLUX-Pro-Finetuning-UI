"""Image generation component for FLUX Pro Finetuning UI."""

from typing import Dict, Any, Optional, Tuple, Union
import gradio as gr
import requests
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from io import BytesIO
import base64

from ui.base import UIComponent
from container import container
from constants import Endpoints
from utils.error_handling.error_handler import ErrorHandler, ErrorContext, APIError
from utils.logging.logger import get_logger
from utils.validation.validator import Validator, ValidationRule

class ImageGenerationComponent(UIComponent):
    """Handles image generation and display in the UI."""

    def __init__(self) -> None:
        """Initialize the image generation component."""
        super().__init__()
        self.manager = container.model_manager
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        self.validator = Validator()

        # Define validation rules
        self.generation_rules = [
            ValidationRule(
                field="model_choice",
                rule_type="required",
                message="Model selection is required"
            ),
            ValidationRule(
                field="prompt",
                rule_type="required",
                message="Prompt is required"
            ),
            ValidationRule(
                field="prompt",
                rule_type="max_length",
                value=1000,
                message="Prompt must not exceed 1000 characters"
            ),
            ValidationRule(
                field="steps",
                rule_type="range",
                value=(1, 50),
                message="Steps must be between 1 and 50"
            ),
            ValidationRule(
                field="guidance",
                rule_type="range",
                value=(1.5, 5.0),
                message="Guidance must be between 1.5 and 5.0"
            ),
            ValidationRule(
                field="strength",
                rule_type="range",
                value=(0.1, 2.0),
                message="Strength must be between 0.1 and 2.0"
            )
        ]

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
    ) -> Tuple[Union[NDArray[Any], str], str]:
        """Generate an image using the selected model and parameters."""
        try:
            # Validate inputs
            self.validator.validate(
                {
                    "model_choice": model_choice,
                    "prompt": prompt,
                    "steps": steps,
                    "guidance": guidance,
                    "strength": strength
                },
                self.generation_rules,
                "ImageGenerationComponent"
            )

            # Get model ID from choice
            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                raise APIError(
                    "Invalid model selection",
                    context=ErrorContext(
                        component="ImageGenerationComponent",
                        operation="generate_image",
                        details={"model_choice": model_choice}
                    )
                )

            # Log generation attempt
            self.logger.info(
                "Starting image generation",
                extra={
                    "endpoint": endpoint,
                    "model_id": model_id,
                    "prompt_length": len(prompt)
                }
            )

            # Common parameters
            params: Dict[str, Any] = {
                "finetune_id": model_id,
                "prompt": prompt.strip(),
                "output_format": output_format.lower(),
                "finetune_strength": strength,
                "safety_tolerance": safety_tolerance,
            }

            # Only include seed if specified
            if seed is not None:
                params["seed"] = seed

            # Add endpoint-specific parameters
            if endpoint == Endpoints.ULTRA:
                params.update({
                    "aspect_ratio": aspect_ratio,
                    "guidance_scale": guidance if guidance is not None else 2.5,
                    "prompt_upsampling": prompt_upsampling
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
            self.logger.debug("Sending generation request", extra={"params": params})
            result = self.manager.generate_image(endpoint=endpoint, **params)

            if not result:
                raise APIError(
                    "No response from generation API",
                    context=ErrorContext(
                        component="ImageGenerationComponent",
                        operation="generate_image"
                    )
                )

            if isinstance(result, dict) and "detail" in result:
                # Handle validation error response
                validation_errors = result["detail"]
                error_msgs = []
                for error in validation_errors:
                    field = " -> ".join(str(loc) for loc in error.get("loc", []))
                    msg = error.get("msg", "Unknown error")
                    error_msgs.append(f"{field}: {msg}")
                raise APIError(
                    "Validation errors",
                    context=ErrorContext(
                        component="ImageGenerationComponent",
                        operation="generate_image",
                        details={"errors": error_msgs}
                    )
                )

            inference_id = result.get("id")
            if not inference_id:
                raise APIError(
                    "No inference ID received",
                    context=ErrorContext(
                        component="ImageGenerationComponent",
                        operation="generate_image"
                    )
                )

            self.logger.info("Generation started", extra={"inference_id": inference_id})

            # Monitor generation progress
            max_attempts = 30
            attempt = 0

            while attempt < max_attempts:
                status = self.manager.get_generation_status(inference_id)
                self.logger.debug(
                    "Generation status",
                    extra={"status": status, "attempt": attempt}
                )

                state = status.get("status", "")

                # Handle all possible status values
                if state == "Task not found":
                    raise APIError(f"Task {inference_id} not found")
                elif state == "Request Moderated":
                    raise APIError("Request was moderated")
                elif state == "Content Moderated":
                    raise APIError("Generated content was moderated")
                elif state == "Error":
                    error_msg = status.get("details", {}).get("error", "Unknown error")
                    raise APIError(f"Generation failed: {error_msg}")
                elif state == "Ready":
                    result = status.get("result", {})
                    if not result:
                        raise APIError("No result data in completed status")

                    image_url = result.get("sample")
                    if not image_url:
                        raise APIError("No image URL in completed status")

                    try:
                        # Download image
                        response = requests.get(image_url)
                        response.raise_for_status()

                        # Convert to PIL Image
                        image = Image.open(BytesIO(response.content))
                        self.logger.info("Image generated successfully")

                        # Convert to base64
                        buffered = BytesIO()
                        image.save(buffered, format=output_format.upper())
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        data_url = f"data:image/{output_format};base64,{img_str}"

                        return (
                            data_url,
                            "Generation completed successfully!"
                        )
                    except Exception as e:
                        raise APIError(
                            f"Error saving generated image: {str(e)}",
                            context=ErrorContext(
                                component="ImageGenerationComponent",
                                operation="save_image"
                            )
                        )
                elif state == "Pending":
                    # Continue waiting
                    pass
                else:
                    self.logger.warning(
                        "Unexpected status state",
                        extra={"state": state}
                    )

                attempt += 1

            raise APIError(
                "Generation timed out",
                context=ErrorContext(
                    component="ImageGenerationComponent",
                    operation="generate_image",
                    details={"max_attempts": max_attempts}
                )
            )

        except Exception as e:
            error_response = self.error_handler.handle_error(
                e,
                context=ErrorContext(
                    component="ImageGenerationComponent",
                    operation="generate_image"
                )
            )
            return ("", error_response["error"]["message"])

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the image generation UI elements."""
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
                        type="pil",
                        interactive=False,
                        show_download_button=True,
                    )

            # Register elements
            self.register_element("generate_button", generate_btn)
            self.register_element("status_text", status_text)
            self.register_element("output_image", output_image)

        return blocks

    def _format_model_choice(self, model: Any) -> str:
        """Format model metadata for dropdown display."""
        if not model or not hasattr(model, 'model_name') or not hasattr(model, 'trigger_word'):
            raise ValueError("Invalid model data")

        # Validate required attributes
        if not all([
            isinstance(getattr(model, attr, None), str)
            for attr in ['model_name', 'trigger_word', 'type', 'mode']
        ]):
            raise ValueError("Invalid model attributes")

        # Format display values
        parts = [
            model.model_name,
            f"({model.trigger_word})",
            f"{model.type.upper()}",
            f"{model.mode.capitalize()}",
        ]

        # Add rank if present
        if hasattr(model, 'rank') and isinstance(model.rank, (int, float)):
            parts.append(f"Rank {int(model.rank)}")

        return " - ".join(parts)

    def _get_model_id_from_choice(self, choice: str) -> str:
        """Extract model ID from formatted choice string."""
        if not isinstance(choice, str) or not choice.strip():
            return ""

        try:
            for model in self.manager.list_models():
                if model and hasattr(model, 'finetune_id'):
                    formatted_choice = self._format_model_choice(model)
                    if formatted_choice == choice:
                        return model.finetune_id
            return ""
        except Exception as e:
            self.logger.error(
                "Error extracting model ID",
                extra={"error": str(e), "choice": choice}
            )
            return ""

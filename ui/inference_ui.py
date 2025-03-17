"""
Inference UI Component for FLUX-Pro-Finetuning-UI.

Provides UI for generating images with fine-tuned models.
"""

import gradio as gr
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ui.base import BaseUI
from services.business.inference_service import InferenceService
from services.business.model_service import ModelService


class InferenceUI(BaseUI):
    """
    UI component for generating images with fine-tuned models.
    
    Provides UI for generating images with fine-tuned models, including
    parameter configuration and result display.
    """
    
    # API endpoints
    ENDPOINT_ULTRA = "flux-pro-1.1-ultra-finetuned"
    ENDPOINT_STANDARD = "flux-pro-finetuned"
    
    def __init__(
        self,
        inference_service: InferenceService,
        model_service: ModelService
    ):
        """
        Initialize the inference UI component.
        
        Args:
            inference_service: Service for inference operations
            model_service: Service for model management
        """
        super().__init__(
            title="Image Generation",
            description="Generate images using your finetuned models."
        )
        self.inference_service = inference_service
        self.model_service = model_service
    
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
            negative_prompt: Things to avoid in generation
            aspect_ratio: Image aspect ratio (ultra endpoint)
            num_steps: Number of generation steps
            guidance_scale: Guidance scale for generation
            strength: Finetune strength for ultra endpoint (0.1-2.0)
            strength_standard: Finetune strength for standard endpoint (0.1-2.0)
            seed: Random seed for reproducibility
            width: Image width (standard endpoint)
            height: Image height (standard endpoint)
            image_prompt: Optional image array to use as a prompt
            output_format: Output image format
            prompt_upsampling: Whether to enhance prompt
            image_prompt_strength: Blend between prompt and image prompt (0-1)
            ultra_prompt_upsampling: Whether to enhance prompt for ultra endpoint
            safety_tolerance: Safety check level (0-6)
            
        Returns:
            Tuple of (numpy array of image or None, status message)
        """
        try:
            # Extract model ID from choice string
            model_id = self._get_model_id_from_choice(model_choice)
            if not model_id:
                return None, "Error: Invalid model selection"
                
            # Generate image
            return self.inference_service.generate_image(
                endpoint=endpoint,
                model_id=model_id,
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                strength_standard=strength_standard,
                seed=seed,
                width=width,
                height=height,
                image_prompt=image_prompt,
                output_format=output_format,
                prompt_upsampling=prompt_upsampling,
                image_prompt_strength=image_prompt_strength,
                ultra_prompt_upsampling=ultra_prompt_upsampling,
                safety_tolerance=safety_tolerance
            )
            
        except Exception as e:
            return None, f"Error generating image: {str(e)}"
    
    def _get_model_id_from_choice(self, choice: str) -> str:
        """
        Extract model ID from formatted choice string.
        
        Args:
            choice: Formatted choice string from dropdown
            
        Returns:
            Model ID or empty string if not found
        """
        if not choice or not isinstance(choice, str):
            return ""
            
        try:
            for model in self.model_service.list_models():
                if model and self.model_service.format_model_choice(model) == choice:
                    return model.finetune_id
            return ""
        except Exception as e:
            print(f"Error extracting model ID: {e}")
            return ""
    
    def create_ui(self) -> gr.Blocks:
        """
        Create the inference UI component.
        
        Returns:
            Gradio Blocks component
        """
        with gr.Blocks() as app:
            # Header
            title_text = self.title if self.title else "Image Generation"
            desc_text = self.description if self.description else "Generate images using your finetuned models."
            self.create_section_header(title_text, desc_text)
            
            # Important note
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
                            ("FLUX 1.1 [pro] ultra Finetune", self.ENDPOINT_ULTRA),
                            ("FLUX.1 [pro] Finetune", self.ENDPOINT_STANDARD),
                        ],
                        value=self.ENDPOINT_ULTRA,
                        label="Generation endpoint",
                        info="Select the generation endpoint to use.",
                    )
                    
                    # Model selection
                    model_choices = self.inference_service.get_model_choices()
                    
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            choices=model_choices,
                            value=model_choices[0] if model_choices else None,
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
                        sources=["upload", "clipboard"],
                        show_download_button=False,
                        height=200,
                        elem_id="image_prompt_upload",
                        interactive=False  # Temporarily disabled
                    )
                    gr.Markdown(
                        """
                        **Image Prompt**: Upload an image to use as a visual reference.
                        The model will blend this with your text prompt based on the
                        "Image prompt strength" slider.
                        
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
                self.model_service.refresh_models()
                choices = self.inference_service.get_model_choices()
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
            
        return app
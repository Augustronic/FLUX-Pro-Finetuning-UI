"""Main UI class that composes all components for FLUX Pro Finetuning UI."""

from typing import Optional
import gradio as gr

from ui.base import UIComponent
from ui.model_selection import ModelSelectionComponent
from ui.parameter_config import ParameterConfigComponent
from ui.image_generation import ImageGenerationComponent
from constants import Endpoints


class MainUI(UIComponent):
    """Main UI class that composes all components."""

    def __init__(self) -> None:
        """Initialize the main UI."""
        super().__init__()
        self.model_selection = ModelSelectionComponent()
        self.parameter_config = ParameterConfigComponent()
        self.image_generation = ImageGenerationComponent()

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the main UI interface.
        
        Args:
            parent: Optional parent Blocks instance
        """
        with gr.Blocks(title="FLUX [pro] Finetuning UI") as interface:
            with gr.Accordion(""):
                gr.Markdown(
                    """
                    <div style="text-align: center; margin: 0 auto; padding: 0 2rem;">
                        <h1 style="font-size: 2.5rem; font-weight: 600; margin: 1rem 0;
                            color: #72a914;">
                            FLUX [pro] Finetuning UI
                        </h1>
                        <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                            Train custom models, browse your collection and generate
                            images.
                        </p>
                    </div>
                    """
                )

            with gr.Tabs():
                with gr.Tab("Generate with Model"):
                    gr.Markdown(
                        """
                        <div style="text-align: center; margin: 1rem 0;">
                            <h2 style="font-size: 1.8rem; font-weight: 600;
                                color: #72a914;">Image Generation</h2>
                            <p>Generate images using your finetuned models.</p>
                        </div>
                        """
                    )

                    # Endpoint selection
                    endpoint = gr.Radio(
                        choices=[
                            ("FLUX 1.1 [pro] ultra Finetune", Endpoints.ULTRA),
                            ("FLUX.1 [pro] Finetune", Endpoints.STANDARD),
                        ],
                        value=Endpoints.ULTRA,
                        label="Generation endpoint",
                        info="Select the generation endpoint to use.",
                    )

                    # Create component UIs
                    self.model_selection.create()
                    self.parameter_config.create()
                    self.image_generation.create()

                    # Register endpoint element
                    self.register_element("endpoint", endpoint)

                    # Connect endpoint change to parameter visibility
                    endpoint.change(
                        fn=self.parameter_config.toggle_endpoint_params,
                        inputs=[endpoint],
                        outputs=[
                            self.parameter_config.ultra_params,
                            self.parameter_config.standard_params
                        ]
                    )

                    # Connect generate button to image generation
                    generate_inputs = [
                        endpoint,
                        self.model_selection.get_element("model_dropdown"),
                        self.parameter_config.get_element("prompt"),
                        self.parameter_config.get_element("negative_prompt"),
                        self.parameter_config.get_element("aspect_ratio"),
                        self.parameter_config.get_element("steps"),
                        self.parameter_config.get_element("guidance"),
                        self.parameter_config.get_element("strength"),
                        self.parameter_config.get_element("seed"),
                        self.parameter_config.get_element("output_format"),
                        self.parameter_config.get_element("prompt_upsampling"),
                        self.parameter_config.get_element("safety_tolerance"),
                        self.parameter_config.get_element("width"),
                        self.parameter_config.get_element("height"),
                        self.parameter_config.get_element("raw_mode"),
                    ]

                    self.image_generation.get_element("generate_button").click(
                        fn=self.image_generation.generate_image,
                        inputs=generate_inputs,
                        outputs=[
                            self.image_generation.get_element("output_image"),
                            self.image_generation.get_element("status_text"),
                        ],
                    )

        self.register_element("interface", interface)
        return interface


def create_ui() -> gr.Blocks:
    """Create and return the main UI instance.
    
    Returns:
        The Gradio Blocks interface
    """
    ui = MainUI()
    return ui.create()
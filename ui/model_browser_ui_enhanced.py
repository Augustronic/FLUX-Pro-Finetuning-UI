"""
Enhanced Model Browser UI Component for FLUX-Pro-Finetuning-UI.

Provides UI for browsing and managing fine-tuned models.
Uses the new service interfaces and feature flags.
"""

import gradio as gr
import logging
from typing import Dict, Any, List, Optional, Tuple

from ui.base import BaseUI
from services.business.model_service import ModelService
from services.core.feature_flag_service import FeatureFlagService


class ModelBrowserUI(BaseUI):
    """
    Enhanced UI component for browsing and managing fine-tuned models.

    Uses the new service interfaces and feature flags.
    """

    def __init__(
        self,
        model_service: ModelService,
        feature_flag_service: FeatureFlagService
    ):
        """
        Initialize the model browser UI component.

        Args:
            model_service: Service for model operations
            feature_flag_service: Service for feature flags
        """
        super().__init__(
            title="Model Browser",
            description="View and manage your fine-tuned models."
        )
        self.model_service = model_service
        self.feature_flags = feature_flag_service
        self.logger = logging.getLogger(__name__)

    def create_ui(self) -> gr.Blocks:
        """
        Create the model browser UI component.

        Returns:
            Gradio Blocks component
        """
        with gr.Blocks() as app:
            # Header
            title_text = self.title if self.title else "Model Browser"
            desc_text = self.description if self.description else "View and manage your fine-tuned models."
            self.create_section_header(title_text, desc_text)

            # Model list and details
            with gr.Row():
                with gr.Column(scale=1):
                    # Model list
                    refresh_btn = gr.Button("🔄 Refresh Models")
                    model_list = gr.Dropdown(
                        choices=self._get_model_choices(),
                        label="Select Model",
                        info="Select a model to view details"
                    )

                    # Model actions
                    with gr.Row():
                        view_btn = gr.Button("👁️ View Details")

                        # Show delete button only if feature is enabled
                        if self.feature_flags.is_enabled("model_deletion", False):
                            delete_btn = gr.Button("🗑️ Delete Model", variant="stop")

                    # Show export button only if feature is enabled
                    if self.feature_flags.is_enabled("model_export", False):
                        export_btn = gr.Button("📤 Export Model")

                with gr.Column(scale=2):
                    # Model details
                    with gr.Group():
                        gr.Markdown("### Model Details")
                        model_name = gr.Textbox(label="Model Name", interactive=False)
                        trigger_word = gr.Textbox(label="Trigger Word", interactive=False)
                        model_type = gr.Textbox(label="Type", interactive=False)
                        model_mode = gr.Textbox(label="Mode", interactive=False)

                        # Show advanced details only if feature is enabled
                        if self.feature_flags.is_enabled("advanced_model_details", True):
                            with gr.Accordion("Advanced Details", open=False):
                                model_id = gr.Textbox(label="Model ID", interactive=False)
                                rank = gr.Number(label="Rank", interactive=False)
                                iterations = gr.Number(label="Iterations", interactive=False)
                                learning_rate = gr.Number(label="Learning Rate", interactive=False)
                                timestamp = gr.Textbox(label="Created", interactive=False)

                    # Status message
                    status_text = gr.Textbox(label="Status", interactive=False)

                    # Show sample images only if feature is enabled
                    if self.feature_flags.is_enabled("model_samples", False):
                        with gr.Group():
                            gr.Markdown("### Sample Images")
                            sample_gallery = gr.Gallery(
                                label="Sample Images",
                                show_label=False,
                                columns=3,
                                height=300
                            )
                            generate_samples_btn = gr.Button("🖼️ Generate Samples")

            # Event handlers

            # Refresh models
            def refresh_models():
                self.logger.info("Refreshing models")
                count = self.model_service.refresh_models()
                choices = self._get_model_choices()
                return gr.update(choices=choices), f"Refreshed {count} models"

            refresh_btn.click(
                fn=refresh_models,
                inputs=[],
                outputs=[model_list, status_text]
            )

            # View model details
            def view_model_details(model_choice):
                self.logger.info(f"Viewing details for model: {model_choice}")

                if not model_choice:
                    return [gr.update() for _ in range(9)], "Please select a model"

                # Get model ID from choice
                model_id = self._get_model_id_from_choice(model_choice)
                if not model_id:
                    return [gr.update() for _ in range(9)], "Invalid model selection"

                # Get model details
                model = self.model_service.get_model(model_id)
                if not model:
                    return [gr.update() for _ in range(9)], "Model not found"

                # Update UI with model details
                return [
                    model.model_name,
                    model.trigger_word,
                    model.type,
                    model.mode,
                    model.finetune_id,
                    model.rank if model.rank is not None else 0,
                    model.iterations if model.iterations is not None else 0,
                    model.learning_rate if model.learning_rate is not None else 0,
                    model.timestamp if model.timestamp is not None else "",
                    f"Loaded details for {model.model_name}"
                ]

            view_btn.click(
                fn=view_model_details,
                inputs=[model_list],
                outputs=[
                    model_name, trigger_word, model_type, model_mode,
                    model_id, rank, iterations, learning_rate, timestamp,
                    status_text
                ]
            )

            # Delete model if feature is enabled
            if self.feature_flags.is_enabled("model_deletion", False):
                def delete_model(model_choice):
                    self.logger.info(f"Deleting model: {model_choice}")

                    if not model_choice:
                        return gr.update(), "Please select a model"

                    # Get model ID from choice
                    model_id = self._get_model_id_from_choice(model_choice)
                    if not model_id:
                        return gr.update(), "Invalid model selection"

                    # Delete model
                    success = self.model_service.delete_model(model_id)
                    if not success:
                        return gr.update(), "Failed to delete model"

                    # Refresh model list
                    choices = self._get_model_choices()
                    return gr.update(choices=choices, value=None), "Model deleted successfully"

                delete_btn.click(
                    fn=delete_model,
                    inputs=[model_list],
                    outputs=[model_list, status_text]
                )

            # Export model if feature is enabled
            if self.feature_flags.is_enabled("model_export", False):
                def export_model(model_choice):
                    self.logger.info(f"Exporting model: {model_choice}")

                    if not model_choice:
                        return "Please select a model"

                    # Get model ID from choice
                    model_id = self._get_model_id_from_choice(model_choice)
                    if not model_id:
                        return "Invalid model selection"

                    # Export model (placeholder)
                    return "Model export is not implemented yet"

                export_btn.click(
                    fn=export_model,
                    inputs=[model_list],
                    outputs=[status_text]
                )

            # Generate samples if feature is enabled
            if self.feature_flags.is_enabled("model_samples", False):
                def generate_samples(model_choice):
                    self.logger.info(f"Generating samples for model: {model_choice}")

                    if not model_choice:
                        return [], "Please select a model"

                    # Get model ID from choice
                    model_id = self._get_model_id_from_choice(model_choice)
                    if not model_id:
                        return [], "Invalid model selection"

                    # Generate samples (placeholder)
                    # In a real implementation, you would call a service to generate samples
                    return [], "Sample generation is not implemented yet"

                generate_samples_btn.click(
                    fn=generate_samples,
                    inputs=[model_list],
                    outputs=[sample_gallery, status_text]
                )

            # Update details when model selection changes
            model_list.change(
                fn=view_model_details,
                inputs=[model_list],
                outputs=[
                    model_name, trigger_word, model_type, model_mode,
                    model_id, rank, iterations, learning_rate, timestamp,
                    status_text
                ]
            )

        return app

    def _get_model_choices(self) -> List[str]:
        """
        Get formatted model choices for dropdown.

        Returns:
            List of formatted model choices
        """
        models = self.model_service.list_models()

        # Filter models based on feature flags
        filtered_models = []
        for model in models:
            if not model or not model.model_name or not model.trigger_word:
                continue

            # Filter by model type if experimental types are disabled
            if not self.feature_flags.is_enabled("experimental_model_types", True):
                if model.type not in ["full", "lora"]:
                    continue

            filtered_models.append(model)

        return [
            self.model_service.format_model_choice(model)
            for model in filtered_models
        ]

    def _get_model_id_from_choice(self, choice: str) -> Optional[str]:
        """
        Get model ID from formatted choice string.

        Args:
            choice: Formatted choice string from dropdown

        Returns:
            Model ID or None if not found
        """
        if not choice or not isinstance(choice, str):
            return None

        try:
            for model in self.model_service.list_models():
                if model and self.model_service.format_model_choice(model) == choice:
                    return model.finetune_id
            return None
        except Exception as e:
            self.logger.error(f"Error extracting model ID: {e}")
            return None

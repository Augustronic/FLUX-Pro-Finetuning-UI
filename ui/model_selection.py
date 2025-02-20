"""Model selection component for FLUX Pro Finetuning UI."""

import re
from typing import List, Any, Optional
import gradio as gr

from ui.base import UIComponent
from container import container
from constants import ValidationConfig
from model_manager import ModelMetadata


class ModelSelectionComponent(UIComponent):
    """Handles model selection and listing in the UI."""
    
    def __init__(self) -> None:
        """Initialize the model selection component."""
        super().__init__()
        self.manager = container.model_manager
    
    def _format_model_choice(self, model: ModelMetadata) -> str:
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
        
        # Only allow valid characters
        text = re.sub(r'[^' + ValidationConfig.VALID_FILENAME_CHARS[1:-1] + ']', '', text)
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
                    if re.match(ValidationConfig.VALID_ID_FORMAT, model.finetune_id):
                        return model.finetune_id
            return ""
        except Exception as e:
            print(f"Error extracting model ID: {e}")
            return ""

    def _get_model_choices(self, models: List[ModelMetadata]) -> List[str]:
        """Get formatted model choices from list of models.
        
        Args:
            models: List of model metadata objects
            
        Returns:
            List of formatted model choices
        """
        choices = []
        for model in models:
            try:
                if isinstance(model, ModelMetadata):
                    choice = self._format_model_choice(model)
                    choices.append(choice)
            except ValueError as e:
                print(f"Skipping invalid model: {e}")
                continue
        return choices

    def refresh_models(self) -> List[str]:
        """Refresh models from API and return updated choices.
        
        Returns:
            List of formatted model choices
        """
        self.manager.refresh_models()
        models = self.manager.list_models()
        return self._get_model_choices(models)

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the model selection UI elements.
        
        Args:
            parent: Optional parent Blocks instance
            
        Returns:
            The created Gradio Blocks interface
        """
        blocks = parent or gr.Blocks()
        with blocks:
            models = self.manager.list_models()
            if not models:
                print("No models found, refreshing from API...")
                self.manager.refresh_models()
                models = self.manager.list_models()

            # Create model choices list
            model_choices = self._get_model_choices(models)

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Select model",
                    info="Model trigger word shown in parentheses. Include in prompt.",
                )
                refresh_btn = gr.Button("🔄 Refresh models")

            # Register elements
            self.register_element("model_dropdown", model_dropdown)
            self.register_element("refresh_button", refresh_btn)

            # Event handlers
            refresh_btn.click(
                fn=self.refresh_models,
                inputs=[],
                outputs=[model_dropdown]
            )

        return blocks
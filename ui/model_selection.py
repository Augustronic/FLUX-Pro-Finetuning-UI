"""Model selection component for FLUX Pro Finetuning UI."""

import re
from typing import List, Any, Optional, Dict
import gradio as gr

from ui.base import UIComponent
from container import container
from constants import ValidationConfig
from model_manager import ModelMetadata
from utils.error_handling.error_handler import ErrorHandler, ErrorContext, ValidationError
from utils.logging.logger import get_logger
from utils.validation.validator import Validator, ValidationRule

class ModelSelectionComponent(UIComponent):
    """Handles model selection and listing in the UI."""
    
    def __init__(self) -> None:
        """Initialize the model selection component."""
        super().__init__()
        self.manager = container.model_manager
        self.logger = get_logger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        self.validator = Validator()
        
        # Define validation rules
        self.model_rules = [
            ValidationRule(
                field="model_name",
                rule_type="required",
                message="Model name is required"
            ),
            ValidationRule(
                field="model_name",
                rule_type="pattern",
                value=r"^[a-zA-Z0-9_-]+$",
                message="Model name can only contain letters, numbers, underscores and hyphens"
            ),
            ValidationRule(
                field="trigger_word",
                rule_type="required",
                message="Trigger word is required"
            ),
            ValidationRule(
                field="type",
                rule_type="enum",
                value=["lora", "textual_inversion", "hypernetwork"],
                message="Invalid model type"
            ),
            ValidationRule(
                field="mode",
                rule_type="enum",
                value=["training", "inference"],
                message="Invalid model mode"
            )
        ]
    
    def _format_model_choice(self, model: ModelMetadata) -> str:
        """Format model metadata for dropdown display."""
        try:
            # Validate model data
            self.validator.validate(
                {
                    "model_name": getattr(model, "model_name", None),
                    "trigger_word": getattr(model, "trigger_word", None),
                    "type": getattr(model, "type", None),
                    "mode": getattr(model, "mode", None)
                },
                self.model_rules,
                "ModelSelectionComponent"
            )
            
            # Format display values with sanitization
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
            
        except Exception as e:
            self.logger.error(
                "Error formatting model choice",
                extra={"error": str(e), "model": str(model)}
            )
            raise ValidationError(
                "Invalid model data",
                context=ErrorContext(
                    component="ModelSelectionComponent",
                    operation="_format_model_choice",
                    details={"model": str(model)}
                )
            )

    def _sanitize_display_text(self, text: str) -> str:
        """Sanitize text for display in UI."""
        try:
            if not isinstance(text, str):
                raise ValidationError(
                    "Invalid text type",
                    context=ErrorContext(
                        component="ModelSelectionComponent",
                        operation="_sanitize_display_text",
                        details={"text": str(text)}
                    )
                )
                
            # Remove control characters and limit length
            text = "".join(char for char in text if char.isprintable())
            text = text[:100]  # Limit length
            
            # Only allow valid characters
            text = re.sub(r'[^' + ValidationConfig.VALID_FILENAME_CHARS[1:-1] + ']', '', text)
            return text.strip()
            
        except Exception as e:
            self.logger.error(
                "Error sanitizing display text",
                extra={"error": str(e), "text": str(text)}
            )
            return ""

    def _get_model_id_from_choice(self, choice: str) -> str:
        """Extract model ID from formatted choice string."""
        try:
            if not isinstance(choice, str) or not choice.strip():
                raise ValidationError(
                    "Invalid choice format",
                    context=ErrorContext(
                        component="ModelSelectionComponent",
                        operation="_get_model_id_from_choice"
                    )
                )
                
            for model in self.manager.list_models():
                if model and self._format_model_choice(model) == choice:
                    # Validate ID format
                    if re.match(ValidationConfig.VALID_ID_FORMAT, model.finetune_id):
                        return model.finetune_id
                        
            raise ValidationError(
                "Model not found",
                context=ErrorContext(
                    component="ModelSelectionComponent",
                    operation="_get_model_id_from_choice",
                    details={"choice": choice}
                )
            )
            
        except Exception as e:
            self.logger.error(
                "Error extracting model ID",
                extra={"error": str(e), "choice": choice}
            )
            return ""

    def _get_model_choices(self, models: List[ModelMetadata]) -> List[str]:
        """Get formatted model choices from list of models."""
        choices = []
        for model in models:
            try:
                if isinstance(model, ModelMetadata):
                    choice = self._format_model_choice(model)
                    choices.append(choice)
            except Exception as e:
                self.logger.warning(
                    "Skipping invalid model",
                    extra={"error": str(e), "model": str(model)}
                )
                continue
        return choices

    def refresh_models(self) -> Dict[str, Any]:
        """Refresh models from API and return updated dropdown state."""
        try:
            self.logger.info("Refreshing models from API")
            self.manager.refresh_models()
            models = self.manager.list_models()
            choices = self._get_model_choices(models)
            
            self.logger.info(
                "Models refreshed successfully",
                extra={"model_count": len(choices)}
            )
            
            return gr.update(
                choices=choices,
                value=choices[0] if choices else None
            )
            
        except Exception as e:
            error_response = self.error_handler.handle_error(
                e,
                context=ErrorContext(
                    component="ModelSelectionComponent",
                    operation="refresh_models"
                )
            )
            self.logger.error(
                "Failed to refresh models",
                extra={"error": error_response}
            )
            return gr.update(choices=[], value=None)

    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the model selection UI elements."""
        blocks = parent or gr.Blocks()
        
        try:
            with blocks:
                # Initialize models
                self.logger.info("Initializing model selection component")
                models = self.manager.list_models()
                
                if not models:
                    self.logger.info("No models found, refreshing from API")
                    self.manager.refresh_models()
                    models = self.manager.list_models()

                # Create model choices list
                model_choices = self._get_model_choices(models)
                self.logger.info(
                    "Model choices created",
                    extra={"choice_count": len(model_choices)}
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=model_choices,
                        value=None,  # Start with no selection
                        label="Select model",
                        info="Model trigger word shown in parentheses. Include in prompt."
                    )
                    refresh_btn = gr.Button("🔄 Refresh models")

                # Register elements
                self.register_element("model_dropdown", model_dropdown)
                self.register_element("refresh_button", refresh_btn)

                # Event handlers
                def update_dropdown():
                    return self.refresh_models()
                    
                refresh_btn.click(
                    fn=update_dropdown,
                    inputs=[],
                    outputs=[model_dropdown]
                )

            return blocks
            
        except Exception as e:
            error_response = self.error_handler.handle_error(
                e,
                context=ErrorContext(
                    component="ModelSelectionComponent",
                    operation="create"
                )
            )
            self.logger.error(
                "Failed to create model selection component",
                extra={"error": error_response}
            )
            raise
"""
Model Browser UI Component for FLUX-Pro-Finetuning-UI.

Provides UI for browsing and managing fine-tuned models.
"""

import gradio as gr
import json
from typing import Dict, Any, Optional, List, Tuple

from ui.base import BaseUI
from services.business.model_service import ModelService


class ModelBrowserUI(BaseUI):
    """
    UI component for browsing and managing fine-tuned models.
    
    Provides UI for browsing and managing fine-tuned models, including
    listing, viewing details, and refreshing from API.
    """
    
    def __init__(self, model_service: ModelService):
        """
        Initialize the model browser UI component.
        
        Args:
            model_service: Service for model management
        """
        super().__init__(
            title="Model Browser",
            description="View and manage your finetuned models."
        )
        self.model_service = model_service
    
    def get_models_df(self) -> List[List[str]]:
        """
        Get all models formatted as a dataframe.
        
        Returns:
            List of model data rows
        """
        models = self.model_service.list_models()
        return [self._format_model_info(model) for model in models]
    
    def _format_model_info(self, model) -> List[str]:
        """
        Format model information for the dataframe.
        
        Args:
            model: Model metadata
            
        Returns:
            List of formatted model data
        """
        return [
            model.model_name,
            model.finetune_id,
            model.trigger_word,
            model.type.upper(),
            model.mode.capitalize(),
            str(model.rank) if model.rank else "N/A",
            str(model.iterations) if model.iterations else "N/A",
            str(model.learning_rate) if model.learning_rate else "N/A",
            model.priority.capitalize() if model.priority else "N/A",
            model.timestamp if model.timestamp else "N/A"
        ]
    
    def refresh_models(self) -> Tuple[List[List[str]], str]:
        """
        Refresh models from API and update display.
        
        Returns:
            Tuple of (model data, status message)
        """
        updated_count = self.model_service.refresh_models()
        models_data = self.get_models_df()
        
        if updated_count > 0:
            msg = f"Models refreshed successfully! Updated {updated_count} models."
        else:
            msg = "No models found or error refreshing."
            
        return models_data, msg
    
    def create_ui(self) -> gr.Blocks:
        """
        Create the model browser UI component.
        
        Returns:
            Gradio Blocks component
        """
        with gr.Blocks() as app:
            # Header
            title_text = self.title if self.title else "Model Browser"
            desc_text = self.description if self.description else "View and manage your finetuned models."
            self.create_section_header(title_text, desc_text)
            
            # Model list
            columns = [
                "Model name",
                "Finetune ID",
                "Trigger word",
                "Type",
                "Mode",
                "Rank",
                "Iterations",
                "Learning rate",
                "Priority",
                "Timestamp"
            ]
            
            with gr.Row():
                with gr.Column(scale=4):
                    model_table = gr.Dataframe(
                        headers=columns,
                        datatype="str",
                        value=self.get_models_df(),
                        label="Click Refresh to fetch the latest models. ➡️",
                        interactive=False,
                        wrap=False
                    )
                    
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh models")
                    status = gr.Textbox(label="Status", interactive=False)
                    
                    # Quick copy section
                    gr.Markdown("### Quick copy")
                    selected_id = gr.Textbox(
                        label="Selected model ID",
                        interactive=False
                    )
                    selected_trigger = gr.Textbox(
                        label="Trigger word",
                        interactive=False
                    )
                    
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### Model details")
                    selected_model = gr.JSON(
                        label="Selected model metadata",
                        value={}
                    )
                with gr.Column(scale=1):
                    gr.Markdown("")
                    
            # Handle refresh
            refresh_btn.click(
                fn=self.refresh_models,
                inputs=[],
                outputs=[model_table, status]
            )
            
            # Update details when model is selected
            def update_selection(evt: gr.SelectData, data) -> Tuple[str, str, str]:
                try:
                    # Get the selected row using iloc for pandas DataFrame
                    row = data.iloc[evt.index[0]].tolist()
                    model_info = {
                        "Model name": row[0],
                        "Finetune ID": row[1],
                        "Trigger word": row[2],
                        "Type": row[3],
                        "Mode": row[4],
                        "Rank": row[5],
                        "Iterations": row[6],
                        "Learning rate": row[7],
                        "Priority": row[8],
                        "Timestamp": row[9]
                    }
                    return (
                        json.dumps(model_info, indent=2),
                        row[1],  # Finetune ID
                        row[2]   # Trigger word
                    )
                except Exception as e:
                    print(f"Error updating selection: {e}")
                    return (
                        "{}",
                        "",
                        ""
                    )
                    
            model_table.select(
                fn=update_selection,
                inputs=[model_table],
                outputs=[selected_model, selected_id, selected_trigger]
            )
            
        return app
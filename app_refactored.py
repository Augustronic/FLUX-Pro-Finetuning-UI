"""
Refactored Main Application for FLUX-Pro-Finetuning-UI.

Uses the new service container with lazy loading for improved performance and modularity.
"""

import os
import logging
import gradio as gr
from gradio.themes import Default
from typing import Dict, Any, List, Optional

from container_lazy import ServiceContainer


def setup_logging() -> None:
    """
    Set up logging configuration.
    
    Returns:
        None
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "app.log")),
            logging.StreamHandler()
        ]
    )


def create_app() -> gr.Blocks:
    """
    Create the main application with all UI components.
    
    Returns:
        Gradio Blocks application
    """
    # Initialize service container
    container = ServiceContainer()
    
    # Get UI components
    finetune_ui = container.get_service('finetune_ui')
    model_browser_ui = container.get_service('model_browser_ui')
    inference_ui = container.get_service('inference_ui')
    
    # Create main application
    with gr.Blocks(
        title="FLUX Pro Finetuning UI",
        theme=Default(
            primary_hue="green",
            secondary_hue="gray",
            neutral_hue="gray",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
        )
    ) as app:
        # Header
        with gr.Row():
            gr.Markdown(
                """
                # FLUX Pro Finetuning UI
                
                A user interface for fine-tuning and using FLUX Pro models.
                """
            )
            
        # Main tabs
        with gr.Tabs() as tabs:
            with gr.TabItem("Generate Images"):
                inference_ui.create_ui()
                
            with gr.TabItem("Browse Models"):
                model_browser_ui.create_ui()
                
            with gr.TabItem("Finetune Models"):
                finetune_ui.create_ui()
                
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## About FLUX Pro Finetuning UI
                    
                    This application provides a user interface for fine-tuning and using FLUX Pro models.
                    
                    ### Features
                    
                    - Generate images using fine-tuned models
                    - Browse and manage fine-tuned models
                    - Fine-tune new models with custom datasets
                    
                    ### Version
                    
                    Version: 1.0.0
                    
                    ### Credits
                    
                    Developed by the FLUX Pro team.
                    """
                )
                
    return app


def main() -> None:
    """
    Main entry point for the application.
    
    Returns:
        None
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting FLUX Pro Finetuning UI")
    
    # Create and launch app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_api=False
    )
    
    logger.info("FLUX Pro Finetuning UI stopped")


if __name__ == "__main__":
    main()
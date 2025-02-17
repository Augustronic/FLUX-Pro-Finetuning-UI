import gradio as gr
from finetune_ui import FineTuneUI
from image_generation_ui import ImageGenerationUI
from model_browser_ui import ModelBrowserUI
from model_manager import ModelManager
from config_manager import ConfigManager


def create_app():
    # Load configuration
    config = ConfigManager()
    config.load_config()

    # Initialize components with config
    model_manager = ModelManager(api_key=config.get_api_key())

    finetune_ui = FineTuneUI()
    image_gen_ui = ImageGenerationUI(model_manager)
    model_browser_ui = ModelBrowserUI(model_manager)

    # Create the combined interface
    with gr.Blocks(title="FLUX Pro Finetuning UI") as app:
        gr.Markdown(
            """
        <div style="text-align: center; margin: 0 auto; padding: 0 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 600; margin: 1rem 0; color: #72a914;">FLUX Pro Finetuning UI</h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">Train custom models, browse your collection and generate images.</p>
            <p style="text-align: right; font-size: 0.8em; color: #666;">Developed by Paresh Ranaut</p>
        </div>
        """
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Finetune Model"):
                gr.Markdown(
                    """
                <div style="text-align: center; padding: 0rem 1rem 2rem;">
                    <h2 style="font-size: 1.8rem; font-weight: 600; color: #72a914;">Model Finetuning</h2>
                    <p>Upload your training dataset and configure finetuning parameters.</p>
                </div>
                """
                 )
                finetune_interface = finetune_ui.create_ui()

            with gr.Tab("Model Browser"):
                gr.Markdown(
                    """
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="font-size: 1.8rem; font-weight: 600; color: #72a914;">Model Browser</h2>
                    <p>View and manage your finetuned models.</p>
                </div>
                """
                )
                browser_interface = model_browser_ui.create_ui()

            with gr.Tab("Generate with Model"):
                gr.Markdown(
                    """
                <div style="text-align: center; margin: 1rem 0;">
                    <h2 style="font-size: 1.8rem; font-weight: 600; color: #72a914;">Image Generation</h2>
                    <p>Generate images using your finetuned models.</p>
                </div>
                """
                )
                generation_interface = image_gen_ui.create_ui()

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)

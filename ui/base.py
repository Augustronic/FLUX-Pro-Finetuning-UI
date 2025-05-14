"""
Base UI Component for FLUX-Pro-Finetuning-UI.

Provides common UI functionality and styling for all UI components.
"""

import gradio as gr
from gradio.themes import Default
from typing import Optional, Dict, Any, List, Tuple


class BaseUI:
    """
    Base class for UI components.

    Provides common UI functionality and styling for all UI components.
    """

    def __init__(self, title: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize the base UI component.

        Args:
            title: Title for the UI component
            description: Description for the UI component
        """
        self.title = title
        self.description = description
        self.theme = self._setup_theme()

    def _setup_theme(self) -> gr.Theme:
        """
        Set up consistent theme and styling.

        Returns:
            Gradio theme
        """
        return Default(
            primary_hue="green",
            secondary_hue="gray",
            neutral_hue="gray",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"]
        )

    def create_header(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        align: str = "center"
    ) -> gr.Markdown:
        """
        Create standardized header component.

        Args:
            title: Title for the header
            description: Description for the header
            align: Text alignment (center, left, right)

        Returns:
            Markdown component with header
        """
        title_text = title or self.title or ""
        desc_text = description or self.description or ""

        header_html = f"""
        <div style="text-align: {align}; margin: 0 auto; padding: 0 2rem;">
            <h1 style="font-size: 2.5rem; font-weight: 600; margin: 1rem 0; color: #72a914;">
                {title_text}
            </h1>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                {desc_text}
            </p>
        </div>
        """

        return gr.Markdown(header_html)

    def create_section_header(
        self,
        title: str,
        description: Optional[str] = None,
        align: str = "center"
    ) -> gr.Markdown:
        """
        Create standardized section header component.

        Args:
            title: Title for the section
            description: Description for the section
            align: Text alignment (center, left, right)

        Returns:
            Markdown component with section header
        """
        desc_html = f"<p>{description}</p>" if description else ""

        header_html = f"""
        <div style="text-align: {align}; padding: 0rem 1rem 1rem;">
            <h2 style="font-size: 1.8rem; font-weight: 600; color: #72a914;">
                {title}
            </h2>
            {desc_html}
        </div>
        """

        return gr.Markdown(header_html)

    def create_info_box(self, content: str, type: str = "info") -> gr.Markdown:
        """
        Create an information box.

        Args:
            content: Content for the info box
            type: Type of info box (info, warning, error, success)

        Returns:
            Markdown component with info box
        """
        colors = {
            "info": "#3498db",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "success": "#2ecc71"
        }

        color = colors.get(type, colors["info"])

        info_html = f"""
        <div style="
            background-color: {color}15;
            border-left: 4px solid {color};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.25rem;
        ">
            {content}
        </div>
        """

        return gr.Markdown(info_html)

    def create_ui(self) -> gr.Blocks:
        """
        Create the UI component.

        This method should be implemented by subclasses.

        Returns:
            Gradio Blocks component
        """
        raise NotImplementedError("Subclasses must implement create_ui()")

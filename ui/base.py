"""Base UI component class for FLUX Pro Finetuning UI."""

from typing import Any, Dict, Optional
import gradio as gr
from abc import ABC, abstractmethod


class UIComponent(ABC):
    """Abstract base class for UI components."""
    
    def __init__(self) -> None:
        """Initialize the UI component."""
        self._elements: Dict[str, Any] = {}
    
    @abstractmethod
    def create(self, parent: Optional[gr.Blocks] = None) -> gr.Blocks:
        """Create the UI elements.
        
        Args:
            parent: Optional parent Blocks instance
            
        Returns:
            The created Gradio Blocks interface
        """
        pass
    
    def get_element(self, name: str) -> Any:
        """Get a UI element by name.
        
        Args:
            name: Name of the element
            
        Returns:
            The UI element if found, None otherwise
        """
        return self._elements.get(name)
    
    def register_element(self, name: str, element: Any) -> None:
        """Register a UI element.
        
        Args:
            name: Name to register the element under
            element: The UI element to register
        """
        self._elements[name] = element
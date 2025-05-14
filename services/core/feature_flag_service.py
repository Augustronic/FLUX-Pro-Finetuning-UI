"""
Feature Flag Service for FLUX-Pro-Finetuning-UI.

Provides functionality for managing feature flags, allowing conditional
enabling/disabling of features.
"""

from typing import Dict, Any, Optional
import logging


class FeatureFlagService:
    """
    Service for managing feature flags.

    Provides methods for checking if features are enabled or disabled.
    """

    def __init__(self, config_service):
        """
        Initialize the feature flag service.

        Args:
            config_service: Configuration service
        """
        self.config = config_service
        self.logger = logging.getLogger(__name__)

        # Load feature flags from config
        self.feature_flags = self.config.get_value('features', {})
        self.logger.info(f"Loaded {len(self.feature_flags)} feature flags")

        # Set default feature flags if not present in config
        self._set_default_flags()

    def _set_default_flags(self) -> None:
        """Set default feature flags if not present in config."""
        default_flags = {
            # UI features
            'advanced_parameters': True,
            'image_prompt_support': False,  # Temporarily disabled
            'model_details_view': True,

            # Generation features
            'prompt_upsampling': True,
            'ultra_prompt_upsampling': True,
            'safety_filters': True,

            # Experimental features
            'batch_generation': False,
            'animation_generation': False,
            'model_comparison': False
        }

        # Add default flags if not present
        for flag, default_value in default_flags.items():
            if flag not in self.feature_flags:
                self.feature_flags[flag] = default_value

    def is_enabled(self, feature_name: str, default: bool = False) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature_name: Name of the feature to check
            default: Default value if feature is not defined

        Returns:
            True if feature is enabled, False otherwise
        """
        return self.feature_flags.get(feature_name, default)

    def get_enabled_features(self) -> Dict[str, bool]:
        """
        Get all enabled features.

        Returns:
            Dictionary of enabled features
        """
        return {
            name: enabled
            for name, enabled in self.feature_flags.items()
            if enabled
        }

    def get_disabled_features(self) -> Dict[str, bool]:
        """
        Get all disabled features.

        Returns:
            Dictionary of disabled features
        """
        return {
            name: enabled
            for name, enabled in self.feature_flags.items()
            if not enabled
        }

    def set_feature_enabled(self, feature_name: str, enabled: bool) -> None:
        """
        Set a feature's enabled status.

        Args:
            feature_name: Name of the feature to set
            enabled: Whether the feature should be enabled
        """
        self.feature_flags[feature_name] = enabled
        self.logger.info(f"Feature '{feature_name}' set to {enabled}")

    def toggle_feature(self, feature_name: str) -> bool:
        """
        Toggle a feature's enabled status.

        Args:
            feature_name: Name of the feature to toggle

        Returns:
            New enabled status
        """
        current_status = self.is_enabled(feature_name)
        new_status = not current_status
        self.set_feature_enabled(feature_name, new_status)
        return new_status

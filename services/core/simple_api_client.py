"""
Simple API Client for FLUX-Pro-Finetuning-UI.

Provides a simplified interface for API communication with minimal dependencies.
"""

import requests
from typing import Dict, Any, Optional, Union, List
import logging
import json


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            response: API response
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class SimpleAPIClient:
    """
    Simplified API client with minimal dependencies.

    Provides basic API communication functionality without complex dependencies.
    """

    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai"):
        """
        Initialize the simple API client.

        Args:
            api_key: API key for authentication
            host: API host
        """
        self.api_key = api_key
        self.host = host
        self.base_url = f"https://{host}"
        self.logger = logging.getLogger(__name__)

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Make a request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            json_data: JSON data for POST requests
            timeout: Request timeout in seconds

        Returns:
            API response as a dictionary

        Raises:
            APIError: If the request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            self.logger.debug(f"Making {method} request to {url}")

            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=timeout
            )

            # Check for errors
            if response.status_code >= 400:
                error_message = f"API request failed with status {response.status_code}"
                error_response = None

                try:
                    error_response = response.json()
                    if "error" in error_response:
                        error_message = f"{error_message}: {error_response['error']}"
                except (ValueError, json.JSONDecodeError):
                    pass

                raise APIError(
                    message=error_message,
                    status_code=response.status_code,
                    response=error_response
                )

            # Parse response
            if response.content:
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError) as e:
                    self.logger.warning(f"Failed to parse JSON response: {e}")
                    return {"raw_content": response.text}

            return {}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise APIError(f"API request failed: {e}")

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            API response as a dictionary
        """
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API.

        Args:
            endpoint: API endpoint
            data: JSON data

        Returns:
            API response as a dictionary
        """
        return self.request("POST", endpoint, json_data=data)

    # Simplified API methods for common operations

    def list_finetunes(self) -> List[str]:
        """
        List all finetunes.

        Returns:
            List of finetune IDs
        """
        try:
            response = self.get("finetunes/list")
            return response.get("finetunes", [])
        except APIError as e:
            self.logger.error(f"Failed to list finetunes: {e}")
            return []

    def get_model_details(self, finetune_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific model.

        Args:
            finetune_id: Finetune ID

        Returns:
            Model details or None if not found
        """
        try:
            return self.get(f"finetunes/{finetune_id}")
        except APIError as e:
            self.logger.error(f"Failed to get model details: {e}")
            return None

    def start_finetune(self, params: Dict[str, Any]) -> Optional[str]:
        """
        Start a finetuning job.

        Args:
            params: Finetuning parameters

        Returns:
            Finetune ID or None if failed
        """
        try:
            response = self.post("finetunes/start", params)
            return response.get("finetune_id")
        except APIError as e:
            self.logger.error(f"Failed to start finetune: {e}")
            return None

    def generate_image(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an image using specified endpoint.

        Args:
            endpoint: Generation endpoint
            params: Generation parameters

        Returns:
            Generation result or None if failed
        """
        try:
            return self.post(f"generate/{endpoint}", params)
        except APIError as e:
            self.logger.error(f"Failed to generate image: {e}")
            return None

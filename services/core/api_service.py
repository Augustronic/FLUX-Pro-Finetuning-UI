"""
API Service for FLUX-Pro-Finetuning-UI.

Provides a unified interface for all API communications with consistent
error handling, response parsing, and authentication.
"""

import json
import requests
import base64
from typing import Dict, Any, Optional, Union, List
import time
import logging


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code
            response: Raw API response
        """
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class APIService:
    """
    Service for handling all API communications.

    Provides methods for making API requests with consistent error handling,
    authentication, and response parsing.
    """

    def __init__(self, config_service):
        """
        Initialize the API service.

        Args:
            config_service: Configuration service for retrieving API settings
        """
        self.config = config_service
        self.api_key = config_service.get_api_key()
        self.host = config_service.get_api_host()
        self.logger = logging.getLogger(__name__)

        # Validate API key
        if not self.api_key or not isinstance(self.api_key, str) or len(self.api_key.strip()) == 0:
            raise ValueError("Invalid API key. Please check your configuration.")

    def _get_headers(self, include_content_type: bool = True) -> Dict[str, str]:
        """
        Get request headers with authentication.

        Args:
            include_content_type: Whether to include Content-Type header

        Returns:
            Dictionary of headers
        """
        headers = {"X-Key": self.api_key}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _build_url(self, endpoint: str) -> str:
        """
        Build full URL for API endpoint.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL
        """
        # Ensure endpoint starts with a slash if not already
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        return f"https://{self.host}{endpoint}"

    def request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        data: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        retries: int = 3,
        retry_delay: int = 1
    ) -> Dict[str, Any]:
        """
        Make an API request with error handling and retries.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Parsed API response

        Raises:
            APIError: If the API request fails
        """
        url = self._build_url(endpoint)
        headers = self._get_headers()

        self.logger.debug(f"Making {method} request to {url}")
        if params:
            self.logger.debug(f"Query params: {params}")
        if data:
            self.logger.debug(f"Request data: {json.dumps(data, indent=2)}")

        attempt = 0
        last_error = None

        while attempt < retries:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data,
                    timeout=timeout
                )

                # Check for HTTP errors
                response.raise_for_status()

                # Parse response
                try:
                    result = response.json()
                    self.logger.debug(f"API response: {json.dumps(result, indent=2)}")
                    return result
                except json.JSONDecodeError:
                    raise APIError(
                        f"Invalid JSON response: {response.text}",
                        response.status_code
                    )

            except requests.exceptions.RequestException as e:
                last_error = e
                self.logger.warning(f"API request failed (attempt {attempt+1}/{retries}): {str(e)}")

                # Check if we should retry
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    attempt += 1
                else:
                    # Extract status code and response if available
                    status_code = None
                    response_data = None

                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        try:
                            response_data = e.response.json()
                        except (json.JSONDecodeError, AttributeError):
                            response_data = {"raw_text": e.response.text}

                    raise APIError(
                        f"API request failed after {retries} attempts: {str(e)}",
                        status_code,
                        response_data
                    ) from e

        # This should never be reached due to the retry logic above,
        # but we include it to satisfy the type checker
        if last_error:
            raise APIError(
                f"API request failed: {str(last_error)}",
                None,
                None
            ) from last_error
        return {}

    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Parsed API response
        """
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a POST request to the API.

        Args:
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Parsed API response
        """
        return self.request("POST", endpoint, params=params, data=data)

    def encode_file(self, file_path: str) -> str:
        """
        Encode file to base64.

        Args:
            file_path: Path to the file

        Returns:
            Base64-encoded file content

        Raises:
            FileNotFoundError: If the file does not exist
        """
        with open(file_path, 'rb') as f:
            file_data = f.read()
            return base64.b64encode(file_data).decode('utf-8')

    # API-specific methods

    def get_model_details(self, finetune_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific model.

        Args:
            finetune_id: ID of the fine-tuned model

        Returns:
            Model details or None if not found
        """
        try:
            response = self.get("/v1/finetune_details", {"finetune_id": finetune_id})
            if response and "finetune_details" in response:
                return response["finetune_details"]
            return None
        except APIError as e:
            self.logger.error(f"Error getting model details: {e}")
            return None

    def list_finetunes(self) -> List[str]:
        """
        List all finetunes.

        Returns:
            List of finetune IDs
        """
        try:
            response = self.get("/v1/my_finetunes")
            if response and "finetunes" in response:
                return response["finetunes"]
            return []
        except APIError as e:
            self.logger.error(f"Error listing finetunes: {e}")
            return []

    def start_finetune(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a finetuning job.

        Args:
            params: Finetuning parameters

        Returns:
            API response with finetune_id
        """
        return self.post("/v1/finetune", params)

    def get_generation_status(self, inference_id: str) -> Dict[str, Any]:
        """
        Check status of a generation task.

        Args:
            inference_id: ID of the generation task

        Returns:
            Generation status
        """
        return self.get("/v1/get_result", {"id": inference_id})

    def generate_image(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an image using specified endpoint.

        Args:
            endpoint: Generation endpoint (e.g., "flux-pro-1.1-ultra")
            params: Generation parameters

        Returns:
            API response with generation task ID
        """
        return self.post(f"/v1/{endpoint}", params)

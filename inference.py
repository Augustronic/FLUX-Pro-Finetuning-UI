import http.client
import json
import os
import time
from typing import Optional, Dict, Any


class ConfigurationError(Exception):
    """Raised when there are configuration issues."""
    pass


def get_api_key() -> str:
    """Get API key from environment variable."""
    api_key = os.getenv("BFL_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "API key not found. Set BFL_API_KEY environment variable."
        )
    return api_key


class ImageGenerator:
    """Handles image generation requests to the BFL API."""

    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai") -> None:
        self.api_key = api_key
        self.host = host

    def _format_endpoint(self, endpoint: str) -> str:
        """Format endpoint string for URL path.

        Args:
            endpoint: Raw endpoint string

        Returns:
            Formatted endpoint string
        """
        # Use endpoint as-is, no formatting needed
        return endpoint.strip()

    def request_inference(
        self,
        finetune_id: str,
        prompt: str,
        finetune_strength: float = 1.2,
        endpoint: str = "flux-pro-1.1-ultra-finetuned",
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        output_format: str = "jpeg",
        scheduler: str = "DPM++ 2M Karras"
    ) -> dict:
        """Request image generation using the fine-tuned model."""
        conn = http.client.HTTPSConnection(self.host)
        headers = {
            'Content-Type': 'application/json',
            'X-Key': self.api_key
        }

        # Prepare payload based on the example code
        payload = {
            "finetune_id": finetune_id,
            "finetune_strength": finetune_strength,
            "prompt": prompt,
            "num_images": num_images,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": output_format,
            "safety_tolerance": 2,
            "scheduler": scheduler
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        try:
            print(f"\nRequesting image generation with prompt: {prompt}")
            # Format endpoint for URL path
            formatted_endpoint = self._format_endpoint(endpoint)

            conn.request(
                "POST",
                f"/v1/{formatted_endpoint}",
                body=json.dumps(payload),
                headers=headers
            )
            res = conn.getresponse()
            data = res.read()

            return json.loads(data.decode('utf-8'))
        finally:
            conn.close()

    def get_result(self, inference_id: str) -> Dict[str, Any]:
        """Get the result of an inference request."""
        conn = http.client.HTTPSConnection(self.host)
        headers = {
            'Content-Type': 'application/json',
            'X-Key': self.api_key
        }

        try:
            endpoint = f"/v1/get_result?id={inference_id}"
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()

            return json.loads(data.decode('utf-8'))
        finally:
            conn.close()


def generate_image(
    finetune_id: str,
    prompt: str,
    endpoint: str = "flux-pro-1.1-ultra-finetuned",
    **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """
    Generate an image using the specified model and parameters.

    Args:
        finetune_id: ID of the fine-tuned model
        prompt: Text prompt for image generation
        endpoint: API endpoint to use
        **kwargs: Additional parameters for image generation

    Returns:
        Optional[Dict[str, Any]]: Generation result or None if failed
    """
    try:
        api_key = get_api_key()
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        return None

    generator = ImageGenerator(api_key)
    check_interval = 5
    max_attempts = 60  # 5 minutes max

    print("\nStarting image generation...")
    try:
        response = generator.request_inference(
            finetune_id=finetune_id,
            prompt=prompt,
            endpoint=endpoint,
            **kwargs
        )

        if 'id' not in response:
            print("Error: No inference ID received")
            print("Response:", json.dumps(response, indent=2))
            return None

        inference_id = response['id']
        print(f"Inference ID: {inference_id}")

        attempts = 0
        while attempts < max_attempts:
            result = generator.get_result(inference_id)
            status = result.get('status', 'Unknown')
            print(f"\nCheck #{attempts + 1} - Status: {status}")

            if status == 'Ready':
                print("\nImage generation complete!")
                print("\nResult:")
                print(json.dumps(result, indent=2))
                return result
            elif status == 'Failed':
                print("\nImage generation failed!")
                print("\nError details:")
                print(json.dumps(result, indent=2))
                return result

            attempts += 1
            time.sleep(check_interval)

        print("\nTimeout: Maximum attempts reached")
        return None

    except Exception as e:
        print(f"Error during image generation: {e}")
        return None

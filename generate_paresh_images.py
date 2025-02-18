import http.client
import json
import time
from typing import Optional


# Refer to https://docs.bfl.ml/ for details on finetune inference parameters.


class ImageGenerator:
    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai"):
        self.api_key = api_key
        self.host = host

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
            "scheduler": scheduler
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        try:
            print(f"\nRequesting image generation with prompt: {prompt}")
            endpoint_url = f"/v1/{endpoint}"
            conn.request(
                "POST",
                endpoint_url,
                body=json.dumps(payload),
                headers=headers
            )
            res = conn.getresponse()
            data = res.read()

            return json.loads(data.decode('utf-8'))
        finally:
            conn.close()

    def get_result(self, inference_id: str) -> dict:
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
    api_key: str,
    finetune_id: str,
    prompt: str,
    check_interval: int = 5,
    max_attempts: int = 60  # 5 minutes max
):
    generator = ImageGenerator(api_key)

    # Request image generation
    print("\nStarting image generation...")
    try:
        response = generator.request_inference(
            finetune_id=finetune_id,
            prompt=prompt
        )

        if 'id' not in response:
            print("Error: No inference ID received")
            print("Response:", json.dumps(response, indent=2))
            return

        inference_id = response['id']
        print(f"Inference ID: {inference_id}")

        # Monitor progress
        attempts = 0
        while attempts < max_attempts:
            result = generator.get_result(inference_id)
            status = result.get('status', 'Unknown')
            print(f"\nCheck #{attempts + 1} - Status: {status}")

            if result.get('status') == 'Ready':
                print("\nImage generation complete!")
                print("\nResult:")
                print(json.dumps(result, indent=2))
                return result
            elif result.get('status') == 'Failed':
                print("\nImage generation failed!")
                print("\nError details:")
                print(json.dumps(result, indent=2))
                return result

            attempts += 1
            print(f"Waiting {check_interval} seconds before next check...")
            time.sleep(check_interval)

        print("\nTimeout: Maximum attempts reached")

    except Exception as e:
        print(f"Error during image generation: {e}")


if __name__ == "__main__":
    # Your configuration
    API_KEY = "21006105-1bcc-4969-abab-97e55051d7a3"
    # Your new model ID
    FINETUNE_ID = "7fca3c9d-ce9f-4e7a-83c4-6761d67d223f"

    # Example prompts using your trigger word
    prompts = [
        (
            "a professional portrait of pareshranaut in a business suit, "
            "high quality, studio lighting"
        ),
        (
            "pareshranaut giving a presentation in a modern office, 8k, "
            "professional photography"
        ),
        (
            "close-up shot of pareshranaut smiling, natural lighting, "
            "detailed, professional headshot"
        )
    ]

    # Generate an image with each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== Generating Image #{i} ===")
        result = generate_image(API_KEY, FINETUNE_ID, prompt)

        if result and result.get('status') == 'Ready':
            result_data = result.get('result', {})
            url = result_data.get('sample')
            print(f"\nImage #{i} URL: {url}")

        # Wait between generations
        if i < len(prompts):
            print("\nWaiting 10 seconds before next generation...")
            time.sleep(10)

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import requests
from pathlib import Path


@dataclass
class ModelMetadata:
    finetune_id: str
    model_name: str
    trigger_word: str
    mode: str
    type: str
    rank: Optional[int] = None
    iterations: Optional[int] = None
    timestamp: Optional[str] = None
    learning_rate: Optional[float] = None
    priority: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ModelManager:
    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai"):
        self.api_key = api_key
        self.host = host

        # Ensure data directory exists
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # Storage files
        self.models_file = self.data_dir / "models.json"
        self.models: Dict[str, ModelMetadata] = {}

        # Load existing models
        self._load_models()

    def _load_models(self):
        """Load models from storage."""
        try:
            if self.models_file.exists():
                with open(self.models_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        self.models[item['finetune_id']] = ModelMetadata(
                            **item)
                print(f"Loaded {len(self.models)} models from storage")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = {}

    def _save_models(self):
        """Save models to storage."""
        try:
            # Convert models to dict format
            data = [model.to_dict() for model in self.models.values()]

            # Save with pretty formatting
            with open(self.models_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.models)} models to storage")
        except Exception as e:
            print(f"Error saving models: {e}")

    def add_model(self, metadata: ModelMetadata):
        """Add or update a model."""
        self.models[metadata.finetune_id] = metadata
        self._save_models()

    def get_model(self, finetune_id: str) -> Optional[ModelMetadata]:
        """Get model by ID."""
        return self.models.get(finetune_id)

    def list_models(self) -> List[ModelMetadata]:
        """List all models."""
        return list(self.models.values())

    def get_model_details(self, finetune_id: str) -> Optional[dict]:
        """Get model details from API."""
        try:
            url = f"https://{self.host}/v1/finetune_details"
            headers = {"X-Key": self.api_key}
            params = {"finetune_id": finetune_id}

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            details = response.json()
            if details and 'finetune_details' in details:
                return details['finetune_details']
            return None
        except Exception as e:
            print(f"Error getting model details: {e}")
            return None

    def update_model_from_api(self, finetune_id: str) -> bool:
        """Update model details from API."""
        details = self.get_model_details(finetune_id)
        if not details:
            return False

        try:
            metadata = ModelMetadata(
                finetune_id=finetune_id,
                model_name=details.get('finetune_comment', ''),
                trigger_word=details.get('trigger_word', ''),
                mode=details.get('mode', ''),
                type=details.get('finetune_type', 'lora'),
                rank=details.get('lora_rank'),
                iterations=details.get('iterations'),
                timestamp=details.get('timestamp'),
                learning_rate=details.get('learning_rate'),
                priority=details.get('priority')
            )
            self.add_model(metadata)
            return True
        except Exception as e:
            print(f"Error updating model {finetune_id}: {e}")
            return False

    def refresh_models(self):
        """Refresh all models from API."""
        try:
            # Get list of models from API
            url = f"https://{self.host}/v1/my_finetunes"
            headers = {"X-Key": self.api_key}

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not data or 'finetunes' not in data:
                return

            # Update each model's details
            for finetune_id in data['finetunes']:
                self.update_model_from_api(finetune_id)

            # Save updated models
            self._save_models()

        except Exception as e:
            print(f"Error refreshing models: {e}")

    def generate_image(self, endpoint: str, **params) -> Dict[str, Any]:
        """Generate an image using the specified endpoint and parameters."""
        try:
            url = f"https://{self.host}/v1/{endpoint}"
            headers = {
                "Content-Type": "application/json",
                "X-Key": self.api_key
            }

            # Remove None values from params
            params = {k: v for k, v in params.items() if v is not None}

            print(f"Sending request to {endpoint}")
            print(f"Parameters: {json.dumps(params, indent=2)}")

            response = requests.post(url, headers=headers, json=params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Error in generate_image: {e}")
            return {}

    def get_generation_status(self, inference_id: str) -> Dict[str, Any]:
        """Get generation status."""
        try:
            url = f"https://{self.host}/v1/get_result"
            headers = {"X-Key": self.api_key}
            params = {"id": inference_id}

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Error checking generation status: {e}")
            return {}


if __name__ == "__main__":
    API_KEY = "21006105-1bcc-4969-abab-97e55051d7a3"
    manager = ModelManager(api_key=API_KEY)

    # List all models
    print("\nAvailable Models:")
    for model in manager.list_models():
        print(f"ID: {model.finetune_id}")
        print(f"Name: {model.model_name}")
        print(f"Trigger Word: {model.trigger_word}")
        print(f"Type: {model.type}")
        print("---")

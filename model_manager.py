import json
import os
import re
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

    @staticmethod
    def validate_input(data: Dict[str, Any]) -> bool:
        """Validate input data for model metadata."""
        if not all(isinstance(data.get(k), str) for k in ['finetune_id', 'model_name', 'trigger_word', 'mode', 'type']):
            return False
        
        # Validate finetune_id format (alphanumeric with hyphens)
        if not re.match(r'^[a-zA-Z0-9-]+$', data['finetune_id']):
            return False
        
        # Validate model_name (alphanumeric with basic punctuation)
        if not re.match(r'^[\w\s\-_.]+$', data['model_name']):
            return False
        
        # Validate trigger_word (alphanumeric with basic punctuation)
        if not re.match(r'^[\w\s\-_.]+$', data['trigger_word']):
            return False
        
        # Validate mode is one of the allowed values
        if data['mode'] not in ['general', 'character', 'style', 'product']:
            return False
        
        # Validate type is either 'lora' or 'full'
        if data['type'] not in ['lora', 'full']:
            return False
        
        return True


class ModelManager:
    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai"):
        """Initialize ModelManager with API key and host.
        
        Args:
            api_key: API key for authentication
            host: API host domain
            
        Raises:
            ValueError: If api_key is invalid or empty
        """
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) == 0:
            raise ValueError("Invalid API key")
        
        if not re.match(r'^[\w.-]+\.[a-zA-Z]{2,}$', host):
            raise ValueError("Invalid host format")

        self.api_key = api_key
        self.host = host

        # Ensure data directory exists with proper permissions
        self.data_dir = Path("data")
        if not self.data_dir.exists():
            self.data_dir.mkdir(mode=0o700)  # Only owner can read/write
        else:
            # Update permissions if directory exists
            os.chmod(self.data_dir, 0o700)

        # Storage files
        self.models_file = self.data_dir / "models.json"
        self.models: Dict[str, ModelMetadata] = {}

        # Load existing models
        self._load_models()

    def _sanitize_path(self, path: str) -> str:
        """Sanitize file path to prevent directory traversal."""
        return os.path.normpath(path).lstrip(os.sep)

    def _load_models(self):
        """Load models from storage with secure file handling."""
        try:
            if self.models_file.exists():
                # Ensure file permissions are secure
                os.chmod(self.models_file, 0o600)  # Only owner can read/write
                
                with open(self.models_file, "r") as f:
                    data = json.load(f)
                    print(f"Loading {len(data)} models from {self.models_file}")
                    for item in data:
                        try:
                            # Validate model data before loading
                            if not ModelMetadata.validate_input(item):
                                print(f"Skipping invalid model data: {item}")
                                continue
                            
                            model = ModelMetadata(**item)
                            self.models[item["finetune_id"]] = model
                        except Exception as e:
                            print(f"Error loading model: {e}")
                            continue
                for model in self.models.values():
                    print(f"- {model.model_name} ({model.trigger_word})")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = {}

    def _save_models(self):
        """Save models to storage with secure file handling."""
        try:
            # Convert models to dict format
            data = [model.to_dict() for model in self.models.values()]

            # Create temporary file with secure permissions
            temp_file = self.models_file.with_suffix('.tmp')
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            
            # Set secure permissions
            os.chmod(temp_file, 0o600)
            
            # Atomic rename for safer file writing
            temp_file.replace(self.models_file)
            
            print(f"Saved {len(self.models)} models to storage")
        except Exception as e:
            print(f"Error saving models: {e}")

    def add_model(self, metadata: ModelMetadata):
        """Add or update a model.
        
        Args:
            metadata: Model metadata to add/update
            
        Raises:
            ValueError: If metadata is invalid
        """
        if not isinstance(metadata, ModelMetadata):
            raise ValueError("Invalid model metadata type")
        
        if not ModelMetadata.validate_input(metadata.to_dict()):
            raise ValueError("Invalid model metadata content")
        
        self.models[metadata.finetune_id] = metadata
        self._save_models()

    def get_model(self, finetune_id: str) -> Optional[ModelMetadata]:
        """Get model by ID.
        
        Args:
            finetune_id: ID of the model to retrieve
            
        Raises:
            ValueError: If finetune_id format is invalid
        """
        if not isinstance(finetune_id, str) or not re.match(r'^[a-zA-Z0-9-]+$', finetune_id):
            raise ValueError("Invalid finetune ID format")
        return self.models.get(finetune_id)

    def list_models(self) -> List[ModelMetadata]:
        """List all models."""
        models = list(self.models.values())
        print(f"Listing {len(models)} models:")
        for model in models:
            print(f"- {model.model_name} ({model.trigger_word})")
        return models

    def get_model_details(self, finetune_id: str) -> Optional[dict]:
        """Get model details from API."""
        try:
            url = f"https://{self.host}/v1/finetune_details"
            headers = {"X-Key": self.api_key}
            params = {"finetune_id": finetune_id}

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            details = response.json()
            if details and "finetune_details" in details:
                return details["finetune_details"]
            return None
        except Exception as e:
            print(f"Error getting model details: {e}")
            return None

    def update_model_from_api(self, finetune_id: str) -> bool:
        """Update model details from API."""
        details = self.get_model_details(finetune_id)
        if not details:
            print(f"No details found for model {finetune_id}")
            return False

        try:
            metadata = ModelMetadata(
                finetune_id=finetune_id,
                model_name=details.get("finetune_comment", ""),
                trigger_word=details.get("trigger_word", ""),
                mode=details.get("mode", ""),
                type=details.get("finetune_type", "lora"),
                rank=details.get("lora_rank"),
                iterations=details.get("iterations"),
                timestamp=details.get("timestamp"),
                learning_rate=details.get("learning_rate"),
                priority=details.get("priority"),
            )
            print(f"Updating model {finetune_id}: {metadata.model_name}")
            self.add_model(metadata)
            return True
        except Exception as e:
            print(f"Error updating model {finetune_id}: {e}")
            return False

    def refresh_models(self):
        """Refresh all models from API."""
        try:
            print("\nRefreshing models from API...")
            # Get list of models from API
            url = f"https://{self.host}/v1/my_finetunes"
            headers = {"X-Key": self.api_key}

            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not data or "finetunes" not in data:
                print("No models found in API response")
                return

            # Update each model's details
            print(f"Found {len(data['finetunes'])} models in API")
            for finetune_id in data["finetunes"]:
                self.update_model_from_api(finetune_id)

            # Save updated models
            self._save_models()

        except Exception as e:
            print(f"Error refreshing models: {e}")

    def generate_image(self, endpoint: str, **params) -> Dict[str, Any]:
        """Generate an image using the specified endpoint and parameters.
        
        Args:
            endpoint: API endpoint to use
            **params: Generation parameters
            
        Raises:
            ValueError: If endpoint or parameters are invalid
        """
        if not isinstance(endpoint, str) or not re.match(r'^[\w-]+$', endpoint):
            raise ValueError("Invalid endpoint format")

        try:
            url = f"https://{self.host}/v1/{endpoint}"
            headers = {"Content-Type": "application/json", "X-Key": self.api_key}

            # Sanitize and validate parameters
            sanitized_params = {}
            for k, v in params.items():
                if v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        # Basic parameter validation
                        if isinstance(v, str):
                            # Validate string parameters
                            if len(v.strip()) == 0:
                                continue
                            if not re.match(r'^[\w\s\-_.,!?()[\]{}@#$%^&*+=<>:/\\|\'\"]+$', v):
                                raise ValueError(f"Invalid characters in parameter: {k}")
                        sanitized_params[k] = v
                    elif isinstance(v, dict):
                        # Only allow simple key-value pairs in nested dicts
                        sanitized_params[k] = {
                            str(dk): str(dv)
                            for dk, dv in v.items()
                            if isinstance(dk, (str, int, float)) and
                               isinstance(dv, (str, int, float))
                        }

            if not sanitized_params:
                raise ValueError("No valid parameters provided")

            print(f"Sending request to {endpoint}")
            print(f"Parameters: {json.dumps(sanitized_params, indent=2)}")

            response = requests.post(url, headers=headers, json=sanitized_params)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"Error in generate_image: {e}")
            return {}

    def get_generation_status(self, inference_id: str) -> Dict[str, Any]:
        """Get generation status.
        
        Args:
            inference_id: ID of the generation to check
            
        Raises:
            ValueError: If inference_id format is invalid
        """
        if not isinstance(inference_id, str) or not re.match(r'^[a-zA-Z0-9-]+$', inference_id):
            raise ValueError("Invalid inference ID format")

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
    import os
    from getpass import getpass
    
    # Get API key from environment variable or prompt
    api_key = os.getenv("FLUX_API_KEY")
    if not api_key:
        api_key = getpass("Enter your FLUX API key: ")
    
    manager = ModelManager(api_key=api_key)

    # List all models
    print("\nAvailable Models:")
    for model in manager.list_models():
        print(f"ID: {model.finetune_id}")
        print(f"Name: {model.model_name}")
        print(f"Trigger Word: {model.trigger_word}")
        print(f"Type: {model.type}")
        print("---")

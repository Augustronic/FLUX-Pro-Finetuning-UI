# FLUX-Pro-Finetuning-UI Migration Guide

This guide provides step-by-step instructions for migrating from the old structure to the new service-oriented architecture. It includes code examples, import changes, and best practices for a smooth transition.

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure Changes](#directory-structure-changes)
3. [Service Container Usage](#service-container-usage)
4. [Import Changes](#import-changes)
5. [UI Component Migration](#ui-component-migration)
6. [Feature Flags](#feature-flags)
7. [Error Handling](#error-handling)
8. [Testing](#testing)
9. [Checklist](#checklist)

## Overview

The refactoring moves from a tightly coupled architecture to a service-oriented architecture with clear separation of concerns. The key changes are:

- **Core Services**: Centralized services for configuration, API communication, storage, and validation
- **Business Services**: Domain-specific services for models, finetuning, and inference
- **UI Components**: Presentation-only components that use services for business logic
- **Dependency Injection**: Services receive their dependencies through constructors
- **Lazy Loading**: Services are initialized only when needed
- **Feature Flags**: Conditional enabling/disabling of features

## Directory Structure Changes

The new directory structure organizes code by responsibility:

```
FLUX-Pro-Finetuning-UI/
├── app.py                 # Main entry point (now uses ServiceContainer)
├── container.py           # Service container for dependency injection
├── config/                # Configuration files
├── services/              # Services layer
│   ├── core/              # Core services
│   │   ├── api_service.py
│   │   ├── config_service.py
│   │   ├── storage_service.py
│   │   ├── validation_service.py
│   │   └── feature_flag_service.py
│   └── business/          # Business services
│       ├── model_service.py
│       ├── finetuning_service.py
│       └── inference_service.py
├── ui/                    # UI components
│   ├── base.py
│   ├── finetune_ui.py
│   ├── model_browser_ui.py
│   └── inference_ui.py
└── utils/                 # Utility modules
    ├── file_utils.py
    ├── image_utils.py
    └── validation_utils.py
```

## Service Container Usage

The ServiceContainer is the central component for dependency injection. It initializes services on-demand and manages their lifecycle.

### Old Approach (Direct Instantiation)

```python
# Old approach
from model_manager import ModelManager
from config_manager import ConfigManager

# Initialize components directly
config = ConfigManager()
model_manager = ModelManager(api_key=config.get_api_key())
inference_ui = ImageGenerationUI(model_manager)
```

### New Approach (Service Container)

```python
# New approach
from container_lazy import ServiceContainer

# Initialize service container
container = ServiceContainer()

# Get services through the container
config_service = container.get_service('config')
model_service = container.get_service('model')
inference_ui = container.get_service('inference_ui')

# Access configuration
api_key = container.get_config('api_key')
```

## Import Changes

Update imports to use the new service structure:

### Old Imports

```python
# Old imports
from model_manager import ModelManager
from config_manager import ConfigManager
from api_client import APIClient
```

### New Imports

```python
# New imports
from services.core.config_service import ConfigService
from services.core.api_service import APIService
from services.business.model_service import ModelService
```

## UI Component Migration

UI components should be updated to use services for business logic:

### Old UI Component

```python
class ImageGenerationUI:
    def __init__(self):
        self.config = ConfigManager()
        self.api_client = APIClient(api_key=self.config.get_api_key())
        
    def generate_image(self, prompt, model_id):
        # Direct API call
        result = self.api_client.generate_image(model_id, prompt)
        # Process result
        return result
```

### New UI Component

```python
class InferenceUI(BaseUI):
    def __init__(self, inference_service, model_service):
        super().__init__()
        self.inference_service = inference_service
        self.model_service = model_service
        
    def generate_image(self, prompt, model_id):
        # Delegate to service
        return self.inference_service.generate_image(
            endpoint="flux-pro-1.1-ultra-finetuned",
            model_id=model_id,
            prompt=prompt
        )
```

## Feature Flags

Feature flags allow conditional enabling/disabling of features:

### Configuration

Add feature flags to your configuration:

```json
{
  "features": {
    "image_prompt_support": false,
    "prompt_upsampling": true,
    "advanced_parameters": true,
    "experimental_endpoints": false
  }
}
```

### Usage in Code

```python
# Check if a feature is enabled
if self.feature_flags.is_enabled('image_prompt_support', False):
    # Implement image prompt feature
    params['image_prompt'] = image_prompt_base64
```

## Error Handling

The new architecture uses specific error types for different categories of errors:

### Error Types

- **ConfigError**: Configuration-related errors
- **APIError**: API communication errors
- **StorageError**: File operation errors
- **ValidationError**: Input validation errors
- **FinetuningError**: Finetuning-specific errors
- **InferenceError**: Inference-specific errors

### Error Handling Example

```python
try:
    result = self.api_service.request('GET', 'models')
    return result
except APIError as e:
    self.logger.error(f"API error: {e}")
    # Handle API error
    return None
except ValidationError as e:
    self.logger.error(f"Validation error: {e}")
    # Handle validation error
    return None
```

## Testing

The new architecture is designed to be more testable:

### Service Testing

```python
def test_model_service():
    # Create mock dependencies
    mock_api = MockAPIService()
    mock_storage = MockStorageService()
    mock_validation = MockValidationService()
    
    # Initialize service with mock dependencies
    model_service = ModelService(mock_api, mock_storage, mock_validation)
    
    # Test service methods
    models = model_service.list_models()
    assert len(models) == 2
```

### UI Testing

```python
def test_inference_ui():
    # Create mock dependencies
    mock_inference_service = MockInferenceService()
    mock_model_service = MockModelService()
    
    # Initialize UI with mock dependencies
    inference_ui = InferenceUI(mock_inference_service, mock_model_service)
    
    # Test UI methods
    result = inference_ui.generate_image("test prompt", "model123")
    assert result is not None
```

## Migration Checklist

Use this checklist to ensure a complete migration:

- [ ] Update app.py to use ServiceContainer
- [ ] Migrate configuration management to ConfigService
- [ ] Migrate API communication to APIService
- [ ] Migrate file operations to StorageService
- [ ] Migrate validation logic to ValidationService
- [ ] Implement feature flags for conditional features
- [ ] Update UI components to use services
- [ ] Update error handling to use specific error types
- [ ] Add tests for services and UI components
- [ ] Update documentation

## Step-by-Step Migration Process

To minimize disruption, follow this step-by-step process:

1. **Prepare**: Create the new directory structure and service files
2. **Core Services**: Implement and test core services first
3. **Business Services**: Implement and test business services
4. **UI Components**: Update UI components to use services
5. **Main Application**: Update app.py to use ServiceContainer
6. **Testing**: Test the entire application
7. **Cleanup**: Remove deprecated code

## Example: Migrating the Model Browser

Here's a complete example of migrating the model browser component:

### Old Implementation

```python
# model_browser_ui.py (old)
class ModelBrowserUI:
    def __init__(self):
        self.config = ConfigManager()
        self.api_client = APIClient(api_key=self.config.get_api_key())
        
    def list_models(self):
        models = self.api_client.list_models()
        return models
        
    def refresh_models(self):
        self.api_client.refresh_models()
```

### New Implementation

```python
# ui/model_browser_ui.py (new)
from ui.base import BaseUI

class ModelBrowserUI(BaseUI):
    def __init__(self, model_service):
        super().__init__()
        self.model_service = model_service
        
    def list_models(self):
        return self.model_service.list_models()
        
    def refresh_models(self):
        return self.model_service.refresh_models()
```

### Usage in Main Application

```python
# app.py (old)
model_browser = ModelBrowserUI()
models = model_browser.list_models()

# app.py (new)
container = ServiceContainer()
model_browser = container.get_service('model_browser_ui')
models = model_browser.list_models()
```

By following this migration guide, you can smoothly transition from the old structure to the new service-oriented architecture while maintaining all existing functionality.
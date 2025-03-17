# FLUX-Pro-Finetuning-UI (Refactored)

This is the refactored version of FLUX-Pro-Finetuning-UI, a user interface for fine-tuning and using AI image generation models. The codebase has been refactored to improve maintainability, scalability, and readability while preserving all existing functionality.

## Key Improvements

- **Service-Oriented Architecture**: Clear separation of concerns with core services, business services, and UI components
- **Dependency Injection**: Components receive their dependencies rather than creating them
- **Lazy Loading**: Services are initialized only when needed to reduce memory usage
- **Improved Error Handling**: Consistent error handling across the application
- **Centralized Configuration**: Unified configuration management with environment variable support
- **Feature Flags**: Conditional enabling/disabling of features to manage complexity
- **Utility Modules**: Centralized common operations for consistency and reduced duplication
- **Comprehensive Documentation**: Detailed documentation of the architecture and how to extend it

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FLUX-Pro-Finetuning-UI.git
   cd FLUX-Pro-Finetuning-UI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example configuration:
   ```bash
   cp config/config.example.json config/config.json
   ```

4. Edit the configuration file to add your API key and other settings.

### Running the Application

Run the application using the refactored entry point:

```bash
python app_refactored.py
```

This will start the Gradio interface on http://localhost:7860.

## Architecture Overview

The refactored codebase is organized into three main layers:

### Core Services

Core services provide fundamental functionality:

- **ConfigService**: Manages application configuration
- **APIService**: Handles API communication
- **StorageService**: Manages file operations
- **ValidationService**: Validates input data
- **FeatureFlagService**: Manages feature flags

### Business Services

Business services implement domain-specific logic:

- **ModelService**: Manages model data and operations
- **FinetuningService**: Handles finetuning operations
- **InferenceService**: Manages image generation

### UI Components

UI components handle presentation and user interaction:

- **BaseUI**: Provides common UI functionality
- **FineTuneUI**: UI for finetuning models
- **ModelBrowserUI**: UI for browsing models
- **InferenceUI**: UI for generating images

## Service Container

The ServiceContainer provides dependency injection for all services and components. It initializes services on-demand (lazy loading) and manages their lifecycle.

```python
# Get a service from the container
model_service = container.get_service('model')

# Get a configuration value
api_key = container.get_config('api_key')
```

## Simplified Interfaces

For complex services, simplified interfaces are provided:

- **SimpleAPIClient**: Simplified interface for API communication
- **SimpleModelInterface**: Simplified interface for model operations

## Feature Flags

Feature flags allow conditional enabling/disabling of features:

```python
# Check if a feature is enabled
if feature_flag_service.is_enabled('image_prompt_support'):
    # Implement image prompt feature
```

Feature flags are defined in the configuration:

```json
{
  "features": {
    "image_prompt_support": false,
    "prompt_upsampling": true,
    "advanced_parameters": true
  }
}
```

## Utility Modules

Utility modules provide common functionality:

- **file_utils.py**: Utilities for file operations
- **image_utils.py**: Utilities for image processing
- **validation_utils.py**: Utilities for input validation

## Migration Guide

If you have existing code that uses the old structure, here's how to migrate to the new structure:

### Old Structure

```python
from model_manager import ModelManager

# Initialize components
model_manager = ModelManager(api_key=config.get_api_key())

# List models
models = model_manager.list_models()

# Generate image
image = model_manager.generate_image(model_id, prompt)
```

### New Structure

```python
from container_lazy import ServiceContainer

# Initialize service container
container = ServiceContainer()

# Get model service
model_service = container.get_service('model')

# List models
models = model_service.list_models()

# Get inference service
inference_service = container.get_service('inference')

# Generate image
image, status = inference_service.generate_image(
    endpoint=endpoint,
    model_id=model_id,
    prompt=prompt
)
```

## Extending the System

### Adding a New Service

To add a new service:

1. Create a new service class in the appropriate directory
2. Add the service to the ServiceContainer's `_initialize_service` method
3. Update the service map in the `get_service` method

### Adding a New Feature

To add a new feature:

1. Add a feature flag in the configuration
2. Implement the feature in the appropriate service
3. Use the feature flag to conditionally enable/disable the feature

### Adding a New UI Component

To add a new UI component:

1. Create a new UI class that extends BaseUI
2. Add the component to the ServiceContainer
3. Update the main application to include the new component

## Documentation

For more detailed documentation, see:

- [Architecture Overview](docs/architecture.md): Comprehensive overview of the architecture
- [API Reference](docs/api_docs.md): Documentation of the API endpoints
- [Refactoring Plan](refactoring_plan.md): Original plan for the refactoring
- [Refactoring Roadmap](refactoring_roadmap.md): Detailed implementation plan

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
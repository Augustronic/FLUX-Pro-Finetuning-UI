# FLUX-Pro-Finetuning-UI Architecture & Planning

## Overview

FLUX-Pro-Finetuning-UI is built on a service-oriented architecture with clear separation of concerns. This document provides a comprehensive overview of the architecture, explaining how components interact and how to extend the system.

## Architecture Principles

The architecture is designed around the following principles:

1. **Separation of Concerns**: Each component has a clear, focused responsibility
2. **Dependency Injection**: Components receive their dependencies rather than creating them
3. **Lazy Loading**: Services are initialized only when needed to reduce memory usage
4. **Modularity**: The system is composed of small, focused modules that can be developed and tested independently
5. **Feature Flags**: Features can be conditionally enabled or disabled to manage complexity

## Service Layers

The architecture is organized into three main layers:

### 1. Core Services

Core services provide fundamental functionality used by other services:

- **ConfigService**: Manages application configuration, including loading from files and environment variables
- **APIService**: Handles API communication with consistent error handling
- **StorageService**: Manages file operations with proper error handling
- **ValidationService**: Validates input data to ensure consistency
- **FeatureFlagService**: Manages feature flags for conditional feature enabling/disabling

### 2. Business Services

Business services implement domain-specific logic:

- **ModelService**: Manages model data and operations, including listing, retrieving, and refreshing models
- **FinetuningService**: Handles finetuning operations, including starting jobs and monitoring progress
- **InferenceService**: Manages image generation and results handling

### 3. UI Components

UI components handle presentation and user interaction:

- **BaseUI**: Provides common UI functionality and styling
- **FineTuneUI**: UI for finetuning models, including parameter configuration
- **ModelBrowserUI**: UI for browsing and managing models
- **InferenceUI**: UI for generating images with models

## Dependency Flow

The dependency flow follows a clear direction:

```
UI Components → Business Services → Core Services
```

This ensures that higher-level components depend on lower-level ones, not the other way around.

## Service Container

The ServiceContainer provides dependency injection for all services and components. It initializes services on-demand (lazy loading) and manages their lifecycle.

### Lazy Loading

Services are initialized only when they're needed, reducing memory usage and making the codebase more manageable. The ServiceContainer maintains a registry of initialized services and creates them on first access.

```python
def get_service(self, service_name: str) -> Any:
    """Get a service by name, initializing it if necessary."""
    if service_name not in self._initialized:
        self._initialize_service(service_name)
        
    return self._services[service_name]
```

## Simplified Interfaces

For complex services, simplified interfaces are provided to reduce cognitive load and make the codebase more manageable:

- **SimpleAPIClient**: Simplified interface for API communication
- **SimpleModelInterface**: Simplified interface for model operations

These interfaces provide a more focused set of methods for common operations, hiding the complexity of the underlying services.

## Feature Flags

Feature flags allow conditional enabling/disabling of features, making the codebase more manageable by reducing complexity:

```python
# Check if a feature is enabled
if feature_flag_service.is_enabled('image_prompt_support'):
    # Implement image prompt feature
```

Feature flags are defined in the configuration and can be overridden at runtime.

## Utility Modules

Utility modules provide common functionality used across the codebase:

- **file_utils.py**: Utilities for file operations
- **image_utils.py**: Utilities for image processing
- **validation_utils.py**: Utilities for input validation

These modules centralize common operations, reducing duplication and ensuring consistency.

## Application Entry Point

The main application entry point is `app.py`, which creates the Gradio interface and initializes the ServiceContainer:

```python
def create_app():
    # Initialize service container
    container = ServiceContainer()
    
    # Create the combined interface
    with gr.Blocks(title="FLUX [pro] Finetuning UI") as demo:
        # ... UI layout ...
        
        with gr.Tabs():
            with gr.Tab("Finetune Model"):
                # Get UI component from container
                container.get_service('finetune_ui').create_ui()
                
            # ... other tabs ...
                
    return demo
```

## Common Operations

### Loading Models

```python
# Get model service
model_service = container.get_service('model')

# List all models
models = model_service.list_models()

# Get a specific model
model = model_service.get_model(finetune_id)
```

### Generating Images

```python
# Get inference service
inference_service = container.get_service('inference')

# Generate an image
image, status = inference_service.generate_image(
    endpoint=endpoint,
    model_id=model_id,
    prompt=prompt,
    # ... other parameters
)
```

### Starting a Finetuning Job

```python
# Get finetuning service
finetuning_service = container.get_service('finetuning')

# Start a finetuning job
job_id = finetuning_service.start_finetune(
    file_path=file_path,
    params=params
)
```

## Error Handling

Error handling is consistent across the codebase, with specific exception types for different error categories:

- **ConfigError**: Configuration-related errors
- **APIError**: API communication errors
- **StorageError**: File operation errors
- **ValidationError**: Input validation errors

Each service catches and handles errors appropriately, providing meaningful error messages and logging for debugging.

## Logging

Logging is used consistently throughout the codebase to provide visibility into the application's behavior:

```python
self.logger = logging.getLogger(__name__)
self.logger.info("Initializing service")
```

Log messages include the source module, timestamp, and log level, making it easier to diagnose issues.

## Configuration

Configuration is managed by the ConfigService, which loads settings from:

1. Configuration files (JSON)
2. Environment variables
3. Default values

Configuration values can be accessed through the ServiceContainer:

```python
# Get a configuration value
api_key = container.get_config('api_key')

# Get a configuration section
storage_config = container.get_config('storage')
```

## API Reference

The application uses the BFL API for finetuning and image generation. The main endpoints are:

### Finetuning Endpoints

- `GET /v1/finetune_details`: Get details about a finetuned model
  - Parameters: `finetune_id` (string, required)

- `POST /v1/finetune`: Start a finetuning job
  - Parameters:
    - `file_data`: Base64-encoded ZIP file containing training images (string, required)
    - `finetune_comment`: Comment or name of the fine-tuned model (string, required)
    - `trigger_word`: Trigger word for the fine-tuned model (string, default: "TOK")
    - `mode`: Mode for the fine-tuned model (string, enum: "general", "character", "style", "product", required)
    - `iterations`: Number of iterations for fine-tuning (integer, min: 100, max: 1000, default: 300)
    - `learning_rate`: Learning rate for fine-tuning (number, min: 0.000001, max: 0.005, nullable)
    - `captioning`: Whether to enable captioning during fine-tuning (boolean, default: true)
    - `priority`: Priority of the fine-tuning process (string, enum: "speed", "quality", default: "quality")
    - `finetune_type`: Type of fine-tuning (string, enum: "lora", "full", default: "full")
    - `lora_rank`: Rank of the fine-tuned model (integer, enum: 16, 32, default: 32)

### Image Generation Endpoints

- `POST /v1/flux-pro-finetuned`: Generate an image using a finetuned model
- `POST /v1/flux-pro-1.1-ultra-finetuned`: Generate an image using a finetuned model with the ultra endpoint

### Utility Endpoints

- `GET /v1/get_result`: Get the result of a generation task
  - Parameters: `id` (string, required)

## Gradio Framework Reference

The UI is built using the Gradio framework, which provides a simple way to create web interfaces for machine learning models.

### Interface Class

The `gr.Interface` class is the main high-level class in Gradio, allowing you to create a web-based GUI around a Python function:

```python
import gradio as gr

def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch()
```

### Key Parameters

- `fn`: The function to wrap an interface around
- `inputs`: Input components (can be strings or component objects)
- `outputs`: Output components (can be strings or component objects)
- `examples`: Sample inputs for the function
- `live`: Whether the interface should automatically rerun if inputs change
- `title`: A title for the interface
- `description`: A description for the interface
- `theme`: A theme object or string representing a theme

### Launching the Interface

```python
demo.launch(share=True, auth=("username", "password"))
```

### Components

Gradio provides various components for inputs and outputs:

- Text: `gr.Textbox`, `gr.Markdown`, `gr.Label`
- Images: `gr.Image`, `gr.Gallery`
- Audio: `gr.Audio`
- Video: `gr.Video`
- Interactive: `gr.Button`, `gr.Checkbox`, `gr.Radio`, `gr.Dropdown`, `gr.Slider`
- Data: `gr.Dataframe`, `gr.JSON`

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
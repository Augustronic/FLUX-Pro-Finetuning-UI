# FLUX Pro Finetuning UI

A professional UI for model finetuning and image generation with robust error handling, logging, and validation. Built on a service-oriented architecture with clear separation of concerns.

## Features

- **Service-Oriented Architecture**: Clear separation of concerns with core services, business services, and UI components
- **Dependency Injection**: Components receive their dependencies rather than creating them
- **Lazy Loading**: Services are initialized only when needed to reduce memory usage
- **Robust Error Handling**: Centralized error management with detailed context and severity levels
- **Structured Logging**: JSON-formatted logs with rotation and comprehensive error tracking
- **Input Validation**: Extensive validation system with customizable rules
- **Configuration Management**: Environment-aware configuration with validation and overrides
- **Feature Flags**: Conditional enabling/disabling of features to manage complexity
- **Comprehensive Testing**: Full test coverage for all core utilities

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FLUX-Pro-Finetuning-UI.git
cd FLUX-Pro-Finetuning-UI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create configuration:
```bash
cp config/config.example.json config/config.json
```

5. Edit `config/config.json` with your settings:
```json
{
    "api_key": "your_api_key",
    "api_endpoint": "https://api.us1.bfl.ai",
    "model_defaults": {
        "steps": 40,
        "guidance": 2.5,
        "strength": 1.0
    },
    "logging": {
        "level": "INFO",
        "file": "app.log",
        "format": "json"
    },
    "performance": {
        "cache_size": 1024,
        "request_timeout": 30,
        "max_retries": 3
    },
    "security": {
        "rate_limit": 60,
        "allowed_origins": ["localhost"],
        "token_expiry": 3600
    },
    "features": {
        "image_prompt_support": false,
        "prompt_upsampling": true,
        "advanced_parameters": true
    }
}
```

## Running the Application

Launch the application:
```bash
python app.py
```

The UI will be available at `http://localhost:7860`

## Architecture Overview

The application is organized into three main layers:

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

## Project Structure

```
FLUX-Pro-Finetuning-UI/
├── app.py                 # Main application entry point
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
├── utils/                 # Utility modules
│   ├── file_utils.py
│   ├── image_utils.py
│   ├── validation_utils.py
│   ├── error_handling/    # Error management
│   ├── logging/           # Logging system
│   ├── validation/        # Input validation
│   └── config/            # Configuration management
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Service Container

The ServiceContainer provides dependency injection for all services and components:

```python
# Get a service from the container
model_service = container.get_service('model')

# Get a configuration value
api_key = container.get_config('api_key')
```

## Running Tests

Run the full test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=utils tests/
```

## Code Quality

Format code:
```bash
black .
isort .
```

Run linting:
```bash
flake8
mypy .
```

## Environment Variables

The following environment variables can override configuration:

- `FLUX_API_KEY`: Override API key
- `FLUX_API_ENDPOINT`: Override API endpoint
- `FLUX_ENV`: Set environment (development, production, test)

Example:
```bash
export FLUX_ENV=development
export FLUX_API_KEY=your_api_key
```

## Error Handling

The application uses a centralized error handling system with different severity levels:

- INFO: Informational messages
- WARNING: Non-critical issues
- ERROR: Serious issues that need attention
- CRITICAL: System-critical issues

Errors are logged with context and can be found in the log files under the `logs/` directory.

## Logging

Logs are stored in JSON format for easy parsing and analysis. Log files are automatically rotated when they reach 10MB, keeping the last 5 files.

Example log entry:
```json
{
    "timestamp": "2025-02-26T12:34:56.789Z",
    "level": "ERROR",
    "logger": "image_generation",
    "message": "Failed to generate image",
    "extra": {
        "component": "ImageGenerationComponent",
        "operation": "generate_image",
        "details": {
            "model_id": "123",
            "error": "API timeout"
        }
    }
}
```

## Documentation

For more detailed documentation, see:

- [Planning & Architecture](PLANNING.md): Comprehensive overview of the architecture and planning
- [Tasks & Roadmap](TASK.md): Current tasks and development roadmap

## API Reference

The application uses the BFL API for finetuning and image generation. The main endpoint is:

```
https://api.us1.bfl.ai/scalar
```

## Gradio Framework

The UI is built using the Gradio framework, which provides a simple way to create web interfaces for machine learning models. For more information, see:

```
https://www.gradio.app/docs/gradio/interface
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

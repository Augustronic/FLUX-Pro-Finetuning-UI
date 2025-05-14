# FLUX-Pro-Finetuning-UI Technical Context

## Technologies Used

### Primary Languages and Frameworks

- **Python**: Primary language for all backend logic and UI components
- **Gradio**: UI framework for creating interactive web interfaces
- **JSON**: Used for configuration files and API communication

### API Integration

- **BFL API**: External API for model finetuning and image generation
  - Base endpoint: `https://api.us1.bfl.ai/scalar`
  - Used for finetuning, image generation, and model management

### Development Tools

- **Black**: Code formatter for consistent Python code style
- **isort**: Import sorting tool for organizing imports
- **pytest**: Testing framework for unit and integration tests
- **flake8**: Linting tool for code quality
- **mypy**: Static type checking for Python

## Development Setup

### Environment Setup

1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Dependencies Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configuration**:
   ```bash
   cp config/config.example.json config/config.json
   # Edit config.json with appropriate values
   ```

### Configuration Structure

The system uses a JSON-based configuration system with environment variable overrides:

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

### Environment Variables

The system supports loading environment variables from .env files using python-dotenv, as well as from the operating system environment. Environment variables can override configuration settings.

#### Environment File (.env)

The system automatically searches for a `.env` file in the project root directory. Example `.env` file:

```
FLUX_API_KEY=your_api_key_here
FLUX_API_ENDPOINT=https://api.us1.bfl.ai
FLUX_ENV=development
```

#### Supported Variables

- `FLUX_API_KEY`: Override API key
- `FLUX_API_ENDPOINT`: Override API endpoint
- `FLUX_ENV`: Set environment (development, production, test)
- `FLUX_LOG_LEVEL`: Set logging level
- `FLUX_CACHE_SIZE`: Set cache size for performance tuning
- `FLUX_FEATURE_*`: Feature flag toggles (e.g., `FLUX_FEATURE_ADVANCED_PARAMETERS=true`)

## Technical Dependencies

### Runtime Dependencies

- **gradio**: Web interface framework
- **requests**: HTTP client for API communication
- **pillow**: Image processing
- **numpy**: Numerical operations
- **python-dotenv**: Environment variable management from .env files
- **pydantic**: Data validation and settings management
- **python-json-logger**: Structured logging

### Development Dependencies

- **pytest**: Testing framework
- **black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter
- **mypy**: Type checker
- **pytest-cov**: Test coverage measurement

## Technical Constraints

### API Limitations

- **Rate Limits**: API calls are subject to rate limiting
- **File Size**: Training data has maximum size limitations
- **Job Concurrency**: Limited number of concurrent finetuning jobs

### Performance Considerations

- **Memory Usage**: Image processing can be memory-intensive
- **API Latency**: External API calls introduce latency
- **Caching**: Model information should be cached to reduce API calls

### Security Constraints

- **API Key Handling**: API keys must be securely stored
- **File Access**: Local storage must be properly secured
- **Input Validation**: All user inputs must be validated

## File Structure

```
FLUX-Pro-Finetuning-UI/
├── app.py                  # Main application entry point
├── container.py            # Service container for dependency injection
├── config/                 # Configuration files
├── services/               # Services layer
│   ├── core/               # Core services
│   │   ├── api_service.py
│   │   ├── config_service.py
│   │   ├── storage_service.py
│   │   ├── validation_service.py
│   │   └── feature_flag_service.py
│   └── business/           # Business services
│       ├── model_service.py
│       ├── finetuning_service.py
│       └── inference_service.py
├── ui/                     # UI components
│   ├── base.py
│   ├── finetune_ui.py
│   ├── model_browser_ui.py
│   └── inference_ui.py
├── utils/                  # Utility modules
│   ├── file_utils.py
│   ├── image_utils.py
│   ├── validation_utils.py
│   ├── env_manager.py      # Environment variable management
│   ├── error_handling/     # Error management
│   ├── logging/            # Logging system
│   ├── validation/         # Input validation
│   └── config/             # Configuration management
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Gradio Framework Usage

### Core Components

- **gr.Blocks**: Main container for UI components
- **gr.Tabs**: Tabbed interface for different functions
- **gr.Tab**: Individual tab components
- **gr.Image**: Image display and upload
- **gr.Textbox**: Text input and display
- **gr.Button**: Action buttons
- **gr.Dropdown**: Selection dropdowns
- **gr.Slider**: Parameter adjustment

### Component Creation Pattern

```python
import gradio as gr

# Create individual UI component
with gr.Blocks(title="FLUX [pro] Finetuning UI") as demo:
    # Tabs for main sections
    with gr.Tabs():
        with gr.Tab("Finetune Model"):
            # Get UI component from container
            container.get_service('finetune_ui').create_ui()

        with gr.Tab("Generate Images"):
            container.get_service('inference_ui').create_ui()

        with gr.Tab("Model Browser"):
            container.get_service('model_browser_ui').create_ui()
```

### Event Handling Pattern

```python
# Create a button
submit_btn = gr.Button("Submit")

# Connect button click to function
submit_btn.click(
    fn=model_service.some_function,
    inputs=[param1, param2],
    outputs=[result]
)
```

## BFL API Integration

### Finetuning Endpoints

- `GET /v1/finetune_details`: Get details about a finetuned model
- `POST /v1/finetune`: Start a finetuning job

### Image Generation Endpoints

- `POST /v1/flux-pro-finetuned`: Generate an image using a finetuned model
- `POST /v1/flux-pro-1.1-ultra-finetuned`: Generate an image using a finetuned model with the ultra endpoint

### Utility Endpoints

- `GET /v1/get_result`: Get the result of a generation task

## Tool Usage Patterns

### Logging Pattern

```python
self.logger = logging.getLogger(__name__)
self.logger.info("Initializing service")
```

### Error Handling Pattern

```python
try:
    result = self.api_client.request(endpoint, params)
except APIError as e:
    self.logger.error(f"API error: {e}")
    raise
except Exception as e:
    self.logger.error(f"Unexpected error: {e}")
    raise
```

### Validation Pattern

```python
def validate_params(self, params):
    if not self.validation_service.is_valid(params, schema=self.schemas.FINETUNE_PARAMS):
        raise ValidationError("Invalid finetuning parameters")
```

### Configuration Access Pattern

```python
api_key = self.config_service.get('api_key')
endpoint = self.config_service.get('api_endpoint')
```

### Environment Variable Access Pattern

```python
from utils.env_manager import get_env, get_bool_env, get_int_env

# Get string value (with default)
api_key = get_env("API_KEY", "default_key")

# Get boolean value
debug_mode = get_bool_env("DEBUG_MODE", False)

# Get integer value
cache_size = get_int_env("CACHE_SIZE", 1024)

# Get float value
timeout = get_float_env("TIMEOUT", 30.0)

# Get list value (comma-separated by default)
allowed_origins = get_list_env("ALLOWED_ORIGINS", ["localhost"])
```

### Feature Flag Pattern

```python
if self.feature_flag_service.is_enabled('advanced_parameters'):
    # Show advanced parameters
```

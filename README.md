# FLUX Pro Finetuning UI

A professional UI for model finetuning and image generation with robust error handling, logging, and validation.

## Features

- **Robust Error Handling**: Centralized error management with detailed context and severity levels
- **Structured Logging**: JSON-formatted logs with rotation and comprehensive error tracking
- **Input Validation**: Extensive validation system with customizable rules
- **Configuration Management**: Environment-aware configuration with validation and overrides
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
cp config.example.json config.json
```

5. Edit `config.json` with your settings:
```json
{
    "api_key": "your_api_key",
    "api_endpoint": "https://api.example.com",
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
    }
}
```

## Running the Application

Launch the application:
```bash
venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

The UI will be available at `http://localhost:7860`

## Development

### Project Structure

```
FLUX-Pro-Finetuning-UI/
├── app.py                 # Main application entry point
├── ui/                    # UI components
│   ├── base.py           # Base UI component class
│   ├── image_generation.py
│   ├── model_selection.py
│   └── parameter_config.py
├── utils/                 # Core utilities
│   ├── error_handling/   # Error management
│   ├── logging/          # Logging system
│   ├── validation/       # Input validation
│   └── config/           # Configuration management
├── tests/                # Test suite
│   └── utils/           # Utility tests
└── docs/                 # Documentation
```

### Running Tests

Run the full test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=utils tests/
```

### Code Quality

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

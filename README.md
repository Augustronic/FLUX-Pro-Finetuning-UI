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

5. Edit `config/config.json` with your settings

## Running the Application

Launch the application:
```bash
python app.py
```

The UI will be available at `http://localhost:7860`

## Environment Variables

The application supports loading configuration from environment variables, which can override settings in the config file. Environment variables can be set directly in your system or through a `.env` file in the project root.

### .env File Support

Create a `.env` file in the project root (see `.env.example` for a template):

```bash
# Copy the example file and customize
cp .env.example .env
```

### Available Environment Variables

#### API Configuration
- `FLUX_API_KEY`: API key for authentication
- `FLUX_API_ENDPOINT`: API endpoint URL
- `FLUX_API_TIMEOUT`: Timeout for API requests in seconds
- `FLUX_API_MAX_RETRIES`: Maximum number of retry attempts

#### Environment
- `FLUX_ENV`: Set environment (`development`, `production`, `test`)

#### Logging
- `FLUX_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `FLUX_LOG_FILE`: Path to log file
- `FLUX_LOG_FORMAT`: Log format (`json`, `text`)

#### Performance
- `FLUX_CACHE_SIZE`: Maximum cache size in memory
- `FLUX_REQUEST_TIMEOUT`: Request timeout in seconds
- `FLUX_MAX_CONCURRENT_JOBS`: Maximum concurrent jobs

#### Feature Flags
- `FLUX_FEATURE_*`: Toggle specific features on/off (e.g., `FLUX_FEATURE_ADVANCED_PARAMETERS=true`)

## Project Structure

```
FLUX-Pro-Finetuning-UI/
├── app.py                 # Main application entry point
├── container.py           # Service container for dependency injection
├── config/                # Configuration files
├── services/              # Services layer
│   ├── core/              # Core services
│   └── business/          # Business services
├── ui/                    # UI components
├── utils/                 # Utility modules
├── tests/                 # Test suite
├── docs/                  # Documentation
└── memory-bank/           # Project memory and documentation
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

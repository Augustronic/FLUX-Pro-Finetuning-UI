# Flux Pro Fine-tuning UI

A professional GUI application for fine-tuning and managing AI models, with support for both LoRA and full fine-tuning approaches.

## Official Documentation
- [API Documentation](https://api.us1.bfl.ai/scalar#tag/tasks/POST/v1/finetune)
- [Fine-tuning Guide](https://docs.bfl.ml/finetuning/)

## Features

- 🎯 **Fine-tuning Interface**
  - Support for LoRA and full fine-tuning
  - Multiple training modes (character, product, style, general)
  - Configurable parameters (iterations, learning rate, etc.)
  - Auto-captioning support
  - Real-time progress monitoring

- 📊 **Model Management**
  - Browse and manage fine-tuned models
  - View model details and parameters
  - Easy model selection for inference

- 🖼️ **Image Generation**
  - Generate images using fine-tuned models
  - Adjust generation parameters
  - Download and manage generated images

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd flux-pro-ft
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Copy `config.example.json` to `config.json`
   - Replace `your-api-key-here` with your actual API key
   ```json
   {
       "api_key": "your-api-key-here",
       "api_host": "api.us1.bfl.ai",
       "storage": {
           "models_dir": "data/models",
           "images_dir": "generated_images"
       }
   }
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

## Usage Guide

### Fine-tuning Models

1. **Prepare Your Data**
   - Create a ZIP file containing your training images
   - Supported formats: JPG, PNG
   - Recommended: 15-20 high-quality images

2. **Start Fine-tuning**
   - Upload your ZIP file
   - Fill in model details:
     - Model Name: A unique identifier
     - Trigger Word: Word to invoke your model
     - Training Mode: character/product/style/general
     - Fine-tuning Type: LoRA (faster) or Full
   - Configure advanced settings if needed
   - Click "Start Fine-tuning"

3. **Monitor Progress**
   - Copy the provided Fine-tune ID
   - Use the "Check Status" button to monitor progress
   - Status will show:
     - Pending: Job is queued/running
     - Ready: Training complete
     - Failed: Check error message

### Model Browser

- View all your fine-tuned models
- Sort and filter by various parameters
- Quick copy of model IDs and trigger words
- View detailed model information

### Image Generation

1. Select a model from the dropdown
2. Enter your prompt (include the trigger word)
3. Adjust generation parameters if needed
4. Click "Generate"
5. Download generated images

## Directory Structure

```
flux-pro-ft/
├── app.py              # Main application entry point
├── config.example.json # Example configuration
├── requirements.txt    # Python dependencies
├── data/              # Model storage directory
└── generated_images/   # Output directory for generated images
```

## Important Notes

- **API Key**: Never commit your `config.json` with real API key
- **Storage**: The application creates necessary directories on first run
- **Models**: Your models are stored locally in `data/`
- **Images**: Generated images are saved in `generated_images/`

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `config.json` exists with valid API key
   - Check API key permissions

2. **Missing Directories**
   - The app creates required directories automatically
   - Ensure write permissions in the app directory

3. **Import Errors**
   - Verify all dependencies are installed
   - Check Python version (3.8+ recommended)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
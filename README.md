# Flux 1.1 Pro & Ultra Fine-tuning UI

A professional GUI application for fine-tuning Flux 1.1 Pro and Ultra models, supporting both LoRA and full fine-tuning approaches.

## Supported Models
- Flux Pro 1.1 Ultra (Latest)
- Flux Pro 1.1 Standard
Both models support LoRA and full fine-tuning methods.

## Official Documentation
- [API Documentation](https://api.us1.bfl.ai/scalar#tag/tasks/POST/v1/finetune)
- [Fine-tuning Guide](https://docs.bfl.ml/finetuning/)

## Features

- 🎯 **Advanced Fine-tuning**
  - Full support for Flux 1.1 Pro & Ultra models
  - LoRA fine-tuning (fast, memory-efficient)
  - Full model fine-tuning (comprehensive)
  - Multiple training modes (character, product, style, general)
  - Configurable parameters (iterations, learning rate, etc.)
  - Auto-captioning support
  - Real-time progress monitoring

- 📊 **Model Management**
  - Browse and manage Flux Pro fine-tuned models
  - View model details and parameters
  - Easy model selection for inference
  - Track LoRA ranks and training progress

- 🖼️ **Pro & Ultra Image Generation**
  - Generate with Flux Pro 1.1 Ultra for highest quality
  - Standard Flux Pro 1.1 generation support
  - Adjust generation parameters
  - Download and manage generated images

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/paresh795/Flux-1.1-Pro-Ultra---Fine-tuning-UI
   cd Flux-1.1-Pro-Ultra---Fine-tuning-UI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   - Copy `config.example.json` to `config.json`
   - Replace `your-api-key-here` with your Flux Pro API key
   ```json
   {
       "api_key": "your-api-key-here",
       "api_host": "api.us1.bfl.ai",
       "storage": {
           "models_dir": "data",
           "images_dir": "generated_images"
       }
   }
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

## Detailed Usage Guide

### Fine-tuning with Flux Pro

1. **Prepare Your Training Data**
   - Create a ZIP file containing your training images
   - Supported formats: JPG, PNG
   - Recommended: 15-20 high-quality images
   - Ensure consistent style/subject matter

2. **Choose Fine-tuning Method**
   - **LoRA Fine-tuning**
     - Faster training
     - Lower resource requirements
     - Great for style and character models
     - Configurable rank (4-128)

   - **Full Fine-tuning**
     - Complete model customization
     - Higher resource requirements
     - Best for comprehensive training
     - Longer training time

3. **Start Fine-tuning**
   - Upload your ZIP file
   - Fill in model details:
     - Model Name: A unique identifier
     - Trigger Word: Word to invoke your model
     - Training Mode: character/product/style/general
     - Fine-tuning Type: LoRA or Full
   - Configure advanced settings if needed
   - Click "Start Fine-tuning"

4. **Monitor Progress**
   - Copy the provided Fine-tune ID
   - Use the "Check Status" button to monitor progress
   - Status will show:
     - Pending: Job is queued
     - Running: Currently training
     - Ready: Training complete
     - Failed: Check error message

### Model Browser

- View all your Flux Pro fine-tuned models
- Sort and filter by various parameters
- Quick copy of model IDs and trigger words
- View detailed model information including:
  - Training method (LoRA/Full)
  - Model parameters
  - Training status

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

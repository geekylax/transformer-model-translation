# Transformer Model Translation (English to Tamil)

A PyTorch-based transformer model for English to Tamil translation, with support for both local training and cloud training via Modal.

## Features

- 🚀 Transformer architecture for machine translation
- 🌐 English to Tamil translation
- ☁️ Cloud training support with Modal
- 📊 TensorBoard integration for monitoring
- 🔧 Configurable hyperparameters
- 📝 Custom tokenizer training

## Security

⚠️ **Important Security Notice**: This project has been updated to address critical PyTorch vulnerabilities including CVE-2025-32434. Always use the latest versions specified in `requirements.txt`.

## Project Structure

```
├── model.py              # Transformer model architecture
├── dataset.py            # Dataset handling and preprocessing
├── train.py              # Local training script
├── train_modal.py        # Modal cloud training script
├── test.py               # Model testing and evaluation
├── config.py             # Configuration parameters
├── requirements.txt      # Python dependencies
├── requirements_modal.txt # Modal-specific dependencies
└── weights/              # Model checkpoints (excluded from git)
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transformer-model-translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Training

```bash
python train.py
```

### Cloud Training with Modal

1. Install Modal CLI:
```bash
pip install modal
```

2. Set up Modal account and run:
```bash
modal run train_modal.py
```

### Testing

```bash
python test.py
```

## Configuration

Modify `config.py` to adjust:
- Model parameters (d_model, num_heads, etc.)
- Training settings (batch_size, learning_rate, etc.)
- Data paths and tokenizer settings

## Model Architecture

- **Model Type**: Transformer (Encoder-Decoder)
- **Source Language**: English (en)
- **Target Language**: Tamil (ta)
- **Sequence Length**: 256 tokens
- **Model Dimension**: 512
- **Attention Heads**: 8
- **Layers**: 6

## Security Best Practices

- ✅ No hardcoded credentials
- ✅ Updated dependencies with security patches
- ✅ Large model files excluded from version control
- ✅ Environment-specific configurations

## Contributing

1. Ensure all dependencies are up to date
2. Run tests before submitting changes
3. Follow the existing code style
4. Add appropriate documentation

## License

[Add your license here]

## Acknowledgments

- PyTorch team for the ML framework
- Modal for cloud compute infrastructure
- HuggingFace for datasets and tokenizers 
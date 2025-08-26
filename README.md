# Gemma 3 270M Fine-tuning with Unsloth

This repository contains multiple ways to fine-tune the `unsloth/gemma-3-270m-it` model using the HuggingFaceH4/Helpful-Instructions dataset.

## Files

- `gemma_finetuning_colab.ipynb`: Google Colab notebook for fine-tuning
- `train_gemma.py`: Standalone Python training script
- `evaluate_model.py`: Script to evaluate the fine-tuned model
- `compare_models.py`: Script to compare base and fine-tuned models
- `visualize_training.py`: Script to visualize training metrics
- `check_env.py`: Utility script to check dependencies and CUDA support
- `config.json`: Configuration file for training parameters
- `requirements.txt`: Python dependencies for local development
- `run_training.sh`: Bash script for Linux/macOS training
- `run_training.bat`: Batch script for Windows training

## Dataset

We use the [HuggingFaceH4/Helpful-Instructions](https://huggingface.co/datasets/HuggingFaceH4/Helpful-Instructions) dataset, which contains 148,000 examples of prompts and completions designed to train helpful AI assistants.

## Model

The notebook fine-tunes the [unsloth/gemma-3-270m-it](https://huggingface.co/unsloth/gemma-3-270m-it) model, which is a quantized and optimized version of Google's Gemma 3 270M parameter model.

## Features

- Uses Unsloth's efficient fine-tuning techniques
- Implements LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Optimized for T4 GPU on Google Colab
- Includes model testing functionality
- Configurable training parameters via JSON
- Training metrics logging and visualization

## Google Colab Usage

1. Open `gemma_finetuning_colab.ipynb` in Google Colab
2. Ensure you're using a GPU runtime (Runtime -> Change runtime type -> GPU -> T4)
3. Run all cells sequentially

## Local Development

### Environment Check

Before training, you can check if your environment is properly set up:

```bash
python check_env.py
```

This script will verify:
- All required dependencies are installed
- CUDA is available for GPU acceleration

### Using the training scripts

To set up the environment locally:

```bash
# On Linux/macOS
chmod +x run_training.sh
./run_training.sh

# On Windows
run_training.bat
```

Alternatively, you can manually set up the environment:

```bash
pip install -r requirements.txt
python train_gemma.py
```

Note: Unsloth has specific requirements for CUDA versions and may require additional setup for local development. Make sure you have:
- Python 3.10+
- CUDA toolkit (for GPU acceleration)
- PyTorch with CUDA support

### Configuration

The training parameters can be adjusted in `config.json`:
- Model settings (name, sequence length, quantization)
- Training hyperparameters (batch size, learning rate, etc.)
- LoRA configuration
- Dataset settings
- Output paths

### Evaluating the Model

After training, you can evaluate the model using:

```bash
python evaluate_model.py
```

To compare the base model with the fine-tuned model:

```bash
python compare_models.py
```

### Visualizing Training Metrics

During training, metrics are logged to `training_logs.json`. You can visualize these metrics using:

```bash
python visualize_training.py
```

This will generate:
- Loss curve plot
- Learning rate schedule plot
- Training summary text file

## Results

The fine-tuned model will be better at following helpful instructions compared to the base model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
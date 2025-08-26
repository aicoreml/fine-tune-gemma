# Gemma Fine-tuning Project Workflow

This document describes the complete workflow for fine-tuning the Gemma 3 270M model using the Unsloth library and the HuggingFaceH4/Helpful-Instructions dataset.

## Overview

The project follows this high-level workflow:
1. Environment Setup
2. Model Training
3. Model Evaluation
4. Training Analysis
5. Model Comparison

## Detailed Workflow

### 1. Environment Setup

Before beginning the fine-tuning process, you need to set up your environment:

1. **Create Virtual Environment**:
   - Run `setup.sh` (Linux/Mac) or `setup.bat` (Windows) to create a virtual environment
   - This installs all required dependencies from `requirements.txt`

2. **Activate Environment**:
   - Linux/Mac: `source gemma_env/bin/activate`
   - Windows: `gemma_env\Scripts\activate`

### 2. Model Training

The training process involves several steps, executed by `train_gemma.py`:

1. **Load Configuration**:
   - Reads parameters from `config.json`
   - Includes model settings, training hyperparameters, and LoRA configuration

2. **Load Model**:
   - Loads the `unsloth/gemma-3-270m-it` model
   - Uses 4-bit quantization for memory efficiency
   - Sets maximum sequence length to 4096 tokens

3. **Load Dataset**:
   - Downloads the `HuggingFaceH4/Helpful-Instructions` dataset
   - Contains 148,000 examples of prompts and completions

4. **Format Dataset**:
   - Formats data in the required Gemma chat format:
     ```
     <start_of_turn>user
     {instruction}<end_of_turn>
     <start_of_turn>model
     {demonstration}<end_of_turn>
     ```

5. **Apply LoRA Adapters**:
   - Adds Low-Rank Adaptation adapters to the model
   - Configurable rank (r=64) and alpha (α=16) parameters
   - Applies to key transformer modules (q_proj, k_proj, v_proj, etc.)

6. **Configure Trainer**:
   - Sets up the SFTTrainer with training arguments
   - Configures batch size, learning rate, and optimizer settings
   - Implements custom logging callback for metrics tracking

7. **Train Model**:
   - Executes the training process for the specified number of steps (default: 60)
   - Logs training metrics to `training_logs.json`

8. **Save Model**:
   - Saves the fine-tuned model to `gemma-3-270m-helpful-instruct/`
   - Also saves the tokenizer

9. **Test Model**:
   - Runs a sample inference on the first test prompt
   - Displays the model's response

#### Running Training

You can execute training in multiple ways:
- **Google Colab**: Open `gemma_finetuning_colab.ipynb`
- **Local Script**: Run `python train_gemma.py`
- **Platform Scripts**: 
  - Linux/Mac: `chmod +x run_training.sh && ./run_training.sh`
  - Windows: `run_training.bat`

### 3. Model Evaluation

After training, evaluate the model's performance using `evaluate_model.py`:

1. **Load Fine-tuned Model**:
   - Loads the saved model from `gemma-3-270m-helpful-instruct/`

2. **Run Inference Tests**:
   - Tests the model on 5 sample prompts covering various topics
   - Formats prompts in Gemma's expected chat format

3. **Display Results**:
   - Shows the model's responses for each prompt
   - Helps assess the quality of fine-tuning

#### Running Evaluation

Execute with: `python evaluate_model.py`

### 4. Training Analysis

Analyze the training process using `visualize_training.py`:

1. **Load Training Logs**:
   - Reads metrics from `training_logs.json`

2. **Generate Reports**:
   - Prints training loss curve data
   - Shows learning rate schedule
   - Creates a comprehensive training summary

3. **Save Summary**:
   - Writes training summary to `training_summary.txt`

#### Running Analysis

Execute with: `python visualize_training.py`

### 5. Model Comparison

Compare the base model with the fine-tuned model using `compare_models.py`:

1. **Load Both Models**:
   - Loads the original base model (`unsloth/gemma-3-270m-it`)
   - Loads the fine-tuned model (`gemma-3-270m-helpful-instruct`)

2. **Run Comparative Tests**:
   - Tests both models on the same set of prompts
   - Formats prompts in Gemma's chat format

3. **Display Side-by-side Results**:
   - Shows responses from both models for each prompt
   - Makes it easy to see improvements from fine-tuning

#### Running Comparison

Execute with: `python compare_models.py`

## Configuration

The `config.json` file controls all aspects of the training process:

### Model Configuration
- `name`: Model identifier on Hugging Face
- `max_seq_length`: Maximum sequence length (4096)
- `load_in_4bit`: Whether to use 4-bit quantization (true)

### Training Configuration
- `per_device_train_batch_size`: Batch size per device (2)
- `gradient_accumulation_steps`: Gradient accumulation steps (4)
- `warmup_steps`: Number of warmup steps (5)
- `max_steps`: Total training steps (60)
- `learning_rate`: Learning rate (0.0002)
- `weight_decay`: Weight decay (0.01)
- `logging_steps`: Steps between logging (10)
- `optim`: Optimizer type (adamw_8bit)
- `lr_scheduler_type`: Learning rate scheduler (linear)
- `seed`: Random seed (3407)

### LoRA Configuration
- `r`: LoRA rank (64)
- `lora_alpha`: LoRA alpha parameter (16)
- `lora_dropout`: LoRA dropout rate (0)
- `bias`: Bias training setting (none)

### Dataset Configuration
- `name`: Dataset identifier on Hugging Face
- `split`: Dataset split to use (train)

### Output Configuration
- `dir`: Training output directory (outputs)
- `model_save_path`: Path to save the fine-tuned model

### Inference Configuration
- `max_new_tokens`: Maximum tokens to generate (200)
- `test_prompts`: Sample prompts for testing

## File Structure

```
├── gemma_finetuning_colab.ipynb  # Google Colab training notebook
├── train_gemma.py                # Main training script
├── evaluate_model.py             # Model evaluation script
├── compare_models.py             # Model comparison script
├── visualize_training.py         # Training metrics analysis
├── check_env.py                  # Environment verification
├── config.json                   # Training configuration
├── requirements.txt              # Python dependencies
├── run_training.sh               # Linux/Mac training script
├── run_training.bat              # Windows training script
├── setup.sh                      # Linux/Mac environment setup
├── setup.bat                     # Windows environment setup
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore patterns
└── outputs/                      # Training outputs (created during training)
```

## Expected Outputs

After running the complete workflow, you'll have:

1. **Fine-tuned Model**: Saved in `gemma-3-270m-helpful-instruct/`
2. **Training Logs**: `training_logs.json` with metrics
3. **Training Summary**: `training_summary.txt` with key metrics
4. **Console Outputs**: Printed results from evaluation and comparison scripts

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**: 
   - Reduce batch size in config.json
   - Reduce max_seq_length
   - Ensure you're using a GPU with sufficient memory

2. **Dependency Issues**:
   - Run `check_env.py` to verify installation
   - Reinstall dependencies with `pip install -r requirements.txt`

3. **Model Loading Errors**:
   - Check internet connection for downloading models/datasets
   - Verify sufficient disk space for model storage

4. **Training Quality Issues**:
   - Increase max_steps in config.json
   - Adjust learning_rate
   - Modify LoRA parameters (r, lora_alpha)
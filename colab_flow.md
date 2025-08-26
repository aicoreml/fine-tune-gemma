# Gemma Fine-tuning Colab Notebook Workflow

This document describes the step-by-step workflow for fine-tuning the Gemma 3 270M model using the `gemma_finetuning_colab.ipynb` notebook on Google Colab.

## Overview

The Colab notebook provides a streamlined approach to fine-tuning the Gemma model with the following key steps:
1. Environment Setup
2. Library Imports
3. Model and Dataset Loading
4. Data Preparation
5. Model Configuration
6. Training
7. Model Saving
8. Model Testing

## Detailed Workflow

### 1. Environment Setup

The first step in the notebook installs all required dependencies:

```bash
!pip install unsloth
```

This command installs:
- Unsloth library for efficient fine-tuning
- Transformers library
- Datasets library
- PyTorch with CUDA support
- All other required dependencies

### 2. Library Imports

The notebook imports the necessary libraries:

```python
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
```

### 3. Model and Dataset Loading

#### Loading the Model
The notebook loads the Gemma 3 270M model with these parameters:
- Model name: `unsloth/gemma-3-270m-it`
- Maximum sequence length: 4096 tokens
- 4-bit quantization enabled for memory efficiency
- Automatic data type selection

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
```

#### Loading the Dataset
The notebook loads the `HuggingFaceH4/Helpful-Instructions` dataset:
- Contains 148,000 examples of prompts and completions
- Designed for training helpful AI assistants

```python
dataset = load_dataset("HuggingFaceH4/Helpful-Instructions", split="train")
```

### 4. Data Preparation

The notebook examines the dataset structure and prepares it for training:

1. **Dataset Inspection**:
   - Displays dataset features
   - Shows sample prompt and completion pairs

2. **Data Formatting**:
   - Converts data to Gemma's chat format:
     ```
     <start_of_turn>user
     {instruction}<end_of_turn>
     <start_of_turn>model
     {demonstration}<end_of_turn>
     ```
   - Applies formatting to the entire dataset using a mapping function

### 5. Model Configuration

The notebook configures the model for efficient fine-tuning using LoRA (Low-Rank Adaptation):

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM usage
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

### 6. Training

The notebook sets up and executes the training process:

#### Training Arguments
The trainer is configured with these parameters:
- Batch size: 2 per device
- Gradient accumulation steps: 4
- Warmup steps: 5
- Maximum steps: 60
- Learning rate: 2e-4
- Optimizer: adamw_8bit
- Weight decay: 0.01
- Learning rate scheduler: linear
- Random seed: 3407

#### Training Execution
The training is executed with:
```python
trainer_stats = trainer.train()
```

The notebook displays training progress with a visual progress bar and logs training loss at regular intervals.

### 7. Model Saving

After training completes, the notebook saves the fine-tuned model:

```python
model.save_pretrained("gemma-3-270m-helpful-instruct")
tokenizer.save_pretrained("gemma-3-270m-helpful-instruct")
```

The saved model can be loaded later for inference:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("gemma-3-270m-helpful-instruct")
```

### 8. Model Testing

The notebook includes a quick test to verify the fine-tuned model:

1. **Enable Fast Inference**:
   ```python
   FastLanguageModel.for_inference(model)  # 2x faster inference
   ```

2. **Run Test Inference**:
   - Formats a test prompt in Gemma's chat format
   - Generates a response using the fine-tuned model
   - Displays the model's output

## Colab-Specific Features

### GPU Acceleration
- The notebook is optimized for T4 GPU on Google Colab
- Uses mixed precision training (FP16/BF16) for faster training
- Implements gradient checkpointing to reduce memory usage

### Interactive Progress Tracking
- Visual progress bar during training
- Real-time loss reporting
- Training summary at completion

## Expected Outputs

After running the complete notebook, you'll have:

1. **Installed Dependencies**: All required libraries for fine-tuning
2. **Loaded Model**: Gemma 3 270M model in memory
3. **Processed Dataset**: Formatted instruction dataset ready for training
4. **Configured Model**: LoRA adapters added to the model
5. **Training Results**: Completed training with loss metrics
6. **Saved Model**: Fine-tuned model saved to `gemma-3-270m-helpful-instruct/`
7. **Test Output**: Sample generation from the fine-tuned model

## Running the Notebook

### Prerequisites
- Google Colab account
- T4 GPU runtime (Runtime → Change runtime type → GPU → T4)

### Execution Steps
1. Open `gemma_finetuning_colab.ipynb` in Google Colab
2. Ensure GPU runtime is selected
3. Run all cells sequentially:
   - First cell installs dependencies
   - Subsequent cells execute the workflow steps
4. Monitor training progress in the output cells
5. Check the final test output to verify the model works

## Customization Options

You can modify these parameters in the notebook:

### Model Parameters
- `max_seq_length`: Maximum sequence length (up to 8192)
- `load_in_4bit`: Quantization setting (True/False)

### LoRA Parameters
- `r`: LoRA rank (8, 16, 32, 64, 128)
- `lora_alpha`: Alpha parameter
- `lora_dropout`: Dropout rate

### Training Parameters
- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Gradient accumulation
- `max_steps`: Total training steps
- `learning_rate`: Learning rate
- `warmup_steps`: Warmup steps

### Dataset Parameters
- Dataset name: Replace `HuggingFaceH4/Helpful-Instructions` with another dataset
- Data formatting function: Modify for different dataset structures

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Reduce sequence length
   - Ensure T4 GPU is selected

2. **Installation Errors**:
   - Restart runtime and reinstall
   - Check internet connection

3. **Training Quality Issues**:
   - Increase max_steps
   - Adjust learning_rate
   - Modify LoRA parameters

4. **Loading Errors**:
   - Verify model path
   - Check disk space availability
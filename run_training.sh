#!/bin/bash

# Script to fine-tune Gemma 3 270M with Unsloth

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create a new conda environment
echo "Creating conda environment..."
conda create -n gemma-finetune python=3.10 -y

# Activate the environment
echo "Activating environment..."
conda activate gemma-finetune

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

# Run the training script
echo "Starting training..."
python train_gemma.py

echo "Training completed!"
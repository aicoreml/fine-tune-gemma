#!/bin/bash
# setup.sh - Setup script for the Gemma fine-tuning project

echo "Setting up Gemma 3 270M fine-tuning environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Please install Python 3.10 or later."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip is not installed. Please install pip."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv gemma_env

# Activate virtual environment
echo "Activating virtual environment..."
source gemma_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source gemma_env/bin/activate"
echo "To deactivate the environment, run: deactivate"
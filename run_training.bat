@echo off
REM Script to fine-tune Gemma 3 270M with Unsloth on Windows

REM Check if conda is installed
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda could not be found. Please install Anaconda or Miniconda first.
    exit /b 1
)

REM Create a new conda environment
echo Creating conda environment...
conda create -n gemma-finetune python=3.10 -y

REM Activate the environment
echo Activating environment...
call conda activate gemma-finetune

REM Install PyTorch with CUDA support
echo Installing PyTorch...
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

REM Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

REM Run the training script
echo Starting training...
python train_gemma.py

echo Training completed!
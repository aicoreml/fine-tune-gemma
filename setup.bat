@echo off
REM setup.bat - Setup script for the Gemma fine-tuning project on Windows

echo Setting up Gemma 3 270M fine-tuning environment...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.10 or later.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv gemma_env

REM Activate virtual environment
echo Activating virtual environment...
call gemma_env\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo Setup complete!
echo To activate the environment, run: gemma_env\Scripts\activate
echo To deactivate the environment, run: deactivate
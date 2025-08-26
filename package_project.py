"""
Script to package all project files into a zip archive
"""

import os
import zipfile

def package_project(output_filename="gemma_finetuning_project.zip"):
    """Package all project files into a zip archive"""
    
    # List of files to include in the package
    files_to_include = [
        "gemma_finetuning_colab.ipynb",
        "train_gemma.py",
        "evaluate_model.py",
        "compare_models.py",
        "visualize_training.py",
        "check_env.py",
        "config.json",
        "requirements.txt",
        "run_training.sh",
        "run_training.bat",
        "README.md",
        "FILE_SUMMARY.md",
        "training_logs.json"
    ]
    
    # Create zip archive
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to archive")
            else:
                print(f"Warning: {file} not found")
    
    print(f"\nProject packaged successfully as {output_filename}")

if __name__ == "__main__":
    package_project()
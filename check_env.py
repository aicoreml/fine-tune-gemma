"""
Utility script to check if required dependencies are installed
"""

import sys
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    required_packages = [
        "torch",
        "transformers",
        "accelerate",
        "trl",
        "datasets",
        "peft",
        "bitsandbytes",
        "unsloth"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except ImportError:
        print("✗ PyTorch is not installed, cannot check CUDA")
        return False

def main():
    print("Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\nChecking CUDA support...")
    cuda_ok = check_cuda()
    
    if deps_ok and cuda_ok:
        print("\n✓ Your environment is ready for training!")
    else:
        print("\n✗ Please fix the issues above before training.")

if __name__ == "__main__":
    main()
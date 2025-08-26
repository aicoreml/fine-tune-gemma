"""
Script to examine the HuggingFaceH4/Helpful-Instructions dataset structure
"""

from datasets import load_dataset

def examine_dataset():
    """Examine the dataset structure"""
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/Helpful-Instructions", split="train")
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    
    # Print first few examples to understand the structure
    print("\nFirst 3 examples:")
    for i in range(min(3, len(dataset))):
        print(f"\nExample {i}:")
        for key in dataset[i].keys():
            # Handle string slicing properly
            value = str(dataset[i][key])
            print(f"  {key}: {value[:100]}...")  # First 100 characters
            
    # Check if 'prompt' and 'completion' keys exist
    first_example = dataset[0]
    print(f"\nAvailable keys in first example: {list(first_example.keys())}")
    
    # Check for common alternative key names\n    possible_prompt_keys = ['prompt', 'instruction', 'input', 'question', 'user']\n    possible_completion_keys = ['completion', 'output', 'response', 'answer', 'model', 'demonstration']\n    \n    print(\"\\nChecking for possible key names:\")\n    for key in possible_prompt_keys:\n        if key in first_example:\n            print(f\"  Found prompt key: {key}\")\n            \n    for key in possible_completion_keys:\n        if key in first_example:\n            print(f\"  Found completion key: {key}\")

if __name__ == "__main__":
    examine_dataset()
"""
Test script to verify the dataset field names and formatting
"""

from datasets import load_dataset

def test_dataset_formatting():
    """Test that we can correctly access the dataset fields"""
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/Helpful-Instructions", split="train")
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")
    
    # Test accessing the fields
    try:
        instruction = dataset[0]["instruction"]
        demonstration = dataset[0]["demonstration"]
        print("\nFirst example:")
        print(f"Instruction: {instruction[:100]}...")
        print(f"Demonstration: {demonstration[:100]}...")
        print("SUCCESS: Fields accessed correctly!")
    except KeyError as e:
        print(f"ERROR: Field not found - {e}")
        return False
    
    # Test the formatting function
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["demonstration"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>\n"
            texts.append(text)
        return {"text": texts}
    
    try:
        formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
        print(f"\nFormatted example: {formatted_dataset[0]['text'][:100]}...")
        print("SUCCESS: Formatting function works correctly!")
        return True
    except Exception as e:
        print(f"ERROR: Formatting failed - {e}")
        return False

if __name__ == "__main__":
    test_dataset_formatting()
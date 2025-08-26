"""
Comparison script to compare base Gemma model with fine-tuned model
"""

import torch
from unsloth import FastLanguageModel

def compare_models(base_model_name, fine_tuned_model_path, test_prompts):
    """Compare base model with fine-tuned model on test prompts"""
    
    # Load base model
    print("Loading base model...")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(fine_tuned_model_path)
    
    # Enable faster inference
    FastLanguageModel.for_inference(base_model)
    FastLanguageModel.for_inference(ft_model)
    
    # Compare models on test prompts
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*50}")
        print(f"Test {i+1}: {prompt}")
        print('='*50)
        
        # Format prompt for Gemma
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Test base model
        print("\n--- Base Model Response ---")
        base_inputs = base_tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        base_outputs = base_model.generate(**base_inputs, max_new_tokens=200, use_cache=True)
        base_response = base_tokenizer.batch_decode(base_outputs)[0]
        base_model_response = base_response.split("<start_of_turn>model\n")[1]
        print(base_model_response)
        
        # Test fine-tuned model
        print("\n--- Fine-tuned Model Response ---")
        ft_inputs = ft_tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        ft_outputs = ft_model.generate(**ft_inputs, max_new_tokens=200, use_cache=True)
        ft_response = ft_tokenizer.batch_decode(ft_outputs)[0]
        ft_model_response = ft_response.split("<start_of_turn>model\n")[1]
        print(ft_model_response)

def main():
    # Define test prompts
    test_prompts = [
        "How can I improve my time management skills?",
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of regular exercise?"
    ]
    
    # Compare models
    compare_models(
        "unsloth/gemma-3-270m-it", 
        "gemma-3-270m-helpful-instruct", 
        test_prompts
    )
    
    print("\nComparison completed!")

if __name__ == "__main__":
    main()
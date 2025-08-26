"""
Evaluation script for the fine-tuned Gemma model
"""

import torch
from unsloth import FastLanguageModel

def evaluate_model(model_path, test_prompts):
    """Load the model and evaluate it on test prompts"""
    
    # Load the fine-tuned model
    print("Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(model_path)
    
    # Enable faster inference
    FastLanguageModel.for_inference(model)
    
    # Test the model on various prompts
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Format prompt for Gemma
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # Tokenize and generate response
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200, use_cache=True)
        response = tokenizer.batch_decode(outputs)[0]
        
        # Extract just the model's response
        model_response = response.split("<start_of_turn>model\n")[1]
        print(f"Response: {model_response}")

def main():
    # Define test prompts
    test_prompts = [
        "How can I improve my time management skills?",
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of regular exercise?",
        "How do I make a delicious pasta dish?",
        "What's the difference between Python and JavaScript?"
    ]
    
    # Evaluate the fine-tuned model
    evaluate_model("gemma-3-270m-helpful-instruct", test_prompts)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
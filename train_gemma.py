"""
Fine-tuning script for Gemma 3 270M using Unsloth
"""

import torch
import json
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from unsloth import is_bfloat16_supported

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

class LoggingCallback(TrainerCallback):
    """Custom callback to log training metrics"""
    
    def __init__(self, log_file="training_logs.json"):
        self.log_file = log_file
        self.logs = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Add step information
            logs['step'] = state.global_step
            self.logs.append(logs)
            
            # Save to file
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)

def main():
    # Load configuration
    config = load_config()
    
    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None,
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(
        config["dataset"]["name"], 
        split=config["dataset"]["split"]
    )
    print(f"Dataset loaded with {len(dataset)} examples")

    # Format dataset
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["demonstration"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = f"<start_of_turn>user\\n{instruction}<end_of_turn>\\n<start_of_turn>model\\n{output}<end_of_turn>\\n"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    # Create logging callback
    logging_callback = LoggingCallback()

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model.config.max_position_embeddings,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            warmup_steps=config["training"]["warmup_steps"],
            max_steps=config["training"]["max_steps"],
            learning_rate=config["training"]["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=config["training"]["logging_steps"],
            optim=config["training"]["optim"],
            weight_decay=config["training"]["weight_decay"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            seed=config["training"]["seed"],
            output_dir=config["output"]["dir"],
        ),
        callbacks=[logging_callback],
    )

    # Train model
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed!")

    # Save model
    print("Saving model...")
    model.save_pretrained(config["output"]["model_save_path"])
    tokenizer.save_pretrained(config["output"]["model_save_path"])
    print("Model saved!")

    # Test model
    print("Testing model...")
    FastLanguageModel.for_inference(model)
    
    test_prompt = config["inference"]["test_prompts"][0]
    inputs = tokenizer(
        [f"<start_of_turn>user\\n{test_prompt}<end_of_turn>\\n<start_of_turn>model\\n"],
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    response = tokenizer.batch_decode(outputs)[0]
    print("Model response:")
    print(response.split("<start_of_turn>model\\n")[1])

if __name__ == "__main__":
    main()
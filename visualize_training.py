"""
Script to analyze training metrics
"""

import json

def load_training_logs(log_file="training_logs.json"):
    """Load training logs from JSON file"""
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
        return logs
    except FileNotFoundError:
        print(f"Log file {log_file} not found.")
        return None

def print_loss_curve(logs):
    """Print training loss curve data"""
    if not logs:
        return
    
    print("\nTraining Loss Curve:")
    print("====================")
    print("Step\tLoss")
    print("----\t----")
    for log in logs:
        if 'loss' in log:
            print(f"{log['step']}\t{log['loss']:.4f}")

def print_lr_schedule(logs):
    """Print learning rate schedule data"""
    if not logs:
        return
    
    print("\nLearning Rate Schedule:")
    print("========================")
    print("Step\tLearning Rate")
    print("----\t-------------")
    for log in logs:
        if 'learning_rate' in log:
            print(f"{log['step']}\t{log['learning_rate']:.6f}")

def create_training_summary(logs, save_path="training_summary.txt"):
    """Create a summary of training metrics"""
    if not logs:
        return
    
    # Extract final metrics
    final_loss = logs[-1].get('loss', 'N/A')
    final_lr = logs[-1].get('learning_rate', 'N/A')
    total_steps = logs[-1].get('step', 'N/A')
    
    # Create summary
    summary = f"""
Training Summary
================

Final Loss: {final_loss}
Final Learning Rate: {final_lr}
Total Training Steps: {total_steps}

Configuration:
- Model: unsloth/gemma-3-270m-it
- Dataset: HuggingFaceH4/Helpful-Instructions
- Max Steps: 60
- Batch Size: 2
- Gradient Accumulation Steps: 4
"""
    
    # Save summary
    with open(save_path, 'w') as f:
        f.write(summary)
    print(f"Training summary saved to {save_path}")
    
    # Also print to console
    print(summary)

def main():
    # Load training logs
    logs = load_training_logs()
    
    if logs:
        # Print metrics
        print_loss_curve(logs)
        print_lr_schedule(logs)
        create_training_summary(logs)
        print("Analysis completed!")
    else:
        print("No training logs found. Run training first.")

if __name__ == "__main__":
    main()

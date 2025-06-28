from transformers import (
    LongformerForSequenceClassification, 
    LongformerTokenizerFast, 
    TrainingArguments, 
    Trainer, 
    LongformerConfig, 
    default_data_collator
)
from datasets import load_from_disk
import json
from datetime import datetime
import os
from pathlib import Path

def load_config():
    """Load configuration from the main config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return None
        
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def main():
    """Train the model using the configured parameters."""
    print("🚀 Starting model training...")
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = config.get("model_output_path", "models")
    output_dir = os.path.join(base_output_dir, f"longformer-esg-{timestamp}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Model will be saved to: {output_dir}")
    
    # Load tokenized dataset
    tokenizer_output_path = config.get("tokenizer_output_path")
    if not tokenizer_output_path or not os.path.exists(tokenizer_output_path):
        print(f"❌ Tokenized dataset not found at: {tokenizer_output_path}")
        print("💡 Please run tokenization step first")
        return False
    
    print(f"📊 Loading tokenized dataset from: {tokenizer_output_path}")
    dataset = load_from_disk(tokenizer_output_path)
    
    # Model configuration
    model_name = config.get("model_name")
    num_labels = config.get("num_labels", 12)
    problem_type = config.get("problem_type", "multi_label_classification")
    
    print(f"⚙️  Model configuration:")
    print(f"   Base model: {model_name}")
    print(f"   Number of labels: {num_labels}")
    print(f"   Problem type: {problem_type}")
    
    try:
        # Load model configuration
        model_config = LongformerConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type
        )
        
        # Load model
        print(f"🤗 Loading model: {model_name}")
        model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            config=model_config
        )
        
        # Prepare training arguments
        training_args_config = config.get("training_args", {})
        
        # Override output_dir with our timestamped directory
        training_args_config["output_dir"] = output_dir
        
        print(f"📋 Training arguments:")
        for key, value in training_args_config.items():
            print(f"   {key}: {value}")
        
        training_args = TrainingArguments(**training_args_config)
        
        # Initialize trainer
        print("🎯 Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=default_data_collator
        )
        
        # Start training
        print("🏃 Starting training process...")
        trainer.train()
        
        # Save the final model
        print(f"💾 Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        
        # Save tokenizer as well
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        # Save configuration used for training
        config_save_path = os.path.join(output_dir, "training_config.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print("✅ Training completed successfully!")
        print(f"📂 Model saved to: {output_dir}")
        print(f"📋 Training configuration saved to: {config_save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
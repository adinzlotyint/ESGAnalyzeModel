from transformers import (
    LongformerForSequenceClassification, 
    LongformerTokenizerFast, 
    TrainingArguments, 
    Trainer, 
    LongformerConfig, 
    default_data_collator
)
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
import mlflow
import mlflow.transformers
from typing import Dict, Any

def load_config():
    """Load configuration from the main config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return None
        
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def compute_metrics(eval_pred):
    """Compute evaluation metrics for multi-label classification."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities, then threshold at 0.5
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int)
    
    # Convert to proper format for sklearn
    labels = labels.astype(int)
    
    # Calculate metrics
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-label F1 scores
    f1_per_label = f1_score(labels, predictions, average=None, zero_division=0)
    
    # Accuracy (exact match for multi-label)
    accuracy = accuracy_score(labels, predictions)
    
    # ESG category names for better tracking
    esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
    
    # Create detailed metrics
    metrics = {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro, 
        'f1_weighted': f1_weighted,
        'accuracy': accuracy,
    }
    
    # Add per-label F1 scores with meaningful names
    for i, f1 in enumerate(f1_per_label):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'f1_{label_name}'] = f1
    
    return metrics

def setup_mlflow(config: Dict[str, Any], output_dir: str) -> None:
    """Initialize MLflow tracking for experiment."""
    # Set experiment name
    experiment_name = "ESG_Longformer_Training"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    mlflow.start_run(run_name=f"longformer-esg-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Log configuration parameters
    training_args = config.get("training_args", {})
    mlflow.log_params({
        "model_name": config.get("model_name"),
        "num_labels": config.get("num_labels"),
        "problem_type": config.get("problem_type"),
        "learning_rate": training_args.get("learning_rate"),
        "batch_size": training_args.get("per_device_train_batch_size"),
        "num_epochs": training_args.get("num_train_epochs"),
        "max_steps": training_args.get("max_steps"),
        "warmup_steps": training_args.get("warmup_steps"),
        "weight_decay": training_args.get("weight_decay"),
        "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps"),
        "fp16": training_args.get("fp16"),
        "gradient_checkpointing": training_args.get("gradient_checkpointing"),
    })
    
    # Log dataset info
    mlflow.log_params({
        "tokenizer_path": config.get("tokenizer_output_path"),
        "model_output_path": output_dir,
    })
    
    print("📊 MLflow tracking initialized")
    print(f"   Experiment: {experiment_name}")
    print(f"   Run: {mlflow.active_run().info.run_name}")

class MLflowCallback:
    """Custom callback to log metrics to MLflow during training."""
    
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, logs):
        """Log metrics to MLflow."""
        if mlflow.active_run():
            # Log all metrics from the logs
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=self.step_count)
            self.step_count += 1

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
    
    # Initialize MLflow tracking
    try:
        setup_mlflow(config, output_dir)
    except Exception as e:
        print(f"⚠️ MLflow initialization failed: {e}")
        print("   Continuing without MLflow tracking...")
    
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
    num_labels = config.get("num_labels", 7)
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
            data_collator=default_data_collator,
            compute_metrics=compute_metrics
        )
        
        # Start training
        print("🏃 Starting training process...")
        trainer.train()
        
        # Log final evaluation metrics to MLflow
        if mlflow.active_run():
            print("📊 Logging final metrics to MLflow...")
            final_metrics = trainer.evaluate()
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"final_{key}", value)
        
        # Save the final model
        print(f"💾 Saving model to: {output_dir}")
        trainer.save_model(output_dir)
        
        # Log model to MLflow
        if mlflow.active_run():
            try:
                # Log the model with MLflow
                mlflow.transformers.log_model(
                    transformers_model={"model": model, "tokenizer": LongformerTokenizerFast.from_pretrained(model_name)},
                    artifact_path="model",
                    registered_model_name="ESG_Longformer"
                )
                print("🏷️ Model logged to MLflow registry")
            except Exception as e:
                print(f"⚠️ MLflow model logging failed: {e}")
        
        # Save tokenizer as well
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        # Save configuration used for training
        config_save_path = os.path.join(output_dir, "training_config.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # Log training artifacts to MLflow
        if mlflow.active_run():
            try:
                mlflow.log_artifact(config_save_path, "config")
                print("📋 Training configuration logged to MLflow")
            except Exception as e:
                print(f"⚠️ MLflow artifact logging failed: {e}")
        
        print("✅ Training completed successfully!")
        print(f"📂 Model saved to: {output_dir}")
        print(f"📋 Training configuration saved to: {config_save_path}")
        
        if mlflow.active_run():
            print(f"🔗 MLflow run: {mlflow.active_run().info.run_id}")
            print(f"📊 View results: mlflow ui --backend-store-uri ./mlruns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()
            print("📊 MLflow run completed")

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
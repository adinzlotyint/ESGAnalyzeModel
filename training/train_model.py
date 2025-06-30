from transformers import (
    LongformerForSequenceClassification, 
    LongformerTokenizerFast, 
    TrainingArguments, 
    Trainer, 
    LongformerConfig, 
    default_data_collator
)
from datasets import load_from_disk
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
import mlflow
import torch
import torch.nn as nn
from typing import Dict, Any

def load_config():
    """Load configuration from the main config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return None
        
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def calculate_moderate_class_weights(dataset, num_labels=7):
    """Calculate moderate class weights using square root scaling."""
    all_labels = np.array(dataset['labels'])
    class_weights = []
    
    for label_idx in range(num_labels):
        label_column = all_labels[:, label_idx]
        pos_count = np.sum(label_column == 1)
        total_count = len(label_column)
        
        if pos_count > 0:
            pos_freq = pos_count / total_count
            pos_weight = np.sqrt(1 / pos_freq)
            pos_weight = min(pos_weight, 1.5)  # Cap at 1.5
        else:
            pos_weight = 1.0
            
        class_weights.append(pos_weight)
    
    return torch.FloatTensor(class_weights)

def compute_metrics(eval_pred):
    """Compute basic evaluation metrics during training."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    y_probs = sigmoid(predictions)
    
    # Use default threshold of 0.5
    y_pred = (y_probs > 0.5).astype(int)
    y_true = labels.astype(int)
    
    # Calculate basic metrics
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1_per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # ESG category names
    esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
    
    metrics = {
        'f1_macro': f1_macro,
        'accuracy': accuracy,
    }
    
    # Add per-label F1 scores
    for i, f1 in enumerate(f1_per_label):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'f1_{label_name}'] = f1
    
    return metrics

def optimize_thresholds_post_training(model, dataloader, device):
    """Optimize thresholds after training on validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    
    # Optimize thresholds for each label
    optimal_thresholds = []
    
    for label_idx in range(y_true.shape[1]):
        best_threshold = 0.5
        best_f1 = 0
        
        # Grid search for optimal threshold
        for threshold in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_probs[:, label_idx] >= threshold).astype(int)
            f1 = f1_score(y_true[:, label_idx], y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
    
    return np.array(optimal_thresholds)

def evaluate_with_thresholds(model, test_dataset, optimal_thresholds, device=None):
    """Evaluate model on test dataset using optimal thresholds."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    
    # Apply optimal thresholds
    y_pred_optimal = np.zeros_like(y_probs)
    for i, threshold in enumerate(optimal_thresholds):
        y_pred_optimal[:, i] = (y_probs[:, i] >= threshold).astype(int)
    
    # Calculate final metrics
    f1_macro = f1_score(y_true, y_pred_optimal, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_optimal)
    f1_per_label = f1_score(y_true, y_pred_optimal, average=None, zero_division=0)
    
    # Build results
    esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
    results = {
        'final_eval_f1_macro': f1_macro,
        'final_eval_accuracy': accuracy,
    }
    
    # Add per-label F1 scores
    for i, f1 in enumerate(f1_per_label):
        label_name = esg_labels[i] if i < len(esg_labels) else f'L{i}'
        results[f'final_eval_f1_{label_name}'] = f1
    
    return results

def setup_mlflow(config: Dict[str, Any], output_dir: str) -> None:
    """Initialize MLflow tracking for experiment."""
    experiment_name = "ESG_Longformer_Training_v2"
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=f"longformer-esg-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Log only essential parameters
    training_args = config.get("training_args", {})
    mlflow.log_params({
        "model_name": config.get("model_name"),
        "num_labels": config.get("num_labels"),
        "num_epochs": training_args.get("num_train_epochs"),
        "learning_rate": training_args.get("learning_rate"),
        "batch_size": training_args.get("per_device_train_batch_size"),
        "class_weight_method": "moderate",
    })

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
    dataset = load_from_disk(tokenizer_output_path)
    if not tokenizer_output_path or not os.path.exists(tokenizer_output_path):
        print(f"❌ Tokenized dataset not found at: {tokenizer_output_path}")
        return False
    
    print(f"📊 Loading tokenized dataset from: {tokenizer_output_path}")
    dataset = load_from_disk(tokenizer_output_path)
    
    # Model configuration
    model_name = config.get("model_name")
    num_labels = config.get("num_labels", 7)
    problem_type = config.get("problem_type", "multi_label_classification")
    
    # Calculate moderate class weights
    class_weights = calculate_moderate_class_weights(dataset["train"], num_labels=num_labels)
    
    try:
        # Load model
        model_config = LongformerConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type
        )
        
        model = LongformerForSequenceClassification.from_pretrained(
            model_name,
            config=model_config
        )
        
        # Prepare training arguments
        training_args_config = config.get("training_args", {})
        training_args_config["output_dir"] = output_dir
        training_args = TrainingArguments(**training_args_config)
        
        # ESG Trainer with moderate class weights
        class ESGTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
                
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                pos_weights = self.class_weights.to(logits.device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='mean')
                loss = loss_fn(logits, labels.float())
                
                return (loss, outputs) if return_outputs else loss
        
        trainer = ESGTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=default_data_collator,
            compute_metrics=compute_metrics
        )
        
        print("🏃 Starting training process...")
        trainer.train()
        
        # Post-training threshold optimization
        val_dataloader = trainer.get_eval_dataloader()
        optimal_thresholds = optimize_thresholds_post_training(
            model, val_dataloader, training_args.device
        )
        
        # Final evaluation with optimal thresholds
        final_results = evaluate_with_thresholds(
            model, dataset["test"], optimal_thresholds, training_args.device
        )
        
        # Log ONLY final results to MLflow
        if mlflow.active_run():
            # Get final epoch number
            final_epoch = training_args.num_train_epochs
            mlflow.log_metric("final_epoch", final_epoch)
            
            # Log final metrics
            for key, value in final_results.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        
        # Save the final model
        trainer.save_model(output_dir)
        
        # Save tokenizer
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        print("✅ Training completed successfully!")
        print(f"📂 Model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False
    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
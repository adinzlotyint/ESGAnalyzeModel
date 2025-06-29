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
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize_scalar
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path
import mlflow
import mlflow.transformers
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

def calculate_class_weights(dataset, num_labels=7, method="balanced"):
    """
    Calculate class weights for multi-label imbalanced dataset.
    
    Args:
        dataset: HuggingFace dataset with 'labels' column
        num_labels: Number of labels (default: 7 for ESG)
        method: "balanced", "sqrt", or "log" weighting method
        
    Returns:
        torch.Tensor: Class weights for each label
    """
    print("📊 Calculating class weights for balanced training...")
    
    # Extract all labels
    all_labels = np.array(dataset['labels'])
    
    # Calculate positive/negative frequencies for each label
    class_weights = []
    
    for label_idx in range(num_labels):
        # Get labels for this specific class
        label_column = all_labels[:, label_idx]
        
        # Count positive and negative examples
        pos_count = np.sum(label_column == 1)
        neg_count = np.sum(label_column == 0)
        total_count = len(label_column)
        
        # Calculate weight based on method
        if method == "balanced":
            # Standard balanced weighting: n_samples / (n_classes * n_samples_class)
            pos_weight = total_count / (2 * pos_count) if pos_count > 0 else 1.0
            
        elif method == "sqrt":
            # Square root of inverse frequency
            pos_freq = pos_count / total_count
            pos_weight = np.sqrt(1 / pos_freq) if pos_freq > 0 else 1.0
            
        elif method == "log":
            # Logarithmic weighting  
            pos_freq = pos_count / total_count
            pos_weight = np.log(1 / pos_freq) if pos_freq > 0 else 1.0
            
        else:
            pos_weight = 1.0
        
        # For multi-label, we primarily care about positive class weight
        class_weights.append(pos_weight)
        
        # ESG label names for logging
        esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
        label_name = esg_labels[label_idx] if label_idx < len(esg_labels) else f'Label_{label_idx}'
        
        print(f"   {label_name}: {pos_count}/{total_count} positive ({pos_count/total_count*100:.1f}%) -> weight: {pos_weight:.3f}")
    
    # Convert to tensor
    weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"✅ Class weights calculated using '{method}' method")
    print(f"   Weights range: {weights_tensor.min():.3f} - {weights_tensor.max():.3f}")
    
    return weights_tensor

def optimize_thresholds(y_true, y_probs, num_labels=7):
    """
    Find optimal thresholds for each label to maximize F1 scores.
    
    Args:
        y_true: True labels (n_samples, n_labels)
        y_probs: Predicted probabilities (n_samples, n_labels)
        num_labels: Number of labels
        
    Returns:
        np.array: Optimal thresholds for each label
    """
    print("🎯 Optimizing classification thresholds...")
    
    optimal_thresholds = []
    esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
    
    for label_idx in range(num_labels):
        # Get true labels and probabilities for this label
        true_labels = y_true[:, label_idx]
        pred_probs = y_probs[:, label_idx]
        
        # Define objective function to maximize F1 score
        def negative_f1(threshold):
            predictions = (pred_probs >= threshold).astype(int)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            return -f1  # Negative because minimize_scalar minimizes
        
        # Optimize threshold between 0.1 and 0.9
        result = minimize_scalar(negative_f1, bounds=(0.1, 0.9), method='bounded')
        optimal_threshold = result.x
        optimal_f1 = -result.fun
        
        optimal_thresholds.append(optimal_threshold)
        
        # Calculate F1 at default 0.5 for comparison
        default_preds = (pred_probs >= 0.5).astype(int)
        default_f1 = f1_score(true_labels, default_preds, zero_division=0)
        
        label_name = esg_labels[label_idx] if label_idx < len(esg_labels) else f'Label_{label_idx}'
        improvement = optimal_f1 - default_f1
        
        print(f"   {label_name}: threshold {optimal_threshold:.3f} (F1: {optimal_f1:.3f}, +{improvement:+.3f})")
    
    optimal_thresholds = np.array(optimal_thresholds)
    print(f"✅ Threshold optimization completed")
    print(f"   Threshold range: {optimal_thresholds.min():.3f} - {optimal_thresholds.max():.3f}")
    
    return optimal_thresholds

def compute_metrics(eval_pred):
    """Compute evaluation metrics with threshold optimization for multi-label classification."""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    y_probs = sigmoid(predictions)
    
    # Convert to proper format for sklearn
    y_true = labels.astype(int)
    
    # Optimize thresholds for each label - ALWAYS ENABLED
    optimal_thresholds = optimize_thresholds(y_true, y_probs, num_labels=7)
    
    # Apply optimized thresholds
    y_pred_optimized = np.zeros_like(y_probs, dtype=int)
    for i, threshold in enumerate(optimal_thresholds):
        y_pred_optimized[:, i] = (y_probs[:, i] >= threshold).astype(int)
    
    # Also calculate metrics with default 0.5 threshold for comparison
    y_pred_default = (y_probs > 0.5).astype(int)
    
    # Calculate optimized metrics
    f1_macro_opt = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0)
    f1_micro_opt = f1_score(y_true, y_pred_optimized, average='micro', zero_division=0)
    f1_weighted_opt = f1_score(y_true, y_pred_optimized, average='weighted', zero_division=0)
    accuracy_opt = accuracy_score(y_true, y_pred_optimized)
    f1_per_label_opt = f1_score(y_true, y_pred_optimized, average=None, zero_division=0)
    
    # Calculate default (0.5) metrics for comparison
    f1_macro_default = f1_score(y_true, y_pred_default, average='macro', zero_division=0)
    f1_micro_default = f1_score(y_true, y_pred_default, average='micro', zero_division=0)
    accuracy_default = accuracy_score(y_true, y_pred_default)
    f1_per_label_default = f1_score(y_true, y_pred_default, average=None, zero_division=0)
    
    # ESG category names for better tracking
    esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
    
    # Create detailed metrics (primary metrics use optimized thresholds)
    metrics = {
        'f1_macro': f1_macro_opt,
        'f1_micro': f1_micro_opt, 
        'f1_weighted': f1_weighted_opt,
        'accuracy': accuracy_opt,
        # Default threshold metrics for comparison
        'f1_macro_default': f1_macro_default,
        'f1_micro_default': f1_micro_default,
        'accuracy_default': accuracy_default,
        # Improvement metrics
        'f1_macro_improvement': f1_macro_opt - f1_macro_default,
        'accuracy_improvement': accuracy_opt - accuracy_default,
    }
    
    # Add per-label F1 scores with meaningful names (optimized)
    for i, f1 in enumerate(f1_per_label_opt):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'f1_{label_name}'] = f1
    
    # Add per-label F1 scores for default thresholds (comparison)
    for i, f1 in enumerate(f1_per_label_default):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'f1_{label_name}_default'] = f1
        
    # Add per-label improvements
    for i, (f1_opt, f1_def) in enumerate(zip(f1_per_label_opt, f1_per_label_default)):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'f1_{label_name}_improvement'] = f1_opt - f1_def
    
    # Add optimal thresholds to metrics for logging
    for i, threshold in enumerate(optimal_thresholds):
        label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
        metrics[f'threshold_{label_name}'] = threshold
    
    # Log threshold optimization summary
    print(f"📊 Threshold Optimization Results:")
    print(f"   F1-macro: {f1_macro_default:.4f} → {f1_macro_opt:.4f} (+{f1_macro_opt-f1_macro_default:+.4f})")
    print(f"   Accuracy: {accuracy_default:.4f} → {accuracy_opt:.4f} (+{accuracy_opt-accuracy_default:+.4f})")
    
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
    
    # Class weighting configuration - ALWAYS ENABLED
    class_weight_method = config.get("class_weight_method", "balanced")
    
    # Calculate class weights
    print(f"⚖️ Calculating balanced class weights with method: {class_weight_method}")
    class_weights = calculate_class_weights(
        dataset["train"], 
        num_labels=num_labels, 
        method=class_weight_method
    )
    
    # Log class weights to MLflow
    if mlflow.active_run():
        mlflow.log_params({
            "use_class_weights": True,
            "class_weight_method": class_weight_method,
            "use_threshold_optimization": True,
        })
        # Log individual weights
        esg_labels = ['C1', 'C2', 'C3', 'C5', 'C8', 'C9', 'C10']
        for i, weight in enumerate(class_weights):
            label_name = esg_labels[i] if i < len(esg_labels) else f'label_{i}'
            mlflow.log_param(f"class_weight_{label_name}", float(weight))
    
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
        
        # ESG Trainer with balanced class weights - ALWAYS ENABLED
        class ESGTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = class_weights
                
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                # Always use balanced weighted loss
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
        
        print(f"   ✅ Balanced class weights + threshold optimization applied")
        
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
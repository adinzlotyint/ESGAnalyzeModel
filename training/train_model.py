import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from databricks.sdk import WorkspaceClient
from datasets import load_from_disk, Dataset
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import (
    LongformerForSequenceClassification,
    LongformerTokenizerFast,
    TrainingArguments,
    Trainer,
    LongformerConfig,
    default_data_collator,
    EvalPrediction,
)

# --- Constants ---
MLFLOW_EXPERIMENT_NAME = "ESGAnalyzeModel-Training"
CRITERIA_NAMES = [
    'c1_transition_plan',
    'c2_risk_management',
    'c4_boundaries',
    'c6_historical_data',
    'c7_intensity_metrics',
    'c8_targets_credibility',
]

# --- Custom Trainer for Weighted Loss ---
class ESGTrainer(Trainer):
    """
    A custom Trainer that overrides the loss function to support class weighting
    for multi-label classification, which is crucial for imbalanced datasets.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

def get_databricks_user_email() -> str:
    try:
        w = WorkspaceClient()
        current_user = w.current_user.me() 
        return current_user.user_name
    except Exception as e:
        print(f"⚠️  Could not automatically get Databricks user email: {e}")
        print("    Falling back to a generic experiment path.")
        return ''
    
    
# --- Helper Functions ---
def _load_config() -> dict:
    # Loads the main project configuration from the root directory.
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at: {config_path}")
        sys.exit(1)

def _calculate_class_weights(dataset: Dataset) -> torch.Tensor:
    # Calculates class weights for handling imbalanced datasets.
    labels = np.array(dataset['labels'])
    num_labels = labels.shape[1]
    pos_counts = np.sum(labels, axis=0)
    total_samples = len(labels)
    
    weights = []
    for count in pos_counts:
        # Formula: total_samples / (2 * num_positive_samples)
        # Add epsilon to avoid division by zero for labels with no positive examples
        weight = total_samples / (2 * count + 1e-6) if count > 0 else 1.0
        weights.append(weight)
        
    print(f"⚖️  Calculated class weights: {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float)

def _compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    # Computes metrics during evaluation using a simple 0.5 threshold.
    logits, labels = p
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs > 0.5).astype(int)
    
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    
    metrics = {'f1_macro': f1_macro, 'accuracy': accuracy}
    f1_per_label = f1_score(labels, preds, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_label):
        metrics[f'f1_{CRITERIA_NAMES[i]}'] = f1
        
    return metrics

def _optimize_thresholds(trainer: Trainer, eval_dataset: Dataset) -> np.ndarray:
    # Finds optimal F1-maximizing thresholds for each label on the validation set.
    print("\n🎯 Optimizing classification thresholds on validation set...")
    preds = trainer.predict(eval_dataset)
    logits, y_true = preds.predictions, preds.label_ids
    y_probs = 1 / (1 + np.exp(-logits))

    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.91, 0.01):
            y_pred = (y_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        optimal_thresholds.append(best_thresh)
        print(f"   - Optimal threshold for '{CRITERIA_NAMES[i]}': {best_thresh:.2f} (F1: {best_f1:.4f})")
        
    return np.array(optimal_thresholds)

def _evaluate_with_optimal_thresholds(trainer: Trainer, test_dataset: Dataset, thresholds: np.ndarray) -> Dict[str, float]:
    # Evaluates the model on the test set using the optimized thresholds.
    print("\n🔍 Final evaluation on test set with optimized thresholds...")
    preds = trainer.predict(test_dataset)
    logits, y_true = preds.predictions, preds.label_ids
    y_probs = 1 / (1 + np.exp(-logits))

    y_pred_optimal = np.zeros_like(y_probs)
    for i in range(y_true.shape[1]):
        y_pred_optimal[:, i] = (y_probs[:, i] >= thresholds[i]).astype(int)
        
    f1_macro = f1_score(y_true, y_pred_optimal, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_optimal)
    
    results = {'final_test_f1_macro': f1_macro, 'final_test_accuracy': accuracy}
    f1_per_label = f1_score(y_true, y_pred_optimal, average=None, zero_division=0)
    for i, f1 in enumerate(f1_per_label):
        results[f'final_test_f1_{CRITERIA_NAMES[i]}'] = f1
        
    return results

def _setup_environment(config: dict) -> Tuple:
    # Loads all necessary configurations, data, and models for training.
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(config.get("model_output_path", "models")) / f"esg-longformer-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 All artifacts will be saved to: {output_dir}")

    # Load tokenized dataset
    dataset_path = config.get("tokenizer_output_path")
    if not dataset_path or not Path(dataset_path).exists():
        print(f"❌ Tokenized dataset not found at: {dataset_path}")
        sys.exit(1)
    dataset = load_from_disk(dataset_path)
    print(f"📊 Loaded tokenized dataset from: {dataset_path}")

    # The number of labels is determined from the defined criteria.
    num_labels = len(CRITERIA_NAMES)
    print(f"📊 Using {num_labels} labels for classification: {CRITERIA_NAMES}")
    
    model_config = LongformerConfig.from_pretrained(
        config["model_name"],
        num_labels=num_labels,
        problem_type=config.get("problem_type", "multi_label_classification")
    )
    model = LongformerForSequenceClassification.from_pretrained(config["model_name"], config=model_config)

    training_args_dict = config.get("training_args", {})
    training_args_dict["output_dir"] = str(output_dir)
    training_args = TrainingArguments(**training_args_dict)

    class_weights = _calculate_class_weights(dataset['train'])
    
    return dataset, model, training_args, class_weights, output_dir

def _save_artifacts(trainer: Trainer, tokenizer: LongformerTokenizerFast, output_dir: Path):
    # Saves the final model and tokenizer.
    print(f"\n💾 Saving final model and tokenizer to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("✅ Artifacts saved successfully.")

# --- Main Execution ---
def main():
    # Main function to run the complete model training and evaluation pipeline.
    config = _load_config()
    
    try:
        user_email = get_databricks_user_email()
        
        if user_email:
            experiment_path = f"/Users/{user_email}/{MLFLOW_EXPERIMENT_NAME}"
        else:
            experiment_path = f"/Shared/{MLFLOW_EXPERIMENT_NAME}"
            
        print(f"🔧 Setting MLflow experiment to: {experiment_path}")
        mlflow.set_experiment(experiment_path)

        mlflow.start_run()
        mlflow.log_params(config.get("training_args", {}))
        mlflow.log_param("model_name", config["model_name"])

        dataset, model, training_args, class_weights, output_dir = _setup_environment(config)
        
        tokenizer = LongformerTokenizerFast.from_pretrained(config["model_name"])

        trainer = ESGTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=default_data_collator,
            compute_metrics=_compute_metrics,
            tokenizer=tokenizer,
            class_weights=class_weights,
        )

        print("\n🏃 Starting model training...")
        trainer.train()
        print("✅ Training finished.")

        # Post-training evaluation and logging
        optimal_thresholds = _optimize_thresholds(trainer, dataset['validation'])
        final_results = _evaluate_with_optimal_thresholds(trainer, dataset['test'], optimal_thresholds)
        
        # Log final metrics and thresholds to MLflow
        mlflow.log_metrics({f"optimal_threshold_{k}": v for k, v in zip(CRITERIA_NAMES, optimal_thresholds)})
        mlflow.log_metrics(final_results)

        # Print final summary
        print("\n✨ FINAL TEST RESULTS (with optimized thresholds):")
        for key, value in final_results.items():
            print(f"   - {key}: {value:.4f}")

        _save_artifacts(trainer, tokenizer, output_dir)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the training pipeline: {e}")
        raise e
    finally:
        if mlflow.active_run():
            print("\n🏁 Finalizing MLflow run.")
            mlflow.end_run()

if __name__ == "__main__":
    main()
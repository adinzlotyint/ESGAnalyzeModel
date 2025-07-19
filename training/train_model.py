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
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, hamming_loss
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
    print("Attempting to connect to Databricks to get user email...")
    w = WorkspaceClient()
    current_user = w.current_user.me() 
    print(f"✅ Successfully connected and found user: {current_user.user_name}")
    return current_user.user_name
    
    
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
    """
    Computes a comprehensive set of multi-label metrics during evaluation.
    Uses a standard 0.5 threshold.
    """
    logits, labels = p
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs > 0.5).astype(int)

    # Obliczenie kluczowych metryk
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    exact_match_ratio = accuracy_score(labels, preds)
    jaccard_macro = jaccard_score(labels, preds, average='macro', zero_division=0)
    h_loss = hamming_loss(labels, preds)

    metrics = {
        'f1_macro': f1_macro,
        'exact_match_ratio': exact_match_ratio,
        'jaccard_macro': jaccard_macro,
        'hamming_loss': h_loss
    }

    # Dodanie F1-score dla każdej etykiety indywidualnie
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
    exact_match_ratio = accuracy_score(y_true, y_pred_optimal)
    jaccard_macro = jaccard_score(y_true, y_pred_optimal, average='macro', zero_division=0)
    h_loss = hamming_loss(y_true, y_pred_optimal)
    
    results = {
        'final_test_f1_macro': f1_macro,
        'final_test_exact_match_ratio': exact_match_ratio,
        'final_test_jaccard_macro': jaccard_macro,
        'final_test_hamming_loss': h_loss
    }

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

def _save_artifacts(trainer: Trainer, tokenizer: LongformerTokenizerFast, output_dir: Path, thresholds: np.ndarray):
    # Saves the final model and tokenizer.
    print(f"\n💾 Saving final model and tokenizer to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    json.dump(thresholds.tolist(), open(output_dir / "thresholds.json", "w"))
    print("✅ Artifacts saved successfully.")

def aggregate_chunks_to_documents(predictions, labels, doc_ids):
    """
    Simple mean pooling aggregation of chunk predictions to document level.
    
    Args:
        predictions: Array of chunk predictions (probabilities)
        labels: Array of chunk labels 
        doc_ids: Array of document IDs for each chunk
        
    Returns:
        doc_predictions: Document-level predictions
        doc_labels: Document-level true labels
        unique_doc_ids: List of unique document IDs
    """
    import pandas as pd
    
    # Create dataframe for easy grouping
    df = pd.DataFrame({
        'doc_id': doc_ids,
        'predictions': predictions.tolist(),
        'labels': labels.tolist()
    })
    
    # Group by document and take max
    doc_results = df.groupby('doc_id').agg({
        'predictions': lambda x: np.max(np.array(x.tolist()), axis=0),
        'labels': lambda x: x.iloc[0]  # Labels are same for all chunks of same doc
    }).reset_index()
    
    doc_predictions = np.array(doc_results['predictions'].tolist())
    doc_labels = np.array(doc_results['labels'].tolist())
    unique_doc_ids = doc_results['doc_id'].tolist()
    
    return doc_predictions, doc_labels, unique_doc_ids

def evaluate_document_level(trainer: Trainer, dataset: Dataset, thresholds: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance at document level using mean pooling and optimized thresholds.
    """
    print("\n📊 Evaluating document-level performance...")
    
    # Get chunk predictions
    preds = trainer.predict(dataset)
    chunk_probs = 1 / (1 + np.exp(-preds.predictions))
    chunk_labels = preds.label_ids
    
    # Aggregate to document level
    doc_probs, doc_labels, doc_ids = aggregate_chunks_to_documents(
        chunk_probs, chunk_labels, dataset['doc_id']
    )
    
    # Apply threshold
    doc_predictions = np.zeros_like(doc_probs)
    for i in range(doc_labels.shape[1]):
        doc_predictions[:, i] = (doc_probs[:, i] >= thresholds[i]).astype(int)
    
    # Calculate metrics
    f1_macro = f1_score(doc_labels, doc_predictions, average='macro', zero_division=0)
    exact_match_ratio = accuracy_score(doc_labels, doc_predictions)
    jaccard_macro = jaccard_score(doc_labels, doc_predictions, average='macro', zero_division=0)
    h_loss = hamming_loss(doc_labels, doc_predictions)
    
    results = {
        'doc_f1_macro': f1_macro,
        'doc_exact_match_ratio': exact_match_ratio,
        'doc_jaccard_macro': jaccard_macro,
        'doc_hamming_loss': h_loss,
        'num_documents': len(doc_ids)
    }

    # Add per-criterion F1
    f1_per_criterion = f1_score(doc_labels, doc_predictions, average=None, zero_division=0)
    for i, criterion in enumerate(CRITERIA_NAMES):
        results[f'doc_f1_{criterion}'] = f1_per_criterion[i]

    print(f"Document-level F1 (macro): {f1_macro:.4f}")
    print(f"Document-level Exact Match Ratio: {exact_match_ratio:.4f}")
    print(f"Number of documents: {len(doc_ids)}")

    return results

# --- Main Execution ---
def main():
    """
    Main function to run the complete model training and evaluation pipeline.
    """
    config = _load_config()

    try:
        try:
            user_email = get_databricks_user_email()
            experiment_path = f"/Users/{user_email}/{MLFLOW_EXPERIMENT_NAME}"
        except Exception as e:
            print(f"\n❌ Critical error: Could not configure MLflow experiment path due to connection issue: {e}")
            print("Please check your Databricks token and host configuration. Aborting.")
            sys.exit(1)
        
        print(f"🔧 Setting MLflow experiment to: {experiment_path}")
        mlflow.set_experiment(experiment_path)

        with mlflow.start_run() as run:
            print(f"🚀 MLflow run started (ID: {run.info.run_id})")
            mlflow.log_params(config.get("training_args", {}))
            mlflow.log_param("model_name", config["model_name"])

            dataset, model, training_args, class_weights, output_dir = _setup_environment(config)
            
            # Check for doc_id column for document-level evaluation.
            if 'doc_id' not in dataset['validation'].features or 'doc_id' not in dataset['test'].features:
                print("⚠️  'doc_id' not found in validation/test sets. Document-level evaluation will be skipped.")
                evaluate_docs = False
            else:
                evaluate_docs = True

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

            # Perform chunk-level evaluation
            optimal_thresholds = _optimize_thresholds(trainer, dataset['validation'])
            chunk_results = _evaluate_with_optimal_thresholds(trainer, dataset['test'], optimal_thresholds)

            # Log metrics and thresholds to MLflow
            mlflow.log_metrics({f"optimal_threshold_{k}": v for k, v in zip(CRITERIA_NAMES, optimal_thresholds)})
            mlflow.log_metrics({f"chunk_{k}": v for k, v in chunk_results.items()})

            # UPDATED SUMMARY
            print("\n✨ FINAL CHUNK-LEVEL RESULTS (with optimized thresholds):")
            for key, value in chunk_results.items():
                print(f"   - {key.replace('final_test_', '')}: {value:.4f}")

            # Perform document-level evaluation
            if evaluate_docs:
                doc_results = evaluate_document_level(trainer, dataset['test'], optimal_thresholds)
                mlflow.log_metrics(doc_results)
                
                # UPDATED SUMMARY
                print("\n🏢 FINAL DOCUMENT-LEVEL RESULTS (with optimized thresholds):")
                for key, value in doc_results.items():
                    print(f"   - {key}: {value:.4f}")

            _save_artifacts(trainer, tokenizer, output_dir, optimal_thresholds)
            mlflow.log_artifacts(str(output_dir), artifact_path="model")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the training pipeline: {e}")
        raise e
    finally:
        if mlflow.active_run():
            print("\n🏁 Finalizing MLflow run.")
            mlflow.end_run()

if __name__ == "__main__":
    main()
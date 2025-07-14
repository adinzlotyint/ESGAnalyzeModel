from datasets import load_dataset, DatasetDict
import json
import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification

def load_config():
    """Load configuration from the main config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return None
        
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def main():
    """Create HuggingFace dataset from processed JSONL file with STRATIFIED split."""
    print("📊 Creating HuggingFace dataset...")
    
    config = load_config()
    if not config:
        return False
    
    final_jsonl_path = config.get("final_jsonl")
    hf_dataset_path = config.get("hf_dataset")
    
    if not final_jsonl_path or not os.path.exists(final_jsonl_path):
        print(f"❌ Final JSONL file not found: {final_jsonl_path}")
        print("💡 Please run conversion step first")
        return False
    
    print(f"📁 Loading data from: {final_jsonl_path}")
    
    try:
        dataset = load_dataset('json', data_files=final_jsonl_path, split='train')
        
        print(f"📊 Full dataset loaded: {len(dataset)} samples")
        print(f"🏷️  Sample features: {dataset.features}")

        # Stratyfikacja
        print("🧬 Performing Stratified Multi-Label Split (80/10/10)...")
        
        X = np.arange(len(dataset)).reshape(-1, 1)
        y = np.array(dataset['labels'])
        
        stratifier_train_temp = IterativeStratification(n_splits=1, order=1, sample_distribution_per_fold=[0.2, 0.8])

        temp_indices_list, train_indices_list = next(stratifier_train_temp.split(X, y))
        
        X_temp = X[temp_indices_list]
        y_temp = y[temp_indices_list]
        
        stratifier_val_test = IterativeStratification(n_splits=1, order=1, sample_distribution_per_fold=[0.5, 0.5])
        test_relative_indices, val_relative_indices = next(stratifier_val_test.split(X_temp, y_temp))

        train_indices = train_indices_list
        val_indices = temp_indices_list[val_relative_indices]
        test_indices = temp_indices_list[test_relative_indices]

        train_dataset = dataset.select(train_indices)
        validation_dataset = dataset.select(val_indices)
        test_dataset = dataset.select(test_indices)
        
        dataset_dict = DatasetDict({
            'train':      train_dataset,
            'validation': validation_dataset,
            'test':       test_dataset,
        })
        
        print(f"📊 Train samples: {len(dataset_dict['train'])}")
        print(f"📊 Validation samples: {len(dataset_dict['validation'])}")
        print(f"📊 Test samples: {len(dataset_dict['test'])}")

        print(f"💾 Saving dataset to: {hf_dataset_path}")
        dataset_dict.save_to_disk(hf_dataset_path)
        
        print(f"✅ Stratified HuggingFace dataset created successfully!")
        print(f"📂 Dataset saved to: {hf_dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
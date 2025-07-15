import json
import os
import sys
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict
from skmultilearn.model_selection import iterative_train_test_split

def load_config() -> dict:
    """
    Loads the main project configuration from the root directory.

    Returns:
        dict: The project configuration.
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at: {config_path}")
        sys.exit(1)

def load_and_split_data(jsonl_path: str) -> DatasetDict:
    """
    Loads data from a JSONL file and performs a stratified multi-label split
    into train, validation, and test sets (80/10/10).

    Args:
        jsonl_path (str): The path to the final processed JSONL data file.

    Returns:
        DatasetDict: A dictionary containing 'train', 'validation', and 'test' splits.
    """
    print(f"📁 Loading data from: {jsonl_path}")
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    print(f"📊 Full dataset loaded with {len(dataset)} samples.")

    print("🧬 Performing stratified multi-label split (80/10/10)...")
    # We use indices as a dummy X for the splitter.
    indices = np.arange(len(dataset)).reshape(-1, 1)
    labels = np.array(dataset['labels'])

    # First split: 80% train, 20% temporary (for validation and test)
    train_idx, _, temp_idx, temp_labels = iterative_train_test_split(
        indices, labels, test_size=0.2
    )

    # Second split: split the temporary set into 50% validation and 50% test
    val_idx, _, test_idx, _ = iterative_train_test_split(
        temp_idx, temp_labels, test_size=0.5
    )

    return DatasetDict({
        'train': dataset.select(train_idx.flatten()),
        'validation': dataset.select(val_idx.flatten()),
        'test': dataset.select(test_idx.flatten()),
    })

def save_dataset(dataset_dict: DatasetDict, save_path: str):
    """
    Saves a DatasetDict to disk and prints the final sample counts.

    Args:
        dataset_dict (DatasetDict): The dataset to save.
        save_path (str): The directory where the dataset will be saved.
    """
    print("\n📊 Final split sizes:")
    print(f"   - Train samples:      {len(dataset_dict['train'])}")
    print(f"   - Validation samples: {len(dataset_dict['validation'])}")
    print(f"   - Test samples:       {len(dataset_dict['test'])}")

    print(f"\n💾 Saving dataset to: {save_path}")
    dataset_dict.save_to_disk(save_path)
    print(f"✅ Stratified HuggingFace dataset created successfully at: {save_path}")

def main():
    """
    Main script to create and save a stratified HuggingFace dataset.
    """
    print("\n🚀 Starting HuggingFace dataset creation...")
    
    config = load_config()
    final_jsonl_path = config.get("final_jsonl")
    hf_dataset_path = config.get("hf_dataset")

    if not final_jsonl_path or not os.path.exists(final_jsonl_path):
        print(f"❌ Final data file not found at: {final_jsonl_path}")
        print("💡 Please run the 'conversion' step first.")
        sys.exit(1)
    
    try:
        stratified_dataset = load_and_split_data(final_jsonl_path)
        save_dataset(stratified_dataset, hf_dataset_path)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during dataset creation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from datasets import load_dataset, DatasetDict
import json
import os

def load_config():
    """Load configuration from the main config.json file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file {config_path} not found!")
        return None
        
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)

def main():
    """Create HuggingFace dataset from processed JSONL file."""
    print("📊 Creating HuggingFace dataset...")
    
    config = load_config()
    if not config:
        return False
    
    # Get paths from config
    final_jsonl_path = config.get("final_jsonl")
    hf_dataset_path = config.get("hf_dataset")
    
    if not final_jsonl_path or not os.path.exists(final_jsonl_path):
        print(f"❌ Final JSONL file not found: {final_jsonl_path}")
        print("💡 Please run conversion step first")
        return False
    
    print(f"📁 Loading data from: {final_jsonl_path}")
    
    try:
        # Load dataset from JSONL file
        dataset = load_dataset('json', data_files=final_jsonl_path, split='train')
        
        print(f"📊 Dataset loaded: {len(dataset)} samples")
        print(f"🏷️  Sample features: {dataset.features}")
        
        # Split into train/validation (80/20)
        print("✂️  Splitting dataset (80% train, 20% validation)...")
        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': dataset_split['train'],
            'validation': dataset_split['test']
        })
        
        print(f"📊 Train samples: {len(dataset_dict['train'])}")
        print(f"📊 Validation samples: {len(dataset_dict['validation'])}")
        
        # Save to disk
        print(f"💾 Saving dataset to: {hf_dataset_path}")
        dataset_dict.save_to_disk(hf_dataset_path)
        
        print(f"✅ HuggingFace dataset created successfully!")
        print(f"📂 Dataset saved to: {hf_dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
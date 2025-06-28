from huggingface_hub import snapshot_download
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
    """Download model snapshot from HuggingFace Hub."""
    print("📥 Starting model snapshot download...")
    
    config = load_config()
    if not config:
        return False
    
    model_name = config.get("model_name")
    if not model_name:
        print("❌ Model name not found in configuration!")
        return False
    
    print(f"🤗 Downloading model: {model_name}")
    
    try:
        snapshot_download(
            repo_id=model_name, 
            repo_type="model",
            cache_dir=".cache/huggingface"  # Optional: specify cache directory
        )
        print(f"✅ Model {model_name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
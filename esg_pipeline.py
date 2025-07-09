import os, sys, json, shutil, subprocess
from pathlib import Path
from datetime import datetime

class ESGPipeline:
    def __init__(self, config_path="config.json"):
        """Initialize the pipeline with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file {self.config_path} not found!")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            sys.exit(1)
    
    def print_step(self, step_num, total_steps, description):
        """Print formatted step information."""
        print(f"\n{'='*60}")
        print(f"🚀 STEP {step_num}/{total_steps}: {description}")
        print(f"{'='*60}")
    
    def check_file_exists(self, filepath, description="File"):
        """Check if a file exists and print status."""
        if os.path.exists(filepath):
            print(f"✅ {description} found: {filepath}")
            return True
        else:
            print(f"❌ {description} not found: {filepath}")
            return False
    
    def clean_directory(self, directory_path, description="Directory"):
        """Remove directory if it exists."""
        if os.path.exists(directory_path):
            print(f"🧹 Cleaning existing {description}: {directory_path}")
            shutil.rmtree(directory_path)
            print(f"✅ {description} cleaned")
        else:
            print(f"ℹ️  {description} doesn't exist, nothing to clean")
    
    def create_directory(self, directory_path, description="Directory"):
        """Create directory if it doesn't exist."""
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 {description} ready: {directory_path}")
    
    def run_script(self, script_name, description):
        """Run a Python script and handle errors."""
        print(f"▶️  Running {script_name}...")
        try:
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, check=True, 
                                  encoding='utf-8', errors='replace')
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("📄 Output:", result.stdout.strip())
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed!")
            print(f"Error: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"❌ Script {script_name} not found!")
            return False
    
    def step_conversion(self, force_clean=False):
        """Step 1: Data conversion and cleaning."""
        self.print_step(1, 4, "DATA CONVERSION & CLEANING")
        
        # Check if intermediate files exist and clean if requested
        intermediate_files = [
            self.config.get("converted_csv_path"),
            self.config.get("cleaned_jsonl_path"),
            self.config.get("merged_jsonl_path"),
            self.config.get("final_jsonl")
        ]
        
        if force_clean:
            for file_path in intermediate_files:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🧹 Removed: {file_path}")
        
        # Check if final output already exists
        final_output = self.config.get("final_jsonl")
        if final_output and os.path.exists(final_output) and not force_clean:
            print(f"ℹ️  Final dataset already exists: {final_output}")
            response = input("Do you want to skip conversion? (y/n): ").lower()
            if response == 'y':
                print("⏭️  Skipping conversion step")
                return True
        
        # Check input files
        raw_csv = self.config.get("raw_csv_path")
        raw_jsonl = self.config.get("raw_jsonl_path")
        
        if not self.check_file_exists(raw_csv, "Raw CSV file"):
            return False
        if not self.check_file_exists(raw_jsonl, "Raw JSONL file"):
            return False
        
        return self.run_script("scripts/conversion.py", "Data conversion")
    
    def step_create_dataset(self, force_clean=False):
        """Step 2: Create HuggingFace dataset."""
        self.print_step(2, 4, "HUGGINGFACE DATASET CREATION")
        
        hf_dataset_path = self.config.get("hf_dataset")
        if force_clean:
            self.clean_directory(hf_dataset_path, "HuggingFace dataset directory")
        
        if hf_dataset_path and os.path.exists(hf_dataset_path) and not force_clean:
            print(f"ℹ️  HuggingFace dataset already exists: {hf_dataset_path}")
            response = input("Do you want to recreate it? (y/n): ").lower()
            if response == 'n':
                print("⏭️  Skipping dataset creation")
                return True
            else:
                self.clean_directory(hf_dataset_path, "HuggingFace dataset directory")
        
        return self.run_script("training/hf_create_dataset.py", "HuggingFace dataset creation")
    
    def step_download_snapshot(self):
        """Step 3: Download model snapshot."""
        self.print_step(3, 4, "MODEL SNAPSHOT DOWNLOAD")
        
        model_name = self.config.get("model_name")
        print(f"📥 Downloading model: {model_name}")
        
        return self.run_script("training/download_snapshot.py", "Model snapshot download")
    
    def step_tokenization(self, force_clean=False):
        """Step 4: Tokenize dataset."""
        self.print_step(4, 4, "DATASET TOKENIZATION")
        
        tokenizer_output = self.config.get("tokenizer_output_path")
        if force_clean:
            self.clean_directory(tokenizer_output, "Tokenizer output directory")
        
        if tokenizer_output and os.path.exists(tokenizer_output) and not force_clean:
            print(f"ℹ️  Tokenized dataset already exists: {tokenizer_output}")
            response = input("Do you want to retokenize? (y/n): ").lower()
            if response == 'n':
                print("⏭️  Skipping tokenization")
                return True
            else:
                self.clean_directory(tokenizer_output, "Tokenizer output directory")
        
        return self.run_script("training/tokenize_dataset.py", "Dataset tokenization")
    
    def step_training(self, method="basic"):
        """Step 5: Train the model with specified method."""
        self.print_step(5, 5, f"MODEL TRAINING ({method.upper()})")
        
        model_output = self.config.get("model_output_path")
        print(f"🎯 Training model using {method} method")
        print(f"📁 Output will be saved to: {model_output}")
        print(f"⏰ Timestamp: {self.timestamp}")
        
        # Map training methods to scripts
        training_scripts = {
            "basic": "training/train_model.py",
            "hierarchical": "training/hierarchical_model.py", 
            "positional": "training/positional_weighting.py",
            "advanced": "training/advanced_training.py"
        }
        
        script_path = training_scripts.get(method, "training/train_model.py")
        return self.run_script(script_path, f"{method.capitalize()} model training")
    
    def run_pipeline(self, steps=None, force_clean=False, skip_download=False):
        """Run the complete pipeline or specific steps."""
        # Default pipeline excludes training - run training separately for better visibility
        default_steps = ['conversion', 'dataset', 'download', 'tokenize']
        all_steps = ['conversion', 'dataset', 'download', 'tokenize', 'train']
        
        if steps is None:
            steps = default_steps
        
        print(f"🎬 Starting ESG Model Training Pipeline")
        print(f"📅 Timestamp: {self.timestamp}")
        print(f"📋 Steps to execute: {', '.join(steps)}")
        print(f"🧹 Force clean: {force_clean}")
        
        # Verify configuration
        print(f"\n📋 Configuration Summary:")
        print(f"   Model: {self.config.get('model_name')}")
        print(f"   Labels: {self.config.get('num_labels')}")
        print(f"   Problem type: {self.config.get('problem_type')}")
        
        success = True
        
        try:
            if 'conversion' in steps:
                if not self.step_conversion(force_clean):
                    success = False
                    
            if 'dataset' in steps and success:
                if not self.step_create_dataset(force_clean):
                    success = False
                    
            if 'download' in steps and success and not skip_download:
                if not self.step_download_snapshot():
                    success = False
                    
            if 'tokenize' in steps and success:
                if not self.step_tokenization(force_clean):
                    success = False
                    
            if 'train' in steps and success:
                if not self.step_training():
                    success = False
            
            if success:
                print(f"\n🎉 Pipeline completed successfully!")
                print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if 'train' not in steps:
                    print(f"\n💡 Ready for training! Run:")
                    print(f"   python main.py --steps train")
                    print(f"   OR")
                    print(f"   python training/train_model.py")
            else:
                print(f"\n❌ Pipeline failed at one of the steps")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            sys.exit(1)
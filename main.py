"""
Main CLI script for ESG model training pipeline.
Orchestrates the entire process from data conversion to model training.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

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
    
    def step_training(self):
        """Step 5: Train the model."""
        self.print_step(5, 5, "MODEL TRAINING")
        
        model_output = self.config.get("model_output_path")
        print(f"🎯 Training model, output will be saved to: {model_output}")
        print(f"⏰ Timestamp: {self.timestamp}")
        
        return self.run_script("training/train_model.py", "Model training")
    
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

def show_interactive_menu():
    """Display interactive menu and get user choice."""
    print("\n" + "="*60)
    print("🤖 ESG MODEL TRAINING PIPELINE")
    print("="*60)
    print("Select an option by typing the number:")
    print()
    print("📊 DATA PIPELINE:")
    print("  1. Run full data pipeline (conversion → dataset → download → tokenize)")
    print("  2. Run data conversion only")
    print("  3. Run dataset creation only") 
    print("  4. Run model download only")
    print("  5. Run tokenization only")
    print()
    print("🎯 TRAINING:")
    print("  6. Run training only (via pipeline)")
    print("  7. Run training directly (full progress bars)")
    print()
    print("🧹 MAINTENANCE:")
    print("  8. Clean all intermediate files and run full pipeline")
    print("  9. Show current configuration")
    print("  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-9): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0-9.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def execute_menu_choice(choice, pipeline):
    """Execute the selected menu option."""
    if choice == '0':
        print("👋 Goodbye!")
        return
    
    elif choice == '1':
        print("🚀 Running full data pipeline...")
        pipeline.run_pipeline()
    
    elif choice == '2':
        print("🔄 Running data conversion only...")
        pipeline.run_pipeline(steps=['conversion'])
    
    elif choice == '3':
        print("📊 Running dataset creation only...")
        pipeline.run_pipeline(steps=['dataset'])
    
    elif choice == '4':
        print("📥 Running model download only...")
        pipeline.run_pipeline(steps=['download'])
    
    elif choice == '5':
        print("🔤 Running tokenization only...")
        pipeline.run_pipeline(steps=['tokenize'])
    
    elif choice == '6':
        print("🎯 Running training via pipeline...")
        pipeline.run_pipeline(steps=['train'])
    
    elif choice == '7':
        print("🎯 Running training directly...")
        print("💡 Launching direct training script for full progress visibility...")
        try:
            result = subprocess.run([sys.executable, "training/train_model.py"], 
                                  encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("✅ Training completed successfully!")
            else:
                print("❌ Training failed!")
        except Exception as e:
            print(f"❌ Error running training: {e}")
    
    elif choice == '8':
        print("🧹 Cleaning all files and running full pipeline...")
        pipeline.run_pipeline(force_clean=True)
    
    elif choice == '9':
        print("📋 Current Configuration:")
        print(json.dumps(pipeline.config, indent=2, ensure_ascii=False))

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="ESG Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive menu
  python main.py --steps train            # Run training only  
  python main.py --steps conversion dataset # Run only specific steps
  python main.py --force-clean            # Clean all intermediate files
  python main.py --skip-download          # Skip model download
  python main.py --config custom.json    # Use custom config file
  
Training:
  python training/train_model.py         # Direct training with progress bars
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['conversion', 'dataset', 'download', 'tokenize', 'train'],
        help='Specific steps to run (default: interactive menu)'
    )
    
    parser.add_argument(
        '--force-clean',
        action='store_true',
        help='Force clean all intermediate files and directories'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip model snapshot download (useful if already downloaded)'
    )
    
    parser.add_argument(
        '--list-config',
        action='store_true',
        help='Show current configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ESGPipeline(args.config)
    
    # List configuration if requested
    if args.list_config:
        print(json.dumps(pipeline.config, indent=2, ensure_ascii=False))
        return
    
    # If no arguments provided, show interactive menu
    if not any([args.steps, args.force_clean, args.skip_download]):
        choice = show_interactive_menu()
        execute_menu_choice(choice, pipeline)
    else:
        # Run with command-line arguments
        pipeline.run_pipeline(
            steps=args.steps,
            force_clean=args.force_clean,
            skip_download=args.skip_download
        )

if __name__ == "__main__":
    main()
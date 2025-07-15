import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

class ESGPipeline:
    """
    Orchestrates the data preparation pipeline for the ESGAnalyzeModel.

    This class manages the sequence of data processing steps, including
    conversion, cleaning, dataset creation, model downloading, and tokenization.
    It is configured via a JSON file and handles intermediate file management,
    logging, and error reporting.
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the pipeline with a specified configuration.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _load_config(self) -> dict:
        # Loads the configuration from a JSON file.
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            sys.exit(1)

    def _print_step_header(self, step_num: int, total_steps: int, description: str):
        # Prints a formatted header for a pipeline step.
        print(f"\n{'='*60}")
        print(f"🚀 STEP {step_num}/{total_steps}: {description}")
        print(f"{'='*60}")

    def _run_script(self, script_path: str, description: str) -> bool:
        """
        Runs a Python script as a subprocess and handles its execution.

        Args:
            script_path (str): The path to the Python script to execute.
            description (str): A brief description of the script's purpose.

        Returns:
            bool: True if the script ran successfully, False otherwise.
        """
        print(f"▶️  Executing script: {script_path}...")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, check=True,
                encoding='utf-8', errors='replace'
            )
            print(f"✅ {description} completed successfully.")
            if result.stdout:
                # Print script output for better traceability.
                print("   └── Script output:", result.stdout.strip().replace("\n", "\n   "))
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed!")
            print(f"   └── Error output:\n{e.stderr.strip()}")
            return False
        except FileNotFoundError:
            print(f"❌ Script not found: {script_path}")
            return False

    def _clean_directory(self, dir_path: str, description: str):
        # Removes a directory and all its contents if it exists.
        if os.path.exists(dir_path):
            print(f"🧹 Cleaning existing {description.lower()}: {dir_path}")
            shutil.rmtree(dir_path)
            print(f"✅ {description} cleaned.")

    def _handle_existing_output(self, path: str, name: str, action: str) -> bool:
        """
        Checks if an output path exists and asks the user whether to skip the step.

        Returns:
            bool: True if the step should be skipped, False otherwise.
        """
        if path and os.path.exists(path):
            print(f"ℹ️  {name} already exists at: {path}")
            response = input(f"Do you want to {action}? (y/n): ").lower()
            if response == 'n':
                print(f"⏭️  Skipping {name.lower()} creation.")
                return True
            else:
                if os.path.isdir(path):
                    self._clean_directory(path, name)
                else:
                    os.remove(path)
                    print(f"🧹 Removed old file: {path}")
        return False
        
    def step_conversion(self, force_clean: bool) -> bool:
        """Step 1: Convert and clean raw data."""
        self._print_step_header(1, 4, "DATA CONVERSION & CLEANING")
        if force_clean:
            for key in ["converted_csv_path", "cleaned_jsonl_path", "merged_jsonl_path", "final_jsonl"]:
                if self.config.get(key) and os.path.exists(self.config[key]):
                    os.remove(self.config[key])
                    print(f"🧹 Removed intermediate file: {self.config[key]}")
        
        if not force_clean and self._handle_existing_output(self.config['final_jsonl'], "Final dataset file", "recreate it"):
            return True

        return self._run_script("scripts/conversion.py", "Data conversion")

    def step_create_dataset(self, force_clean: bool) -> bool:
        """Step 2: Create a HuggingFace dataset."""
        self._print_step_header(2, 4, "HUGGINGFACE DATASET CREATION")
        if force_clean:
            self._clean_directory(self.config['hf_dataset'], "HuggingFace dataset directory")
        elif self._handle_existing_output(self.config['hf_dataset'], "HuggingFace dataset", "recreate it"):
            return True

        return self._run_script("training/hf_create_dataset.py", "HuggingFace dataset creation")

    def step_download_snapshot(self) -> bool:
        """Step 3: Download the model snapshot from Hugging Face Hub."""
        self._print_step_header(3, 4, "MODEL SNAPSHOT DOWNLOAD")
        print(f"📥 Model to download: {self.config.get('model_name')}")
        return self._run_script("training/download_snapshot.py", "Model snapshot download")

    def step_tokenization(self, force_clean: bool) -> bool:
        """Step 4: Tokenize the dataset."""
        self._print_step_header(4, 4, "DATASET TOKENIZATION")
        if force_clean:
            self._clean_directory(self.config['tokenizer_output_path'], "Tokenized dataset directory")
        elif self._handle_existing_output(self.config['tokenizer_output_path'], "Tokenized dataset", "re-tokenize it"):
            return True

        return self._run_script("training/tokenize_dataset.py", "Dataset tokenization")

    def run_pipeline(self, steps: list = None, force_clean: bool = False, skip_download: bool = False):
        """
        Runs the data preparation pipeline.

        Args:
            steps (list, optional): A list of specific steps to run. 
                                    Defaults to all data preparation steps.
            force_clean (bool): If True, cleans all intermediate files before starting.
            skip_download (bool): If True, skips the model download step.
        """
        pipeline_steps = {
            'conversion': self.step_conversion,
            'dataset': self.step_create_dataset,
            'download': self.step_download_snapshot,
            'tokenize': self.step_tokenization,
        }
        
        if steps is None:
            steps_to_run = list(pipeline_steps.keys())
        else:
            steps_to_run = steps

        print(f"\n🎬 Starting ESG Data Preparation Pipeline")
        print(f"   Timestamp: {self.timestamp}")
        print(f"   Steps to execute: {', '.join(steps_to_run)}")
        print(f"   Force clean: {force_clean}")

        try:
            for step_name in steps_to_run:
                if step_name == 'download' and skip_download:
                    print("\n⏭️  Skipping model download as requested.")
                    continue
                
                step_func = pipeline_steps.get(step_name)
                if not step_func(force_clean=force_clean):
                    print(f"\n❌ Pipeline failed at step: '{step_name}'. Aborting.")
                    sys.exit(1)

            print(f"\n🎉 Data pipeline completed successfully!")
            print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\n💡 Ready for training! You can now run the model training step from the main menu.")

        except KeyboardInterrupt:
            print(f"\n\n⚠️ Pipeline interrupted by user. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ An unexpected error occurred in the pipeline: {e}")
            sys.exit(1)
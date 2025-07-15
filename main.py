import os
import sys
import json
import argparse
import subprocess
import threading
import webbrowser
import time
from pathlib import Path

from esg_pipeline import ESGPipeline

# Set UTF-8 encoding for Windows compatibility to ensure proper character display.
if os.name == 'nt':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def show_interactive_menu():
    """
    Displays the interactive main menu and retrieves the user's choice.
    """
    print("\n" + "="*60)
    print("🤖 ESGAnalyzeModel Training Pipeline")
    print("="*60)
    print("Select an option by typing its number:")
    print()
    print("📊 DATA PIPELINE:")
    print("  1. Run full data pipeline (conversion → dataset → download → tokenize)")
    print("  2. Run data conversion only")
    print("  3. Run dataset creation only")
    print("  4. Run model download only")
    print("  5. Run tokenization only")
    print()
    print("🎯 TRAINING:")
    print("  6. Train ESG model")
    print("  7. Launch MLflow UI")
    print()
    print("🔍 INFERENCE:")
    print("  8. Test model inference (interactive mode)")
    print()
    print("🧹 MAINTENANCE:")
    print("  9. Clean intermediate files and run full data pipeline")
    print("  10. Show current configuration")
    print("  0. Exit")
    print()

    while True:
        try:
            choice = input("Enter your choice (0-10): ").strip()
            if choice in map(str, range(11)):
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 10.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def execute_menu_choice(choice: str, pipeline: ESGPipeline):
    """
    Executes the action corresponding to the user's menu choice.

    Args:
        choice (str): The user's selected option number.
        pipeline (ESGPipeline): An instance of the data pipeline orchestrator.
    """
    if choice == '0':
        print("👋 Goodbye!")
        return

    action_map = {
        '1': ("🚀 Running full data pipeline...", lambda p: p.run_pipeline()),
        '2': ("🔄 Running data conversion only...", lambda p: p.run_pipeline(steps=['conversion'])),
        '3': ("📊 Running dataset creation only...", lambda p: p.run_pipeline(steps=['dataset'])),
        '4': ("📥 Running model download only...", lambda p: p.run_pipeline(steps=['download'])),
        '5': ("🔤 Running tokenization only...", lambda p: p.run_pipeline(steps=['tokenize'])),
        '9': ("🧹 Cleaning files and running full pipeline...", lambda p: p.run_pipeline(force_clean=True)),
    }

    if choice in action_map:
        print(action_map[choice][0])
        action_map[choice][1](pipeline)
    elif choice == '6':
        print("🎯 Training ESG model...")
        print("📊 MLflow tracking will be automatically enabled.")
        print("💡 After training, use option 7 to view results in the MLflow UI.")
        try:
            # We run training in a subprocess to ensure a clean environment.
            subprocess.run([sys.executable, "training/train_model.py"], check=True, encoding='utf-8', errors='replace')
            print("\n✅ ESG model training completed successfully!")
            print("📊 Training metrics have been logged to MLflow.")
            response = input("Do you want to open the MLflow UI now? (y/n): ").lower()
            if response == 'y':
                launch_mlflow_ui()
        except subprocess.CalledProcessError:
            print("❌ ESG model training failed with an error.")
        except Exception as e:
            print(f"❌ An unexpected error occurred during training: {e}")
    elif choice == '7':
        launch_mlflow_ui()
    elif choice == '8':
        print("🔍 Testing model inference (interactive mode)...")
        try:
            subprocess.run([sys.executable, "scripts/inference.py", "--interactive"], check=True, encoding='utf-8', errors='replace')
            print("✅ Interactive inference completed successfully.")
        except subprocess.CalledProcessError:
            print("❌ Interactive inference failed with an error.")
        except Exception as e:
            print(f"❌ An unexpected error occurred during inference: {e}")
    elif choice == '10':
        print("📋 Current Configuration:")
        print(json.dumps(pipeline.config, indent=2, ensure_ascii=False))


def launch_mlflow_ui():
    # Launches the MLflow UI in a separate background thread and opens it in a web browser.
    print("🚀 Launching MLflow UI server...")

    mlruns_path = Path("mlruns")
    if not mlruns_path.exists():
        print("⚠️ No MLflow experiments found.")
        print("💡 Run training first to generate MLflow data.")
        response = input("Do you want to create an empty MLflow setup? (y/n): ").lower()
        if response == 'y':
            mlruns_path.mkdir(exist_ok=True)
            print("📁 'mlruns' directory created.")
        else:
            return

    def start_mlflow_server():
        # Command to start the MLflow UI server.
        cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", "./mlruns", "--host", "127.0.0.1", "--port", "5000"]
        subprocess.run(cmd, capture_output=True, text=True)

    try:
        # Run MLflow in a daemon thread so it doesn't block the main program.
        mlflow_thread = threading.Thread(target=start_mlflow_server, daemon=True)
        mlflow_thread.start()

        print("⏳ Waiting for the MLflow server to start...")
        time.sleep(3)

        mlflow_url = "http://127.0.0.1:5000"
        print(f"🌐 Opening MLflow UI at: {mlflow_url}")

        try:
            webbrowser.open(mlflow_url)
            print("\n✅ MLflow UI launched successfully!")
            print("📊 You can now browse your experiments and metrics.")
            print("⚠️ Keep this terminal open to keep the MLflow server running.")
            print("   Press Ctrl+C to stop the server when you are finished.")
            print("\n🔄 MLflow server is running in the background...")
            input("   Press Enter to return to the main menu (the server will continue to run).")
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"🌐 Please open this URL manually: {mlflow_url}")
            input("   Press Enter to continue...")

    except FileNotFoundError:
        print("❌ The 'mlflow' package appears to be not installed.")
        print("💡 Install it using: pip install mlflow>=2.0.0")
    except Exception as e:
        print(f"❌ An error occurred while launching MLflow UI: {e}")
        print("💡 You can try to launch it manually: mlflow ui --backend-store-uri ./mlruns")

def main():
    """
    Main entry point for the ESGAnalyzeModel pipeline command-line interface.

    This script serves as a dual-mode orchestrator:
    1. Interactive Mode: If run without arguments, it displays a user-friendly menu
       for executing different stages of the pipeline.
    2. Command-Line Mode: Allows for programmatic execution of specific pipeline
       steps, useful for automation and scripting.
    """
    parser = argparse.ArgumentParser(
        description="ESGAnalyzeModel Training and Data Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Show interactive menu
  python main.py --steps conversion dataset # Run only specific steps
  python main.py --force-clean            # Clean all intermediate files before running
  python main.py --skip-download          # Skip model download step
  python main.py --config custom.json     # Use a custom configuration file
  python main.py --list-config            # Display current config and exit

Direct script execution:
  python training/train_model.py          # Run model training directly (with MLflow)
  mlflow ui --backend-store-uri ./mlruns  # Manually launch MLflow UI
  python scripts/inference.py             # Run inference on sample data
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to the JSON configuration file (default: config.json)'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['conversion', 'dataset', 'download', 'tokenize'],
        help='Specific data pipeline steps to run. Training is invoked separately.'
    )
    parser.add_argument(
        '--force-clean',
        action='store_true',
        help='Force-clean all intermediate files and directories before execution.'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip the model snapshot download step (useful if already downloaded).'
    )
    parser.add_argument(
        '--list-config',
        action='store_true',
        help='Display the current configuration and exit.'
    )

    args = parser.parse_args()
    pipeline = ESGPipeline(config_path=args.config)

    if args.list_config:
        print(json.dumps(pipeline.config, indent=2, ensure_ascii=False))
        return

    # If no specific command-line arguments are given, run the interactive menu.
    if not any([args.steps, args.force_clean, args.skip_download]):
        while True:
            choice = show_interactive_menu()
            execute_menu_choice(choice, pipeline)
            if choice == '0':
                break
    else:
        # Otherwise, run the pipeline with the specified command-line arguments.
        pipeline.run_pipeline(
            steps=args.steps,
            force_clean=args.force_clean,
            skip_download=args.skip_download
        )

if __name__ == "__main__":
    main()
import os, sys, json, argparse, subprocess
from esg_pipeline import ESGPipeline
from pathlib import Path

# Set UTF-8 encoding for Windows compatibility
if os.name == 'nt':  # Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

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
    print("  6. Train ESG model")
    print("  7. Launch MLflow UI")
    print()
    print("🔍 INFERENCE:")
    print("  8. Test PyTorch inference (interactive)")
    print()
    print("🧹 MAINTENANCE:")
    print("  9. Clean all intermediate files and run full pipeline")
    print("  10. Show current configuration")
    print("  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-10): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                return choice
            else:
                print("❌ Invalid choice. Please enter a number between 0-10.")
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
        print("🎯 Training ESG model...")
        print("📊 MLflow tracking will be automatically enabled")
        print("💡 After training, use option 7 to view results in MLflow UI")
        try:
            result = subprocess.run([sys.executable, "training/train_model.py"], 
                                  encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("✅ ESG model training completed successfully!")
                print("📊 Training metrics logged to MLflow")
                response = input("Do you want to open MLflow UI now? (y/n): ").lower()
                if response == 'y':
                    launch_mlflow_ui()
            else:
                print("❌ ESG model training failed!")
        except Exception as e:
            print(f"❌ Error running training: {e}")
    
    elif choice == '7':
        print("📊 Launching MLflow UI...")
        launch_mlflow_ui()
    
    elif choice == '8':
        print("🔍 Testing PyTorch inference (interactive)...")
        try:
            result = subprocess.run([sys.executable, "scripts/inference.py", "--interactive"], 
                                  encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("✅ Interactive inference completed!")
            else:
                print("❌ Interactive inference failed!")
        except Exception as e:
            print(f"❌ Error running interactive inference: {e}")
    
    elif choice == '9':
        print("🧹 Cleaning all files and running full pipeline...")
        pipeline.run_pipeline(force_clean=True)
    
    elif choice == '10':
        print("📋 Current Configuration:")
        print(json.dumps(pipeline.config, indent=2, ensure_ascii=False))

def launch_mlflow_ui():
    """Launch MLflow UI in a separate process."""
    import threading
    import webbrowser
    import time
    
    print("🚀 Starting MLflow UI server...")
    
    # Check if mlruns directory exists
    mlruns_path = Path("mlruns")
    if not mlruns_path.exists():
        print("⚠️ No MLflow experiments found yet.")
        print("💡 Run training first to generate MLflow data.")
        response = input("Do you want to create an empty MLflow setup? (y/n): ").lower()
        if response == 'y':
            mlruns_path.mkdir(exist_ok=True)
            print("📁 MLflow directory created.")
        else:
            return
    
    try:
        # Start MLflow UI in background
        def start_mlflow():
            cmd = [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", "./mlruns", "--host", "127.0.0.1", "--port", "5000"]
            subprocess.run(cmd, capture_output=True, text=True)
        
        mlflow_thread = threading.Thread(target=start_mlflow, daemon=True)
        mlflow_thread.start()
        
        # Give MLflow time to start
        print("⏳ Starting MLflow server...")
        time.sleep(3)
        
        # Open browser
        mlflow_url = "http://127.0.0.1:5000"
        print(f"🌐 Opening MLflow UI: {mlflow_url}")
        
        try:
            webbrowser.open(mlflow_url)
            print("✅ MLflow UI launched successfully!")
            print("📊 You can now view your experiments and metrics")
            print("⚠️ Keep this terminal open to maintain the MLflow server")
            print("Press Ctrl+C to stop the MLflow server when done")
            
            # Keep the server running
            print("\n🔄 MLflow server is running...")
            print("Press Enter to return to main menu (server will continue in background)")
            input()
            
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"🌐 Manually open: {mlflow_url}")
            print("Press Enter to continue...")
            input()
            
    except FileNotFoundError:
        print("❌ MLflow not installed!")
        print("💡 Install with: pip install mlflow>=2.0.0")
    except Exception as e:
        print(f"❌ Error launching MLflow UI: {e}")
        print("💡 Try manually: mlflow ui --backend-store-uri ./mlruns")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="ESG Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive menu
  python main.py --steps conversion dataset # Run only specific steps
  python main.py --force-clean            # Clean all intermediate files
  python main.py --skip-download          # Skip model download
  python main.py --config custom.json    # Use custom config file
  
Training:
  python training/train_model.py         # ESG model training (with MLflow)
  
MLflow:
  mlflow ui --backend-store-uri ./mlruns  # Manual MLflow UI launch
  
Inference:
  python scripts/inference.py --interactive    # Interactive inference
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
        choices=['conversion', 'dataset', 'download', 'tokenize'],
        help='Specific pipeline steps to run (training done via menu)'
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
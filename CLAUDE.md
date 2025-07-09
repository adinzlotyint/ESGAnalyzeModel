# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESGAnalyzeModel is a Polish ESG (Environmental, Social, Governance) text classification system that trains transformer models to analyze corporate sustainability reports. The project processes 393 Polish corporate reports with expert annotations across 7 ESG criteria to create a multi-label classification model using Polish Longformer.

## Data Processing Pipeline

The codebase follows a sequential data processing pipeline:

1. **Raw Data**: Corporate reports in `data/reports/` (393 .txt files)
2. **Expert Annotations**: `data/expert_scores.csv` and `data/rule_based_scores.csv`
3. **Data Cleaning**: Scripts in `scripts/` directory clean and transform data
4. **Dataset Creation**: HuggingFace datasets in `data/processed/hf_dataset/`
5. **Tokenization**: Chunked datasets in `data/processed/tokenized_dataset/`
6. **Model Training**: Trained models saved to `models/` directory

## Common Commands

### Environment Setup

```bash
# Activate virtual environment (if exists)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Interactive Pipeline (Recommended)

```bash
# Launch interactive menu
python main.py

# The menu provides options 1-10:
# 1. Full data pipeline (conversion → dataset → download → tokenize)
# 2-5. Individual data processing steps
# 6. Training via pipeline (basic output)
# 7. Direct training (full progress bars)
# 8. Model inference (removed - use direct model loading)
# 9. Clean all files and run full pipeline
# 10. Show configuration
# 0. Exit
```

### Command Line Pipeline

```bash
# Full data pipeline (excludes training)
python main.py

# Specific steps
python main.py --steps conversion dataset
python main.py --steps train
python main.py --force-clean

# Direct training (with progress bars)
python training/train_model.py

# Direct model inference
python scripts/inference.py
```

### Individual Scripts (Advanced)

```bash
# Data processing
python scripts/conversion.py
python training/hf_create_dataset.py
python training/download_snapshot.py
python training/tokenize_dataset.py
python training/train_model.py

# Model inference
python scripts/inference.py
```

### Exploratory Data Analysis

```bash
# Launch Jupyter for data exploration
jupyter notebook exploratory_data_analysis/
```

## Code Architecture

### Key Components

**Data Scripts (`scripts/`):**

- `csv_konwersja.py`: Converts European CSV format (semicolon separators, comma decimals) to US format
- `jsonl_tekst_czyszczenie.py`: Removes XML artifacts and formatting errors from corporate reports
- `csv_jsonl_concat.py`: Merges cleaned text reports with expert ESG scores
- `float_to_int.py`: Converts continuous ESG scores to discrete classification labels
- `huggingface_datasets.py`: Creates standardized train/validation splits (80/20)

**Training Pipeline (`training/`):**

- `config.json`: Central configuration for model, training parameters, and paths
- `download_snapshot.py`: Downloads `sdadas/polish-longformer-base-4096` base model
- `tokenize_dataset.py`: Implements document chunking with 4096 tokens and 512 stride overlap
- `train_model.py`: Orchestrates multi-label classification training with HuggingFace Trainer

### Model Configuration

The system uses Polish Longformer for long document processing:

- **Context Length**: 4096 tokens with sliding window chunking
- **Task**: Multi-label classification (7 ESG criteria)
- **Base Model**: `sdadas/polish-longformer-base-4096`
- **Training**: 3 epochs, 2e-5 learning rate, gradient accumulation for memory efficiency
- **Class Balancing**: Configurable weighted loss functions for imbalanced datasets

### Data Formats

- **Input**: Plain text corporate reports in Polish
- **Intermediate**: JSONL format with text and ESG score arrays
- **Final**: HuggingFace Arrow datasets with tokenized chunks
- **Labels**: Integer arrays for multi-label classification (C1-C12 criteria)

### Memory Management

The pipeline handles large documents through:

- Document chunking with 512-token stride for overlap
- Gradient checkpointing and FP16 precision during training
- Batch size 1 with gradient accumulation (8 steps)
- Preservation of document-to-chunk mapping for evaluation

### Balanced Training with Threshold Optimization

This training script is **dedicated to optimized ESG classification** with two key features - always enabled:

**Configuration** (in `config.json`):
```json
{
  "class_weight_method": "balanced"
}
```

**Available Methods**:
- `"balanced"`: Standard sklearn-style balanced weighting (n_samples / (n_classes * n_samples_class))
- `"sqrt"`: Square root of inverse frequency weighting  
- `"log"`: Logarithmic weighting for moderate adjustment

**Features**:
- ✅ **Always enabled** - no optional toggles
- ✅ **Balanced Class Weights**: Automatic calculation from training data
- ✅ **Threshold Optimization**: Per-ESG category optimal thresholds (instead of 0.5)
- ✅ **Comprehensive Metrics**: Default vs optimized comparison
- ✅ **MLflow Integration**: Full experiment tracking
- ✅ **Optimized for imbalanced ESG datasets**

**Usage**:
```bash
# Set weighting method in config.json
{
  "class_weight_method": "balanced"
}

# Run balanced training
python training/train_model.py
```

**What it does automatically**:
1. **Training-time**: Calculate optimal weights for each ESG category (C1, C2, C3, C5, C8, C9, C10)
2. **Training-time**: Apply weighted Binary Cross Entropy loss during training
3. **Evaluation-time**: Optimize classification thresholds for each ESG category (instead of default 0.5)
4. **Evaluation-time**: Apply optimized thresholds to maximize F1 scores per category
5. **Logging**: Log all weights, thresholds, and comparison metrics to MLflow
6. **Focus**: More attention on underrepresented classes + optimal decision boundaries

**Expected Results**:
- **Balanced weights**: C3 (28.5%) gets more focus, C2 (62.8%) gets less focus
- **Threshold optimization**: C3 might use 0.3 threshold, C2 might use 0.6 threshold
- **Combined impact**: 2-5% improvement in F1-macro scores

## Model Deployment

### Direct Model Loading

The system uses direct model loading from the trained HuggingFace model saved in `models/` directory:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load trained model and tokenizer
model_path = "models/longformer-esg-2024-01-01_10-30-00"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare input text
text = "Your ESG report content here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=4096)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).int()
```

### Key Features:

- **Multi-label ESG classification** - 7 ESG criteria
- **Long document support** - Up to 4096 tokens with chunking
- **Direct HuggingFace model loading** - No export/conversion needed
- **Polish language optimized** - Based on `sdadas/polish-longformer-base-4096`
- **Optimized thresholds** - Per-category thresholds from training

## Model Inference

### Quick Start

```bash
# Interactive mode (recommended for testing)
python scripts/inference.py --interactive

# Single text analysis
python scripts/inference.py --model models/longformer-esg-2024-01-01_10-30-00 --text "Your ESG report text here"

# File analysis
python scripts/inference.py --model models/longformer-esg-2024-01-01_10-30-00 --file report.txt --output results.json

# Batch processing
python scripts/inference.py --model models/longformer-esg-2024-01-01_10-30-00 --batch-input data/reports --batch-output results/
```

### Output Format

The inference scripts provide:

- **Binary predictions** for each ESG criterion (YES/NO)
- **Confidence scores** (0.0-1.0 probability)
- **Text statistics** (length, tokens processed)
- **JSON output** for programmatic use

## Important Notes

- All scripts should be run from the project root directory
- The pipeline expects specific file paths defined in `training/config.json`
- Document chunking preserves semantic coherence while fitting model constraints
- Expert annotations use different scales per ESG criterion (binary, 3-level, 5-level)
- The system is optimized for Polish language corporate ESG disclosures
- Trained models are saved to `models/` directory in HuggingFace format
- Models can be loaded directly with transformers library for inference

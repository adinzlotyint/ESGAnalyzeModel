# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ESGAnalyzeModel is a Polish ESG (Environmental, Social, Governance) text classification system that trains transformer models to analyze corporate sustainability reports. The project processes 393 Polish corporate reports with expert annotations across 12 ESG criteria to create a multi-label classification model using Polish Longformer.

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

# The menu provides options 1-9:
# 1. Full data pipeline (conversion → dataset → download → tokenize)
# 2-5. Individual data processing steps
# 6. Training via pipeline (basic output)
# 7. Direct training (full progress bars)
# 8. Clean all files and run full pipeline
# 9. Show configuration
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
```

### Individual Scripts (Advanced)
```bash
# Data processing
python scripts/conversion.py
python training/hf_create_dataset.py
python training/download_snapshot.py
python training/tokenize_dataset.py
python training/train_model.py
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
- **Task**: Multi-label classification (12 ESG criteria)
- **Base Model**: `sdadas/polish-longformer-base-4096`
- **Training**: 3 epochs, 2e-5 learning rate, gradient accumulation for memory efficiency

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

## Important Notes

- All scripts should be run from the project root directory
- The pipeline expects specific file paths defined in `training/config.json`
- Document chunking preserves semantic coherence while fitting model constraints
- Expert annotations use different scales per ESG criterion (binary, 3-level, 5-level)
- The system is optimized for Polish language corporate ESG disclosures
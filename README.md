This repository contains a complete and reproducible pipeline created for a master's thesis research project. Its purpose was to train and evaluate a multi-label classification model capable of analyzing Polish corporate sustainability reports based on selected ESRS standards.

**Note:** This repository documents the model training process. The final, trained model is deployed in a separate project (`ESGAnalyze`: https://github.com/adinzlotyint/ESGAnalyze), which serves it via an API for inference. Due to data confidentiality, the raw reports and expert annotations are not included in this repository.

## Project Overview

This project focuses on fine-tuning the `sdadas/polish-longformer-base-4096` model for the specialized task of analyzing non-financial report content. It utilizes a dataset of 393 Polish corporate reports and their corresponding expert annotations. The entire experimental workflow is tracked using MLflow to ensure transparency and reproducibility of the results.

## Implemented Pipeline Features

The pipeline was designed to ensure a robust and reproducible research process:

- **Data Preprocessing**: Scripts handle text cleaning, merging data from various sources, and tokenization using a "sliding window" technique to manage long documents (4096 tokens).
- **Multi-Label Stratification**: The dataset is split using iterative stratification (`scikit-multilearn`) to preserve the distribution of label combinations across the train, validation, and test sets. This is a critical step for handling multi-label data.
- **Two-Level Evaluation**: Model performance is assessed at two granularities. It is first measured on text chunks (4096 tokens), and these predictions are then aggregated to the document level to produce a final, holistic score for each report.
- **Multi-Label Evaluation Metrics**: The evaluation employs a suite of metrics appropriate for multi-label tasks, including F1-score (macro), Jaccard Score, Hamming Loss, and Exact Match Ratio.
- **Threshold Optimization**: The evaluation strategy includes a step to find F1-score-maximizing decision thresholds for each category individually using the validation set.
- **Memory-Efficient Training**: The pipeline uses `gradient_checkpointing`, `gradient_accumulation`, and `mixed-precision training (FP16)` to enable the training of a large model on consumer-grade hardware.
- **Polish Language Model**: The project leverages a model pre-trained specifically on the Polish language to achieve better performance on local documents.

## Classification Criteria

The model was trained to assess reports against 6 criteria that require a deep contextual understanding:

#### 1. Climate Transition Plan (ESRS E1-1)

- **Description:** Assesses whether the report presents a climate transition plan integrated with the business strategy.

#### 2. Risk Identification and Management (ESRS E1-2)

- **Description:** Assesses whether the report describes active management processes for identified climate-related risks.

#### 4. Definition of Consolidation Boundaries (ESRS E1-6 Methodology)

- **Description:** Assesses whether the report transparently defines the organizational boundaries for reported emissions.

#### 6. Historical Data Reporting (ESRS 1 + ESRS E1)

- **Description:** Assesses whether the report provides historical data for at least a 3-year period.

#### 7. Disclosure of Intensity Metrics (ESRS E1-6)

- **Description:** Verifies whether the report presents emission intensity metrics in addition to absolute values.

#### 8. Credibility of Defined Targets (ESRS E1-4)

- **Description:** Assesses whether the reduction strategy links defined targets with a concrete action plan.

## Pipeline Architecture

The system is based on a modular architecture where each stage is managed by a dedicated script. The central `config.json` file stores all parameters, ensuring flexibility and ease of reconfiguring experiments.

#### Project Structure

```
ESGAnalyzeModel/
├── config.json # Central experiment configuration
├── main.py # Main control script (CLI)
├── esg_pipeline.py # Class orchestrating the data processing workflow
├── training/ # Scripts related to model training
│ ├── train_model.py # Main training script
│ ├── hf_create_dataset.py # Dataset creation and stratified splitting
│ └── tokenize_dataset.py # Tokenization with a "sliding window"
├── scripts/ # Utility scripts
│ └── conversion.py # Preprocessing and merging of source data
└── exploratory_data_analysis/ # Notebooks with Exploratory Data Analysis (EDA)
```

#### Data Flow

1.  **Data Merging and Cleaning** (`scripts/conversion.py`): Combines text data and annotations.
2.  **Dataset Creation** (`training/hf_create_dataset.py`): Performs an iterative stratified split of the data into training (70%), validation (15%), and test (15%) sets.
3.  **Tokenization** (`training/tokenize_dataset.py`): Processes texts into 4096-token chunks.
4.  **Training and Evaluation** (`training/train_model.py`): Fine-tunes the model, optimizes decision thresholds, and performs a final evaluation on the test set at both chunk and document levels.

## MLflow Experiment Tracking

All training runs are logged to MLflow. The following items are tracked:

- **Parameters**: All hyperparameters from `config.json` (e.g., learning rate, batch size).
- **Metrics**: Key metrics for both chunk-level and document-level evaluations, including F1-score (macro and per-category), Jaccard Score, Hamming Loss, and Exact Match Ratio.
- **Artifacts**: The trained model, tokenizer, configuration files, and the optimized decision thresholds, which are essential for reproducing the final test results.

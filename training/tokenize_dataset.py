import json
import sys
from pathlib import Path
from datasets import (
    load_from_disk,
    concatenate_datasets,
    DatasetDict,
    Sequence,
    Value
)
from transformers import LongformerTokenizerFast

# Tokenizer settings
MAX_LENGTH = 4096
STRIDE = 512

def load_config() -> dict:
    """
    Loads the main project configuration from the root directory.

    Returns:
        dict: The project configuration.
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found at: {config_path}")
        sys.exit(1)

def tokenize_dataset(dataset: DatasetDict, tokenizer: LongformerTokenizerFast) -> 'Dataset':
    """
    Tokenizes a dataset of long documents using a sliding window approach.

    It first merges train, validation, and test splits to process them
    jointly, adding metadata to preserve the original split information.
    Each long document is chunked into smaller, overlapping segments.

    Args:
        dataset (DatasetDict): The input dataset with 'train', 'validation', 'test' splits.
        tokenizer (LongformerTokenizerFast): The tokenizer to use.

    Returns:
        Dataset: A single, tokenized dataset containing chunks from all splits.
    """
    train_size = len(dataset['train'])
    val_size = len(dataset['validation'])

    # 1. Merge all splits to process them together
    merged_dataset = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test']
    ])
    print(f"🔗 Merged all splits into a single dataset of {len(merged_dataset)} documents.")

    # 2. Add a unique document ID and original split info
    def add_metadata(example, idx):
        example['doc_id'] = idx
        if idx < train_size:
            example['original_split'] = 'train'
        elif idx < train_size + val_size:
            example['original_split'] = 'validation'
        else:
            example['original_split'] = 'test'
        return example

    dataset_with_meta = merged_dataset.map(add_metadata, with_indices=True)
    print("ℹ️  Added document IDs and original split metadata.")

    # 3. Define the tokenization and chunking function
    def chunk_and_tokenize(examples):
        tokenized_output = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            stride=STRIDE,
            return_overflowing_tokens=True,
        )

        # Propagate metadata (doc_id, labels, split) to each new chunk
        sample_map = tokenized_output.pop("overflow_to_sample_mapping")
        for key in ["doc_id", "labels", "original_split"]:
            tokenized_output[key] = [examples[key][i] for i in sample_map]
        
        return tokenized_output

    # 4. Apply tokenization
    print(f"🔄 Tokenizing and chunking documents (max_length={MAX_LENGTH}, stride={STRIDE})...")
    tokenized = dataset_with_meta.map(
        chunk_and_tokenize,
        batched=True,
        remove_columns=dataset_with_meta.column_names,
        desc="Tokenizing & chunking"
    )
    
    # 5. Cast labels to float32 for loss computation
    tokenized = tokenized.cast_column("labels", Sequence(Value("float32")))
    print("✅ Tokenization complete.")
    return tokenized

def reconstruct_splits(tokenized_dataset: 'Dataset') -> DatasetDict:
    """
    Reconstructs the train, validation, and test splits from a tokenized
    dataset containing chunks from all original documents.

    Args:
        tokenized_dataset (Dataset): The tokenized dataset with 'original_split' metadata.

    Returns:
        DatasetDict: The final dataset with 'train', 'validation', and 'test' splits.
    """
    print("🧬 Reconstructing train, validation, and test splits...")
    
    final_splits = DatasetDict({
        split: tokenized_dataset.filter(lambda x: x["original_split"] == split)
        for split in ["train", "validation", "test"]
    })

    # Remove the temporary metadata column
    for split_name in final_splits:
        final_splits[split_name] = final_splits[split_name].remove_columns(["original_split", "doc_id"])

    print("✅ Splits reconstructed.")
    return final_splits

def save_tokenized_dataset(dataset: DatasetDict, save_path: str):
    """
    Saves the final tokenized DatasetDict to disk and prints chunk counts.

    Args:
        dataset (DatasetDict): The final dataset to save.
        save_path (str): The directory where the dataset will be saved.
    """
    print("\n📊 Final chunk counts per split:")
    total_chunks = 0
    for split_name, split_data in dataset.items():
        count = len(split_data)
        total_chunks += count
        print(f"   - {split_name.capitalize()}: {count} chunks")
    print(f"   - Total: {total_chunks} chunks")

    print(f"\n💾 Saving tokenized dataset to: {save_path}")
    dataset.save_to_disk(save_path)
    print(f"✅ Tokenized dataset saved successfully at: {save_path}")

def main():
    # Main script to tokenize and chunk the stratified HuggingFace dataset.
    print("\n🚀 Starting dataset tokenization process...")
    
    config = load_config()
    model_name = config.get("model_name")
    hf_dataset_path = config.get("hf_dataset")
    tokenizer_output_path = config.get("tokenizer_output_path")

    if not all([model_name, hf_dataset_path, tokenizer_output_path]):
        print("❌ Missing required paths in config.json (model_name, hf_dataset, tokenizer_output_path).")
        sys.exit(1)
        
    if not Path(hf_dataset_path).exists():
        print(f"❌ Input dataset not found at: {hf_dataset_path}")
        print("💡 Please run the 'dataset creation' step first.")
        sys.exit(1)

    try:
        tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        raw_dataset = load_from_disk(hf_dataset_path)
        
        tokenized_full = tokenize_dataset(raw_dataset, tokenizer)
        final_dataset = reconstruct_splits(tokenized_full)
        save_tokenized_dataset(final_dataset, tokenizer_output_path)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred during tokenization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from transformers import LongformerTokenizerFast
from datasets import load_from_disk, Sequence, Value, concatenate_datasets, DatasetDict
import json

with open("config.json") as f:
    cfg = json.load(f)

tokenizer = LongformerTokenizerFast.from_pretrained(cfg["model_name"])

raw = load_from_disk(cfg["hf_dataset"])

# Przechowuj oryginalne rozmiary podziałów do późniejszej rekonstrukcji
train_size = len(raw['train'])
val_size   = len(raw['validation'])
test_size  = len(raw['test'])

# Połącz wszystkie zbiory danych, aby zapewnić unikalny doc_id we wszystkich podziałach
dataset = concatenate_datasets([raw['train'], raw['validation'], raw['test']])

# doc_id = indeks wiersza, żeby wiedzieć skąd chunk pochodzi
dataset = dataset.map(lambda _, idx: {"doc_id": idx}, with_indices=True)

# Dodaj informacje o oryginalnym podziale, aby zachować rozróżnienie między treningiem a walidacją
def add_split_info(examples, indices):
    original_split = [
        "train"      if idx < train_size
        else "validation" if idx < train_size + val_size
        else "test"
        for idx in indices
    ]
    return {
        "original_split": original_split
    }

dataset = dataset.map(add_split_info, with_indices=True, batched=True)

def chunk_and_tokenize(examples):
    out = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
        stride=512,
        return_overflowing_tokens=True,
        return_token_type_ids =True,
    )

    sample_map = out.pop("overflow_to_sample_mapping")
    out["doc_id"] = [examples["doc_id"][i] for i in sample_map]
    out["labels"] = [examples["labels"][i] for i in sample_map]
    out["original_split"] = [examples["original_split"][i] for i in sample_map]
    
    return out

tokenized = dataset.map(
    chunk_and_tokenize,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing & chunking"
)

tokenized = tokenized.cast_column(
    "labels", Sequence(Value("float32"))
)

train_tokenized = tokenized.filter(lambda x: x["original_split"] == "train")
val_tokenized = tokenized.filter(lambda x: x["original_split"] == "validation")
test_tokenized = tokenized.filter(lambda x: x["original_split"] == "test")

train_tokenized = train_tokenized.remove_columns(["original_split"])
val_tokenized = val_tokenized.remove_columns(["original_split"])
test_tokenized = test_tokenized.remove_columns(["original_split"])

final_dataset = DatasetDict({
    "train":       train_tokenized,
    "validation":  val_tokenized,
    "test":        test_tokenized
})

final_dataset.save_to_disk(cfg["tokenizer_output_path"])
print("✅ Zapisano dataset:")
print(f"   Train: {len(final_dataset['train'])} chunks")
print(f"   Validation: {len(final_dataset['validation'])} chunks")
print(f"   Test: {len(final_dataset['test'])} chunks")
print(f"   Total: {len(final_dataset['train']) + len(final_dataset['validation']) + len(final_dataset['test'])} chunks")
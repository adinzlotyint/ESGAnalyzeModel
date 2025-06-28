from transformers import LongformerTokenizerFast
from datasets import load_from_disk, Sequence, Value
import json

with open("config.json") as f:
    cfg = json.load(f)

tokenizer = LongformerTokenizerFast.from_pretrained(cfg["model_name"])

raw = load_from_disk(cfg["hf_dataset"])

# doc_id = indeks wiersza, żeby wiedzieć skąd chunk pochodzi
raw = raw.map(lambda _, idx: {"doc_id": idx}, with_indices=True)

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
    
    return out

tokenized = raw.map(
    chunk_and_tokenize,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing & chunking"
)

tokenized = tokenized.cast_column(
    "labels", Sequence(Value("float32"))
)

tokenized.save_to_disk(cfg["tokenizer_output_path"])
print("✅ Zapisano dataset:", tokenized)
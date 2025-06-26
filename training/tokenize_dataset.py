from transformers import LongformerTokenizerFast
from datasets import load_from_disk
import json

# Wczytaj config z modelem
with open("training/config.json", "r") as file:
    cfg = json.load(file)

# Wczytaj tokenizer Longformer
tokenizer = LongformerTokenizerFast.from_pretrained(cfg["model_name"])

# Wczytaj wstępnie przetworzony dataset (text + labels)
dataset = load_from_disk(cfg["hf_dataset"])

# Tokenizacja (max 4096 tokenów, padding do pełnej długości)
def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=4096,
    )
    tokenized["labels"] = example["labels"]
    return tokenized

# Mapowanie tokenizacji na cały zbiór
tokenized_dataset = dataset.map(tokenize_function, batched=False)

print(tokenized_dataset)
print("\nDostępne pola:")
print(tokenized_dataset["train"].features)

# Zapis tokenizowanego zbioru na dysk
tokenized_dataset.save_to_disk(cfg["tokenizer_output_path"])
print(f"✅ Tokenizacja zakończona. Zapisano do: {cfg['tokenizer_output_path']}")
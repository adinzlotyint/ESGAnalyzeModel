from transformers import LongformerForSequenceClassification, LongformerTokenizerFast, TrainingArguments, Trainer, LongformerConfig, default_data_collator
from datasets import load_from_disk
import json
from datetime import datetime
import os

# Wczytaj konfigurację modelu
with open("training/config.json", "r") as file:
    cfg = json.load(file)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(cfg["model_output_path"], f"longformer-{timestamp}")

# Wczytaj tokenizowany dataset z dysku
dataset = load_from_disk(cfg["tokenizer_output_path"])

# Konfiguracja modelu
config = LongformerConfig.from_pretrained(
    cfg["model_name"],
    num_labels=cfg["num_labels"],
    problem_type=cfg["problem_type"]
)

# Wczytaj model
model = LongformerForSequenceClassification.from_pretrained(
    cfg["model_name"],
    config=config
)

# Argumenty treningowe
args_cfg = cfg["training_args"]
training_args = TrainingArguments(**args_cfg)

# Dodaj default_data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=default_data_collator
)

trainer.train()
trainer.save_model(output_dir)
print("✅ Trening zakończony. Model zapisany.")
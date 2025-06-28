from datasets import load_dataset, DatasetDict
import os

# Ścieżka do JSONL
data_path = 'data/processed/dane_do_uczenia_int.jsonl'

# Wczytanie jako dataset
dataset = load_dataset('json', data_files=data_path, split='train')

# Podział na train/val (np. 80/20)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['test']
})

# Zapis do katalogu
output_dir = 'data/processed/hf_dataset'
dataset_dict.save_to_disk(output_dir)

print(f'Dataset zapisany do {output_dir}')
import json
from pathlib import Path

input_path = Path("data/processed/dane_do_uczenia.jsonl")
output_path = Path("data/processed/dane_do_uczenia_int.jsonl")

with input_path.open(encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        rec = json.loads(line)
        rec["labels"] = [int(x) for x in rec.get("labels", [])]
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("✅ Gotowe: zapisano z int-labels do:", output_path)

import pandas as pd
import json
from pathlib import Path

# === Ścieżki ===
jsonl_path = Path("data/processed/teksty_raportow_wyczyszczone.jsonl")
csv_path = Path("data/processed/expert_scores_uproszczone_po_konwersji.csv")
output_path = Path("data/processed/dane_do_uczenia.jsonl")

# === Wczytaj CSV z ocenami ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # usuń ewentualne spacje
df["Raport"] = df["Raport"].str.strip()
df = df.set_index("Raport")

# === Wczytaj JSONL i dopisz etykiety z CSV ===
with jsonl_path.open(encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        rec = json.loads(line)
        report_name = rec.get("nazwa_raportu", "").strip()

        if report_name in df.index:
            labels = df.loc[report_name].tolist()
        else:
            print(f"⚠ Brak ocen dla: {report_name}")
            labels = [0.0] * 12  # albo pomiń, albo wstaw zera

        f_out.write(json.dumps({
            "text": rec.get("text", ""),
            "labels": labels
        }, ensure_ascii=False) + "\n")

print("✅ Gotowe. Zapisano:", output_path)
import pandas as pd
import json
import re
import os
from pathlib import Path

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_csv_format(input_file, output_file):
    # Konwertuj format CSV: zamień przecinki na kropki, średniki na przecinki
    print(f"📝 Konwertowanie formatu CSV: {input_file} -> {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(",", ".")
    content = content.replace(";", ",")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ Konwersja CSV zakończona: {output_file}")

def clean_text_jsonl(input_path, output_path):
    # Czyszczenie tekstu w pliku JSONL przy użyciu wyrażeń regularnych
    print(f"🧹 Czyszczenie tekstu w JSONL: {input_path} -> {output_path}")
    
    RE_DOTS = re.compile(r"\.{2,}")
    RE_SPACED_DOTS = re.compile(r"(?:\.\s*){2,}")
    RE_ERROR = re.compile(r"This page contains the following errors: error on line.{0,10}at column.{0,10}: Extra content at the end of the document Below is a rendering of the page up to the first error.")

    processed_count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            if "text" in obj:
                text = obj["text"]
                text = RE_DOTS.sub(".", text)
                text = RE_SPACED_DOTS.sub(". ", text)
                text = RE_ERROR.sub("", text)
                text = text.replace("", "")
                text = text.replace("", "")
                obj["text"] = text.strip()
                processed_count += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"✅ Czyszczenie zakończone: przetworzono {processed_count} rekordów")

def merge_jsonl_with_csv(jsonl_path, csv_path, output_path, num_labels=12):
    # Połącz dane z JSONL z etykietami z pliku CSV
    print(f"🔗 Łączenie JSONL z etykietami z CSV: {jsonl_path} + {csv_path} -> {output_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Raport"] = df["Raport"].str.strip()
    df = df.set_index("Raport")
    
    print(f"📊 Wczytano CSV: {len(df)} rekordów, {len(df.columns)} kolumn z etykietami")

    matched_count = 0
    missing_count = 0
    
    with open(jsonl_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            rec = json.loads(line)
            report_name = rec.get("nazwa_raportu", "").strip()

            if report_name in df.index:
                labels = df.loc[report_name].tolist()
                matched_count += 1
            else:
                print(f"⚠️  Brak etykiet dla: {report_name}")
                labels = [0.0] * num_labels
                missing_count += 1

            f_out.write(json.dumps({
                "text": rec.get("text", ""),
                "labels": labels
            }, ensure_ascii=False) + "\n")

    print(f"✅ Łączenie zakończone: dopasowano {matched_count}, brakujących {missing_count}")

def convert_labels_to_int(input_path, output_path):
    # Konwertuj etykiety z floatów na liczby całkowite
    print(f"🔢 Konwertowanie etykiet na liczby całkowite: {input_path} -> {output_path}")
    
    converted_count = 0
    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            rec = json.loads(line)
            rec["labels"] = [int(x) for x in rec.get("labels", [])]
            converted_count += 1
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Konwersja zakończona: przetworzono {converted_count} rekordów")

def main():
    print("🚀 Rozpoczynanie procesu przetwarzania danych...")

    config = load_config()
    
    raw_csv_path = config["raw_csv_path"]
    raw_jsonl_path = config["raw_jsonl_path"]
    
    converted_csv_path = config["converted_csv_path"]
    cleaned_jsonl_path = config["cleaned_jsonl_path"]
    merged_data_path = config["merged_jsonl_path"]
    
    final_output_path = Path(config["final_jsonl"])
    
    num_labels = config["num_labels"]
    
    print(f"📁 Pliki do przetworzenia:")
    print(f"   Surowy CSV: {raw_csv_path}")
    print(f"   Surowy JSONL: {raw_jsonl_path}")
    print(f"   Wyjściowy zbiór danych: {final_output_path}")
    print(f"   Liczba etykiet: {num_labels}")
    
    # Krok 1: Konwersja formatu CSV
    convert_csv_format(raw_csv_path, converted_csv_path)
    
    # Krok 2: Czyszczenie tekstów JSONL
    clean_text_jsonl(raw_jsonl_path, cleaned_jsonl_path)
    
    # Krok 3: Łączenie danych JSONL z etykietami z CSV
    merge_jsonl_with_csv(cleaned_jsonl_path, converted_csv_path, merged_data_path, num_labels)
    
    # Krok 4: Konwersja etykiet do liczb całkowitych
    convert_labels_to_int(merged_data_path, final_output_path)
    
    print(f"🎉 Proces zakończony pomyślnie!")
    print(f"📂 Zbiór danych gotowy w: {final_output_path}")
    print(f"💡 Możesz teraz uruchomić hf_create_dataset.py, aby stworzyć zbiór HuggingFace")

if __name__ == "__main__":
    main()
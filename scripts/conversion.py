import pandas as pd
import json
import re
import os
from pathlib import Path

def load_config() -> dict:
    """
    Loads the main project configuration from the root directory.

    Returns:
        dict: The project configuration.
    """
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_csv_format(input_path: str, output_path: str):
    """
    Standardizes a CSV file by replacing decimal commas with dots and
    semicolon separators with commas.

    Args:
        input_path (str): Path to the source CSV file.
        output_path (str): Path to save the converted CSV file.
    """
    print(f"🔄 Standardizing CSV format: {input_path} -> {output_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(",", ".")
    content = content.replace(";", ",")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ CSV format standardized.")

def clean_jsonl_text(input_path: str, output_path: str):
    """
    Cleans text content within a JSONL file using predefined regex patterns
    to remove common text extraction artifacts.

    Args:
        input_path (str): Path to the source JSONL file.
        output_path (str): Path to save the cleaned JSONL file.
    """
    print(f"🧹 Cleaning text in JSONL: {input_path} -> {output_path}")
    
    RE_DOTS = re.compile(r"\.{2,}")
    RE_SPACED_DOTS = re.compile(r"(?:\.\s*){2,}")
    RE_XML_ERROR = re.compile(r"This page contains the following errors:.*?Below is a rendering of the page up to the first error\.", re.DOTALL)
    
    processed_count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue

            record = json.loads(line)
            if "text" in record and isinstance(record["text"], str):
                text = record["text"]
                text = RE_DOTS.sub(".", text)
                text = RE_SPACED_DOTS.sub(". ", text)
                text = RE_XML_ERROR.sub("", text)
                record["text"] = text.strip()
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1
            
    print(f"✅ Text cleaning complete. Processed {processed_count} records.")

def merge_data(jsonl_path: str, csv_path: str, output_path: str, num_labels: int):
    """
    Merges text data from a JSONL file with labels from a CSV file,
    matching records based on a report name identifier.

    Args:
        jsonl_path (str): Path to the cleaned JSONL file with texts.
        csv_path (str): Path to the standardized CSV file with labels.
        output_path (str): Path to save the merged JSONL file.
        num_labels (int): The expected number of labels per record.
    """
    print(f"🔗 Merging JSONL texts with CSV labels...")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Raport"] = df["Raport"].str.strip()
    df = df.set_index("Raport")
    
    print(f"   - Loaded CSV with {len(df)} records and {len(df.columns)} label columns.")

    matched_count = 0
    missing_count = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            record = json.loads(line)
            report_name = record.get("nazwa_raportu", "").strip()

            if report_name in df.index:
                labels = df.loc[report_name].tolist()
                matched_count += 1
            else:
                print(f"   ⚠️ Warning: No labels found for report '{report_name}'. Assigning zeros.")
                labels = [0.0] * num_labels
                missing_count += 1
            
            output_record = {"text": record.get("text", ""), "labels": labels}
            f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"✅ Merge complete. Matched: {matched_count}, Missing: {missing_count}.")

def finalize_labels(input_path: str, output_path: str):
    """
    Converts float labels in a JSONL file to integers, binarizing them
    for multi-label classification.

    Args:
        input_path (str): Path to the merged JSONL file with float labels.
        output_path (str): Path to the final JSONL file with integer labels.
    """
    print(f"🔢 Finalizing labels (float to int): {input_path} -> {output_path}")
    
    converted_count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            record = json.loads(line)
            if "labels" in record and isinstance(record["labels"], list):
                record["labels"] = [int(x) for x in record["labels"]]
            
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            converted_count += 1
            
    print(f"✅ Label finalization complete. Converted {converted_count} records.")

def main():
    # Executes the full data conversion and merging pipeline.
    print("\n🚀 Starting data conversion and preparation process...")
    config = load_config()
    
    raw_csv_path = config["raw_csv_path"]
    raw_jsonl_path = config["raw_jsonl_path"]
    final_output_path = config["final_jsonl"]
    num_labels = config["num_labels"]
    
    proc_dir = Path(final_output_path).parent
    proc_dir.mkdir(parents=True, exist_ok=True)
    converted_csv_path = proc_dir / "expert_scores_standardized.csv"
    cleaned_jsonl_path = proc_dir / "reports_texts_cleaned.jsonl"
    merged_data_path = proc_dir / "data_merged.jsonl"
    
    print(f"\n📋 Configuration:")
    print(f"   - Raw CSV: {raw_csv_path}")
    print(f"   - Raw JSONL: {raw_jsonl_path}")
    print(f"   - Final Output: {final_output_path}")
    print(f"   - Number of Labels: {num_labels}")
    
    print("\n--- [Step 1/4] Standardizing CSV ---")
    convert_csv_format(raw_csv_path, str(converted_csv_path))
    
    print("\n--- [Step 2/4] Cleaning JSONL Text ---")
    clean_jsonl_text(raw_jsonl_path, str(cleaned_jsonl_path))
    
    print("\n--- [Step 3/4] Merging Texts and Labels ---")
    merge_data(str(cleaned_jsonl_path), str(converted_csv_path), str(merged_data_path), num_labels)
    
    print("\n--- [Step 4/4] Finalizing Labels ---")
    finalize_labels(str(merged_data_path), final_output_path)
    
    print(f"\n🎉 Data preparation process completed successfully!")
    print(f"📂 Final dataset is ready at: {final_output_path}")

if __name__ == "__main__":
    main()
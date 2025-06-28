import json
import re

input_path = "data/processed/teksty_raportow.jsonl"
output_path = "data/processed/teksty_raportow_wyczyszczone.jsonl"

RE_DOTS = re.compile(r"\.{2,}")
RE_SPACED_DOTS = re.compile(r"(?:\.\s*){2,}")
RE_ERROR = re.compile(r"This page contains the following errors: error on line.{0,10}at column.{0,10}: Extra content at the end of the document Below is a rendering of the page up to the first error.")

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
            text = text.replace("", "")
            text = text.replace("", "")
            obj["text"] = text.strip()

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
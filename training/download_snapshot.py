from huggingface_hub import snapshot_download
import json

with open("training/config.json", "r") as file:
    cfg = json.load(file)

snapshot_download(repo_id=cfg["model_name"], repo_type="model")
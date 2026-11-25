"""Download the journal-extension VLMs into $HF_HOME (run with HF_XET_HIGH_PERFORMANCE=1)."""
from huggingface_hub import snapshot_download

for repo in ["Qwen/Qwen2.5-VL-7B-Instruct", "Salesforce/instructblip-vicuna-7b"]:
    print(repo, snapshot_download(repo, max_workers=16, allow_patterns=[
        "*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]), flush=True)

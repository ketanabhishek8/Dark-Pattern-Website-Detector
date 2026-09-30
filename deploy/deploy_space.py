"""Creates (or updates) the Hugging Face Space and uploads the app with its trained model.

Log in first with `hf auth login`, and train the model with `python app/train.py`.
Usage: python deploy/deploy_space.py
"""
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "dark-pattern-detector"

if not (ROOT / "app" / "model" / "distilbert").exists():
    raise SystemExit("Model not found. Run `python app/train.py` first.")

api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{SPACE_NAME}"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=ROOT,
                  allow_patterns=["Dockerfile", "app/**"], ignore_patterns=["**/__pycache__/**", "**/.DS_Store"],
                  commit_message="Deploy dashboard and model")
api.upload_file(repo_id=repo_id, repo_type="space", path_or_fileobj=ROOT / "deploy" / "space_README.md",
                path_in_repo="README.md", commit_message="Space card")
print(f"Space page: https://huggingface.co/spaces/{repo_id}")
print(f"App link (live once the build finishes, about 5-10 minutes): https://{user}-{SPACE_NAME}.hf.space")

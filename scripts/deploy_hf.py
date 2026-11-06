"""Publish the trained adapter to the Hugging Face Hub (and optionally a Gradio Space).

Prerequisite (once):  hf auth login   (token needs WRITE access)

Usage:
    python scripts/deploy_hf.py                  # model repo only (free)
    python scripts/deploy_hf.py --with-space     # also a Gradio Space (needs HF PRO)
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "models" / "tinyllama-hatexplain-lora"
METRICS = ROOT / "results" / "tinyllama_lora" / "metrics.json"
GITHUB_URL = "https://github.com/venkateshmedasani/SLM-based-hate-speech-recognition"


def model_card(repo_id: str, metrics: dict) -> str:
    per_class = "\n".join(
        f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} |"
        for name, m in metrics["per_class"].items()
    )
    return f"""---
base_model: {metrics['base_model']}
library_name: peft
pipeline_tag: text-classification
datasets:
  - Hate-speech-CNERG/hatexplain
tags:
  - lora
  - hate-speech
  - content-moderation
---

# {repo_id.split('/')[-1]}

LoRA adapter (plus classification head) that turns **{metrics['base_model']}** into a 3-way
hate speech classifier (`hate`, `normal`, `offensive`), trained on HateXplain.

Code, training script and evaluation: {GITHUB_URL}

## Test-set results (2,015 held-out posts)

| Metric | Value |
|---|---|
| Accuracy | {metrics['accuracy']:.3f} |
| Macro F1 | {metrics['macro_f1']:.3f} |
| Weighted F1 | {metrics['weighted_f1']:.3f} |

| Class | Precision | Recall | F1 |
|---|---|---|---|
{per_class}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
base = AutoModelForSequenceClassification.from_pretrained("{metrics['base_model']}", num_labels=3)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, "{repo_id}").merge_and_unload().eval()
```

Input text should be normalised with `clean_text` from `src/text_utils.py` in the GitHub repo
(lower-case, punctuation stripped, `@mentions` to `<user>`), matching HateXplain's format.

Labels: `0 = hate`, `1 = normal`, `2 = offensive`.

## Limitations

Research artefact. Inherits the annotation biases of HateXplain, is English-only,
and confuses *hate* with *offensive* more than any other pair of classes.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-repo", default="tinyllama-hatexplain-lora")
    p.add_argument("--space", default="slm-hate-speech-detector")
    p.add_argument(
        "--with-space",
        action="store_true",
        help="also deploy the Gradio Space (Gradio Spaces require a Hugging Face PRO plan)",
    )
    args = p.parse_args()

    if not (ADAPTER_DIR / "adapter_model.safetensors").exists():
        raise SystemExit(f"No trained adapter at {ADAPTER_DIR}. Run `python src/train.py` first.")

    api = HfApi()
    user = api.whoami()["name"]
    model_id, space_id = f"{user}/{args.model_repo}", f"{user}/{args.space}"

    print(f"Uploading adapter -> https://huggingface.co/{model_id}")
    api.create_repo(model_id, exist_ok=True)
    api.upload_folder(
        repo_id=model_id,
        folder_path=ADAPTER_DIR,
        ignore_patterns=["checkpoints/*", "README.md"],
        commit_message="Upload LoRA adapter",
    )
    api.upload_file(
        repo_id=model_id,
        path_or_fileobj=model_card(model_id, json.loads(METRICS.read_text())).encode(),
        path_in_repo="README.md",
        commit_message="Add model card",
    )

    if not args.with_space:
        print(f"\nDone. Model: https://huggingface.co/{model_id}")
        return

    print(f"Deploying Space -> https://huggingface.co/spaces/{space_id}")
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(space_id, "MODEL_ID", model_id)
    for local, remote in [
        (ROOT / "demo" / "app.py", "app.py"),
        (ROOT / "demo" / "requirements.txt", "requirements.txt"),
        (ROOT / "demo" / "README.md", "README.md"),
        (ROOT / "src" / "text_utils.py", "text_utils.py"),
    ]:
        api.upload_file(repo_id=space_id, repo_type="space", path_or_fileobj=str(local), path_in_repo=remote)

    print("\nDone. The Space takes a few minutes to build on first deploy.")
    print(f"  Model: https://huggingface.co/{model_id}")
    print(f"  Demo:  https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()

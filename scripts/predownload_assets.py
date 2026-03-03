from __future__ import annotations

import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'").strip('"')
    return out


def _ensure_hf_token_env() -> None:
    if os.getenv("HF_TOKEN"):
        return
    env = _read_dotenv(Path(".env"))
    token = (
        env.get("HF_TOKEN")
        or env.get("HUGGINGFACE_HUB_TOKEN")
        or env.get("HUGGING_FACE_TOKEN")
    )
    if token:
        os.environ["HF_TOKEN"] = token


def _download_model(model_id: str) -> None:
    print(f"Downloading tokenizer: {model_id}")
    AutoTokenizer.from_pretrained(model_id)
    print(f"Downloading model weights: {model_id}")
    AutoModelForSeq2SeqLM.from_pretrained(model_id)
    print(f"✅ Cached: {model_id}\n")


def main() -> None:
    _ensure_hf_token_env()
    _download_model("t5-small")
    _download_model("facebook/bart-base")
    _download_model("google/pegasus-cnn_dailymail")

    print("Downloading SAMSum dataset...")
    ds = load_dataset("knkarthick/samsum")
    print(
        f"✅ SAMSum cached — Train: {len(ds['train'])}, "
        f"Val: {len(ds['validation'])}, Test: {len(ds['test'])}"
    )


if __name__ == "__main__":
    main()

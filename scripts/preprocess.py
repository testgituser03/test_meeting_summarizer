#!/usr/bin/env python3
"""
preprocess.py — Tokenization and dataset caching pipeline.

Produces two tokenized variants from SAMSum and saves them to data/cache/.
Run once before training; subsequent runs load from disk in milliseconds.

Variants:
  samsum_with_speakers_<model>  — speaker tags kept in input (baseline)
  samsum_no_speakers_<model>    — speaker tags stripped (E2 ablation)

Usage:
  python3 scripts/preprocess.py                         # uses config.yaml
  python3 scripts/preprocess.py --model t5-small        # override model
  python3 scripts/preprocess.py --variants with_speakers no_speakers
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_preprocess_fn(tokenizer, max_source: int, max_target: int, task_prefix: str):
    """Return a preprocessing function that keeps speaker tags."""

    def preprocess_with_speakers(batch: dict) -> dict:
        inputs = [task_prefix + d for d in batch["dialogue"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source,
            truncation=True,
            padding=False,
        )
        # text_target= is the Transformers 5.x replacement for
        # the removed as_target_tokenizer() context manager.
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_with_speakers


def make_preprocess_no_speakers_fn(tokenizer, max_source: int, max_target: int, task_prefix: str):
    """Return a preprocessing function that strips 'Speaker: ' prefixes."""

    _tag_re = re.compile(r"^[^\n:]+:\s*", re.MULTILINE)

    def preprocess_no_speakers(batch: dict) -> dict:
        stripped = [_tag_re.sub("", d) for d in batch["dialogue"]]
        inputs   = [task_prefix + d for d in stripped]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["summary"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_no_speakers


def main() -> None:
    # Enforce offline mode BEFORE any HuggingFace imports — all assets must
    # already be cached in ~/.cache/huggingface/ by predownload_assets.py.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    parser = argparse.ArgumentParser(description="Tokenize SAMSum into data/cache/")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--model",    default=None, help="Override model_name")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["with_speakers", "no_speakers"],
        choices=["with_speakers", "no_speakers"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model

    MODEL_NAME     = cfg["model_name"]
    MAX_SOURCE_LEN = cfg["max_source_length"]
    MAX_TARGET_LEN = cfg["max_target_length"]
    TASK_PREFIX    = cfg["task_prefix"]

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    cache_dir    = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset           # noqa: PLC0415
        from transformers import AutoTokenizer      # noqa: PLC0415
    except ImportError as exc:
        print(f"❌  Missing package: {exc}")
        sys.exit(1)

    print(f"\n  Model      : {MODEL_NAME}")
    print(f"  Source len : {MAX_SOURCE_LEN}   Target len : {MAX_TARGET_LEN}")
    print(f"  Task prefix: '{TASK_PREFIX}'")
    print(f"  Variants   : {args.variants}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_raw    = load_dataset("knkarthick/samsum")

    variant_fns = {
        "with_speakers": make_preprocess_fn(tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX),
        "no_speakers":   make_preprocess_no_speakers_fn(tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX),
    }

    model_slug = MODEL_NAME.replace("/", "_")

    for variant in args.variants:
        out_path = cache_dir / f"samsum_{variant}_{model_slug}"
        if out_path.exists():
            print(f"  ⚡  {variant}: already cached at {out_path.name} — skipping")
            continue

        print(f"  Processing '{variant}'...")
        fn = variant_fns[variant]
        tokenized = ds_raw.map(
            fn,
            batched=True,
            remove_columns=ds_raw["train"].column_names,
            load_from_cache_file=True,   # reuse HF's internal .map() cache if available
            desc=f"Tokenizing ({variant})",
        )
        tokenized.save_to_disk(str(out_path))
        print(f"  ✅  Saved: {out_path.relative_to(project_root)}")
        print(f"       Train: {len(tokenized['train']):,}  "
              f"Val: {len(tokenized['validation']):,}  "
              f"Test: {len(tokenized['test']):,}")

    print()
    print("  Preprocessing complete.")
    print()


if __name__ == "__main__":
    main()

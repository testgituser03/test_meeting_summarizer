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
  python3 scripts/preprocess.py --online --model flan-t5-base   # allow HF download if tokenizer not cached
  python3 scripts/preprocess.py --variants with_speakers no_speakers
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from model_registry import effective_task_prefix, resolve_model_name


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


def make_preprocess_split_speakers_fn(
    tokenizer, max_source: int, max_target: int, task_prefix: str, stride: int = 256
):
    """Return a preprocessing function that splits long dialogues into overlapping windows.

    For dialogues that tokenize to > max_source tokens, this creates multiple
    training examples using a sliding window with the given stride. Each window
    gets the same summary label. Short dialogues (≤ max_source tokens) pass
    through unchanged (single example).

    This helps the model learn from the full content of long conversations that
    would otherwise be truncated.
    """

    def preprocess_split_speakers(batch: dict) -> dict:
        all_input_ids: list[list[int]] = []
        all_labels:    list[list[int]] = []

        for dialogue, summary in zip(batch["dialogue"], batch["summary"]):
            # Tokenize full dialogue without truncation
            full_input = task_prefix + dialogue
            full_tokens = tokenizer(
                full_input,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            # Tokenize summary (always within max_target)
            label_tokens = tokenizer(
                text_target=summary,
                max_length=max_target,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(full_tokens) <= max_source:
                # Short dialogue — no splitting needed
                all_input_ids.append(full_tokens)
                all_labels.append(label_tokens)
            else:
                # Long dialogue — sliding window with overlap
                for start in range(0, len(full_tokens), stride):
                    window = full_tokens[start : start + max_source]
                    if len(window) < 32:  # skip tiny trailing fragments
                        break
                    all_input_ids.append(window)
                    all_labels.append(label_tokens)

        return {"input_ids": all_input_ids, "labels": all_labels}

    return preprocess_split_speakers


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize SAMSum into data/cache/")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--model",    default=None, help="Override model_name")
    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Allow Hugging Face downloads for tokenizer/dataset (default: offline-only). "
            "Use for a new hub id before cache is warm, or run predownload_assets.py online first."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["with_speakers", "no_speakers"],
        choices=["with_speakers", "no_speakers", "split_speakers"],
    )
    args = parser.parse_args()

    # Offline by default (reproducible, no surprise network). --online opts in.
    if args.online:
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    else:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model

    MODEL_NAME     = resolve_model_name(cfg["model_name"])
    MAX_SOURCE_LEN = cfg["max_source_length"]
    MAX_TARGET_LEN = cfg["max_target_length"]
    TASK_PREFIX    = effective_task_prefix(MODEL_NAME, cfg.get("task_prefix"))

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
    print(f"  HF hub     : {'online (downloads OK)' if args.online else 'offline-only'}")
    print(f"  Source len : {MAX_SOURCE_LEN}   Target len : {MAX_TARGET_LEN}")
    print(f"  Task prefix: '{TASK_PREFIX}'")
    print(f"  Variants   : {args.variants}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_raw    = load_dataset("knkarthick/samsum")

    variant_fns = {
        "with_speakers": make_preprocess_fn(tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX),
        "no_speakers":   make_preprocess_no_speakers_fn(tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX),
        "split_speakers": make_preprocess_split_speakers_fn(tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, TASK_PREFIX, stride=256),
    }

    model_slug = MODEL_NAME.replace("/", "_")
    cache_suffix = str(cfg.get("tokenized_cache_suffix") or "").strip()

    for variant in args.variants:
        out_path = cache_dir / f"samsum_{variant}_{model_slug}{cache_suffix}"
        if out_path.exists():
            print(f"  ⚡  {variant}: already cached at {out_path.name} — skipping")
            continue

        print(f"  Processing '{variant}'...")
        fn = variant_fns[variant]

        # split_speakers changes example count (sliding windows), so we must
        # use batched=True with remove_columns and allow the function to
        # return a different number of rows than the input.
        tokenized = ds_raw.map(
            fn,
            batched=True,
            remove_columns=ds_raw["train"].column_names,
            load_from_cache_file=True,
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

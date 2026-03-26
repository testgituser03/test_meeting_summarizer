#!/usr/bin/env python3
"""
extract_activations.py — Task 3 / activation extraction for focus steering.

Grading note: default “topic vs action” buckets use **heuristic** labels on gold summaries,
not PDF-perfect **human** dialogue labels. See **docs/rev-v2/TASK3_METHODOLOGY.md**.

Design:
  • Uses decoder hidden states from teacher-forced forward pass (deterministic).
  • Pools each selected decoder layer to one vector per example (mask-aware mean).
  • Labels summaries as action/topic via heuristic rules, with optional manual override.
  • Writes a compact tensor artifact for downstream steering-vector computation.

Output:
  results/activations/<run_name>_<split>_layers-<spec>.pt

Usage:
  python3 scripts/extract_activations.py
  python3 scripts/extract_activations.py --model_path models/best/facebook_bart-base_with_speakers
  python3 scripts/extract_activations.py --layers 6 8 10 12 --max_samples 4000
  python3 scripts/extract_activations.py --manual_labels results/activations/manual_labels.json
"""

# Set env vars before importing torch/transformers/datasets.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_ACTION_RE = re.compile(
    r"\b("  # modal + action/future cues
    r"will|going to|needs? to|should|must|have to|plan to|"  # modal
    r"send|call|email|schedule|book|prepare|review|check|bring|"  # action verbs
    r"follow up|confirm|share|update|deliver|submit|finish|complete"
    r")\b",
    re.IGNORECASE,
)


def heuristic_label(summary: str) -> int:
    """Return 1 for action-focused, 0 for topic-focused."""
    return 1 if _ACTION_RE.search(summary or "") else 0


def load_manual_labels(path: str | None) -> dict[tuple[str, int], int]:
    """Expected JSON: [{"split": "train", "index": 12, "label": "action"}, ...]."""
    if not path:
        return {}

    with open(path) as f:
        rows = json.load(f)

    mapping: dict[tuple[str, int], int] = {}
    for row in rows:
        split = str(row["split"])
        idx = int(row["index"])
        label_raw = str(row["label"]).strip().lower()
        if label_raw not in {"action", "topic", "0", "1"}:
            raise ValueError(f"Invalid label '{label_raw}' at ({split}, {idx}).")
        label = 1 if label_raw in {"action", "1"} else 0
        mapping[(split, idx)] = label
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract decoder activations for steering")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_path", default=None, help="Checkpoint dir (defaults to models/best/<run_name>)")
    parser.add_argument("--variant", default=None, help="Override dataset variant")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layers", nargs="+", type=int, default=[6, 8, 10, 12],
                        help="1-based decoder layer numbers")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for faster runs")
    parser.add_argument("--manual_labels", default=None,
                        help="Optional JSON path for manual labels (split/index/label)")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.variant:
        cfg["dataset_variant"] = args.variant
    if args.seed is not None:
        cfg["seed"] = args.seed

    set_all_seeds(int(cfg["seed"]))

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    model_name = cfg["model_name"]
    variant = cfg["dataset_variant"]
    run_name = cfg.get("run_name", f"{model_name.replace('/', '_')}_{variant}")

    model_path = Path(args.model_path) if args.model_path else (project_root / "models" / "best" / run_name)
    if not model_path.exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        sys.exit(1)

    dataset_path = project_root / "data" / "cache" / f"samsum_{variant}_{model_name.replace('/', '_')}"
    if not dataset_path.exists():
        print(f"❌ Tokenized dataset not found: {dataset_path}")
        sys.exit(1)

    try:
        from datasets import load_dataset, load_from_disk  # noqa: PLC0415
        from torch.utils.data import DataLoader            # noqa: PLC0415
        from transformers import (                         # noqa: PLC0415
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
        )
    except ImportError as exc:
        print(f"❌ Missing package: {exc}")
        sys.exit(1)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32,
    ).to(device)
    model.eval()

    ds_tok = load_from_disk(str(dataset_path))[args.split]
    ds_raw = load_dataset(cfg["dataset_name"])[args.split]

    n_total = len(ds_tok)
    if len(ds_raw) != n_total:
        print(f"❌ Raw/tokenized split length mismatch: {len(ds_raw)} vs {n_total}")
        sys.exit(1)

    if args.max_samples:
        n_take = min(args.max_samples, n_total)
        ds_tok = ds_tok.select(range(n_take))
        ds_raw = ds_raw.select(range(n_take))
    else:
        n_take = n_total

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    loader = DataLoader(ds_tok, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # Resolve requested decoder layers against model depth.
    n_decoder_layers = int(getattr(model.config, "decoder_layers", 0) or getattr(model.config, "num_decoder_layers", 0) or 0)
    if n_decoder_layers <= 0:
        print("❌ Could not infer decoder depth from model config.")
        sys.exit(1)

    valid_layers = sorted({layer for layer in args.layers if 1 <= layer <= n_decoder_layers})
    if not valid_layers:
        print(f"❌ None of requested layers {args.layers} are valid for decoder depth {n_decoder_layers}.")
        sys.exit(1)

    dropped_layers = [layer for layer in args.layers if layer not in valid_layers]
    if dropped_layers:
        print(f"⚠️ Dropping out-of-range layers for this model: {dropped_layers}")

    manual_labels = load_manual_labels(args.manual_labels)

    # Build deterministic labels from raw summaries.
    label_ids = []
    label_source = []
    for i in range(n_take):
        key = (args.split, i)
        if key in manual_labels:
            label_ids.append(manual_labels[key])
            label_source.append("manual")
        else:
            label_ids.append(heuristic_label(ds_raw[i]["summary"]))
            label_source.append("heuristic")

    # Extract pooled activations: [N, L, D].
    pooled_chunks: list[torch.Tensor] = []
    start_idx = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
            decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # decoder_hidden_states[0] = embedding output; [1..L] = per-layer outputs.
            selected = []
            token_mask = decoder_attention_mask.unsqueeze(-1).to(outputs.decoder_hidden_states[1].dtype)
            denom = token_mask.sum(dim=1).clamp(min=1.0)

            for layer in valid_layers:
                hs = outputs.decoder_hidden_states[layer]  # 1-based logical layer index.
                pooled = (hs * token_mask).sum(dim=1) / denom
                selected.append(pooled)

            # [B, n_layers, hidden]
            block = torch.stack(selected, dim=1).to("cpu", dtype=torch.float16)
            pooled_chunks.append(block)

            if step % 50 == 0:
                seen = min(step * args.batch_size, n_take)
                print(f"  Processed {seen:,}/{n_take:,}")

            # Keep memory pressure low on MPS.
            if device.type == "mps":
                torch.mps.empty_cache()

            start_idx += input_ids.size(0)

    activations = torch.cat(pooled_chunks, dim=0)
    labels_tensor = torch.tensor(label_ids, dtype=torch.int64)

    if activations.size(0) != n_take:
        print(f"❌ Activation count mismatch: {activations.size(0)} vs {n_take}")
        sys.exit(1)

    action_count = int((labels_tensor == 1).sum().item())
    topic_count = int((labels_tensor == 0).sum().item())

    out_dir = project_root / "results" / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_spec = "-".join(str(x) for x in valid_layers)
    out_path = out_dir / f"{run_name}_{args.split}_layers-{layer_spec}.pt"

    payload: dict[str, Any] = {
        "meta": {
            "run_name": run_name,
            "model_name": model_name,
            "model_path": str(model_path),
            "variant": variant,
            "split": args.split,
            "n_samples": int(n_take),
            "layers": valid_layers,
            "hidden_size": int(activations.size(-1)),
            "seed": int(cfg["seed"]),
            "label_schema": {"0": "topic", "1": "action"},
            "label_counts": {"topic": topic_count, "action": action_count},
            "manual_label_count": int(sum(1 for x in label_source if x == "manual")),
            "heuristic_label_count": int(sum(1 for x in label_source if x == "heuristic")),
        },
        "activations": activations,
        "labels": labels_tensor,
        "indices": torch.arange(n_take, dtype=torch.int64),
    }

    torch.save(payload, out_path)

    print("\n✅ Activation extraction complete")
    print(f"   Output: {out_path.relative_to(project_root)}")
    print(f"   Shape : {tuple(activations.shape)} [N, n_layers, hidden]")
    print(f"   Labels: topic={topic_count}, action={action_count}")


if __name__ == "__main__":
    main()

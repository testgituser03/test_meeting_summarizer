#!/usr/bin/env python3
"""
decoding_ablation.py — Experiment 3: decoding strategy ablation.

Tests 5 generation configurations against the best fine-tuned BART-base
checkpoint.  NO RETRAINING — only generation parameters vary per config.

  D1  num_beams=4, length_penalty=0.8            (shorter outputs)
  D2  num_beams=4, length_penalty=1.0  ← baseline (matches training config)
  D3  num_beams=4, length_penalty=1.2            (longer outputs)
  D4  num_beams=8, length_penalty=1.0            (wider beam search)
  D5  num_beams=1, do_sample=True, top_p=0.9, temperature=0.8  (nucleus)

Timing — MPS deferred execution:
  MPS queues GPU work asynchronously. model.generate() returns when the
  last Metal command is *submitted*, not *completed*.  Reading perf_counter()
  immediately after generate() measures CPU→GPU submission latency, not actual
  GPU compute time.  torch.mps.synchronize() blocks until the MPS command queue
  is fully drained, so the elapsed time below is a true wall-clock measurement.

Per-config outputs (5 separate JSON files):
  results/metrics/decoding_D1_beam4_lp0.8.json
  results/metrics/decoding_D2_beam4_lp1.0.json
  results/metrics/decoding_D3_beam4_lp1.2.json
  results/metrics/decoding_D4_beam8_lp1.0.json
  results/metrics/decoding_D5_nucleus_p0.9.json

Aggregated summary:
  results/metrics/experiment_3_decoding_summary.json

Usage:
  python3 scripts/decoding_ablation.py
  python3 scripts/decoding_ablation.py --model_path models/best/facebook_bart-base_with_speakers
  python3 scripts/decoding_ablation.py --n_samples 100   # fast subset for debugging
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",   "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE",   "1")

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR_STR = str(SCRIPT_DIR)
if sys.path and sys.path[0] == SCRIPT_DIR_STR:
    sys.path.pop(0)


# ── Experiment 3 configurations ────────────────────────────────────────────────
# Control variable: generation kwargs only.
# Model, tokenizer, dataset, and all other conditions are held constant.

CONFIGS: list[dict] = [
    {
        "id":          "D1",
        "filename":    "decoding_D1_beam4_lp0.8",
        "label":       "beam4_lp0.8",
        "description": "4 beams, length_penalty=0.8 — favours shorter outputs",
        "gen_kwargs":  {"num_beams": 4, "length_penalty": 0.8, "do_sample": False,
                        "early_stopping": True},
    },
    {
        "id":          "D2",
        "filename":    "decoding_D2_beam4_lp1.0",
        "label":       "beam4_lp1.0 (baseline)",
        "description": "4 beams, length_penalty=1.0 — identical to training config",
        "gen_kwargs":  {"num_beams": 4, "length_penalty": 1.0, "do_sample": False,
                        "early_stopping": True},
    },
    {
        "id":          "D3",
        "filename":    "decoding_D3_beam4_lp1.2",
        "label":       "beam4_lp1.2",
        "description": "4 beams, length_penalty=1.2 — favours longer outputs",
        "gen_kwargs":  {"num_beams": 4, "length_penalty": 1.2, "do_sample": False,
                        "early_stopping": True},
    },
    {
        "id":          "D4",
        "filename":    "decoding_D4_beam8_lp1.0",
        "label":       "beam8_lp1.0",
        "description": "8 beams, length_penalty=1.0 — wider search, ~2× compute cost",
        "gen_kwargs":  {"num_beams": 8, "length_penalty": 1.0, "do_sample": False,
                        "early_stopping": True},
    },
    {
        "id":          "D5",
        "filename":    "decoding_D5_nucleus_p0.9",
        "label":       "nucleus_p0.9",
        "description": "nucleus sampling: top_p=0.9, temperature=0.8 (stochastic baseline)",
        "gen_kwargs":  {"num_beams": 1, "do_sample": True, "top_p": 0.9,
                        "temperature": 0.8, "length_penalty": 1.0},
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _pad_batch(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    """Right-pad variable-length sequences to the length of the longest."""
    max_len = max(len(s) for s in sequences)
    return torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in sequences],
        dtype=torch.long,
    )


def mps_memory_mb() -> float:
    return torch.mps.driver_allocated_memory() / 1_000_000 \
        if torch.backends.mps.is_available() else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E3: decoding strategy ablation (5 configs, no retraining)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Override model checkpoint path. "
             "Default: models/best/<model_name>_with_speakers",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Subsample test set for fast debugging (default: full 819 examples)",
    )
    args = parser.parse_args()

    cfg          = load_config(args.config)
    project_root = Path.cwd()
    MODEL_NAME   = cfg["model_name"]

    # ── Model path ─────────────────────────────────────────────────────────────
    # E3 always uses the with_speakers best checkpoint — the confirmed best model
    # from E1, regardless of the current dataset_variant in config.yaml.
    e3_run_name = f"{MODEL_NAME.replace('/', '_')}_with_speakers"
    model_path  = (
        Path(args.model_path)
        if args.model_path
        else project_root / "models" / "best" / e3_run_name
    )
    if not model_path.exists():
        print(f"  ❌  Checkpoint not found: {model_path}")
        print(f"       Train with: python3 scripts/train.py")
        sys.exit(1)

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        from datasets import load_from_disk                             # noqa: PLC0415
        from evaluate import load as load_metric                        # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  Import error: {exc}")
        sys.exit(1)

    # ── Device + model ─────────────────────────────────────────────────────────
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,  # Transformers 5.x API
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M  |  BF16: {cfg['use_bf16']}")
    print(f"  MPS memory : {mps_memory_mb():.1f} MB")

    # ── Dataset ────────────────────────────────────────────────────────────────
    # Always use the with_speakers tokenized cache: E3 isolates decoding only.
    dataset_cache = (
        project_root / "data" / "cache"
        / f"samsum_with_speakers_{MODEL_NAME.replace('/', '_')}"
    )
    if not dataset_cache.exists():
        print(f"  ❌  Tokenized dataset not found: {dataset_cache}")
        print(f"       Run: python3 scripts/preprocess.py")
        sys.exit(1)

    ds = load_from_disk(str(dataset_cache))["test"]
    if args.n_samples:
        ds = ds.select(range(min(args.n_samples, len(ds))))
    n = len(ds)
    print(f"  Test set   : {n:,} examples\n")

    rouge      = load_metric("rouge")
    batch_size = cfg["batch_size"]
    max_tokens = cfg["max_target_length"]
    out_dir    = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # ── Column header ──────────────────────────────────────────────────────────
    W = (30, 8, 8, 8, 10, 10)
    print(
        f"  {'Config':<{W[0]}} {'ROUGE-1':>{W[1]}} {'ROUGE-2':>{W[2]}} "
        f"{'ROUGE-L':>{W[3]}} {'avg_words':>{W[4]}} {'ms/sample':>{W[5]}}"
    )
    print(f"  {'-'*W[0]} {'-'*W[1]} {'-'*W[2]} {'-'*W[3]} {'-'*W[4]} {'-'*W[5]}")

    # ── Per-config loop ────────────────────────────────────────────────────────
    for cfg_item in CONFIGS:
        label      = cfg_item["label"]
        file_stem  = cfg_item["filename"]
        gen_kwargs = cfg_item["gen_kwargs"].copy()   # never mutate CONFIGS

        all_preds:      list[str] = []
        all_refs:       list[str] = []
        total_time_s:   float     = 0.0
        total_word_len: int       = 0

        for i in range(0, n, batch_size):
            batch = ds[i : i + batch_size]

            # Pad variable-length sequences from the tokenized on-disk cache
            input_ids      = _pad_batch(batch["input_ids"], tokenizer.pad_token_id).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                # ── MPS timing: deferred execution fix ─────────────────────────
                # MPS queues commands asynchronously. generate() returns when
                # the last command is *submitted*, not *completed*. Reading
                # perf_counter() here would measure submission latency only.
                # torch.mps.synchronize() blocks until the Metal command queue
                # is empty, so the delta below is true GPU wall-clock time.
                t_start = time.perf_counter()
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    **gen_kwargs,
                )
                if device.type == "mps":
                    torch.mps.synchronize()   # ← drain MPS queue BEFORE clock read
                total_time_s += time.perf_counter() - t_start

            # Decode predictions
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

            # Decode references (replace -100 padding before decoding)
            labels_list = [
                [tokenizer.pad_token_id if token == -100 else token for token in seq]
                for seq in batch["labels"]
            ]
            refs = tokenizer.batch_decode(labels_list, skip_special_tokens=True)

            all_preds.extend([p.strip() for p in preds])
            all_refs.extend([r.strip()  for r in refs])

            # Length = whitespace-split words in decoded output.
            # "decoded tokens" means words, NOT BPE subwords, NOT characters.
            total_word_len += sum(len(p.split()) for p in preds)

        # ── Metrics ────────────────────────────────────────────────────────────
        scores = rouge.compute(
            predictions=all_preds,
            references=all_refs,
            use_stemmer=True,
        )
        r1 = round(scores["rouge1"] * 100, 4)
        r2 = round(scores["rouge2"] * 100, 4)
        rL = round(scores["rougeL"] * 100, 4)
        avg_words     = round(total_word_len / n, 2)
        ms_per_sample = round((total_time_s  / n) * 1000, 2)

        row = {
            "config_id":            cfg_item["id"],
            "label":                label,
            "description":          cfg_item["description"],
            "rouge1":               r1,
            "rouge2":               r2,
            "rougeL":               rL,
            "avg_summary_tokens":   avg_words,     # whitespace-split words
            "ms_per_sample":        ms_per_sample,
            "n_samples":            n,
            "model_path":           str(model_path),
            "gen_kwargs":           cfg_item["gen_kwargs"],
        }
        all_rows.append(row)

        # ── Write per-config JSON ──────────────────────────────────────────────
        out_path = out_dir / f"{file_stem}.json"
        with open(out_path, "w") as fh:
            json.dump(row, fh, indent=2)

        # ── Print row ──────────────────────────────────────────────────────────
        print(
            f"  {label:<{W[0]}} {r1:>{W[1]}.2f} {r2:>{W[2]}.2f} "
            f"{rL:>{W[3]}.2f} {avg_words:>{W[4]}.1f} {ms_per_sample:>{W[5]}.1f}"
        )

        # Free MPS allocations before next config (important for beam=8 which
        # holds significantly more intermediate tensors on-device)
        if device.type == "mps":
            torch.mps.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    baseline_rL = next(r["rougeL"] for r in all_rows if r["config_id"] == "D2")
    best_row    = max(all_rows, key=lambda r: r["rougeL"])

    print(f"\n  {'─'*68}")
    print(f"  Δ ROUGE-L vs D2 baseline (beam=4, length_penalty=1.0):")
    for r in all_rows:
        delta  = r["rougeL"] - baseline_rL
        sign   = "+" if delta >= 0 else ""
        marker = " ← baseline" if r["config_id"] == "D2" else (
            " ← best" if r["rougeL"] == best_row["rougeL"] else ""
        )
        print(f"    {r['label']:<32}  {sign}{delta:.2f}{marker}")

    print(f"\n  Best ROUGE-L : {best_row['rougeL']:.2f}  ({best_row['label']})")
    print(f"  {'─'*68}\n")

    # ── Write aggregated summary ───────────────────────────────────────────────
    summary_path = out_dir / "experiment_3_decoding_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "experiment":         "E3 decoding strategy ablation",
                "model_path":         str(model_path),
                "n_samples":          n,
                "best_rougeL_config": best_row["label"],
                "configs":            all_rows,
            },
            fh,
            indent=2,
        )

    print(f"  Per-config JSONs : results/metrics/decoding_D{{1..5}}_*.json")
    print(f"  Summary          : {summary_path.relative_to(project_root)}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
error_analysis_helper.py — Manual error analysis: 20-example sample.

Selects 20 examples from the FULL 819-example SAMSum test set (seed=42,
reproducible) using the best BART-base checkpoint, then writes:
  results/error_analysis_raw.json

Each record contains:
  {
    "idx":          int,        # index in the 819-example test set
    "source":       str,        # full dialogue (raw text)
    "reference":    str,        # human-written reference summary
    "generated":    str,        # model-generated summary
    "rouge_l_score": float      # per-example ROUGE-L (0–100 scale)
  }

After running this script, open results/error_analysis_raw.json and
annotate each entry in results/error_analysis.md using the rubric there.

Usage:
  python3 scripts/error_analysis_helper.py
  python3 scripts/error_analysis_helper.py --model_path models/best/facebook_bart-base_with_speakers
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

N_SAMPLES = 20
SEED      = 42          # fixed — same 20 examples every run


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis: 20-example sample")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model_path", default=None,
                        help="Override checkpoint. Default: models/best/<model>_with_speakers")
    args = parser.parse_args()

    cfg          = _load_config(args.config)
    project_root = Path.cwd()
    MODEL_NAME   = cfg["model_name"]

    e4_run_name = f"{MODEL_NAME.replace('/', '_')}_with_speakers"
    model_path  = (
        Path(args.model_path) if args.model_path
        else project_root / "models" / "best" / e4_run_name
    )
    if not model_path.exists():
        print(f"  ❌  Checkpoint not found: {model_path}")
        sys.exit(1)

    try:
        from datasets import load_from_disk                             # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
        from rouge_score import rouge_scorer as rs                      # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  {exc}")
        sys.exit(1)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\n  Loading checkpoint: {model_path.name}  →  {device}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    # ── Load FULL 819-example test set, then sample 20 ────────────────────────
    dataset_cache = (
        project_root / "data" / "cache"
        / f"samsum_with_speakers_{MODEL_NAME.replace('/', '_')}"
    )
    if not dataset_cache.exists():
        print(f"  ❌  Tokenized dataset not found: {dataset_cache}")
        sys.exit(1)

    ds_full = load_from_disk(str(dataset_cache))["test"]
    total   = len(ds_full)          # must be 819

    # Deterministic 20-sample selection from full test set — seed=42
    rng     = random.Random(SEED)
    indices = sorted(rng.sample(range(total), N_SAMPLES))
    ds_sub  = ds_full.select(indices)

    print(f"  Test set : {total} examples  →  {N_SAMPLES} sampled (seed={SEED})")
    print(f"  Indices  : {indices}\n")

    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    records: list[dict] = []

    for pos, (sample_idx, row) in enumerate(zip(indices, ds_sub), 1):
        # dialogue: try raw text column first
        dialogue  = row.get("dialogue", "")
        reference = row.get("summary", "")

        # If raw columns are absent, decode from input_ids
        if not dialogue:
            dialogue  = tokenizer.decode(row["input_ids"], skip_special_tokens=True)
        if not reference:
            ref_ids   = [t for t in row["labels"] if t != -100]
            reference = tokenizer.decode(ref_ids, skip_special_tokens=True)

        # ── Generate summary ──────────────────────────────────────────────────
        enc = tokenizer(
            dialogue,
            return_tensors="pt",
            max_length=cfg["max_source_length"],
            truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=cfg["max_target_length"],
                num_beams=cfg["num_beams"],
                length_penalty=cfg["length_penalty"],
                early_stopping=True,
            )
            if device.type == "mps":
                torch.mps.synchronize()

        generated = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # ── Per-example ROUGE-L ───────────────────────────────────────────────
        rl = scorer.score(reference, generated)["rougeL"].fmeasure * 100

        records.append({
            "idx":           sample_idx,
            "source":        dialogue,
            "reference":     reference,
            "generated":     generated,
            "rouge_l_score": round(rl, 2),
        })

        print(f"  [{pos:>2}/{N_SAMPLES}]  idx={sample_idx:>3}  ROUGE-L={rl:.1f}")
        print(f"          REF : {reference[:90]}{'...' if len(reference)>90 else ''}")
        print(f"          GEN : {generated[:90]}{'...' if len(generated)>90 else ''}")
        print()

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = project_root / "results" / "error_analysis_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    avg_rl = sum(r["rouge_l_score"] for r in records) / len(records)
    print(f"  ── Summary ──────────────────────────────────────────────────────────")
    print(f"  {N_SAMPLES} examples  |  avg ROUGE-L = {avg_rl:.2f}")
    print(f"  Saved → {out_path.relative_to(project_root)}")
    print(f"\n  Next step: annotate results/error_analysis.md using the template.\n")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

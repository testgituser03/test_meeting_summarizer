#!/usr/bin/env python3
"""
decoding_ablation.py — Experiment 3: decoding strategy comparison.

Tests 5 generation configurations against the best BART-base checkpoint.
No retraining; only generation parameters vary.

Configurations:
  D1  beam=4,  length_penalty=0.8,  greedy
  D2  beam=4,  length_penalty=1.0,  greedy  (baseline from config.yaml)
  D3  beam=4,  length_penalty=1.2,  greedy
  D4  beam=8,  length_penalty=1.0,  greedy
  D5  beam=1,  nucleus sampling p=0.9, temp=0.8

Output: results/metrics/experiment_3_decoding.json

Usage:
  python3 scripts/decoding_ablation.py
  python3 scripts/decoding_ablation.py --model_path models/best/facebook_bart-base_with_speakers
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CONFIGS = [
    {"label": "beam4_lp0.8",   "num_beams": 4, "length_penalty": 0.8, "do_sample": False},
    {"label": "beam4_lp1.0",   "num_beams": 4, "length_penalty": 1.0, "do_sample": False},
    {"label": "beam4_lp1.2",   "num_beams": 4, "length_penalty": 1.2, "do_sample": False},
    {"label": "beam8_lp1.0",   "num_beams": 8, "length_penalty": 1.0, "do_sample": False},
    {
        "label": "nucleus_p0.9",
        "num_beams": 1, "do_sample": True,
        "top_p": 0.9, "temperature": 0.8, "length_penalty": 1.0,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="E3: decoding strategy ablation")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Subsample test set (None = full 819)")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    MODEL_NAME   = cfg["model_name"]
    VARIANT      = cfg["dataset_variant"]
    run_name     = f"{MODEL_NAME.replace('/', '_')}_{VARIANT}"

    model_path = Path(args.model_path) if args.model_path else (
        project_root / "models" / "best" / run_name
    )
    if not model_path.exists():
        print(f"  ❌  Model not found: {model_path}\n  Run train.py first.")
        sys.exit(1)

    try:
        from datasets import load_from_disk                         # noqa: PLC0415
        from evaluate import load as load_metric                    # noqa: PLC0415
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  {exc}"); sys.exit(1)

    device    = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    dataset_path = (
        project_root / "data" / "cache"
        / f"samsum_{VARIANT}_{MODEL_NAME.replace('/', '_')}"
    )
    ds = load_from_disk(str(dataset_path))["test"]
    if args.n_samples:
        ds = ds.select(range(min(args.n_samples, len(ds))))

    rouge = load_metric("rouge")
    results = []

    print(f"\n  Running {len(CONFIGS)} decoding configs on {len(ds):,} test examples...\n")

    for cfg_decoding in CONFIGS:
        label   = cfg_decoding.pop("label")
        gen_kw  = {k: v for k, v in cfg_decoding.items()}
        cfg_decoding["label"] = label  # restore

        all_preds: list[str] = []
        all_refs:  list[str] = []
        bs = cfg["batch_size"]

        for i in range(0, len(ds), bs):
            batch     = ds[i: i + bs]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=cfg["max_target_length"],
                    **gen_kw,
                )
            import numpy as np  # noqa: PLC0415
            labels = np.array(batch["labels"])
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            refs  = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
            all_preds.extend([p.strip() for p in preds])
            all_refs.extend([r.strip() for r in refs])

        scores = rouge.compute(predictions=all_preds, references=all_refs, use_stemmer=True)
        avg_len = sum(len(p.split()) for p in all_preds) / len(all_preds)
        row = {
            "label":         label,
            "rouge1":        round(scores["rouge1"] * 100, 4),
            "rouge2":        round(scores["rouge2"] * 100, 4),
            "rougeL":        round(scores["rougeL"] * 100, 4),
            "avg_len_words": round(avg_len, 1),
            **gen_kw,
        }
        results.append(row)
        print(f"  {label:<20}  R1={row['rouge1']:.2f}  R2={row['rouge2']:.2f}  RL={row['rougeL']:.2f}  avg_len={avg_len:.1f}")
        torch.mps.empty_cache() if device.type == "mps" else None

    out_path = project_root / "results" / "metrics" / "experiment_3_decoding.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"configs": results}, fh, indent=2)
    print(f"\n  ✅  Saved → {out_path.relative_to(project_root)}\n")


if __name__ == "__main__":
    main()

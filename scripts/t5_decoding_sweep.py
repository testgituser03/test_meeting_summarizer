#!/usr/bin/env python3
"""
T5-small decoding sweep — maximize ROUGE-L without retraining.

Why a separate script (not decoding_ablation.py):
  - decoding_ablation uses config.yaml model_name for the token cache path; default is BART.
  - It writes decoding_D*.json names shared with the BART E3 artifact set — do not clobber.

This sweep writes only:
  results/metrics/t5_decode_*.json
  results/metrics/t5_decoding_sweep_summary.json

Dataset: ``data/cache/samsum_with_speakers_<slug>`` from ``preprocess.py --model <alias>``.
Default checkpoint: ``models/best/t5-small_with_speakers``. Use ``--model flan-t5-base`` (etc.)
to point at FLAN checkpoints + matching cache without spelling full paths.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/t5_decoding_sweep.py --n_samples 200   # quick probe
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/t5_decoding_sweep.py                  # full 819 test set
  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/t5_decoding_sweep.py --model flan-t5-base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
# Avoid shadowing the external `evaluate` package with local scripts/evaluate.py
# when this script is invoked as `python scripts/t5_decoding_sweep.py`.
_SCRIPT_DIR_STR = str(_SCRIPT_DIR)
if sys.path and sys.path[0] == _SCRIPT_DIR_STR:
    sys.path.pop(0)
if _SCRIPT_DIR_STR not in sys.path:
    sys.path.append(_SCRIPT_DIR_STR)
from model_registry import infer_hf_id_from_run_name, resolve_model_name

# Grid: coarse + BART-sweet-spot-adjacent (T5 may prefer different lp/beam).
SWEEP_CONFIGS: list[dict] = [
    {
        "id": "T00",
        "file": "t5_decode_T00_baseline_b4_lp1",
        "label": "beam4_lp1.0_baseline",
        "desc": "Matches train.py / evaluate.py defaults",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.0, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T01",
        "file": "t5_decode_T01_b4_lp0_9",
        "label": "beam4_lp0.9",
        "desc": "Slightly shorter summaries",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 0.9, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T02",
        "file": "t5_decode_T02_b4_lp1_1",
        "label": "beam4_lp1.1",
        "desc": None,
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.1, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T03",
        "file": "t5_decode_T03_b4_lp1_2",
        "label": "beam4_lp1.2",
        "desc": None,
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.2, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T04",
        "file": "t5_decode_T04_b4_lp1_25",
        "label": "beam4_lp1.25",
        "desc": None,
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.25, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T05",
        "file": "t5_decode_T05_b4_lp1_3",
        "label": "beam4_lp1.3",
        "desc": None,
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.3, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T06",
        "file": "t5_decode_T06_b5_lp1_2",
        "label": "beam5_lp1.2",
        "desc": None,
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.2, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T07",
        "file": "t5_decode_T07_b5_lp1_33",
        "label": "beam5_lp1.33",
        "desc": "Analogous to BART D27 neighborhood",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.33, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T08",
        "file": "t5_decode_T08_b6_lp1_15",
        "label": "beam6_lp1.15",
        "desc": None,
        "gen_kwargs": {"num_beams": 6, "length_penalty": 1.15, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T09",
        "file": "t5_decode_T09_b8_lp1_0",
        "label": "beam8_lp1.0",
        "desc": "Wider beam, neutral lp",
        "gen_kwargs": {"num_beams": 8, "length_penalty": 1.0, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T10",
        "file": "t5_decode_T10_b8_lp1_2",
        "label": "beam8_lp1.2",
        "desc": None,
        "gen_kwargs": {"num_beams": 8, "length_penalty": 1.2, "do_sample": False, "early_stopping": True},
    },
    {
        "id": "T11",
        "file": "t5_decode_T11_b8_lp1_25",
        "label": "beam8_lp1.25",
        "desc": None,
        "gen_kwargs": {"num_beams": 8, "length_penalty": 1.25, "do_sample": False, "early_stopping": True},
    },
]


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pad_batch(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    return torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in sequences],
        dtype=torch.long,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T5-class (T5 / FLAN-T5) beam/length_penalty sweep on SAMSum test"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint/cache shortcut: flan-t5-base | flan-t5-small | t5-small (overrides default paths)",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Explicit best checkpoint dir (default: from --model or t5-small_with_speakers)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Explicit tokenized cache dir (default: from --model / t5-small)",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Subset of test set (default: all)")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse JSON on disk for configs that already ran (resume-friendly)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.model:
        hid = resolve_model_name(args.model)
        slug = hid.replace("/", "_")
        model_path = args.model_path or (
            PROJECT_ROOT / "models" / "best" / f"{slug}_with_speakers"
        )
        cache = args.cache or (
            PROJECT_ROOT / "data" / "cache" / f"samsum_with_speakers_{slug}"
        )
    else:
        model_path = args.model_path or (
            PROJECT_ROOT / "models" / "best" / "t5-small_with_speakers"
        )
        cache = args.cache or (
            PROJECT_ROOT / "data" / "cache" / "samsum_with_speakers_t5-small"
        )

    hf_model_id = (
        resolve_model_name(args.model)
        if args.model
        else infer_hf_id_from_run_name(model_path.name)
    )

    if not model_path.is_dir():
        print(f"  ❌  Model not found: {model_path}")
        sys.exit(1)

    if not cache.exists():
        print(f"  ❌  Tokenized cache missing: {cache}")
        print(f"      Run: python scripts/preprocess.py --model {hf_model_id}")
        sys.exit(1)

    try:
        from datasets import load_from_disk  # noqa: PLC0415
        from evaluate import load as load_metric  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as e:
        print(f"  ❌  {e}")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device      : {device}")
    print(f"  HF model id : {hf_model_id}")
    print(f"  Model       : {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Cache       : {cache.relative_to(PROJECT_ROOT)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    dtype = torch.bfloat16 if cfg.get("use_bf16") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), dtype=dtype).to(device)
    model.eval()

    ds = load_from_disk(str(cache))["test"]
    n_full = len(ds)
    n = min(args.n_samples, n_full) if args.n_samples else n_full
    ds = ds.select(range(n))
    print(f"  Test samples: {n:,} (of {n_full:,})\n")

    rouge = load_metric("rouge")
    batch_size = int(cfg.get("batch_size", 8))
    max_tokens = int(cfg.get("max_target_length", 128))
    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rl: float | None = None
    train_run_stem = model_path.name
    baseline_path = out_dir / f"{train_run_stem}_test.json"
    if baseline_path.is_file():
        try:
            baseline_rl = float(json.load(open(baseline_path, encoding="utf-8"))["rougeL"])
        except (KeyError, json.JSONDecodeError, TypeError):
            baseline_rl = None

    rows: list[dict] = []
    W = (34, 8, 8, 8, 10, 10)
    print(f"  {'Config':<{W[0]}} {'R-1':>{W[1]}} {'R-2':>{W[2]}} {'R-L':>{W[3]}} {'words':>{W[4]}} {'ms/sp':>{W[5]}}")
    print(f"  {'-'*W[0]} {'-'*W[1]} {'-'*W[2]} {'-'*W[3]} {'-'*W[4]} {'-'*W[5]}")

    for item in SWEEP_CONFIGS:
        out_path = out_dir / f"{item['file']}.json"
        if args.skip_existing and out_path.is_file():
            row = json.load(open(out_path, encoding="utf-8"))
            rows.append(row)
            print(
                f"  {row['label']:<{W[0]}} {row['rouge1']:>{W[1]}.2f} {row['rouge2']:>{W[2]}.2f} "
                f"{row['rougeL']:>{W[3]}.2f} {row['avg_summary_tokens']:>{W[4]}.1f} "
                f"{row['ms_per_sample']:>{W[5]}.1f}  (cached)"
            )
            continue

        gen_kwargs = dict(item["gen_kwargs"])
        preds_all: list[str] = []
        refs_all: list[str] = []
        total_time = 0.0
        total_words = 0

        for i in range(0, n, batch_size):
            batch = ds[i : i + batch_size]
            input_ids = _pad_batch(batch["input_ids"], tokenizer.pad_token_id).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            with torch.no_grad():
                t0 = time.perf_counter()
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    **gen_kwargs,
                )
                if device.type == "mps":
                    torch.mps.synchronize()
                total_time += time.perf_counter() - t0

            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            labels_list = [
                [tokenizer.pad_token_id if t == -100 else t for t in seq]
                for seq in batch["labels"]
            ]
            refs = tokenizer.batch_decode(labels_list, skip_special_tokens=True)
            preds_all.extend(p.strip() for p in preds)
            refs_all.extend(r.strip() for r in refs)
            total_words += sum(len(p.split()) for p in preds)

        scores = rouge.compute(predictions=preds_all, references=refs_all, use_stemmer=True)
        r1 = round(float(scores["rouge1"]) * 100, 4)
        r2 = round(float(scores["rouge2"]) * 100, 4)
        rL = round(float(scores["rougeL"]) * 100, 4)
        avg_words = round(total_words / n, 2)
        ms_sp = round((total_time / n) * 1000, 2)

        row = {
            "experiment": "t5_decoding_sweep",
            "config_id": item["id"],
            "label": item["label"],
            "description": item.get("desc"),
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rL,
            "avg_summary_tokens": avg_words,
            "ms_per_sample": ms_sp,
            "n_samples": n,
            "hf_model_id": hf_model_id,
            "model_path": str(model_path),
            "gen_kwargs": gen_kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        mark = ""
        if baseline_rl is not None:
            d = rL - baseline_rl
            mark = f"  (Δ vs {baseline_path.name} {d:+.2f})" if abs(d) >= 0.005 else ""

        print(
            f"  {item['label']:<{W[0]}} {r1:>{W[1]}.2f} {r2:>{W[2]}.2f} {rL:>{W[3]}.2f} "
            f"{avg_words:>{W[4]}.1f} {ms_sp:>{W[5]}.1f}{mark}"
        )

        if device.type == "mps":
            torch.mps.empty_cache()

    best = max(rows, key=lambda r: r["rougeL"])
    print(f"\n  Best ROUGE-L: {best['rougeL']:.4f}  ({best['label']})")
    if baseline_rl is not None:
        print(f"  Baseline ({baseline_path.name}): {baseline_rl:.4f}  "
              f"(delta {best['rougeL'] - baseline_rl:+.4f})")

    summary = {
        "experiment": "t5_decoding_sweep",
        "hf_model_id": hf_model_id,
        "model_path": str(model_path),
        "n_samples": n,
        "baseline_rougeL_from_train_artifact": baseline_rl,
        "best_config_id": best["config_id"],
        "best_label": best["label"],
        "best_rougeL": best["rougeL"],
        "best_gen_kwargs": best["gen_kwargs"],
        "configs": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = out_dir / "t5_decoding_sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {summary_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()

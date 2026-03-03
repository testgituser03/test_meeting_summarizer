#!/usr/bin/env python3
"""
train_lora.py — LoRA fine-tuning of facebook/bart-base on SAMSum.

Uses peft (Parameter-Efficient Fine-Tuning) to train only low-rank adapter
matrices on the attention q_proj and v_proj layers of BART-base.  This reduces
trainable parameters to ~1–3% of the full model while often matching or
exceeding full fine-tuning quality.

The merged model is saved to models/best/facebook_bart-base_lora/ in standard
HuggingFace format, compatible with all existing evaluation/inference scripts.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_lora.py
"""

# ── MPS fallback BEFORE torch is imported ────────────────────────────────────
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",   "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE",   "1")

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("⚠️  MPS not available — falling back to CPU", file=sys.stderr)
    return torch.device("cpu")


def mps_memory_mb() -> float:
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e6
    return 0.0


def _best_epoch_from_history(log_history: list) -> float:
    best_rl    = -1.0
    best_epoch = 0.0
    for entry in log_history:
        if "eval_rougeL" in entry and entry["eval_rougeL"] > best_rl:
            best_rl    = entry["eval_rougeL"]
            best_epoch = entry.get("epoch", 0.0)
    return best_epoch


def make_compute_metrics(tokenizer):
    from rouge_score import rouge_scorer as _rs  # noqa: PLC0415

    _scorer = _rs.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds  = [p.strip()  for p in decoded_preds]
        decoded_labels = [lb.strip() for lb in decoded_labels]

        r1_vals, r2_vals, rL_vals = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            s = _scorer.score(ref, pred)
            r1_vals.append(s["rouge1"].fmeasure)
            r2_vals.append(s["rouge2"].fmeasure)
            rL_vals.append(s["rougeL"].fmeasure)

        n = max(len(r1_vals), 1)
        return {
            "rouge1": round(sum(r1_vals) / n * 100, 4),
            "rouge2": round(sum(r2_vals) / n * 100, 4),
            "rougeL": round(sum(rL_vals) / n * 100, 4),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune BART-base on SAMSum")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Force BART-base with_speakers for LoRA experiment
    MODEL_NAME = "facebook/bart-base"
    VARIANT    = "with_speakers"
    run_name   = "facebook_bart-base_lora"

    project_root = Path.cwd()

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        from datasets import load_from_disk                            # noqa: PLC0415
        from transformers import (                                     # noqa: PLC0415
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            set_seed,
        )
        from peft import LoraConfig, get_peft_model, TaskType          # noqa: PLC0415
    except ImportError as exc:
        print(f"❌  Missing package: {exc}")
        print("   Install with: pip install peft>=0.9.0")
        sys.exit(1)

    set_seed(cfg["seed"])
    device = get_device()

    print(f"\n{'='*62}")
    print(f"  LoRA Fine-tuning : {MODEL_NAME}")
    print(f"  Variant          : {VARIANT}")
    print(f"  run_name         : {run_name}")
    print(f"  Device           : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"{'='*62}\n")

    # ── Model + tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Print trainable parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    pct = trainable_params / total_params * 100
    print(f"  Total parameters     : {total_params / 1e6:.1f}M")
    print(f"  Trainable (LoRA)     : {trainable_params / 1e6:.2f}M  ({pct:.2f}%)")
    print(f"  MPS memory (loaded)  : {mps_memory_mb():.1f} MB\n")

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_path = (
        project_root / "data" / "cache"
        / f"samsum_{VARIANT}_{MODEL_NAME.replace('/', '_')}"
    )
    if not dataset_path.exists():
        print(f"\n  ❌  Cached dataset not found: {dataset_path}")
        print("      Run preprocess.py first: python3 scripts/preprocess.py")
        sys.exit(1)

    ds = load_from_disk(str(dataset_path))
    print(f"  Dataset : {dataset_path.name}")
    print(f"  Train   : {len(ds['train']):,}  |  Val : {len(ds['validation']):,}  |  Test : {len(ds['test']):,}")

    # ── Training arguments ─────────────────────────────────────────────────────
    checkpoint_dir = project_root / "models" / "checkpoints" / run_name
    best_dir       = project_root / "models" / "best" / run_name
    metrics_dir    = project_root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(checkpoint_dir),
        num_train_epochs            = cfg["num_epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"],
        learning_rate               = cfg["learning_rate"],
        weight_decay                = cfg["weight_decay"],
        warmup_steps                = cfg["warmup_steps"],
        max_grad_norm               = cfg["gradient_clip_max_norm"],
        bf16                        = cfg["use_bf16"],
        fp16                        = cfg["use_fp16"],
        eval_strategy               = cfg["evaluation_strategy"],
        save_strategy               = cfg["save_strategy"],
        load_best_model_at_end      = cfg["load_best_model_at_end"],
        metric_for_best_model       = cfg["metric_for_best_model"],
        greater_is_better           = cfg["greater_is_better"],
        predict_with_generate       = cfg["predict_with_generate"],
        generation_max_length       = cfg["max_target_length"],
        generation_num_beams        = cfg["num_beams"],
        dataloader_num_workers      = cfg["dataloader_num_workers"],
        dataloader_pin_memory       = cfg["dataloader_pin_memory"],
        save_total_limit            = cfg["save_total_limit"],
        seed                        = cfg["seed"],
        report_to                   = "none",
        run_name                    = run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Seq2SeqTrainer(
        model              = model,
        args               = training_args,
        train_dataset      = ds["train"],
        eval_dataset       = ds["validation"],
        processing_class   = tokenizer,
        data_collator      = data_collator,
        compute_metrics    = make_compute_metrics(tokenizer),
        callbacks          = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            ),
        ],
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n  Starting LoRA training  ({MODEL_NAME})")
    print(f"  Max epochs       : {cfg['num_epochs']}  "
          f"(early stopping patience: {cfg['early_stopping_patience']})")
    print(f"  Batch size       : {cfg['batch_size']}  "
          f"|  LR: {cfg['learning_rate']}  "
          f"|  Warmup: {cfg['warmup_steps']} steps\n")

    train_start = time.time()
    trainer.train()
    training_time_sec = time.time() - train_start
    training_time_min = training_time_sec / 60.0

    print(f"\n  Training complete.  Wall time: {training_time_min:.1f} min")

    # ── Merge LoRA weights and save as standard HF checkpoint ─────────────────
    print("  Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    best_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"  Merged model saved → {best_dir.relative_to(project_root)}")

    # ── Test set evaluation ──────────────────────────────────────────────────
    best_epoch  = _best_epoch_from_history(trainer.state.log_history)
    best_val_rl = trainer.state.best_metric

    print(f"\n  Evaluating on test split ({len(ds['test']):,} examples)...")
    compute_metrics_fn = make_compute_metrics(tokenizer)
    test_output  = trainer.predict(ds["test"])
    test_metrics = compute_metrics_fn(
        (test_output.predictions, test_output.label_ids)
    )

    # ── Write output JSON ────────────────────────────────────────────────────
    result_json = {
        "model":                 MODEL_NAME,
        "variant":               "lora",
        "dataset":               cfg["dataset_name"],
        "split":                 "test",
        "n_samples":             len(ds["test"]),
        "rouge1":                test_metrics["rouge1"],
        "rouge2":                test_metrics["rouge2"],
        "rougeL":                test_metrics["rougeL"],
        "training_time_minutes": round(training_time_min, 2),
        "best_epoch":            best_epoch,
        "best_val_rougeL":       round(float(best_val_rl), 4) if best_val_rl else None,
        "lora_config": {
            "r":              16,
            "lora_alpha":     32,
            "lora_dropout":   0.05,
            "target_modules": ["q_proj", "v_proj"],
            "trainable_pct":  round(pct, 2),
        },
        "memory_profile_mb": {
            "post_load":  round(mps_memory_mb(), 1),
        },
        "generation_config": {
            "num_beams":      cfg["num_beams"],
            "max_new_tokens": cfg["max_target_length"],
            "length_penalty": cfg["length_penalty"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = metrics_dir / f"{run_name}_test.json"
    with open(out_path, "w") as fh:
        json.dump(result_json, fh, indent=2)

    print(f"\n{'='*62}")
    print(f"  LoRA Test ROUGE Results — {MODEL_NAME}")
    print(f"{'='*62}")
    print(f"  ROUGE-1 : {test_metrics['rouge1']:.2f}")
    print(f"  ROUGE-2 : {test_metrics['rouge2']:.2f}")
    print(f"  ROUGE-L : {test_metrics['rougeL']:.2f}")
    print(f"\n  Training time     : {training_time_min:.1f} min")
    print(f"  Trainable params  : {trainable_params / 1e6:.2f}M ({pct:.2f}%)")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Saved JSON        → {out_path.relative_to(project_root)}")
    print(f"{'='*62}\n")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

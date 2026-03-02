#!/usr/bin/env python3
"""
train.py — Fine-tune seq2seq model (BART-base or T5-small) on SAMSum.

Reads ALL hyperparameters from config.yaml — nothing is hardcoded.
Requires pre-tokenized dataset in data/cache/ (run preprocess.py first).

Outputs:
  models/checkpoints/<run_name>/  — all per-epoch checkpoints
  models/best/<run_name>/         — best checkpoint by rougeL
  results/metrics/<run_name>_test.json — final test-set metrics

Usage:
  python3 scripts/train.py
  python3 scripts/train.py --model t5-small
  python3 scripts/train.py --model facebook/bart-base --variant no_speakers
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Strict MPS mode: crash on unsupported ops rather than silent CPU fallback.
# Set to "1" in config or shell if you hit "MPS not supported" errors.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("⚠️  MPS not available — falling back to CPU", file=sys.stderr)
    return torch.device("cpu")


def make_compute_metrics(tokenizer):
    """Return a compute_metrics function bound to the given tokenizer."""
    from evaluate import load as load_metric  # noqa: PLC0415
    rouge = load_metric("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # -100 is the ignore index; replace before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        # Multiply by 100 and round to 4 dp; rouge_score returns 0–1
        return {k: round(v * 100, 4) for k, v in result.items()}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune seq2seq on SAMSum")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--model",   default=None, help="Override model_name in config")
    parser.add_argument("--variant", default=None, help="Override dataset_variant (with/no_speakers)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    if args.variant:
        cfg["dataset_variant"] = args.variant

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()

    from datasets import load_from_disk                         # noqa: PLC0415
    from transformers import (                                  # noqa: PLC0415
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(cfg["seed"])
    device     = get_device()
    MODEL_NAME = cfg["model_name"]
    VARIANT    = cfg["dataset_variant"]
    run_name   = f"{MODEL_NAME.replace('/', '_')}_{VARIANT}"

    print(f"\n{'='*62}")
    print(f"  Fine-tuning  : {MODEL_NAME}")
    print(f"  Variant      : {VARIANT}")
    print(f"  Device       : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"{'='*62}\n")

    # ── Model + tokenizer ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")
    if device.type == "mps":
        print(f"  MPS memory : {torch.mps.driver_allocated_memory() / 1e6:.1f} MB  (post-load)")

    # ── Dataset ────────────────────────────────────────────────────────────
    dataset_path = (
        project_root / "data" / "cache"
        / f"samsum_{VARIANT}_{MODEL_NAME.replace('/', '_')}"
    )
    if not dataset_path.exists():
        print(f"\n  ❌  Cached dataset not found: {dataset_path}")
        print("      Run preprocess.py first: python3 scripts/preprocess.py")
        sys.exit(1)

    ds = load_from_disk(str(dataset_path))
    print(f"\n  Dataset : {dataset_path.name}")
    print(f"  Train   : {len(ds['train']):,}  |  Val : {len(ds['validation']):,}  |  Test : {len(ds['test']):,}")

    # ── Training arguments ─────────────────────────────────────────────────
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
        model           = model,
        args            = training_args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["validation"],
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = make_compute_metrics(tokenizer),
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        ],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Starting training  (up to {cfg['num_epochs']} epochs)...\n")
    trainer.train()

    # ── Save best ──────────────────────────────────────────────────────────
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"\n  Best model saved → {best_dir.relative_to(project_root)}")

    # ── Evaluate on test set ───────────────────────────────────────────────
    print("  Evaluating on test set...")
    test_output = trainer.predict(ds["test"])
    metrics = make_compute_metrics(tokenizer)(
        (test_output.predictions, test_output.label_ids)
    )
    metrics.update({
        "model":               MODEL_NAME,
        "variant":             VARIANT,
        "epochs_trained":      round(trainer.state.epoch, 2),
        "best_val_rougeL":     trainer.state.best_metric,
    })

    out_path = metrics_dir / f"{run_name}_test.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n{'='*62}")
    print("  Test ROUGE Results")
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if k in metrics:
            print(f"    {k:>10} : {metrics[k]:.2f}")
    print(f"\n  Saved → {out_path.relative_to(project_root)}")
    print(f"{'='*62}\n")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

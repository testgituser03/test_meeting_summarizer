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

# ── CRITICAL: Set MPS fallback BEFORE torch is imported ─────────────────────
# Must be the very first executable statement — even a bare `import torch`
# initialises the MPS driver, making any subsequent setenv a no-op.
# "1" = CPU fallback for unsupported MPS ops (safe for seq2seq edge cases).
# "0" = strict mode (crash on unsupported op) — enable once loop is MPS-clean.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("⚠️  MPS not available — falling back to CPU", file=sys.stderr)
    return torch.device("cpu")


def mps_memory_mb() -> float:
    """Return MPS driver-allocated memory in MB; 0.0 if MPS is unavailable."""
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1e6
    return 0.0


def _best_epoch_from_history(log_history: list) -> float:
    """
    Scan trainer.state.log_history to find the epoch at which eval_rougeL
    peaked.  Returns 0.0 if no eval entries are present (e.g. early abort).
    Early stopping fires when rougeL fails to improve for `patience` epochs;
    best_epoch will typically be 2–4 for T5-small on SAMSum.
    """
    best_rl    = -1.0
    best_epoch = 0.0
    for entry in log_history:
        if "eval_rougeL" in entry and entry["eval_rougeL"] > best_rl:
            best_rl    = entry["eval_rougeL"]
            best_epoch = entry.get("epoch", 0.0)
    return best_epoch


def make_compute_metrics(tokenizer):
    """
    Return a compute_metrics closure for Seq2SeqTrainer.

    Called after EVERY validation epoch (training eval path) AND by
    trainer.predict() for the final test evaluation (test eval path).
    Both paths use beam-search generation because predict_with_generate=True
    is set in Seq2SeqTrainingArguments — the same generate() call, same
    num_beams / max_new_tokens.  Scores are directly comparable between the
    two paths and are NOT inflated by teacher forcing.

    Implementation contract
    ───────────────────────
    • predict_with_generate=True → eval_preds.predictions are integer token
      IDs produced by model.generate(), NOT softmax logits.
    • Labels contain -100 (Transformers ignore-index) at padded positions;
      must be replaced with tokenizer.pad_token_id before decoding.
    • ROUGE: macro-average F-measure across all samples (equal weight per
      sample, no length bias).  Consistent with baseline_zeroshot.py.
    • use_stemmer=True: Porter stemming ("running" → "run").  Standard for
      SAMSum leaderboard and published BART/T5 SAMSum results.
    • Values × 100 → 0–100 scale for human-readable Trainer log lines.
    """
    from rouge_score import rouge_scorer as _rs  # noqa: PLC0415

    _scorer = _rs.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # Step 1: Replace -100 (ignore index) with pad_token_id.
        # In Transformers 5.x, both generated predictions AND labels are padded
        # with -100 when sequences in a batch differ in length.  The fast
        # tokenizer's Rust backend raises OverflowError on negative token IDs,
        # so we must sanitise both tensors before calling batch_decode.
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Step 2: Decode token IDs → strings.
        # skip_special_tokens=True removes pad / eos / bos / sentinel tokens.
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Step 3: Strip leading/trailing whitespace.
        # T5 SentencePiece tokenizer occasionally prefixes decoded text with " ".
        decoded_preds  = [p.strip()  for p in decoded_preds]
        decoded_labels = [lb.strip() for lb in decoded_labels]

        # Step 4: Compute macro-average ROUGE F-measure.
        r1_vals, r2_vals, rL_vals = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            s = _scorer.score(ref, pred)
            r1_vals.append(s["rouge1"].fmeasure)
            r2_vals.append(s["rouge2"].fmeasure)
            rL_vals.append(s["rougeL"].fmeasure)

        n = max(len(r1_vals), 1)  # guard: avoid ZeroDivisionError on empty batch
        return {
            "rouge1": round(sum(r1_vals) / n * 100, 4),
            "rouge2": round(sum(r2_vals) / n * 100, 4),
            "rougeL": round(sum(rL_vals) / n * 100, 4),
        }

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
        TrainerCallback,
        set_seed,
    )

    set_seed(cfg["seed"])

    # ── MPS memory callback: records driver-allocated MB after epoch 1 eval ──
    # on_evaluate() is called by Seq2SeqTrainer at the end of each epoch's
    # validation run — the first call captures peak memory after the GPU graph
    # has been compiled and a full forward+generate pass has been executed.
    class _MpsMemoryCallback(TrainerCallback):  # noqa: PLC0415
        """Record MPS driver-allocated memory after the first evaluation call."""
        def __init__(self) -> None:
            self.post_epoch1_mb: float = 0.0
            self._recorded: bool = False

        def on_evaluate(self, args, state, control, **kwargs):  # noqa: ANN001
            if not self._recorded:
                self.post_epoch1_mb = mps_memory_mb()
                self._recorded = True
                print(f"  MPS memory (post-epoch1): {self.post_epoch1_mb:.1f} MB")

    memory_cb  = _MpsMemoryCallback()

    device     = get_device()
    MODEL_NAME = cfg["model_name"]
    VARIANT    = cfg["dataset_variant"]
    run_name   = cfg.get("run_name", f"{MODEL_NAME.replace('/', '_')}_{VARIANT}")

    print(f"\n{'='*62}")
    print(f"  Fine-tuning  : {MODEL_NAME}")
    print(f"  Variant      : {VARIANT}")
    print(f"  run_name     : {run_name}")
    print(f"  Device       : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"  MPS fallback : {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'unset')}")
    print(f"{'='*62}\n")

    # ── Memory baseline (before model load) ───────────────────────────────
    mem_pre_load = mps_memory_mb()
    print(f"  MPS memory (pre-load)  : {mem_pre_load:.1f} MB")

    # ── Model + tokenizer ───────────────────────────────────────────────
    # Init with torch_dtype=bfloat16 BEFORE .to(device) to avoid a transient
    # FP32 copy in memory when converting an already-loaded FP32 model to MPS.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)

    n_params      = sum(p.numel() for p in model.parameters()) / 1e6
    mem_post_load = mps_memory_mb()
    print(f"  Parameters             : {n_params:.1f}M")
    print(f"  MPS memory (post-load) : {mem_post_load:.1f} MB")

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
        lr_scheduler_type           = cfg.get("lr_scheduler_type", "linear"),
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
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            ),
            memory_cb,
        ],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    # Epoch 1 is slower: MPSGraph compiles and caches Metal GPU kernels on
    # first execution.  T5-small estimate: epoch 1 ≈ 7–9 min, epochs 2+
    # ≈ 4–6 min (graph cache warm).  BART-base: epoch 1 ≈ 14–18 min, 2+
    # ≈ 10–13 min.  Early stopping at epoch 3 or 4 is EXPECTED and CORRECT
    # — it signals convergence, not failure.  Record best_epoch from JSON.
    print(f"\n  Starting training  ({MODEL_NAME}, {VARIANT})")
    print(f"  Max epochs       : {cfg['num_epochs']}  "
          f"(early stopping patience: {cfg['early_stopping_patience']})")
    print(f"  Batch size       : {cfg['batch_size']}  "
          f"|  LR: {cfg['learning_rate']}  "
          f"|  Warmup: {cfg['warmup_steps']} steps")
    print(f"  MPS memory (pre) : {mps_memory_mb():.1f} MB\n")

    train_start = time.time()
    trainer.train()
    training_time_sec = time.time() - train_start
    training_time_min = training_time_sec / 60.0

    mem_post_train  = mps_memory_mb()
    mem_post_epoch1 = memory_cb.post_epoch1_mb  # 0.0 if training aborted before first eval
    print(f"\n  Training complete.")
    print(f"  Wall time        : {training_time_min:.1f} min  ({training_time_sec:.0f}s)")
    print(f"  MPS memory (post): {mem_post_train:.1f} MB")

    # ── Save best model ─────────────────────────────────────────────────────
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"  Best model saved → {best_dir.relative_to(project_root)}")

    # ── Extract best epoch from training history ────────────────────────────
    best_epoch  = _best_epoch_from_history(trainer.state.log_history)
    best_val_rl = trainer.state.best_metric   # rougeL on validation, 0–100 scale
    print(f"  Best epoch       : {best_epoch}")
    if best_val_rl:
        print(f"  Best val rougeL  : {best_val_rl:.4f}")

    # ── Test set evaluation ─────────────────────────────────────────────────
    # trainer.predict() uses model.generate() because predict_with_generate=True
    # is set in Seq2SeqTrainingArguments.  This is the SAME generation path as
    # validation eval: beam=4, max_new_tokens=128 from training_args.
    # Scores are directly comparable to per-epoch validation rougeL values.
    print(f"\n  Evaluating on test split ({len(ds['test']):,} examples)...")
    compute_metrics_fn = make_compute_metrics(tokenizer)
    test_output  = trainer.predict(ds["test"])
    test_metrics = compute_metrics_fn(
        (test_output.predictions, test_output.label_ids)
    )

    # ── Build and write output JSON ─────────────────────────────────────────
    result_json = {
        "model":                 MODEL_NAME,
        "variant":               VARIANT,
        "run_name":              run_name,
        "dataset":               cfg["dataset_name"],
        "split":                 "test",
        "n_samples":             len(ds["test"]),
        "rouge1":                test_metrics["rouge1"],
        "rouge2":                test_metrics["rouge2"],
        "rougeL":                test_metrics["rougeL"],
        "training_time_minutes": round(training_time_min, 2),
        "best_epoch":            best_epoch,
        "best_val_rougeL":       round(float(best_val_rl), 4) if best_val_rl else None,
        "memory_profile_mb": {
            "pre_load":    round(mem_pre_load,    1),
            "post_load":   round(mem_post_load,   1),
            "post_epoch1": round(mem_post_epoch1, 1),
            "post_train":  round(mem_post_train,  1),
        },
        "generation_config": {
            "num_beams":      cfg["num_beams"],
            "max_new_tokens": cfg["max_target_length"],
            "length_penalty": cfg["length_penalty"],
            "early_stopping": cfg["early_stopping_beam"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = metrics_dir / f"{run_name}_test.json"
    with open(out_path, "w") as fh:
        json.dump(result_json, fh, indent=2)

    print(f"\n{'='*62}")
    print(f"  Test ROUGE Results — {MODEL_NAME} ({VARIANT})")
    print(f"{'='*62}")
    print(f"  ROUGE-1 : {test_metrics['rouge1']:.2f}")
    print(f"  ROUGE-2 : {test_metrics['rouge2']:.2f}")
    print(f"  ROUGE-L : {test_metrics['rougeL']:.2f}")
    print(f"\n  Training time : {training_time_min:.1f} min")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Saved JSON    → {out_path.relative_to(project_root)}")
    print(f"{'='*62}\n")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    if device.type == "mps":
        torch.mps.empty_cache()
        print(f"  MPS memory (post-cleanup): {mps_memory_mb():.1f} MB")


if __name__ == "__main__":
    main()

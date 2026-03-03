#!/usr/bin/env python3
"""
pegasus_experiment.py — Experiment 5: PEGASUS-CNN/DailyMail on SAMSum.

Tests google/pegasus-cnn_dailymail (568M parameters) as a third architecture
alongside T5-small (60M) and BART-base (139M).  Provides zero-shot evaluation
and fine-tuning with the same evaluation protocol as the existing experiments.

PEGASUS was pre-trained with gap sentence generation (GSG) on a diverse news
corpus, making it strong at extractive-leaning abstractive summarization.
However, SAMSum dialogues are structurally different from news — this tests
cross-domain transfer.

Outputs:
  results/metrics/zeroshot_google_pegasus-cnn_dailymail.json  (zero-shot E0)
  results/metrics/google_pegasus-cnn_dailymail_with_speakers_test.json  (fine-tuned E1)
  models/best/google_pegasus-cnn_dailymail_with_speakers/  (fine-tuned checkpoint)

Usage:
  # Step 1: Download model (online, run once)
  python3 scripts/pegasus_experiment.py --download

  # Step 2: Zero-shot evaluation (offline, ~5 min)
  python3 scripts/pegasus_experiment.py --zeroshot

  # Step 3: Preprocess tokenized cache (offline, ~10 sec)
  python3 scripts/pegasus_experiment.py --preprocess

  # Step 4: Fine-tune (offline, ~2-3 hours on MPS)
  python3 scripts/pegasus_experiment.py --train

  # Step 5: Run all steps in sequence
  python3 scripts/pegasus_experiment.py --all
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

PEGASUS_MODEL = "google/pegasus-cnn_dailymail"
PEGASUS_SLUG  = PEGASUS_MODEL.replace("/", "_")

# PEGASUS has no task prefix — same as BART
TASK_PREFIX = ""

# Reduced batch sizes for 568M model on 24GB UMA
EVAL_BATCH_SIZE  = 1
TRAIN_BATCH_SIZE = 1  # 768M params requires batch=1 on 24GB MPS

# PEGASUS-specific: reduce source length to fit in MPS memory
PEGASUS_MAX_SOURCE = 256  # 512 causes OOM; 256 covers 95%+ of SAMSum dialogues


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


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_pegasus():
    """Download PEGASUS model and tokenizer to HuggingFace cache (online)."""
    # Remove offline flags for download
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from transformers import AutoModelForSeq2SeqLM, PegasusTokenizer

    print(f"\n  Downloading tokenizer: {PEGASUS_MODEL}")
    PegasusTokenizer.from_pretrained(PEGASUS_MODEL)
    print(f"  Downloading model weights: {PEGASUS_MODEL} (~2.2GB)")
    AutoModelForSeq2SeqLM.from_pretrained(PEGASUS_MODEL)
    print(f"  ✅ PEGASUS cached successfully\n")


# ── Step 2: Zero-shot evaluation ──────────────────────────────────────────────

def zeroshot_eval(cfg: dict):
    """Zero-shot ROUGE evaluation on 100-sample SAMSum test subset."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datasets import load_dataset
    from rouge_score import rouge_scorer
    from transformers import AutoModelForSeq2SeqLM, PegasusTokenizer

    device = get_device()
    n_samples = 100

    print(f"\n{'─' * 62}")
    print(f"  PEGASUS Zero-Shot Evaluation (E0)")
    print(f"  Model   : {PEGASUS_MODEL}")
    print(f"  Samples : {n_samples}")
    print(f"  Device  : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"{'─' * 62}")

    tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        PEGASUS_MODEL,
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")
    print(f"  MPS memory : {mps_memory_mb():.1f} MB")

    ds = load_dataset("knkarthick/samsum")
    ds_test = ds["test"].shuffle(seed=cfg["seed"]).select(range(n_samples))
    dialogues  = ds_test["dialogue"]
    references = ds_test["summary"]

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    max_target = cfg["max_target_length"]
    gen_kwargs = {
        "num_beams": 4,
        "length_penalty": 1.0,
        "do_sample": False,
        "early_stopping": True,
    }

    all_preds = []
    t_start = time.perf_counter()

    for i in range(0, n_samples, EVAL_BATCH_SIZE):
        batch_texts = dialogues[i : i + EVAL_BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            max_length=PEGASUS_MAX_SOURCE,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_target,
                **gen_kwargs,
            )
            if device.type == "mps":
                torch.mps.synchronize()

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in preds])

        if (i // EVAL_BATCH_SIZE) % 5 == 0:
            print(f"    Batch {i // EVAL_BATCH_SIZE + 1}/{(n_samples + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE}")

    elapsed = time.perf_counter() - t_start

    # Compute ROUGE
    r1_vals, r2_vals, rL_vals = [], [], []
    for pred, ref in zip(all_preds, references):
        s = scorer.score(ref.lower(), pred.lower())
        r1_vals.append(s["rouge1"].fmeasure)
        r2_vals.append(s["rouge2"].fmeasure)
        rL_vals.append(s["rougeL"].fmeasure)

    result = {
        "model":      PEGASUS_MODEL,
        "n_samples":  n_samples,
        "rouge1":     round(np.mean(r1_vals) * 100, 2),
        "rouge2":     round(np.mean(r2_vals) * 100, 2),
        "rougeL":     round(np.mean(rL_vals) * 100, 2),
        "generation_config": gen_kwargs,
        "elapsed_seconds":   round(elapsed, 1),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zeroshot_{PEGASUS_SLUG}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ROUGE-1 : {result['rouge1']:.2f}")
    print(f"  ROUGE-2 : {result['rouge2']:.2f}")
    print(f"  ROUGE-L : {result['rougeL']:.2f}")
    print(f"  Time    : {elapsed:.1f}s ({elapsed / n_samples * 1000:.0f} ms/sample)")
    print(f"  Saved   : {out_path}\n")

    # Cleanup GPU memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


# ── Step 3: Preprocess ────────────────────────────────────────────────────────

def preprocess_pegasus(cfg: dict):
    """Tokenize SAMSum with PEGASUS tokenizer and save to data/cache/."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datasets import load_dataset
    from transformers import PegasusTokenizer

    tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)
    ds_raw = load_dataset("knkarthick/samsum")

    max_source = PEGASUS_MAX_SOURCE  # reduced for memory; 256 covers 95%+ of SAMSum
    max_target = cfg["max_target_length"]

    def preprocess_with_speakers(batch: dict) -> dict:
        inputs = [TASK_PREFIX + d for d in batch["dialogue"]]
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

    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"samsum_with_speakers_{PEGASUS_SLUG}"

    if out_path.exists():
        print(f"  ⚡  PEGASUS tokenized cache already exists: {out_path.name}")
        return

    print(f"\n  Tokenizing SAMSum for PEGASUS...")
    tokenized = ds_raw.map(
        preprocess_with_speakers,
        batched=True,
        remove_columns=ds_raw["train"].column_names,
        desc="Tokenizing (PEGASUS with_speakers)",
    )
    tokenized.save_to_disk(str(out_path))
    print(f"  ✅ Saved: {out_path}")
    print(f"     Train: {len(tokenized['train']):,}  "
          f"Val: {len(tokenized['validation']):,}  "
          f"Test: {len(tokenized['test']):,}\n")


# ── Step 4: Fine-tuning ──────────────────────────────────────────────────────

def fine_tune_pegasus(cfg: dict):
    """Fine-tune PEGASUS on SAMSum with_speakers variant."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datasets import load_from_disk
    from rouge_score import rouge_scorer as _rs
    from transformers import (
        AutoModelForSeq2SeqLM,
        PegasusTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    device = get_device()
    run_name = f"{PEGASUS_SLUG}_with_speakers"
    model_out = Path("models/best") / run_name
    ckpt_dir  = Path("models/checkpoints") / run_name

    print(f"\n{'─' * 62}")
    print(f"  PEGASUS Fine-Tuning on SAMSum")
    print(f"  Model     : {PEGASUS_MODEL}")
    print(f"  Output    : {model_out}")
    print(f"  Device    : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE} (reduced for 568M model)")
    print(f"{'─' * 62}")

    # Load tokenized dataset
    cache_path = Path("data/cache") / f"samsum_with_speakers_{PEGASUS_SLUG}"
    if not cache_path.exists():
        print(f"  ❌ Tokenized cache not found: {cache_path}")
        print(f"     Run: python3 scripts/pegasus_experiment.py --preprocess")
        sys.exit(1)

    ds = load_from_disk(str(cache_path))
    tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)

    # Disable MPS memory watermark to allow full 24GB usage
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Load model with gradient checkpointing for memory efficiency
    model = AutoModelForSeq2SeqLM.from_pretrained(
        PEGASUS_MODEL,
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    )
    model.gradient_checkpointing_enable()  # trade compute for memory

    # Aggressively free cached MPS memory before training
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M (gradient checkpointing ON)")

    # ROUGE compute_metrics
    _scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [lb.strip() for lb in tokenizer.batch_decode(labels, skip_special_tokens=True)]

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

    # Training arguments — conservative for large model
    num_epochs = 3  # fewer epochs for larger model
    grad_accum = 8  # effective batch = 1 * 8 = 8 (batch=1 for MPS memory)
    warmup_steps = 300
    lr = 2e-5  # lower lr for larger pre-trained model

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=cfg["use_bf16"],
        predict_with_generate=True,
        generation_max_length=cfg["max_target_length"],
        generation_num_beams=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=cfg["seed"],
        logging_steps=50,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        run_name=run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    t_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - t_start
    train_minutes = round(train_time / 60, 1)

    print(f"\n  Training complete: {train_minutes} min")

    # Save best model
    model_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    print(f"  ✅ Model saved: {model_out}")

    # Evaluate on test set
    print(f"\n  Evaluating on test set ({len(ds['test'])} examples)...")
    test_results = trainer.evaluate(eval_dataset=ds["test"])

    best_epoch = 0.0
    for entry in trainer.state.log_history:
        if "eval_rougeL" in entry:
            if entry["eval_rougeL"] > best_epoch:
                best_epoch = entry.get("epoch", 0.0)

    result = {
        "model":       PEGASUS_MODEL,
        "variant":     "with_speakers",
        "n_samples":   len(ds["test"]),
        "rouge1":      round(test_results.get("eval_rouge1", 0), 2),
        "rouge2":      round(test_results.get("eval_rouge2", 0), 2),
        "rougeL":      round(test_results.get("eval_rougeL", 0), 2),
        "best_epoch":  best_epoch,
        "training_time_minutes": train_minutes,
        "batch_size":  TRAIN_BATCH_SIZE,
        "gradient_accumulation": grad_accum,
        "learning_rate": lr,
        "num_epochs":  num_epochs,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{PEGASUS_SLUG}_with_speakers_test.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  ROUGE-1 : {result['rouge1']:.2f}")
    print(f"  ROUGE-2 : {result['rouge2']:.2f}")
    print(f"  ROUGE-L : {result['rougeL']:.2f}")
    print(f"  Time    : {train_minutes} min")
    print(f"  Saved   : {out_path}\n")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PEGASUS experiment on SAMSum")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--download",   action="store_true", help="Download PEGASUS model (online)")
    parser.add_argument("--zeroshot",   action="store_true", help="Run zero-shot evaluation")
    parser.add_argument("--preprocess", action="store_true", help="Tokenize SAMSum for PEGASUS")
    parser.add_argument("--train",      action="store_true", help="Fine-tune PEGASUS on SAMSum")
    parser.add_argument("--all",        action="store_true", help="Run all steps in sequence")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.all:
        args.download = args.zeroshot = args.preprocess = args.train = True

    if not any([args.download, args.zeroshot, args.preprocess, args.train]):
        parser.print_help()
        sys.exit(0)

    if args.download:
        download_pegasus()

    if args.zeroshot:
        zeroshot_eval(cfg)

    if args.preprocess:
        preprocess_pegasus(cfg)

    if args.train:
        fine_tune_pegasus(cfg)


if __name__ == "__main__":
    main()

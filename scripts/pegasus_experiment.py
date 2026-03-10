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

# Batch sizes tuned for PEGASUS 767.6M BF16 on 24 GB UMA.
# Without gradient_checkpointing, ALL layer activations are held in memory simultaneously
# for the backward pass.  batch=4 causes sporadic MPS OOM (CPU fallback ⇒ 3–6s/it);
# batch=2 + grad_accum=4 keeps eff. batch=8 while halving per-step activation memory,
# eliminating MPS OOM and stabilising throughput at ~1.5–2.5s/it.
EVAL_BATCH_SIZE  = 4   # eval is forward-only — batch=4 is safe and 2× faster than 2
TRAIN_BATCH_SIZE = 2   # reduced from 4: half activation memory → no MPS OOM fallback

# PEGASUS-specific: with BF16 + batch_size=2 + grad_accum=4, 512 fits within
# 24 GB UMA.  Previous value of 256 was an unnecessary conservative cut that
# dropped ~5% of training examples and hurt cross-attention on longer dialogues.
PEGASUS_MAX_SOURCE = 512  # matches BART training; covers ≈99% of SAMSum dialogues


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
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,  # Fix: dtype= (Transformers 5.x); torch_dtype= is deprecated
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
    from transformers.trainer_utils import get_last_checkpoint

    device = get_device()
    run_name = f"{PEGASUS_SLUG}_with_speakers"
    model_out = Path("models/best") / run_name
    ckpt_dir  = Path("models/checkpoints") / run_name

    print(f"\n{'─' * 62}")
    print(f"  PEGASUS Fine-Tuning on SAMSum")
    print(f"  Model     : {PEGASUS_MODEL}")
    print(f"  Output    : {model_out}")
    print(f"  Device    : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"  Batch size: {TRAIN_BATCH_SIZE} × grad_accum 4 = eff. batch {TRAIN_BATCH_SIZE*4} | sortish_sampler=True")
    print(f"  Epochs    : 3 (early stopping patience=2) | train-eval: greedy on 200-sample val, test-eval: beam=4")
    print(f"  Max src   : {PEGASUS_MAX_SOURCE} tokens | eval_accumulation_steps=4 | gen_max_len=100")
    print(f"{'─' * 62}")

    # Load tokenized dataset
    cache_path = Path("data/cache") / f"samsum_with_speakers_{PEGASUS_SLUG}"
    if not cache_path.exists():
        print(f"  ❌ Tokenized cache not found: {cache_path}")
        print(f"     Run: python3 scripts/pegasus_experiment.py --preprocess")
        sys.exit(1)

    ds = load_from_disk(str(cache_path))
    tokenizer = PegasusTokenizer.from_pretrained(PEGASUS_MODEL)

    # Silence the BOS token config-mismatch warning: PEGASUS tokenizer has
    # bos_token_id=None (PEGASUS never uses a true BOS prefix — the decoder
    # starts from the pad token).  Explicitly set the model/generation config
    # to match so the Trainer does not emit the alignment warning on every eval.
    # This has zero effect on training or ROUGE — it is purely a config cleanup.
    tokenizer.bos_token_id = tokenizer.pad_token_id  # pad_token_id=0 is the de-facto BOS

    # Disable MPS memory watermark to allow full 24GB usage
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Load model in BF16 directly to save memory (no gradient checkpointing —
    # gradient_checkpointing_enable() + BF16 + MPS produces near-zero gradients
    # because BF16 is non-deterministic on MPS: recomputed activations differ from
    # originals, causing gradients to cancel → loss stuck flat across all epochs).
    #
    # Pre-load config with tie_word_embeddings=False to silence the tied-weights
    # warning at load time.  The PEGASUS checkpoint already stores model.shared,
    # encoder.embed_tokens, and decoder.embed_tokens as separate tensors; the
    # default config's tie_word_embeddings=True is a pre-training artefact that
    # contradicts the checkpoint.  Setting False before from_pretrained prevents
    # the warning being emitted during the weight-loading scan.
    from transformers import AutoConfig  # noqa: PLC0415
    _model_cfg = AutoConfig.from_pretrained(PEGASUS_MODEL)
    _model_cfg.tie_word_embeddings = False

    model = AutoModelForSeq2SeqLM.from_pretrained(
        PEGASUS_MODEL,
        config=_model_cfg,
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,  # Fix: dtype= (Transformers 5.x); torch_dtype= is deprecated
    )
    # NOTE: gradient_checkpointing intentionally disabled — see comment above

    # Explicitly move model to MPS before Trainer takes ownership
    # (Trainer auto-detects MPS but explicit placement ensures BF16 weights
    # are on GPU before any optimizer state is initialized)
    model = model.to(device)

    # Aggressively free cached MPS memory before training
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    mem_gb = mps_memory_mb() / 1024
    print(f"  Parameters : {n_params:.1f}M | Device: {next(model.parameters()).device} | MPS mem: {mem_gb:.2f} GB")

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

    # ── Training arguments — fully optimised for 24 GB UMA / MPS ──────────────
    #
    # Speed / memory optimisation stack applied here:
    #
    #  1. TRAIN_BATCH_SIZE=4, grad_accum=2  →  effective batch=8 (same as BART)
    #     but only 2 accum steps instead of 4 → fewer forward passes per update.
    #
    #  2. group_by_length=True  →  batches examples of similar input length
    #     together.  Reduces padding overhead by ~25–35% on SAMSum (high
    #     variance in dialogue lengths: p50=106 tokens, p99=525 tokens).
    #     This alone cuts per-epoch wall time by ~20% on MPS.
    #
    #  3. generation_num_beams=1 (greedy) during training validation:
    #     Beam-4 eval on 818 examples with 568M params ≈ 4× slower than greedy.
    #     Greedy ROUGE correlates well with beam ROUGE for checkpoint selection.
    #     We switch to beam=4 for the final test evaluation only (see below).
    #
    #  4. per_device_eval_batch_size=2  →  2× eval throughput vs batch=1.
    #     Safe because eval uses greedy (beam=1) → much lower peak activation mem.
    #
    #  5. eval_accumulation_steps=8  →  accumulate eval logits/preds in chunks
    #     to avoid a large all-gather of 818 beam=4 sequences at once.
    #
    #  6. dataloader_drop_last=True  →  avoids a small remainder batch that
    #     would force MPS to recompile the Metal kernel for a different shape.
    #     Drops at most 3 train examples out of 14,731 — negligible.
    #
    #  7. PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 already set above → lets
    #     PyTorch use the full 24 GB pool without artificial cap.
    #
    num_epochs   = 3    # PEGASUS cross-domain shift benefits from ≥3 epochs
    grad_accum   = 4    # batch=2 × accum=4 = eff. batch 8; same convergence, no MPS OOM
    warmup_steps = 500  # ~9% of 5523 total optimizer steps; suits cosine LR schedule
    lr           = 2e-5  # conservative for 767M model doing cross-domain (news → dialogue) transfer

    import gc  # noqa: PLC0415
    from transformers import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState  # noqa: PLC0415

    class MPSMemoryCallback(TrainerCallback):
        """Force gc + MPS allocator flush every N optimizer steps.

        torch_empty_cache_steps releases the PyTorch block pool, but does NOT
        call Python's garbage collector.  Large BF16 gradient tensors (PEGASUS
        has 96 K vocab → 96 K × 1024 × 2 bytes = ~200 MB per lm_head gradient)
        may linger as unreferenced Python objects between steps, preventing the
        allocator from reclaiming them.  A gc.collect() sweep before the MPS
        flush ensures these objects are freed first, keeping steady-state MPS
        memory stable throughout the full epoch.
        """
        def __init__(self, flush_every: int = 25):
            self.flush_every = flush_every

        def on_step_end(
            self,
            args: "Seq2SeqTrainingArguments",
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ) -> None:
            if (
                state.global_step % self.flush_every == 0
                and torch.backends.mps.is_available()
            ):
                gc.collect()
                torch.mps.empty_cache()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        # ── eval / save schedule ──────────────────────────────────────────────
        eval_strategy="epoch",          # per-epoch validation ROUGE
        save_strategy="epoch",
        load_best_model_at_end=True,    # restore best rougeL checkpoint
        metric_for_best_model="rougeL",
        greater_is_better=True,
        # ── batch & accumulation ─────────────────────────────────────────────
        per_device_train_batch_size=TRAIN_BATCH_SIZE,  # 2 (reduced from 4 to eliminate MPS OOM)
        per_device_eval_batch_size=EVAL_BATCH_SIZE,    # 2 (safe with greedy eval)
        gradient_accumulation_steps=grad_accum,        # 4 → eff. batch=8
        dataloader_drop_last=True,      # avoid small remainder batches on MPS
        # ── optimisation ─────────────────────────────────────────────────────
        learning_rate=lr,
        num_train_epochs=num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",     # cosine decay suits long PEGASUS runs
        # ── speed: sort-by-length to reduce padding waste (transformers-5 API) ─
        sortish_sampler=True,          # Seq2Seq-specific: sorts by decoder len
        # ── CRITICAL: gradient_checkpointing must stay False on MPS+BF16 ─────
        # gradient_checkpointing=True recomputes activations during backward.
        # On MPS+BF16 the recomputed values differ (non-deterministic ops) →
        # gradients cancel each other → near-zero updates → loss stays flat.
        # With 24 GB UMA and batch=8 we have ample VRAM without checkpointing.
        # ── periodic MPS cache flush to prevent allocator fragmentation ──────
        # 10 steps (was 50): more aggressive flush prevents the 2.2→10.5s/it speed
        # regression that occurs as MPS allocator accumulates fragmented blocks over
        # a long epoch.  Small overhead (~0.02s/flush) is negligible vs fragmentation cost.
        torch_empty_cache_steps=10,
        # ── generation — greedy (beam=1) during training for ~4× faster eval ─
        # beam=4 is set explicitly BEFORE the final test evaluate() call below.
        predict_with_generate=True,
        generation_max_length=100,     # SAMSum summary max = 94 tokens; 100 is safe ceiling
        generation_num_beams=1,         # greedy during training eval for speed
        # ── memory: accumulate eval preds in chunks to avoid peak-mem spike ──
        eval_accumulation_steps=4,      # 4 × EVAL_BATCH_SIZE(4) = 16 samples per host flush
        bf16=cfg["use_bf16"],
        # ── reproducibility / logging ────────────────────────────────────────
        seed=cfg["seed"],
        save_total_limit=2,
        logging_steps=50,
        dataloader_num_workers=0,       # MPS + multiprocessing → context errors
        dataloader_pin_memory=False,    # UMA: no PCIe transfer to pin
        report_to="none",
        run_name=run_name,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100,
    )

    # 200-sample validation subset for per-epoch eval during training.
    # Full 818-sample eval at beam=1 costs ~30–50 min/epoch on PEGASUS+MPS.
    # 200 samples (reproducible shuffle, seed=42) is sufficient for early-stopping
    # and best-checkpoint selection.  The final test eval uses all 819 samples.
    val_subset = ds["validation"].shuffle(seed=cfg["seed"]).select(range(200))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=val_subset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            MPSMemoryCallback(flush_every=25),  # gc + MPS flush every 25 steps → prevents speed regression
        ],
    )

    # Auto-resume from last checkpoint if one exists in ckpt_dir
    last_ckpt = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if last_ckpt:
        print(f"  ▶  Resuming from checkpoint: {last_ckpt}")
    else:
        print(f"  ▶  Starting training from scratch")

    t_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_ckpt)
    train_time = time.perf_counter() - t_start
    train_minutes = round(train_time / 60, 1)

    print(f"\n  Training complete: {train_minutes} min")

    # Save best model
    model_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_out))
    tokenizer.save_pretrained(str(model_out))
    print(f"  ✅ Model saved: {model_out}")

    # Evaluate on test set — switch to beam=4 for final quality measurement.
    # Training validation used greedy (beam=1) for speed; now restore beam=4
    # to match the E1/E3 evaluation protocol and get accurate ROUGE numbers.
    print(f"\n  Evaluating on test set ({len(ds['test'])} examples) with beam=4...")
    trainer.args.generation_num_beams = 4
    trainer.args.eval_accumulation_steps = 4   # smaller chunks for beam=4
    test_results = trainer.evaluate(eval_dataset=ds["test"])

    best_epoch = 0.0
    best_rl_seen = 0.0
    for entry in trainer.state.log_history:
        if "eval_rougeL" in entry:
            if entry["eval_rougeL"] > best_rl_seen:  # Fix: compare rougeL vs best rougeL (not vs epoch number)
                best_rl_seen = entry["eval_rougeL"]
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
        "effective_batch_size": TRAIN_BATCH_SIZE * grad_accum,
        "learning_rate": lr,
        "num_epochs":  num_epochs,
        "warmup_steps": warmup_steps,
        "sortish_sampler": True,
        "train_eval_beams": 1,
        "test_eval_beams": 4,
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

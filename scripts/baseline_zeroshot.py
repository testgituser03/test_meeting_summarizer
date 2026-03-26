#!/usr/bin/env python3
"""
baseline_zeroshot.py — Zero-shot ROUGE evaluation for Experiment 0 (E0).

Tests t5-small and facebook/bart-base with NO fine-tuning on a deterministic
100-sample subset of the SAMSum test split. Establishes the performance floor
before any fine-tuning; results are consumed by compare_experiments.py.

Subset selection:
  ds_test.shuffle(seed=cfg["seed"]).select(range(n_samples))
  HuggingFace Dataset.shuffle(seed=42) applies a deterministic Fisher–Yates
  permutation — the SAME 100 samples are returned on every run.

ROUGE computation:
  Macro-average: per-sample F-measures (rouge_score library, use_stemmer=True,
  both prediction and reference lowercased), then arithmetic mean across samples.
  Each sample contributes weight 1/n — no length bias.

Outputs:
  results/metrics/zeroshot_t5-small.json
  results/metrics/zeroshot_facebook_bart-base.json

JSON schema: model, n_samples, rouge1, rouge2, rougeL, generation_config, timestamp

Usage:
  python3 scripts/baseline_zeroshot.py
  python3 scripts/baseline_zeroshot.py --n_samples 50
  python3 scripts/baseline_zeroshot.py --models t5-small
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Enforce offline mode BEFORE any HuggingFace imports ─────────────────────
# All models and the SAMSum dataset must already be in ~/.cache/huggingface/
# (run scripts/predownload_assets.py first if not yet cached).
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402 (import after env-var setup)
import yaml   # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from model_registry import effective_task_prefix, resolve_model_name

# Conservative batch size — avoids OOM spikes during first-run MPSGraph
# kernel compilation, which temporarily peaks higher than steady-state.
_BATCH_SIZE = 4


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("⚠️  MPS not available — falling back to CPU", file=sys.stderr)
    return torch.device("cpu")


# ── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    model_id: str,
    dialogues: list,
    references: list,
    cfg: dict,
    device: torch.device,
) -> dict:
    """
    Zero-shot generation + ROUGE for one model.

    Args:
        model_id:   HuggingFace model ID (e.g. "t5-small")
        dialogues:  List of raw dialogue strings (NOT fine-tuning inputs)
        references: List of gold summary strings
        cfg:        Loaded config.yaml dict
        device:     torch.device("mps") or torch.device("cpu")

    Returns:
        Dict matching the required JSON schema:
        { model, n_samples, rouge1, rouge2, rougeL, generation_config, timestamp }

    ROUGE values are on the 0–100 scale (consistent with train.py / evaluate.py).
    """
    from rouge_score import rouge_scorer                        # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    resolved    = resolve_model_name(model_id)
    task_prefix = effective_task_prefix(resolved, cfg.get("task_prefix"))
    n_samples   = len(dialogues)

    print(f"\n{'─' * 62}")
    print(f"  Model      : {resolved}" + (f"  (from '{model_id}')" if resolved != model_id else ""))
    print(f"  Samples    : {n_samples}")
    print(f"  Task prefix: '{task_prefix}'")
    print(f"  Device     : {device}  |  BF16: {cfg['use_bf16']}")
    print(f"{'─' * 62}")

    # ── Load model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        resolved,
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    if device.type == "mps":
        mem_mb = torch.mps.driver_allocated_memory() / 1e6
        print(f"  MPS memory (post-load) : {mem_mb:.1f} MB")

    # ── Batched beam-search generation ──────────────────────────────────
    predictions: list[str] = []

    for batch_start in range(0, n_samples, _BATCH_SIZE):
        batch_dialogues   = dialogues[batch_start : batch_start + _BATCH_SIZE]
        inputs_with_prefix = [task_prefix + d for d in batch_dialogues]

        # Pad to longest sequence in batch; attention_mask handles variable lengths.
        encoded = tokenizer(
            inputs_with_prefix,
            max_length=cfg["max_source_length"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                num_beams        = cfg["num_beams"],       # 4 (from config)
                max_new_tokens   = cfg["max_target_length"],  # 128
                early_stopping   = cfg["early_stopping_beam"],
                length_penalty   = cfg["length_penalty"],  # 1.0
            )

        batch_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.extend(batch_preds)

        done = min(batch_start + _BATCH_SIZE, n_samples)
        print(f"  Generated {done:>3}/{n_samples}", end="\r", flush=True)

    print(f"  Generated {n_samples}/{n_samples} ✓                     ")

    # ── ROUGE: macro-average over samples ───────────────────────────────
    # Macro-average: compute F-measure for each (prediction, reference) pair
    # independently, then take the arithmetic mean. Equal weight per sample
    # regardless of dialogue or summary length.
    # Both strings are lowercased; use_stemmer=True for Porter stemming.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1_list: list[float] = []
    r2_list: list[float] = []
    rL_list: list[float] = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref.lower(), pred.lower())
        r1_list.append(scores["rouge1"].fmeasure)
        r2_list.append(scores["rouge2"].fmeasure)
        rL_list.append(scores["rougeL"].fmeasure)

    # Multiply by 100 → 0–100 scale; round to 4 dp (matches train.py)
    rouge1 = round(sum(r1_list) / len(r1_list) * 100, 4)
    rouge2 = round(sum(r2_list) / len(r2_list) * 100, 4)
    rougeL = round(sum(rL_list) / len(rL_list) * 100, 4)

    print(f"\n  ROUGE-1 : {rouge1:.2f}")
    print(f"  ROUGE-2 : {rouge2:.2f}")
    print(f"  ROUGE-L : {rougeL:.2f}")

    result = {
        "model":    resolved,
        "n_samples": n_samples,
        "rouge1":   rouge1,
        "rouge2":   rouge2,
        "rougeL":   rougeL,
        "generation_config": {
            "num_beams":      cfg["num_beams"],
            "max_new_tokens": cfg["max_target_length"],
            "length_penalty": cfg["length_penalty"],
            "early_stopping": cfg["early_stopping_beam"],
            "task_prefix":    task_prefix,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── Cleanup ─────────────────────────────────────────────────────────
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
        mem_mb = torch.mps.driver_allocated_memory() / 1e6
        print(f"  MPS memory (post-cleanup): {mem_mb:.1f} MB")

    return result


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot ROUGE baseline for E0 — no fine-tuning"
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Test-set samples to evaluate (default: 100; max: 819)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["t5-small", "facebook/bart-base"],
        help=(
            "Hub ids or aliases (flan-t5-base, flan-t5-small, t5-small, bart-base, …). "
            "Default: t5-small facebook/bart-base"
        ),
    )
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    metrics_dir  = project_root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        print(f"❌  Missing package: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Load SAMSum test split from local HF cache ───────────────────────
    print("\n  Loading SAMSum test split from local cache (offline mode)...")
    ds_test = load_dataset(cfg["dataset_name"], split="test")

    # ── Deterministic subset ─────────────────────────────────────────────
    # Dataset.shuffle(seed=N) applies a deterministic Fisher–Yates permutation
    # seeded by N. The same seed always yields the same shuffled order, so
    # the same 100 examples are selected on every run — fully reproducible.
    n_samples = min(args.n_samples, len(ds_test))
    ds_subset = ds_test.shuffle(seed=cfg["seed"]).select(range(n_samples))

    dialogues  = ds_subset["dialogue"]
    references = ds_subset["summary"]

    print(f"  SAMSum test: {len(ds_test):,} total examples")
    print(f"  Subset     : {n_samples} samples  "
          f"(shuffle seed={cfg['seed']}, select first {n_samples})")
    print(f"  Zero-shot  : no fine-tuning — establishes performance floor (E0)")

    # ── Evaluate each model in sequence ─────────────────────────────────
    # Models are evaluated sequentially (not in parallel) so that
    # torch.mps.empty_cache() fully reclaims GPU memory between runs.
    all_results: dict = {}

    for model_id in args.models:
        result = evaluate_model(
            model_id   = model_id,
            dialogues  = dialogues,
            references = references,
            cfg        = cfg,
            device     = device,
        )
        resolved_key = resolve_model_name(model_id)
        all_results[resolved_key] = result

        # Write immediately after each model — don't lose results if the
        # second model crashes (e.g. OOM on first run of BART).
        slug     = resolved_key.replace("/", "_")
        out_path = metrics_dir / f"zeroshot_{slug}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\n  ✅  Saved → {out_path.relative_to(project_root)}")

    # ── Comparison table ─────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'═' * 62}")
        print("  E0 Zero-Shot Baseline — Comparison Table")
        print(f"{'═' * 62}")
        print(f"  {'Model':<28} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}")
        print(f"  {'─' * 28} {'─' * 8} {'─' * 8} {'─' * 8}")
        for model_id, res in sorted(all_results.items()):
            print(
                f"  {model_id:<28} "
                f"{res['rouge1']:>8.2f} "
                f"{res['rouge2']:>8.2f} "
                f"{res['rougeL']:>8.2f}"
            )
        print(f"{'═' * 62}")
        print("\n  NOTE: These are pre-training baselines with NO fine-tuning.")
        print("  Fine-tuned models should significantly exceed these values.")
        print("  BART-base zero-shot is expected to be LOWER than T5-small")
        print("  because T5 was pre-trained on summarization tasks (C4 + multi-task).")
        print()


if __name__ == "__main__":
    main()

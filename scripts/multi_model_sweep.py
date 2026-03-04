#!/usr/bin/env python3
"""
multi_model_sweep.py — Run best decoding configs on every fine-tuned model.

For each model checkpoint that isn't the main BART-base-with-speakers, this
script applies the top decoding configurations and writes per-result JSONs to
results/metrics/.  The experiment_3_decoding_summary.json (BART-only) is NOT
touched — this script writes its own aggregate file.

Models tested
─────────────
  facebook_bart-base_lora       (E2 LoRA, currently RL=37.59 at beam4/lp1.0)
  facebook_bart-base_extended   (E5 extended training, RL=38.46 at beam4/lp1.0)
  t5-small_with_speakers        (E1 T5, RL=31.95 at beam4/lp1.0)

Decoding configs applied to each model
───────────────────────────────────────
  baseline  beam=4, lp=1.0   (reproduce *_test.json baseline for comparison)
  D10       beam=6, lp=1.2   (best on BART-base → 40.03)
  D12       beam=6, lp=1.25  (new sweet spot, also run on BART)
  D13       beam=6, lp=1.3   (wide beam + high lp)
  D14       beam=8, lp=1.2   (maximum beam at optimal lp)

Output
──────
  results/metrics/sweep_<model_tag>_<config_id>.json    (per run)
  results/metrics/multi_model_sweep_summary.json        (all models × all configs)

Usage
─────
  python3 scripts/multi_model_sweep.py
  python3 scripts/multi_model_sweep.py --n_samples 50   # fast smoke test
  python3 scripts/multi_model_sweep.py --models lora extended
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE",   "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE",  "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Remove the scripts/ directory from sys.path so that `import evaluate`
# resolves to the HuggingFace `evaluate` package, not scripts/evaluate.py
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPTS_DIR:
    sys.path.pop(0)
# Also remove it if it appears elsewhere (edge case when run from scripts/)
sys.path = [p for p in sys.path if p != _SCRIPTS_DIR]

# ── Model registry ─────────────────────────────────────────────────────────────
# key        : short tag used in filenames and CLI --models filter
# model_path : relative to PROJECT_ROOT
# dataset_cache : relative to PROJECT_ROOT  (with_speakers tokenized dataset)
# model_type : "bart" | "t5"  — controls decoder_start_token_id handling
MODEL_REGISTRY: dict[str, dict] = {
    "lora": {
        "tag":           "lora",
        "label":         "BART-base LoRA (r=16)",
        "model_path":    "models/best/facebook_bart-base_lora",
        "dataset_cache": "data/cache/samsum_with_speakers_facebook_bart-base",
        "model_type":    "bart",
        "baseline_rl":   37.59,   # from *_test.json for reference
    },
    "extended": {
        "tag":           "extended",
        "label":         "BART-base Extended (8ep, cosine)",
        "model_path":    "models/best/facebook_bart-base_extended",
        "dataset_cache": "data/cache/samsum_with_speakers_facebook_bart-base",
        "model_type":    "bart",
        "baseline_rl":   38.46,
    },
    "t5": {
        "tag":           "t5",
        "label":         "T5-small (with speakers)",
        "model_path":    "models/best/t5-small_with_speakers",
        "dataset_cache": "data/cache/samsum_with_speakers_t5-small",
        "model_type":    "t5",
        "baseline_rl":   31.95,
    },
    "no_speakers": {
        "tag":           "no_speakers",
        "label":         "BART-base (no speakers)",
        "model_path":    "models/best/facebook_bart-base_no_speakers",
        "dataset_cache": "data/cache/samsum_no_speakers_facebook_bart-base",
        "model_type":    "bart",
        "baseline_rl":   33.23,
    },
}

# ── Decoding configs to sweep ──────────────────────────────────────────────────
SWEEP_CONFIGS: list[dict] = [
    {
        "id":       "baseline",
        "label":    "beam4_lp1.0",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.0,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D10",
        "label":    "beam6_lp1.2",
        "gen_kwargs": {"num_beams": 6, "length_penalty": 1.2,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D12",
        "label":    "beam6_lp1.25",
        "gen_kwargs": {"num_beams": 6, "length_penalty": 1.25,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D13",
        "label":    "beam6_lp1.3",
        "gen_kwargs": {"num_beams": 6, "length_penalty": 1.3,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D14",
        "label":    "beam8_lp1.2",
        "gen_kwargs": {"num_beams": 8, "length_penalty": 1.2,
                       "do_sample": False, "early_stopping": True},
    },
    # ── beam=4 ridge: also crossed 40 on BART-base ─────────────────────────────
    {
        "id":       "D7",
        "label":    "beam4_lp1.25",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.25,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D8",
        "label":    "beam4_lp1.3",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.3,
                       "do_sample": False, "early_stopping": True},
    },
    # ── beam=5 ridge: the dominant zone discovered in full ablation on BART-base ──
    # All configs below achieved ROUGE-L ≥ 40.0 on the base BART-base checkpoint.
    # Testing them here reveals whether the beam=5 optimum transfers to other models.
    {
        "id":       "D17",
        "label":    "beam5_lp1.2",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.2,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D19",
        "label":    "beam5_lp1.3",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.3,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D21",
        "label":    "beam4_lp1.28",
        "gen_kwargs": {"num_beams": 4, "length_penalty": 1.28,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D22",
        "label":    "beam5_lp1.28",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.28,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D23",
        "label":    "beam5_lp1.32",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.32,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D24",
        "label":    "beam5_lp1.35",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.35,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D25",
        "label":    "beam5_lp1.4",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.4,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D27",
        "label":    "beam5_lp1.33",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.33,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D28",
        "label":    "beam5_lp1.37",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.37,
                       "do_sample": False, "early_stopping": True},
    },
    {
        "id":       "D29",
        "label":    "beam5_lp1.45",
        "gen_kwargs": {"num_beams": 5, "length_penalty": 1.45,
                       "do_sample": False, "early_stopping": True},
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pad_batch(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    return torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in sequences],
        dtype=torch.long,
    )


def evaluate_config(
    model,
    tokenizer,
    ds,
    gen_kwargs: dict,
    batch_size: int,
    max_tokens: int,
    device: torch.device,
    rouge_metric,
) -> tuple[float, float, float, float, float]:
    """Run inference on ds with gen_kwargs.  Returns (r1, r2, rL, avg_words, ms/sample)."""
    all_preds: list[str] = []
    all_refs:  list[str] = []
    total_time_s  = 0.0
    total_word_len = 0
    n = len(ds)

    for i in range(0, n, batch_size):
        batch = ds[i: i + batch_size]
        input_ids      = _pad_batch(batch["input_ids"], tokenizer.pad_token_id).to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            t_start = time.perf_counter()
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                **gen_kwargs,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            total_time_s += time.perf_counter() - t_start

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        labels_list = [
            [tokenizer.pad_token_id if tok == -100 else tok for tok in seq]
            for seq in batch["labels"]
        ]
        refs = tokenizer.batch_decode(labels_list, skip_special_tokens=True)

        all_preds.extend([p.strip() for p in preds])
        all_refs.extend([r.strip()  for r in refs])
        total_word_len += sum(len(p.split()) for p in preds)

    scores = rouge_metric.compute(
        predictions=all_preds, references=all_refs, use_stemmer=True
    )
    r1 = round(scores["rouge1"] * 100, 4)
    r2 = round(scores["rouge2"] * 100, 4)
    rL = round(scores["rougeL"] * 100, 4)
    avg_words     = round(total_word_len / n, 2)
    ms_per_sample = round((total_time_s / n) * 1000, 2)
    return r1, r2, rL, avg_words, ms_per_sample


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model decoding sweep — run top configs on LoRA / Extended / T5"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model(s) to sweep (default: all)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Subsample test set for fast debugging",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml for batch_size / use_bf16 / max_target_length",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    batch_size = cfg["batch_size"]
    max_tokens = cfg["max_target_length"]
    use_bf16   = cfg.get("use_bf16", True)

    try:
        from datasets import load_from_disk                              # noqa: PLC0415
        from evaluate import load as load_metric                         # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer   # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  Import error: {exc}")
        sys.exit(1)

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    rouge = load_metric("rouge")
    out_dir = PROJECT_ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict] = []

    print(f"\n  Device : {device}")
    print(f"  BF16   : {use_bf16}")
    print(f"  Models : {args.models}\n")

    for model_key in args.models:
        minfo      = MODEL_REGISTRY[model_key]
        model_path = PROJECT_ROOT / minfo["model_path"]
        ds_path    = PROJECT_ROOT / minfo["dataset_cache"]

        if not model_path.exists():
            print(f"  ⚠️   Checkpoint missing: {model_path}  — skipping {model_key}")
            continue
        if not ds_path.exists():
            print(f"  ⚠️   Dataset cache missing: {ds_path}  — skipping {model_key}")
            continue

        print(f"  ═══ {minfo['label']} ═══")
        print(f"      Checkpoint : {model_path.relative_to(PROJECT_ROOT)}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
        ).to(device)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"      Parameters : {n_params:.1f}M")

        ds = load_from_disk(str(ds_path))["test"]
        if args.n_samples:
            ds = ds.select(range(min(args.n_samples, len(ds))))
        n = len(ds)
        print(f"      Test set   : {n:,} examples")
        print(f"      Baseline   : ROUGE-L = {minfo['baseline_rl']:.2f} (beam4/lp1.0)\n")

        # ── Column header ──────────────────────────────────────────────────────
        W = (22, 8, 8, 8, 10, 10)
        print(
            f"      {'Config':<{W[0]}} {'ROUGE-1':>{W[1]}} {'ROUGE-2':>{W[2]}} "
            f"{'ROUGE-L':>{W[3]}} {'avg_words':>{W[4]}} {'ms/samp':>{W[5]}}"
        )
        print(f"      {'-'*W[0]} {'-'*W[1]} {'-'*W[2]} {'-'*W[3]} {'-'*W[4]} {'-'*W[5]}")

        model_rows: list[dict] = []

        for dcfg in SWEEP_CONFIGS:
            cfg_id  = dcfg["id"]
            label   = dcfg["label"]
            gk      = dcfg["gen_kwargs"].copy()

            # Cache file name: sweep_<model_tag>_<cfg_id>.json
            cache_file = out_dir / f"sweep_{model_key}_{cfg_id}.json"
            if cache_file.exists():
                with open(cache_file) as fh:
                    row = json.load(fh)
                model_rows.append(row)
                print(
                    f"      {label:<{W[0]}} {row['rouge1']:>{W[1]}.2f} "
                    f"{row['rouge2']:>{W[2]}.2f} {row['rougeL']:>{W[3]}.2f} "
                    f"{row['avg_summary_tokens']:>{W[4]}.1f} "
                    f"{row['ms_per_sample']:>{W[5]}.1f}  (cached)"
                )
                continue

            r1, r2, rL, avg_words, ms = evaluate_config(
                model, tokenizer, ds, gk, batch_size, max_tokens, device, rouge
            )
            row = {
                "model_key":          model_key,
                "model_label":        minfo["label"],
                "model_path":         str(model_path),
                "config_id":          cfg_id,
                "label":              label,
                "gen_kwargs":         gk,
                "rouge1":             r1,
                "rouge2":             r2,
                "rougeL":             rL,
                "avg_summary_tokens": avg_words,
                "ms_per_sample":      ms,
                "n_samples":          n,
            }
            model_rows.append(row)
            with open(cache_file, "w") as fh:
                json.dump(row, fh, indent=2)

            delta = rL - minfo["baseline_rl"]
            sign  = "+" if delta >= 0 else ""
            marker = f"  Δ{sign}{delta:.2f}" if cfg_id != "baseline" else ""
            print(
                f"      {label:<{W[0]}} {r1:>{W[1]}.2f} {r2:>{W[2]}.2f} "
                f"{rL:>{W[3]}.2f} {avg_words:>{W[4]}.1f} {ms:>{W[5]}.1f}{marker}"
            )

            if device.type == "mps":
                torch.mps.empty_cache()

        # ── Per-model best ─────────────────────────────────────────────────────
        best = max(model_rows, key=lambda r: r["rougeL"])
        print(f"\n      Best ROUGE-L: {best['rougeL']:.4f}  ({best['label']})")
        delta = best["rougeL"] - minfo["baseline_rl"]
        sign  = "+" if delta >= 0 else ""
        print(f"      vs beam4/lp1.0 baseline: {sign}{delta:.2f}\n")

        all_summary_rows.extend(model_rows)

        # Free GPU memory before loading next model
        del model
        del tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()

    # ── Overall best ───────────────────────────────────────────────────────────
    if all_summary_rows:
        best_overall = max(all_summary_rows, key=lambda r: r["rougeL"])
        print(f"  ═══ Overall best across all swept models ═══")
        print(f"      {best_overall['model_label']}  "
              f"{best_overall['label']}  →  ROUGE-L {best_overall['rougeL']:.4f}")

    # ── Write aggregate summary ────────────────────────────────────────────────
    summary_path = out_dir / "multi_model_sweep_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(
            {
                "experiment": "Multi-model decoding sweep",
                "description": (
                    "Top decoding configs applied to LoRA, Extended, and T5 models."
                ),
                "n_models": len(args.models),
                "n_configs_per_model": len(SWEEP_CONFIGS),
                "results": all_summary_rows,
            },
            fh,
            indent=2,
        )
    print(f"\n  Summary written → {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

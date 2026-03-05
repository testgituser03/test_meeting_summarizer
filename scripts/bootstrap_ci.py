#!/usr/bin/env python3
"""
bootstrap_ci.py — Compute bootstrap confidence intervals for E1 architecture comparison.

Loads BART-base and T5-small best checkpoints, generates summaries for the full
819-sample SAMSum test set, computes per-sample ROUGE-1/2/L scores, and runs
bootstrap resampling (1000 iterations) to produce 95% confidence intervals.

Also computes the bootstrap CI for the Δ ROUGE-L (BART − T5) to test whether
the architecture difference is statistically significant at p < 0.05.

Outputs:
  results/metrics/bootstrap_ci_e1.json  — per-model CIs + paired Δ CI
  Updates: results/metrics/facebook_bart-base_with_speakers_test.json  (adds bootstrap_ci key)
  Updates: results/metrics/t5-small_with_speakers_test.json            (adds bootstrap_ci key)

Usage:
  python3 scripts/bootstrap_ci.py
  python3 scripts/bootstrap_ci.py --n_bootstrap 5000  # more iterations for tighter CI
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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


def _pad_batch(
    sequences: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length sequences; return (input_ids, attention_mask)."""
    max_len = max(len(s) for s in sequences)
    ids  = torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in sequences], dtype=torch.long
    )
    mask = (ids != pad_id).long()
    return ids, mask


def generate_per_sample_rouge(
    model_path: str,
    dataset,
    cfg: dict,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, list[float]]:
    """Generate summaries and return per-sample ROUGE-{1,2,L} F-measures (0–100 scale)."""
    from rouge_score import rouge_scorer as rs  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    n = len(dataset)

    r1_vals: list[float] = []
    r2_vals: list[float] = []
    rL_vals: list[float] = []

    max_new_tokens = cfg["max_target_length"]
    num_beams      = cfg["num_beams"]
    length_penalty = cfg["length_penalty"]

    t_start = time.perf_counter()

    for i in range(0, n, batch_size):
        batch = dataset[i : i + batch_size]
        ids_list = batch["input_ids"]
        input_ids, attention_mask = _pad_batch(ids_list, tokenizer.pad_token_id)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True,
            )
            if device.type == "mps":
                torch.mps.synchronize()

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Decode reference labels (replace -100 → pad)
        label_seqs = batch["labels"]
        refs = tokenizer.batch_decode(
            [
                [tokenizer.pad_token_id if t == -100 else t for t in seq]
                for seq in label_seqs
            ],
            skip_special_tokens=True,
        )

        for pred, ref in zip(preds, refs):
            pred = pred.strip()
            ref  = ref.strip()
            s = scorer.score(ref, pred)
            r1_vals.append(s["rouge1"].fmeasure * 100)
            r2_vals.append(s["rouge2"].fmeasure * 100)
            rL_vals.append(s["rougeL"].fmeasure * 100)

        done = min(i + batch_size, n)
        if (done // batch_size) % 20 == 0 or done == n:
            elapsed = time.perf_counter() - t_start
            print(f"    {done:>4}/{n}  ({elapsed:.0f}s elapsed)")

    elapsed = time.perf_counter() - t_start
    print(f"    Generation complete: {elapsed:.1f}s ({elapsed / n * 1000:.0f} ms/sample)")

    # Cleanup
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    return {"rouge1": r1_vals, "rouge2": r2_vals, "rougeL": rL_vals}


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 42
) -> dict:
    """Compute bootstrap confidence interval for the mean of `values`.

    Returns: {mean, ci_low, ci_high, std, n_bootstrap, confidence}
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()

    alpha = 1 - confidence
    ci_low  = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean":         round(float(values.mean()), 4),
        "ci_low":       round(ci_low, 4),
        "ci_high":      round(ci_high, 4),
        "std":          round(float(values.std()), 4),
        "n_bootstrap":  n_bootstrap,
        "confidence":   confidence,
    }


def paired_delta_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for the difference in means (A − B), paired by sample index.

    If the CI excludes zero, the difference is statistically significant.
    """
    assert len(values_a) == len(values_b), "Must have same number of samples for paired test"
    deltas = values_a - values_b  # per-sample differences
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = deltas[rng.integers(0, n, size=n)]
        boot_means[b] = sample.mean()

    alpha = 1 - confidence
    ci_low  = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "delta_mean":   round(float(deltas.mean()), 4),
        "ci_low":       round(ci_low, 4),
        "ci_high":      round(ci_high, 4),
        "significant":  bool(ci_low > 0 or ci_high < 0),  # CI excludes zero
        "n_bootstrap":  n_bootstrap,
        "confidence":   confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for E1 architecture comparison"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = get_device()

    project_root = Path.cwd()
    metrics_dir  = project_root / "results" / "metrics"

    # Model paths
    models = {
        "facebook_bart-base_with_speakers": str(
            project_root / "models" / "best" / "facebook_bart-base_with_speakers"
        ),
        "t5-small_with_speakers": str(
            project_root / "models" / "best" / "t5-small_with_speakers"
        ),
    }

    # Check both models exist
    for label, path in models.items():
        if not Path(path).exists():
            print(f"  ❌  Checkpoint not found: {path}")
            print(f"       Run train.py for {label} first.")
            sys.exit(1)

    # Load test dataset (use BART tokenized cache)
    from datasets import load_from_disk  # noqa: PLC0415

    print(f"\n{'=' * 66}")
    print(f"  Bootstrap Confidence Intervals — E1 Architecture Comparison")
    print(f"  Models     : BART-base vs T5-small (both with_speakers)")
    print(f"  Bootstrap  : {args.n_bootstrap} iterations, 95% CI")
    print(f"  Device     : {device}")
    print(f"{'=' * 66}\n")

    per_sample_scores: dict[str, dict] = {}

    for label, model_path in models.items():
        print(f"\n  ── Generating: {label} ──────────────────────────────────────")

        # Each model needs its own tokenized cache
        model_slug = label.rsplit("_", 1)[0]  # e.g., facebook_bart-base
        # Map model slug to appropriate dataset cache
        if "bart" in label.lower():
            cache_name = "samsum_with_speakers_facebook_bart-base"
        elif "t5" in label.lower():
            cache_name = "samsum_with_speakers_t5-small"
        else:
            cache_name = f"samsum_with_speakers_{model_slug}"

        cache_path = project_root / "data" / "cache" / cache_name
        if not cache_path.exists():
            print(f"  ❌  Dataset cache not found: {cache_path}")
            sys.exit(1)

        ds_test = load_from_disk(str(cache_path))["test"]
        print(f"    Test samples: {len(ds_test)}")

        scores = generate_per_sample_rouge(
            model_path, ds_test, cfg, device, batch_size=args.batch_size
        )
        per_sample_scores[label] = scores

    # ── Compute bootstrap CIs ──────────────────────────────────────────────
    print(f"\n  ── Computing bootstrap CIs ({args.n_bootstrap} iterations) ──────")

    ci_results: dict = {
        "experiment": "E1 architecture comparison — bootstrap confidence intervals",
        "n_bootstrap": args.n_bootstrap,
        "confidence_level": 0.95,
        "n_samples": 819,
        "models": {},
    }

    for label, scores in per_sample_scores.items():
        model_cis = {}
        for metric in ["rouge1", "rouge2", "rougeL"]:
            vals = np.array(scores[metric])
            ci = bootstrap_ci(vals, n_bootstrap=args.n_bootstrap)
            model_cis[metric] = ci
            print(f"    {label} {metric}: {ci['mean']:.2f} [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
        ci_results["models"][label] = model_cis

    # ── Paired Δ CI (BART − T5) ───────────────────────────────────────────
    bart_key = "facebook_bart-base_with_speakers"
    t5_key   = "t5-small_with_speakers"

    print(f"\n  ── Paired Δ CI (BART − T5) ───────────────────────────────────")
    delta_cis: dict = {}
    for metric in ["rouge1", "rouge2", "rougeL"]:
        bart_vals = np.array(per_sample_scores[bart_key][metric])
        t5_vals   = np.array(per_sample_scores[t5_key][metric])
        d = paired_delta_ci(bart_vals, t5_vals, n_bootstrap=args.n_bootstrap)
        delta_cis[metric] = d
        sig_marker = "✅ significant" if d["significant"] else "❌ not significant"
        print(
            f"    Δ {metric}: {d['delta_mean']:+.2f} "
            f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] — {sig_marker}"
        )

    ci_results["paired_delta_BART_minus_T5"] = delta_cis
    ci_results["timestamp"] = datetime.now(timezone.utc).isoformat()

    # ── Save per-sample scores (for future analysis) ──────────────────────
    ci_results["per_sample_scores"] = {
        label: {metric: [round(v, 4) for v in vals] for metric, vals in scores.items()}
        for label, scores in per_sample_scores.items()
    }

    out_path = metrics_dir / "bootstrap_ci_e1.json"
    with open(out_path, "w") as f:
        json.dump(ci_results, f, indent=2)
    print(f"\n  ✅  Full results saved → {out_path}")

    # ── Update existing test JSONs with bootstrap CIs ─────────────────────
    for label in [bart_key, t5_key]:
        test_json_path = metrics_dir / f"{label}_test.json"
        if test_json_path.exists():
            with open(test_json_path) as f:
                existing = json.load(f)
            existing["bootstrap_ci"] = ci_results["models"][label]
            with open(test_json_path, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  ✅  Updated: {test_json_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print(f"  E1 Bootstrap CI Summary (95% confidence)")
    print(f"{'=' * 66}")
    for label in [bart_key, t5_key]:
        cis = ci_results["models"][label]
        rl = cis["rougeL"]
        print(f"  {label}:")
        print(f"    ROUGE-L = {rl['mean']:.2f}  (95% CI: [{rl['ci_low']:.2f}, {rl['ci_high']:.2f}])")
    d = delta_cis["rougeL"]
    print(f"\n  Δ ROUGE-L (BART − T5) = {d['delta_mean']:+.2f}  "
          f"(95% CI: [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}])")
    print(f"  Statistically significant: {'YES ✅' if d['significant'] else 'NO ❌'}")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()

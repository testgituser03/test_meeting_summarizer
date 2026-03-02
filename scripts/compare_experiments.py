#!/usr/bin/env python3
"""
compare_experiments.py — Aggregate and compare all SAMSum experiment results.

Reads from results/metrics/:
  zeroshot_*.json    — E0 zero-shot baselines (no fine-tuning)
  *_test.json        — Fine-tuned model test results (E1+)

Skips any file that is missing or unreadable — future experiments not yet run
will be absent without causing a crash.

Outputs:
  stdout                               — Markdown comparison table
  results/experiment_1_architecture.csv — CSV of all results + delta row

Delta convention:  Δ ROUGE-L = BART fine-tuned − T5 fine-tuned
  Positive value means BART wins (expected threshold: ≥ +2.0 pts).

Sanity check: each fine-tuned model must beat its zero-shot baseline
  by ≥ 10 ROUGE-1 points. Failure → training loop is likely broken.

Usage:
    python3 scripts/compare_experiments.py
    python3 scripts/compare_experiments.py --results-dir results/metrics
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# ── JSON loading ───────────────────────────────────────────────────────────────

def _load_json_safe(path: Path) -> dict | None:
    """
    Load a JSON file.  Returns None on FileNotFoundError, JSONDecodeError,
    or any OSError so callers can skip missing / malformed files gracefully.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ── Data loading ───────────────────────────────────────────────────────────────

def load_results(metrics_dir: Path) -> list[dict]:
    """
    Load and normalise all result JSON files from metrics_dir.

    File patterns:
      zeroshot_*.json   — E0 zero-shot baselines; no variant field
      *_test.json       — fine-tuned results; must have model, variant fields

    Missing or unreadable files are skipped with a stderr warning.
    Returns rows sorted: zero-shot first, then fine-tuned; each group
    sorted alphabetically by model name.
    """
    zeroshot_rows: list[dict] = []
    finetuned_rows: list[dict] = []

    # ── E0: zero-shot baselines ────────────────────────────────────────────
    for path in sorted(metrics_dir.glob("zeroshot_*.json")):
        data = _load_json_safe(path)
        if data is None:
            print(f"  ⚠️  Skipping (unreadable): {path.name}", file=sys.stderr)
            continue
        zeroshot_rows.append({
            "model":      data.get("model", path.stem.replace("zeroshot_", "")),
            "variant":    "—",
            "training":   "zero-shot",
            "rouge1":     float(data.get("rouge1", 0.0)),
            "rouge2":     float(data.get("rouge2", 0.0)),
            "rougeL":     float(data.get("rougeL", 0.0)),
            "n_samples":  str(data.get("n_samples", "—")),
            "best_epoch": "—",
            "train_min":  "—",
        })

    # ── E1+: fine-tuned results ────────────────────────────────────────────
    for path in sorted(metrics_dir.glob("*_test.json")):
        data = _load_json_safe(path)
        if data is None:
            print(f"  ⚠️  Skipping (unreadable): {path.name}", file=sys.stderr)
            continue
        finetuned_rows.append({
            "model":      data.get("model", "—"),
            "variant":    data.get("variant", "—"),
            "training":   "fine-tuned",
            "rouge1":     float(data.get("rouge1", 0.0)),
            "rouge2":     float(data.get("rouge2", 0.0)),
            "rougeL":     float(data.get("rougeL", 0.0)),
            "n_samples":  str(data.get("n_samples", "—")),
            "best_epoch": str(data.get("best_epoch", "—")),
            "train_min":  str(data.get("training_time_minutes", "—")),
        })

    return zeroshot_rows + finetuned_rows


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt(val: object, decimals: int = 2) -> str:
    """Format a numeric value to fixed decimal places; return as-is if not numeric."""
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def _sign(x: float) -> str:
    return "+" if x >= 0 else ""


# ── Markdown table ─────────────────────────────────────────────────────────────

def print_markdown_table(rows: list[dict]) -> None:
    """Print a markdown-formatted comparison table to stdout."""
    # Column widths: model(28), variant(16), training(12), scores(8 each), N(7)
    W = (28, 16, 12, 8, 8, 8, 7)
    header = (
        f"| {'Model':<{W[0]}} | {'Variant':<{W[1]}} | {'Training':<{W[2]}} "
        f"| {'ROUGE-1':>{W[3]}} | {'ROUGE-2':>{W[4]}} | {'ROUGE-L':>{W[5]}} "
        f"| {'N':>{W[6]}} |"
    )
    sep = (
        f"|{'-'*(W[0]+2)}|{'-'*(W[1]+2)}|{'-'*(W[2]+2)}"
        f"|{'-'*(W[3]+2)}|{'-'*(W[4]+2)}|{'-'*(W[5]+2)}|{'-'*(W[6]+2)}|"
    )

    print()
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['model']:<{W[0]}} | {r['variant']:<{W[1]}} "
            f"| {r['training']:<{W[2]}} "
            f"| {_fmt(r['rouge1']):>{W[3]}} | {_fmt(r['rouge2']):>{W[4]}} "
            f"| {_fmt(r['rougeL']):>{W[5]}} | {r['n_samples']:>{W[6]}} |"
        )
    print()


# ── Delta + sanity checks ──────────────────────────────────────────────────────

def compute_deltas_and_flags(rows: list[dict]) -> None:
    """
    Compute and print:

    1. Δ ROUGE (BART fine-tuned − T5 fine-tuned, with_speakers variant).
       Convention: positive = BART outperforms T5.
       Expected threshold: Δ ROUGE-L ≥ +2.0 points.

    2. Sanity check per model: fine-tuned ROUGE-1 vs zero-shot ROUGE-1.
       Expected improvement: ≥ 10 ROUGE-1 points.
       Failure implies the training loop is broken (teacher forcing leak,
       wrong metric, or model not fine-tuned at all).
    """
    # Index by (model, variant, training) for O(1) lookup
    idx: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["variant"], r["training"])
        idx[key] = r

    print(f"\n{'─'*62}")
    print("  Δ Analysis — E1 Architecture Comparison")
    print(f"{'─'*62}")

    # ── 1. Architecture delta: BART vs T5 (both fine-tuned, with_speakers) ──
    t5_ft   = idx.get(("t5-small",           "with_speakers", "fine-tuned"))
    bart_ft = idx.get(("facebook/bart-base", "with_speakers", "fine-tuned"))

    if t5_ft and bart_ft:
        δr1 = bart_ft["rouge1"] - t5_ft["rouge1"]
        δr2 = bart_ft["rouge2"] - t5_ft["rouge2"]
        δrL = bart_ft["rougeL"] - t5_ft["rougeL"]
        print(f"\n  Δ ROUGE (BART fine-tuned − T5 fine-tuned, with_speakers):")
        print(f"    ROUGE-1  :  {_sign(δr1)}{δr1:.2f}")
        print(f"    ROUGE-2  :  {_sign(δr2)}{δr2:.2f}")
        print(f"    ROUGE-L  :  {_sign(δrL)}{δrL:.2f}")
        if δrL >= 2.0:
            print(f"\n  ✅ BART beats T5 by {δrL:.2f} ROUGE-L pts — ≥2.0 threshold met")
        elif δrL >= 0.0:
            print(
                f"\n  ⚠️  BART leads T5 by {δrL:.2f} ROUGE-L pts — "
                f"positive but below the ≥2.0 expected threshold"
            )
        else:
            print(
                f"\n  ❌ T5 leads BART by {abs(δrL):.2f} ROUGE-L pts — "
                f"unexpected; check pre-training objective alignment"
            )
    elif t5_ft and not bart_ft:
        print("\n  ⏳ BART fine-tuned result not yet available — run BART-base training first.")
    elif bart_ft and not t5_ft:
        print("\n  ⏳ T5 fine-tuned result not yet available — run T5-small training first.")
    else:
        print("\n  ⏳ No fine-tuned results found yet. Run training first.")

    # ── 2. Sanity check: fine-tuned must beat zero-shot by ≥10 ROUGE-1 ─────
    print(f"\n  Sanity check — fine-tuned ROUGE-1 vs zero-shot baseline")
    print(f"  Threshold: ≥ +10.0 pts. Failure → training loop may be broken.\n")

    for model in ["t5-small", "facebook/bart-base"]:
        zs = idx.get((model, "—",            "zero-shot"))
        ft = idx.get((model, "with_speakers", "fine-tuned"))
        if zs is None:
            print(f"    ⏳  {model:<30}  zero-shot baseline missing — skipping")
            continue
        if ft is None:
            print(f"    ⏳  {model:<30}  fine-tuned result missing — skipping")
            continue
        gain = ft["rouge1"] - zs["rouge1"]
        flag = "✅" if gain >= 10.0 else "❌"
        print(
            f"    {flag}  {model:<30}  "
            f"zero-shot={zs['rouge1']:.2f}  "
            f"fine-tuned={ft['rouge1']:.2f}  "
            f"Δ={gain:+.2f}"
        )

    print()


# ── CSV output ─────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], out_path: Path) -> None:
    """Write all result rows to a CSV file."""
    fieldnames = [
        "model", "variant", "training",
        "rouge1", "rouge2", "rougeL",
        "n_samples", "best_epoch", "train_min",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "—") for k in fieldnames})
    print(f"  CSV saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all SAMSum experiment results"
    )
    parser.add_argument(
        "--results-dir",
        default="results/metrics",
        help="Directory containing *_test.json and zeroshot_*.json files",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.results_dir)
    if not metrics_dir.exists():
        print(f"ERROR: results directory not found: {metrics_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n  Loading results from: {metrics_dir.resolve()}")
    rows = load_results(metrics_dir)

    if not rows:
        print(
            "  No result files found. "
            "Run baseline_zeroshot.py and train.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loaded {len(rows)} result entr{'y' if len(rows) == 1 else 'ies'}.")

    # ── Markdown table ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  Experiment 1 — Architecture Comparison")
    print(f"  (E0 zero-shot baselines + E1 fine-tuned results)")
    print(f"{'='*62}")
    print_markdown_table(rows)

    # ── Delta + flags ──────────────────────────────────────────────────────
    compute_deltas_and_flags(rows)

    # ── CSV ───────────────────────────────────────────────────────────────
    out_path = metrics_dir.parent / "experiment_1_architecture.csv"
    save_csv(rows, out_path)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()

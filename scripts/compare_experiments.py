#!/usr/bin/env python3
"""
compare_experiments.py — Aggregate all experiment results into a summary table.

Reads all JSON files from results/metrics/ and produces:
  - A markdown table printed to stdout (paste into README.md)
  - results/metrics/experiment_summary.csv for plotting
  - results/metrics/experiment_summary.json for programmatic access

Usage:
  python3 scripts/compare_experiments.py
"""

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    metrics_dir  = project_root / "results" / "metrics"

    if not metrics_dir.exists():
        print("  ❌  results/metrics/ not found. Run training scripts first.")
        sys.exit(1)

    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        print("  ❌  pandas not installed: pip install pandas")
        sys.exit(1)

    rows = []

    # ── Collect all *_test.json files ─────────────────────────────────────
    for p in sorted(metrics_dir.glob("*_test.json")):
        data = load_json(p)
        if not data:
            continue
        rows.append({
            "experiment":  p.stem.replace("_test", ""),
            "model":       data.get("model", "—"),
            "variant":     data.get("variant", "—"),
            "rouge1":      data.get("rouge1", None),
            "rouge2":      data.get("rouge2", None),
            "rougeL":      data.get("rougeL", None),
            "epochs":      data.get("epochs_trained", "—"),
            "best_val_RL": data.get("best_val_rougeL", None),
        })

    # ── Collect zero-shot baselines ───────────────────────────────────────
    for p in sorted(metrics_dir.glob("zeroshot_*.json")):
        data = load_json(p)
        if not data:
            continue
        rows.append({
            "experiment":  p.stem,
            "model":       data.get("model", "—"),
            "variant":     "zero-shot",
            "rouge1":      data.get("rouge1", None),
            "rouge2":      data.get("rouge2", None),
            "rougeL":      data.get("rougeL", None),
            "epochs":      0,
            "best_val_RL": None,
        })

    if not rows:
        print("  ⚠️   No result files found in results/metrics/")
        print("      Run train.py and evaluate.py first.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # ── Print markdown table ───────────────────────────────────────────────
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v) if v is not None else "—"

    print("\n  EXPERIMENT SUMMARY TABLE")
    print("  " + "=" * 80)
    header = f"  {'Experiment':<45} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8}"
    print(header)
    print("  " + "-" * 80)
    for _, row in df.iterrows():
        print(
            f"  {row['experiment']:<45} "
            f"{fmt(row['rouge1']):>8} "
            f"{fmt(row['rouge2']):>8} "
            f"{fmt(row['rougeL']):>8}"
        )
    print("  " + "=" * 80)

    # ── Markdown for README ────────────────────────────────────────────────
    print("\n  MARKDOWN TABLE (copy into README.md Results section):\n")
    print("  | Experiment | Model | Variant | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    print("  |---|---|---|---|---|---|")
    for _, row in df.iterrows():
        print(
            f"  | {row['experiment']} | {row['model']} | {row['variant']} "
            f"| {fmt(row['rouge1'])} | {fmt(row['rouge2'])} | {fmt(row['rougeL'])} |"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    csv_path  = metrics_dir / "experiment_summary.csv"
    json_path = metrics_dir / "experiment_summary.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n  ✅  Saved → {csv_path.relative_to(project_root)}")
    print(f"  ✅  Saved → {json_path.relative_to(project_root)}\n")


if __name__ == "__main__":
    main()

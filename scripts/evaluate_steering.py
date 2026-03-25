#!/usr/bin/env python3
"""
evaluate_steering.py — Task 3 / quality-focus trade-off evaluation.

Inputs:
  • results/steering/*_steering_generations.json from steering_inference.py
  • optional human ratings CSV (filled after manual review)

Outputs:
  • results/metrics/<run_name>_steering_eval.json
  • results/steering/<run_name>_human_eval_template.csv (50-sample rubric sheet)
  • results/steering/human_eval_rubric.md
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate steering trade-offs")
    parser.add_argument("--input", required=True, help="Path to steering_generations.json")
    parser.add_argument("--ratings_csv", default=None,
                        help="Optional completed ratings CSV with columns: layer,alpha,sample_id,clarity_score")
    parser.add_argument("--max_human_samples", type=int, default=50)
    parser.add_argument("--max_rouge_drop_pct", type=float, default=2.0)
    return parser.parse_args()


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _load_ratings(path: str) -> dict[tuple[int, float], list[float]]:
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            alpha = float(row["alpha"])
            score = float(row["clarity_score"])
            grouped[(layer, alpha)].append(score)
    return grouped


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)

    with open(in_path) as f:
        payload = json.load(f)

    meta = payload["meta"]
    runs = payload["runs"]
    dialogues = payload["dialogues"]
    refs = payload["references"]

    run_name = meta["run_name"]
    project_root = Path.cwd()

    # Index runs by layer and alpha.
    by_layer: dict[int, dict[float, dict]] = defaultdict(dict)
    for row in runs:
        by_layer[int(row["layer"])][float(row["alpha"])] = row

    # Auto metrics summary and optimal alpha per layer (ROUGE constraint only).
    layer_summary: list[dict] = []
    for layer, alpha_map in sorted(by_layer.items()):
        if 0.0 not in alpha_map:
            raise ValueError(f"Layer {layer} missing baseline alpha=0.0 run")

        base_rouge = float(alpha_map[0.0]["rougeL"])
        candidates = []

        for alpha, row in sorted(alpha_map.items()):
            rouge = float(row["rougeL"])
            drop_pct = ((base_rouge - rouge) / max(base_rouge, 1e-8)) * 100.0
            candidates.append(
                {
                    "alpha": float(alpha),
                    "rougeL": round(rouge, 4),
                    "rouge_drop_pct_vs_alpha0": round(drop_pct, 4),
                    "action_proxy_rate": float(row["action_proxy_rate"]),
                    "ms_per_sample": float(row["ms_per_sample"]),
                    "n_samples": int(row["n_samples"]),
                }
            )

        valid = [c for c in candidates if c["rouge_drop_pct_vs_alpha0"] <= args.max_rouge_drop_pct]
        if valid:
            best = max(valid, key=lambda x: (x["action_proxy_rate"], x["alpha"]))
        else:
            best = max(candidates, key=lambda x: x["rougeL"])

        layer_summary.append(
            {
                "layer": layer,
                "baseline_alpha0_rougeL": round(base_rouge, 4),
                "candidates": candidates,
                "best_alpha_under_rouge_constraint": best["alpha"],
                "best_action_proxy_rate": best["action_proxy_rate"],
            }
        )

    # Build manual-eval template (50 samples × all layer/alpha combinations).
    out_steering_dir = project_root / "results" / "steering"
    out_steering_dir.mkdir(parents=True, exist_ok=True)

    template_path = out_steering_dir / f"{run_name}_human_eval_template.csv"
    n_eval = min(args.max_human_samples, len(dialogues))

    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "alpha", "sample_id", "dialogue", "reference", "prediction",
            "clarity_score", "notes",
        ])

        for layer, alpha_map in sorted(by_layer.items()):
            for alpha, row in sorted(alpha_map.items()):
                preds = row["predictions"]
                for i in range(n_eval):
                    writer.writerow([
                        layer,
                        alpha,
                        i,
                        dialogues[i],
                        refs[i],
                        preds[i],
                        "",
                        "",
                    ])

    rubric_path = out_steering_dir / "human_eval_rubric.md"
    rubric_text = """# Human Evaluation Rubric — Action Item Clarity (1-5)

Score each generated summary for whether concrete next actions are clear.

- 1: No actionable item; vague or purely descriptive.
- 2: Weakly actionable; actor/action unclear.
- 3: Some action present, but missing owner, timing, or specificity.
- 4: Clear action item(s) with mostly clear owner and intent.
- 5: Highly clear, specific action item(s) with explicit intent/owner and minimal ambiguity.

Protocol:
1. Use only the first 50 samples from the template.
2. Rate every (layer, alpha, sample) row.
3. Keep the rubric constant across all settings.
4. Save completed CSV and pass via --ratings_csv.
"""
    with open(rubric_path, "w") as f:
        f.write(rubric_text)

    ratings_summary = None
    global_best = None

    if args.ratings_csv:
        grouped = _load_ratings(args.ratings_csv)
        ratings_rows = []
        for layer, alpha_map in sorted(by_layer.items()):
            for alpha in sorted(alpha_map.keys()):
                scores = grouped.get((layer, alpha), [])
                ratings_rows.append(
                    {
                        "layer": layer,
                        "alpha": float(alpha),
                        "n_rated": len(scores),
                        "clarity_mean": round(_mean(scores), 4) if scores else None,
                        "clarity_std": round(float(statistics.pstdev(scores)), 4) if len(scores) > 1 else None,
                    }
                )

        ratings_summary = ratings_rows

        # Combine human clarity + rouge constraints to pick best operating point.
        merged_candidates = []
        layer_map = {(ls["layer"], c["alpha"]): c for ls in layer_summary for c in ls["candidates"]}
        for row in ratings_rows:
            key = (row["layer"], row["alpha"])
            if row["clarity_mean"] is None or key not in layer_map:
                continue
            cand = layer_map[key]
            merged_candidates.append(
                {
                    "layer": row["layer"],
                    "alpha": row["alpha"],
                    "clarity_mean": row["clarity_mean"],
                    "rougeL": cand["rougeL"],
                    "rouge_drop_pct_vs_alpha0": cand["rouge_drop_pct_vs_alpha0"],
                }
            )

        valid = [m for m in merged_candidates if m["rouge_drop_pct_vs_alpha0"] <= args.max_rouge_drop_pct]
        if valid:
            global_best = max(valid, key=lambda x: (x["clarity_mean"], x["rougeL"]))

    out_metrics_dir = project_root / "results" / "metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_metrics_dir / f"{run_name}_{meta['split']}_{meta['method']}_steering_eval.json"

    output = {
        "experiment": "Task 3 steering for focus control",
        "meta": meta,
        "constraint": {"max_rouge_drop_pct": args.max_rouge_drop_pct},
        "layer_summary": layer_summary,
        "human_eval": {
            "template_csv": str(template_path),
            "rubric": str(rubric_path),
            "n_requested_samples": n_eval,
            "ratings_csv": args.ratings_csv,
            "ratings_summary": ratings_summary,
        },
        "global_best": global_best,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Steering evaluation complete")
    print(f"   Metrics: {out_path.relative_to(project_root)}")
    print(f"   Human template: {template_path.relative_to(project_root)}")
    print(f"   Rubric: {rubric_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()

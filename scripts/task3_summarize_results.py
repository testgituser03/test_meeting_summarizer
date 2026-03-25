#!/usr/bin/env python3
"""
Task 3 — Consolidate steering sweep results and document layer effectiveness.

Reads per-method steering eval JSONs and produces task3_full_sweep_summary.json
with best operating points under ROUGE-drop constraint.

Layer documentation:
  BART-base has 6 decoder layers (1–6). The Task 3 spec requested layers 6–12;
  only layer 6 exists in BART-base, so all steering experiments use layer 6.
  For models with more decoder layers (e.g. T5-large, BART-large), layers 8–12
  would enable stronger steering; layer 6 is the strongest available in BART-base.
"""
import json
from pathlib import Path

ROOT = Path("results/metrics")
FILES = [
    ROOT / "facebook_bart-base_with_speakers_test_mean_diff_steering_eval.json",
    ROOT / "facebook_bart-base_with_speakers_test_pca_delta_steering_eval.json",
    ROOT / "facebook_bart-base_with_speakers_test_logistic_steering_eval.json",
]

rows = []
for p in FILES:
    d = json.loads(p.read_text())
    method = d["meta"]["method"]
    layer_block = d["layer_summary"][0]
    layer = layer_block["layer"]
    base = layer_block["baseline_alpha0_rougeL"]
    for c in layer_block["candidates"]:
        x = dict(c)
        x["method"] = method
        x["layer"] = layer
        x["baseline_rougeL"] = base
        rows.append(x)

best_per_method = {}
for r in rows:
    if r["rouge_drop_pct_vs_alpha0"] <= 2.0:
        key = r["method"]
        prev = best_per_method.get(key)
        if prev is None or (r["action_proxy_rate"], r["rougeL"]) > (prev["action_proxy_rate"], prev["rougeL"]):
            best_per_method[key] = r

valid = [r for r in rows if r["rouge_drop_pct_vs_alpha0"] <= 2.0]
global_best = max(valid, key=lambda x: (x["action_proxy_rate"], x["rougeL"])) if valid else None

summary = {
    "experiment": "task3_full_sweep_bart_with_speakers",
    "n_points": len(rows),
    "constraint": {"max_rouge_drop_pct": 2.0},
    "layer_documentation": {
        "model": "facebook/bart-base",
        "n_decoder_layers": 6,
        "requested_layers": [6, 8, 10, 12],
        "effective_layers": [6],
        "strongest_steering_layer": 6,
        "note": "BART-base has 6 decoder layers. Only layer 6 is in range. "
        "For larger models (e.g. BART-large, T5-large), layers 8–12 would allow "
        "richer steering analysis; layer 6 is the sole valid layer for BART-base.",
    },
    "best_per_method_under_constraint": best_per_method,
    "global_best_under_constraint": global_best,
    "rows": rows,
}

out = ROOT / "task3_full_sweep_summary.json"
out.write_text(json.dumps(summary, indent=2))

print(f"Wrote {out}")
print("Global best under constraint:")
print(json.dumps(global_best, indent=2))

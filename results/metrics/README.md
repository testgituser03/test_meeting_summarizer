# results/metrics — Output Files Reference

This directory contains all per-experiment JSON output files. Every JSON file
is a direct product of a deterministic, reproducible script run and serves as
the primary evidence for reported metric values.

---

## File Count

| Pattern | Count | Produced by |
|---------|-------|-------------|
| `zeroshot_*.json` | 3 | `scripts/baseline_zeroshot.py` + `scripts/pegasus_experiment.py --zeroshot` |
| `*_test.json` | 7 | `scripts/train.py`, `scripts/train_lora.py`, `scripts/pegasus_experiment.py --train` |
| `decoding_D*.json` | 29 | `scripts/decoding_ablation.py` |
| `experiment_3_decoding_summary.json` | 1 | `scripts/decoding_ablation.py` (aggregated) |
| `sweep_*.json` | 68 | `scripts/multi_model_sweep.py` (4 models × 17 configs) |
| `multi_model_sweep_summary.json` | 1 | `scripts/multi_model_sweep.py` (aggregated) |
| `bootstrap_ci_e1.json` | 1 | `scripts/bootstrap_ci.py` |
| `faithfulness_report.json` | 1 | `scripts/evaluate_faithfulness.py` |
| `data_audit.json` | 1 | `scripts/data_audit.py` |
| `task2_*.json` | 5 | `scripts/task2_quantization.py`, `scripts/task2_benchmark.py` |
| `*_steering_eval.json` | 1+ | `scripts/evaluate_steering.py` |

---

## Naming Conventions

### Zero-Shot Baselines
```
zeroshot_{model_slug}.json
```
Schema: `{ model, n_samples, rouge1, rouge2, rougeL, generation_config, timestamp }`

### Trained Model Test Results
```
{run_name}_test.json
```
Run names: `facebook_bart-base_with_speakers`, `facebook_bart-base_no_speakers`,
`facebook_bart-base_split_speakers`, `t5-small_with_speakers`, `facebook_bart-base_lora`,
`facebook_bart-base_extended`, `google_pegasus-cnn_dailymail_with_speakers`

Schema: `{ model, variant, rouge1, rouge2, rougeL, best_epoch, training_time_min, memory_profile_mb, ... }`

### Decoding Ablation (E3) — 29 Configs
```
decoding_{config_id}_{description}.json
```
Examples: `decoding_D27_beam5_lp1.33.json`, `decoding_D5_nucleus_p0.9.json`

Schema: `{ config_id, label, description, rouge1, rouge2, rougeL, avg_summary_tokens, ms_per_sample, n_samples, model_path, gen_kwargs }`

Champion: **`decoding_D27_beam5_lp1.33.json`** — ROUGE-L 40.12

### Multi-Model Sweep
```
sweep_{model}_{config_id}.json
```
Model slugs: `extended`, `lora`, `t5`, `no_speakers`  
Config IDs: `baseline`, `D7`, `D8`, `D10`, `D12`, `D13`, `D14`, `D17`, `D19`, `D21`–`D29`

### Other Outputs
| File | Description |
|------|-------------|
| `bootstrap_ci_e1.json` | Bootstrap 95% CIs for E1 (T5-small and BART-base), 1,000 iterations |
| `faithfulness_report.json` | E4: NER hallucination rate, NLI entailment score, speaker preservation |
| `data_audit.json` | Token-length distributions, speaker stats, leakage check |
| `experiment_3_decoding_summary.json` | All 29 D-configs ranked by ROUGE-L |
| `multi_model_sweep_summary.json` | Best config per model across 68 sweep runs |

### Task 2 Outputs — Quantization + Real-Time Benchmarking

| File | Description |
|------|-------------|
| `task2_quantization_manifest.json` | Feasibility report + runtime mapping + quantized artifact locations |
| `task2_benchmark_table.json` | Per-quant/per-length latency, throughput, memory, ROUGE-L rows |
| `task2_streaming_vs_batch.json` | Streaming vs batch quality and efficiency comparison |
| `task2_parallel_scaling.json` | 1/2/4-process throughput and memory contention results |
| `task2_eval_rougel.json` | Object: `benchmark_args` (eval_samples, seed, paths), `results[]` rows; quant folder names are **nominal** — runtime is CTranslate2 (see `task2_quantization_manifest.json`) |

To refresh only this file (skip length/streaming/parallel benchmarks): `python scripts/task2_benchmark.py --eval-only --eval_samples 256`.

### Task 3 Outputs — Steering for Focus Control

| File Pattern | Description |
|------|-------------|
| `*_steering_eval.json` | Layer/alpha steering trade-off summary (ROUGE-L vs. action focus), plus optimal α under ROUGE-drop constraint |

### Task 4 Outputs — Adversarial Robustness

| File | Description |
|------|-------------|
| `task4_robustness_eval.json` | ROUGE-L on 150 original + 150 adversarial; failure modes; action completeness |
| `task4_coherence_template.csv` | **Template only** — empty human rating scaffold (1–5 coherence); not filled automatically |
| `task4_robustness_comparison.json` | Pre/post ROUGE-L on held-out adversarial; `model_*_resolved` = checkpoint path used (usually `merged/`) |
| `task4_retrain_manifest.json` | Retrain hyperparameters (70/30 mix, LR, epochs, early stop, best held-out ROUGE-L) when `retrain` was run |

### Task 5 Outputs — LoRA Rank & Structured Output

| File | Description |
|------|-------------|
| `task5_rank_ablation.json` | ROUGE-L, latency, model size per LoRA rank (2,4,8,16,32) |
| `task5_structured_output.json` | Per rank: `parse_success_rate`, `heuristic_fallback_rate`, `api_envelope_valid_rate`, `rougeL_structured_json_vs_gold`, etc.; see `metric_notes` in JSON |
| `task5_structured_train_r*.json` | Per-rank manifest from `train_structured` (samples, LoRA config, `json_target_format`) |
| `task5_structured_training_summary.json` | Aggregate summary of structured training run |
| `task5_sweet_spot.json` | Sweet spot: `parse_success_rate ≥ min_parse_success` (default 0.2) + ROUGE window; `package` uses `--default_rank` when null |

Related non-metrics artifacts written outside this folder:

- `results/activations/*.pt` — pooled decoder activations + labels from `scripts/extract_activations.py`
- `results/steering/*_steering.pt` — per-layer steering vectors from `scripts/compute_steering_vector.py`
- `results/steering/*_steering_generations.json` — steered predictions for all α/layer configs
- `results/steering/*_human_eval_template.csv` — 50-sample manual rating sheet
- `results/steering/human_eval_rubric.md` — action-item clarity rubric

---

## Reproducing Any File

Every JSON file can be regenerated from scratch. See [README.md](../../README.md)
for the full reproduction command sequence, or run `make help` from the project root.

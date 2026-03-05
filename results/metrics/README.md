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

---

## Reproducing Any File

Every JSON file can be regenerated from scratch. See [README.md](../../README.md)
for the full reproduction command sequence, or run `make help` from the project root.

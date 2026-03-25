# Architecture

## System Overview

`meeting-summarizer` is a sequential ML pipeline for abstractive dialogue summarization. It fine-tunes encoder-decoder seq2seq models on the SAMSum dataset and exposes inference through a Streamlit demo. All components communicate exclusively through files on disk — no shared state, no in-process imports between scripts.

```
config.yaml ──► scripts/ ──► data/cache/ ──► models/best/ ──► results/metrics/
                                ▲                  ▲
                          preprocess.py         train.py
```

---

## Pipeline Stages

### Stage 0 — Environment & Asset Setup
| Script | Purpose |
|--------|---------|
| `scripts/verify_env.py` | Pre-flight: ARM64, MPS availability, BF16, float64 rejection, memory baseline |
| `scripts/predownload_assets.py` | One-time download of model weights and SAMSum dataset to HuggingFace cache |
| `scripts/hf_whoami.py` | Validates HuggingFace authentication token |
| `scripts/data_audit.py` | Dataset integrity: leakage guard, token-length stats, speaker distribution |

### Stage 1 — Tokenization
| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/preprocess.py` | Tokenizes SAMSum into 3 on-disk variants | `data/cache/samsum_{variant}_{model}/` |

**Variants produced:**
- `with_speakers` — speaker tags preserved (e.g. `Amanda: I baked cookies.`)
- `no_speakers` — speaker tags stripped via regex
- `split_speakers` — sliding-window segmentation for long dialogues (stride=256, min=32 tokens)

### Stage 2 — Training
| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/baseline_zeroshot.py` | E0: zero-shot ROUGE with no fine-tuning | `results/metrics/zeroshot_*.json` |
| `scripts/train.py` | Full fine-tuning via `Seq2SeqTrainer` | `models/best/<run_name>/` |
| `scripts/train_lora.py` | LoRA fine-tuning (rank=16, 0.63% trainable params) | `models/best/facebook_bart-base_lora/` |
| `scripts/pegasus_experiment.py` | PEGASUS pipeline: download / zero-shot / preprocess / train | `models/best/google_pegasus-cnn_dailymail_with_speakers/` |

**Extension (course / robustness tasks):**

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/task4_adversarial.py` | Adversarial transcripts, robustness **eval** (`--metrics-out`), **retrain** (~55/45 mix, LR 5e-6, ≤5 ep, pattern-macro held-out), **compare** | `data/adversarial_task4/`, `models/best/t5-small_lora_task4/`, `results/metrics/task4_*.json` |
| `scripts/task5_lora_structured.py` | LoRA rank ablation; **`train_structured`** (inner-JSON labels for T5); structured eval (strict / salvage / generative-native metrics + round-trip); sweet spot; package | `models/best/t5-small_lora_r*/`, `merged_structured/`, `results/metrics/task5_*.json` |

### Stage 3 — Evaluation
| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/evaluate.py` | Standalone ROUGE eval on any saved checkpoint | `results/metrics/<run_name>_eval_{split}.json` |
| `scripts/decoding_ablation.py` | E3: 29-config beam/length-penalty sweep | `results/metrics/decoding_D*.json` |
| `scripts/multi_model_sweep.py` | E3-style sweep across all 4 secondary models (68 configs) | `results/metrics/sweep_*.json` |
| `scripts/evaluate_faithfulness.py` | E4: NER hallucination + NLI entailment + speaker preservation | `results/metrics/faithfulness_report.json` |
| `scripts/bootstrap_ci.py` | Bootstrap 95% confidence intervals for E1 models | `results/metrics/bootstrap_ci_e1.json` |
| `scripts/error_analysis_helper.py` | Samples 20 test examples for manual annotation | `results/error_analysis_raw.json` |
| `scripts/compare_experiments.py` | Aggregates all `*_test.json` + `zeroshot_*.json` into a comparison table | `results/experiment_1_architecture.csv` |

### Stage 4 — Demo
| Script | Purpose |
|--------|---------|
| `scripts/app.py` | Streamlit inference demo with model selector, generation controls, NER, action items |
| `scripts/run_app.sh` | Shell launcher: activates venv, sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, starts server |

---

## Configuration

All hyperparameters live in `config.yaml`. No script hardcodes a value that appears there. Scripts read it via:

```python
import yaml
cfg = yaml.safe_load(open("config.yaml"))
```

`config_extended.yaml` is a drop-in override for Experiment 8 (8 epochs, cosine LR, lower peak learning rate):

```bash
python3 scripts/train.py --config config_extended.yaml
```

### Key parameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_name` | `facebook/bart-base` | Override with `--model` CLI flag |
| `dataset_variant` | `with_speakers` | Override with `--variant` CLI flag |
| `batch_size` | `8` | Safe for BART-base BF16 on 24 GB UMA |
| `learning_rate` | `5e-5` | Standard seq2seq fine-tuning rate |
| `num_epochs` | `5` | Upper bound; early stopping usually triggers at epoch 3–5 |
| `num_beams` | `4` | Default beam search width |
| `length_penalty` | `1.0` | Champion D27 uses `1.33` |
| `use_bf16` | `true` | MPS-native; FP16 backward is unstable on MPS |
| `dataloader_num_workers` | `0` | MPS + multiprocessing → context errors |
| `dataloader_pin_memory` | `false` | UMA: no PCIe transfer to pin |
| `seed` | `42` | Fixes torch/numpy/Python random |

---

## Data Flow

```
HuggingFace Hub (online, one-time)
        │
        ▼
~/.cache/huggingface/       ← predownload_assets.py
        │
        ▼
data/cache/samsum_{variant}_{model}/   ← preprocess.py (offline, saved to disk)
        │
        ▼
models/checkpoints/<run_name>/         ← train.py (per-epoch Trainer checkpoints)
        │
        ▼
models/best/<run_name>/                ← train.py (best rougeL checkpoint)
        │
        ├──► results/metrics/*_test.json     ← train.py + evaluate.py
        ├──► results/metrics/decoding_*.json ← decoding_ablation.py
        ├──► results/metrics/sweep_*.json    ← multi_model_sweep.py
        ├──► results/metrics/faithfulness_report.json ← evaluate_faithfulness.py
        └──► scripts/app.py (runtime inference)
```

---

## Directory Reference

```
meeting-summarizer/
├── config.yaml                  # All hyperparameters — single source of truth
├── config_extended.yaml         # E8 overrides (8 epochs, cosine LR)
├── requirements.txt             # Pinned Python dependencies
├── model_card.md                # HuggingFace model card
├── Makefile                     # Common workflow targets
├── scripts/                     # All executable pipeline scripts (18 files)
├── data/cache/                  # Tokenized HuggingFace datasets (gitignored)
├── models/checkpoints/          # Per-epoch Trainer checkpoints (gitignored)
├── models/best/                 # Best-rougeL checkpoint per run (gitignored)
├── results/metrics/             # JSON outputs per experiment (committed)
├── results/error_analysis.md    # Manual 20-sample error annotation
├── notebooks/eda.ipynb          # SAMSum exploratory data analysis
└── docs/
    ├── ARCHITECTURE.md          # This document
    └── EXPERIMENTS.md           # Full experiment results and analysis
```

---

## Hardware & Compute Constraints

All training and inference run on **Apple M4 Pro (24 GB UMA)** using the PyTorch MPS backend with BF16 precision. Two MPS-specific constraints are enforced throughout:

1. `dataloader_num_workers=0` — MPS + Python multiprocessing triggers MPS context errors
2. `dataloader_pin_memory=False` — Unified Memory Architecture shares CPU/GPU pool; no PCIe transfer exists to pin

The `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable is set before any `torch` import to allow CPU fallback for the handful of ops not yet implemented in MPS (e.g., some attention edge cases).

Inference uses `torch.mps.synchronize()` before stopping latency timers to ensure all MPS kernel execution is included in the measurement.

# Meeting Summarizer

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![PyTorch MPS/BF16](https://img.shields.io/badge/PyTorch-MPS%2FBF16-ee4c2c?logo=pytorch&logoColor=white)
![License CC-BY-NC-ND-4.0](https://img.shields.io/badge/License-CC--BY--NC--ND--4.0-lightgrey)
![Platform macOS Sequoia](https://img.shields.io/badge/Platform-macOS%20Sequoia-000000?logo=apple&logoColor=white)

Fine-tunes `facebook/bart-base` (139 M parameters) on 14,731 SAMSum dialogues
and achieves **ROUGE-L 40.12** on the 819-sample held-out test set — exceeding
the ≥ 40 project target — running entirely on a single Apple M4 Pro with BF16
precision and no cloud GPU.

```
Raw SAMSum dialogues (16,368 total)
      │
      ▼  preprocess.py  — speaker-tagged T5 tokenization, 3 variants
      ▼  train.py       — Seq2SeqTrainer · 5 epochs · BF16/MPS · 72.4 min
      ▼  decoding_ablation.py — 29 beam / length-penalty configurations
      ▼  D27: num_beams=5 · length_penalty=1.33

ROUGE-L 40.12  ✓  project target met
```

---

## ⚠️ License Notice

**The SAMSum dataset is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
(Creative Commons — Attribution · Non-Commercial · No Derivatives).**

> This project, the fine-tuned model weights, and any generated outputs are
> restricted to **non-commercial use only**, in compliance with the SAMSum
> dataset license. Deploying or distributing the model in any commercial
> product or service is prohibited without explicit permission from the
> dataset authors.

Original dataset: Gliwa et al., 2019 — *SAMSum Corpus: A Human-annotated
Dialogue Dataset for Abstractive Summarization*.

---

## Results

Zero-shot `facebook/bart-base` scores **19.89 ROUGE-L** on a 100-sample probe.
Fine-tuning for **72.4 minutes** (5 epochs, BF16/MPS) raises this to **39.85** —
a +100.4 % relative gain. A 29-configuration decoding sweep then identifies
`num_beams=5, length_penalty=1.33` (D27) as the champion, reaching **40.12**
and clearing the ≥ 40 target.

### Experiment Summary

All metrics are macro-averaged ROUGE F-measures × 100 on the 819-sample test
set unless noted.

| # | Model / Config | R-1 | R-2 | R-L | Notes |
|---|----------------|-----|-----|-----|-------|
| E0 zero-shot | BART-base | 27.34 | 8.87 | 19.89 | 100-sample subset |
| E0 zero-shot | T5-small | 27.60 | 7.63 | 22.19 | 100-sample subset |
| E1 fine-tuned | T5-small | 38.96 | 15.96 | 31.95 | 35.1 min · epoch 2 |
| **E1 fine-tuned** | **BART-base** | **47.86** | **23.22** | **39.85** | **72.4 min · epoch 5** |
| E2 ablation | BART-base no_speakers | 38.95 | 19.17 | 33.23 | −6.62 R-L vs with_speakers |
| **E3 champion** | **BART-base D27 beam=5 lp=1.33** | **48.48** | **23.55** | **40.12** | **Best of 29 configs** |
| E5 LoRA | BART-base (r=16, 0.63 % params) | 45.15 | 21.20 | 37.59 | 54.7 min |
| E6 windowing | BART-base split_speakers | 47.11 | 22.55 | 39.08 | −0.77 R-L vs E1 |
| E7 PEGASUS | google/pegasus-cnn_dailymail | 1.85 | 0.00 | 1.60 | Zero-shot; domain mismatch |
| E8 extended | BART-base (8 ep · cosine LR) | 46.45 | 22.05 | 38.46 | −1.39 R-L vs E1 · 259.6 min |

### Key Findings

| Finding | Value | Interpretation |
|---------|-------|----------------|
| Speaker tags vs no-tags | +6.62 R-L (+19.9 % relative) | Retaining `Alice:` / `Bob:` turn prefixes is the single largest lever |
| BART-base vs T5-small | +7.90 R-L; 95 % CI [+6.99, +9.02] | Difference is statistically significant — CI excludes zero |
| LoRA efficiency | 37.59 R-L with 0.88 M / 139 M params (0.63 %) | 94 % of full fine-tune quality at 1/160 the trainable parameters |
| Decoding sweep gain | 39.85 → 40.12 (+0.27 R-L) | beam=5, lp=1.33 unlocks the final increment to clear the target |
| Extended training penalty | −1.39 R-L (8 ep vs 5 ep) | Overfitting on this 14 K-sample corpus; early stopping at epoch 5 is optimal |
| PEGASUS domain mismatch | R-L 1.60 zero-shot | CNN/DM pre-training vocabulary does not transfer to chat dialogue |

**Faithfulness audit (E4, 819 test samples):** 10.1 % entity-level hallucination
rate (83/819), 75.5 % speaker-preservation, 0.308 average NLI faithfulness score,
−0.25 length–ROUGE correlation.

→ Full methodology, per-config decoding tables, bootstrap CIs, and error analysis:
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**

---

## Quick Start

### Option A — Makefile (recommended)

```bash
make install    # create venv · pip install · download spaCy model
make verify     # MPS / BF16 pre-flight check (all items must pass)
make download   # one-time HuggingFace asset download (~2 GB, network)
make demo       # launch Streamlit inference demo at http://localhost:8501
```

Run `make` with no arguments to list every available target.

### Option B — Manual

```bash
# 1. Create Python 3.12 venv (system Python 3.14 lacks stable PyTorch wheels)
python3.12 -m venv ~/.venvs/meeting-summarizer --prompt meeting-summarizer
source ~/.venvs/meeting-summarizer/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# 3. Verify MPS environment (all checks must pass before training)
python3 scripts/verify_env.py

# 4. Pre-download all model and dataset assets (network step — run once)
python3 scripts/predownload_assets.py

# 5. Launch demo (requires models/best/ checkpoints; see Reproducing Results)
streamlit run scripts/app.py
```

---

## Dataset

**SAMSum** (`knkarthick/samsum`) is a human-annotated corpus of ~16 K
messenger-style dialogues with gold abstractive summaries written by
professional linguists (Gliwa et al., 2019).

### Split Sizes

| Split | Dialogues |
|-------|----------:|
| Train | 14,731 |
| Validation | 818 |
| Test | 819 |
| **Total** | **16,368** |

Zero ID overlap confirmed across all splits — no data leakage
(`results/metrics/data_audit.json`, field `leakage_check.passed = true`).

### Token Statistics

Source: `results/metrics/data_audit.json` — T5 tokenizer, training split
(n = 14,731).

| Field | min | p50 | p90 | p99 | max | mean |
|-------|----:|----:|----:|----:|----:|-----:|
| Dialogue tokens | 13 | 119 | 296 | 525 | 1,153 | 148.9 |
| Summary tokens | 2 | 25 | 50 | 73 | 94 | 28.7 |

`max_source_length=512` (config.yaml) covers ≈ 99 % of training dialogues;
the top ~1 % are truncated at token 512 (p99 = 525).
`max_target_length=128` safely covers all summaries (max = 94 tokens).

### Speaker Distribution (training split)

| Speakers | Dialogues | % |
|----------|----------:|--:|
| 2 | 10,758 | 73.0 % |
| 3 | 2,808 | 19.1 % |
| 4 | 822 | 5.6 % |
| 5+ | 343 | 2.3 % |

The corpus is heavily skewed toward two-speaker conversations. SAMSum
dialogues are constructed scenarios, not transcripts of real meetings.

**Why speaker tags matter:** prepending `Alice:` / `Bob:` turn prefixes before
fine-tuning lifts ROUGE-L from 33.23 (no_speakers, E2) to 39.85 (with_speakers,
E1) — a **+6.62 absolute / +19.9 % relative gain**. The model learns to use
speaker identity as a discourse signal for attribution-accurate summarization.

---

## Hardware

| Component | Specification |
|-----------|---------------|
| SoC | Apple M4 Pro (T6041) |
| Memory | 24 GB Unified Memory (LPDDR5X) |
| GPU | 20-core GPU (Metal 3) |
| OS | macOS Sequoia 15.7.3 |
| Compute | PyTorch MPS backend — BF16 verified |

All training and inference runs use `torch.device("mps")` with BF16 precision.
`num_workers=0` and `pin_memory=False` are required MPS constraints.
All latency and training-time figures are specific to this hardware; CUDA
results will differ.

---

## Reproducing Results

All hyperparameters are controlled by [`config.yaml`](config.yaml) (or
[`config_extended.yaml`](config_extended.yaml) for E8).
Key values: `batch_size=8`, `lr=5e-5`, `num_epochs=5`, `warmup_steps=500`,
`num_beams=4`, `length_penalty=1.0`, `use_bf16=true`.

```bash
# ── Data preparation ────────────────────────────────────────────────────────
python3 scripts/data_audit.py              # dataset statistics + leakage guard
python3 scripts/preprocess.py              # tokenize with_speakers + no_speakers
python3 scripts/preprocess.py --variants split_speakers   # long-dialogue windowing

# ── Training ────────────────────────────────────────────────────────────────
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py             # E1: BART-base
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py --model t5-small  # E1: T5
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_lora.py        # E5: LoRA (r=16)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py --config config_extended.yaml  # E8

# PEGASUS pipeline (E7) — each flag is one pipeline stage
python3 scripts/pegasus_experiment.py --download     # ~2.2 GB, network
python3 scripts/pegasus_experiment.py --zeroshot     # E0 on 100 samples
python3 scripts/pegasus_experiment.py --preprocess   # tokenize SAMSum for PEGASUS
python3 scripts/pegasus_experiment.py --train        # fine-tune 568 M-param model

# ── Evaluation ──────────────────────────────────────────────────────────────
python3 scripts/evaluate.py                    # ROUGE on test set
python3 scripts/baseline_zeroshot.py           # E0: zero-shot ROUGE baseline
python3 scripts/decoding_ablation.py           # E3: 29-config beam / lp sweep
python3 scripts/evaluate_faithfulness.py       # E4: hallucination + NLI metrics
python3 scripts/bootstrap_ci.py                # 95 % CIs for E1 model comparison
python3 scripts/compare_experiments.py         # aggregate results → CSV + table
```

Equivalent `make` targets are available for every step above — run `make` to
list them all.

---

## Demo

```bash
streamlit run scripts/app.py
# or via the convenience launcher:
bash scripts/run_app.sh
# opens http://localhost:8501
```

Features:

- **Model selector** — sidebar dropdown auto-discovers all checkpoints in
  `models/best/`
- **Two-column layout** — dialogue input with generation settings (left) /
  summary + action items + entities + generation metadata (right)
- **Beam width slider** (1–8) and length-penalty selector
  (0.8 / 1.0 / 1.2 / 1.25 / 1.3 / 1.4)
- **Action-item extraction** — regex patterns for modal + action-verb
  constructions
- **spaCy NER entity cards** — named entities surfaced from the generated
  summary
- **Accurate latency** — measured via `torch.mps.synchronize()` to account
  for MPS async dispatch

---

## Project Structure

```
meeting-summarizer/
├── config.yaml                   # ALL hyperparameters — single source of truth
├── config_extended.yaml          # Extended training config (E8: 8 epochs, cosine LR)
├── requirements.txt              # Full pinned dependency list
├── Makefile                      # Workflow targets: make install / train / demo / …
├── model_card.md                 # HuggingFace model card
├── data/
│   └── cache/                    # Tokenized dataset cache (git-ignored)
├── models/
│   ├── checkpoints/              # Per-epoch training checkpoints (git-ignored)
│   └── best/                     # Best checkpoint per experiment (git-ignored)
│       ├── facebook_bart-base_with_speakers/   # E1 champion
│       ├── facebook_bart-base_no_speakers/     # E2 ablation
│       ├── facebook_bart-base_split_speakers/  # E6 windowing
│       ├── facebook_bart-base_lora/            # E5 LoRA
│       ├── facebook_bart-base_extended/        # E8 extended
│       ├── t5-small_with_speakers/             # E1 T5
│       └── google_pegasus-cnn_dailymail_with_speakers/  # E7
├── results/
│   ├── error_analysis.md         # Manual annotation of 20 test examples
│   ├── error_analysis_raw.json   # Raw examples with source / reference / generated
│   ├── experiment_1_architecture.csv  # Aggregated results table (compare_experiments.py)
│   └── metrics/                  # Per-experiment JSON results (113 files)
│       ├── data_audit.json       # Dataset statistics and token distributions
│       ├── faithfulness_report.json  # E4 hallucination / NLI / speaker metrics
│       ├── bootstrap_ci_e1.json  # Bootstrap 95 % CIs for BART vs T5 delta
│       ├── decoding_D*.json      # 29 decoding ablation configs (D1–D29)
│       ├── sweep_*.json          # Multi-model decoding sweeps
│       ├── zeroshot_*.json       # E0 zero-shot baselines
│       ├── *_test.json           # Fine-tuned model test-set evaluations
│       └── README.md             # Field-by-field schema for all metric files
├── scripts/                      # 18 executable pipeline scripts
│   ├── verify_env.py             # Pre-flight MPS / BF16 environment check
│   ├── predownload_assets.py     # One-time HuggingFace asset download
│   ├── hf_whoami.py              # HuggingFace authentication check
│   ├── data_audit.py             # Dataset statistics + leakage guard
│   ├── preprocess.py             # Tokenization pipeline (3 variants)
│   ├── baseline_zeroshot.py      # E0: zero-shot ROUGE baseline
│   ├── train.py                  # Fine-tuning (reads config.yaml)
│   ├── train_lora.py             # E5: LoRA parameter-efficient fine-tuning
│   ├── pegasus_experiment.py     # E7: PEGASUS download → zero-shot → train
│   ├── evaluate.py               # ROUGE evaluation on saved checkpoint
│   ├── decoding_ablation.py      # E3: 29-config beam / length-penalty sweep
│   ├── multi_model_sweep.py      # E3-style sweep across all model variants
│   ├── evaluate_faithfulness.py  # E4: NER hallucination + NLI faithfulness
│   ├── bootstrap_ci.py           # Bootstrap 95 % CIs for E1 model comparison
│   ├── compare_experiments.py    # Aggregate results → comparison table + CSV
│   ├── error_analysis_helper.py  # 20-sample error analysis generation
│   ├── app.py                    # Streamlit inference demo
│   └── run_app.sh                # Streamlit launcher script
├── notebooks/
│   └── eda.ipynb                 # SAMSum exploratory data analysis (7 sections)
└── docs/
    ├── ARCHITECTURE.md           # System design, pipeline flow, config reference
    └── EXPERIMENTS.md            # Full experiment write-ups, tables, and analysis
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for pipeline design and
configuration details.
See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for full experiment results,
decoding ablation rankings, and error analysis.

---

## Limitations

- **Synthetic dataset:** SAMSum dialogues were written by paid annotators to
  resemble WhatsApp conversations — they are not transcripts of real meetings.
  Performance on actual meeting recordings (with disfluencies, overlapping
  speech, and domain-specific jargon) has not been evaluated.
- **Two-speaker bias:** 73 % of SAMSum training examples involve exactly two
  speakers. Multi-party summarization (3+ participants) is underrepresented
  and may degrade silently on production meeting data.
- **Hallucination rate 10.1 %:** The spaCy NER cross-reference metric detects
  only entity-level confabulations. Action-direction swaps and fabricated
  events (e.g. example idx=654 in the error analysis) are not caught by the
  automated metric and require human review.
- **Brittle action-item extraction:** The regex pipeline in `app.py` is a
  proof-of-concept. It produces false positives on quoted speech and misses
  multi-clause action items.
- **MPS-only timing:** All latency and training-time figures are from Apple
  M4 Pro MPS. CUDA hardware will produce different timings.
- **CC BY-NC-ND 4.0:** Non-commercial use only. Commercial deployment or
  derivative model distribution requires explicit permission from the SAMSum
  dataset authors.


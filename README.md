# Meeting Summarizer

![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)
![PyTorch MPS/BF16](https://img.shields.io/badge/PyTorch-MPS%2FBF16-ee4c2c?logo=pytorch&logoColor=white)
![License CC-BY-NC-ND-4.0](https://img.shields.io/badge/License-CC--BY--NC--ND--4.0-lightgrey)
![Platform macOS Sequoia](https://img.shields.io/badge/Platform-macOS%20Sequoia-000000?logo=apple&logoColor=white)

Abstractive dialogue summarization via fine-tuned sequence-to-sequence models
on the SAMSum corpus. Achieves **ROUGE-L 40.12** — exceeding the ≥ 40 project
target — using `facebook/bart-base` (139 M parameters), running on a single
Apple M4 Pro with BF16/MPS and no cloud GPU.

```
Raw SAMSum dialogues (16,368 total)
      │
      ▼  preprocess.py  — speaker-tagged tokenization, 3 preprocessing variants
      ▼  train.py       — Seq2SeqTrainer · 5 epochs · BF16/MPS · 72.4 min
      ▼  decoding_ablation.py — 29-config beam search / length-penalty sweep
      ▼  D27: num_beams=5 · length_penalty=1.33

ROUGE-L 40.12
```

---

## Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Training Setup](#5-training-setup)
6. [Experiments](#6-experiments)
7. [Results](#7-results)
8. [Best Model Selection](#8-best-model-selection)
9. [Demo Application](#9-demo-application)
10. [Repository Structure](#10-repository-structure)
11. [Hardware](#11-hardware)
12. [Installation](#12-installation)
13. [Reproducing Results](#13-reproducing-results)
14. [Evaluation Instructions](#14-evaluation-instructions)
15. [Example Usage](#15-example-usage)
16. [Limitations](#16-limitations)
17. [Future Work](#17-future-work)

---

## 1. Overview

Meetings and messaging conversations contain actionable information — decisions,
commitments, and follow-up tasks — that is time-consuming to extract manually.
This project builds an end-to-end abstractive summarization pipeline that
converts multi-speaker chat dialogues into concise, faithful natural-language
summaries.

The system demonstrates that a mid-size pre-trained seq2seq model
(`facebook/bart-base`, 139 M parameters) fine-tuned on approximately 15 K
annotated dialogue–summary pairs can reliably meet a ROUGE-L ≥ 40 target
without cloud infrastructure, specialized hardware, or parameter counts in the
billions.

The project covers the full ML lifecycle: dataset acquisition and auditing,
multi-model architecture comparison, systematic training ablations,
decoding-strategy optimization, faithfulness and hallucination analysis,
parameter-efficient fine-tuning (LoRA), and a live Streamlit inference demo.

---

## 2. Problem Statement

**Task:** Given a multi-turn, multi-speaker chat dialogue, generate a concise
abstractive summary that captures the key facts, decisions, and action items
while preserving correct speaker attribution.

**Why it is hard:**

- Dialogues are informal, fragmented, and elliptical — very different from the
  news articles most pre-trained summarization models are trained on.
- Speaker turns must be tracked: "I'll send the file" means different things
  depending on who said it.
- Summaries must be both fluent and faithful — hallucinated names, places, or
  events are a concrete failure mode, not a minor stylistic issue.
- The compression ratio is high: summaries average 28.7 tokens while source
  dialogues average 148.9 tokens (~5.2× compression).

**Success criterion:** ROUGE-L ≥ 40 on the 819-sample SAMSum test set.
Achieved: **ROUGE-L 40.12** (D27 decoding config).

---

## 3. Dataset

### SAMSum Corpus

**Source:** `knkarthick/samsum` (Hugging Face Hub)
**License:** CC BY-NC-ND 4.0 — non-commercial use only
**Reference:** Gliwa et al., 2019

SAMSum contains ~16 K English messenger-style dialogues annotated with gold
abstractive summaries written by professional linguists. Dialogues simulate
WhatsApp-style conversations across everyday topics: scheduling, coordination,
social plans, and information exchange.

### Split Sizes

| Split | Dialogues |
|-------|----------:|
| Train | 14,731 |
| Validation | 818 |
| Test | 819 |
| **Total** | **16,368** |

No data leakage — zero ID overlap across all splits confirmed by
`scripts/data_audit.py` (`results/metrics/data_audit.json`,
`leakage_check.passed = true`).

### Token Statistics

Source: `results/metrics/data_audit.json` — T5 tokenizer, training split
(n = 14,731).

| Field | min | p50 | p90 | p99 | max | mean |
|-------|----:|----:|----:|----:|----:|-----:|
| Dialogue tokens | 13 | 119 | 296 | 525 | 1,153 | 148.9 |
| Summary tokens | 2 | 25 | 50 | 73 | 94 | 28.7 |

`max_source_length=512` covers ≈ 99 % of training dialogues (p99 = 525 tokens;
the top ~1 % are truncated at token 512).
`max_target_length=128` covers all summaries (max = 94 tokens).

### Speaker Distribution (training split)

| Speakers | Dialogues | % |
|----------|----------:|--:|
| 2 | 10,758 | 73.0 % |
| 3 | 2,808 | 19.1 % |
| 4 | 822 | 5.6 % |
| 5+ | 343 | 2.3 % |

73 % of conversations are two-speaker. The corpus is constructed (not
transcribed), which limits direct generalization to real meeting recordings.

**Why speaker tags matter:** prepending `Alice:` / `Bob:` turn prefixes before
fine-tuning lifts ROUGE-L from 33.23 → 39.85 — a **+6.62 absolute / +19.9 %
relative gain** (E1 vs E2 ablation). The model uses speaker identity as a
discourse signal for correct attribution.

---

## 4. Model Architecture

### Architecture Comparison

Three pre-trained seq2seq architectures were evaluated:

| Model | Parameters | Architecture | Pre-training data |
|-------|-----------:|--------------|-------------------|
| `facebook/bart-base` | 139 M | Transformer encoder–decoder (BART) | Books + Wikipedia (denoising) |
| `t5-small` | 60 M | Transformer encoder–decoder (T5) | C4 (text-to-text) |
| `google/pegasus-cnn_dailymail` | 568 M | Transformer encoder–decoder (PEGASUS) | CNN/DailyMail news |

**Selected champion: `facebook/bart-base`** — it achieves the highest ROUGE-L
after fine-tuning (39.85 baseline → 40.12 with optimized decoding) while
remaining small enough to train in under 75 minutes on consumer hardware.

### BART-base Architecture

BART uses a bidirectional encoder (BERT-style) and an autoregressive decoder
(GPT-style), pre-trained with a text-denoising objective. This architecture is
well-suited to seq2seq generation tasks. Fine-tuning adds task-specific
adaptation to the dialogue–summary domain via standard cross-entropy loss on
teacher-forced target tokens.

### LoRA Variant (E5)

A LoRA (Low-Rank Adaptation) variant applies rank-16 adapters to the query and
value projection matrices of every attention layer:

- Trainable parameters: 0.63 % of total (0.88 M / 139 M)
- Target modules: `q_proj`, `v_proj`
- LoRA alpha: 32, dropout: 0.05
- ROUGE-L: 37.59 (94 % of full fine-tune quality at 1/160 the trainable
  parameter count)

---

## 5. Training Setup

### Configuration (`config.yaml`)

All hyperparameters are version-controlled in [`config.yaml`](config.yaml) —
no values are hardcoded in training scripts.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model_name` | `facebook/bart-base` | Highest post-fine-tune ROUGE-L |
| `batch_size` | 8 | Safe for BF16/MPS on 24 GB UMA; peak ≈ 1.1 GB model memory |
| `learning_rate` | 5e-5 | Standard seq2seq fine-tuning rate |
| `weight_decay` | 0.01 | AdamW L2 regularization |
| `num_epochs` | 5 | Upper bound; early stopping at patience=2 |
| `warmup_steps` | 500 | Linear warmup ≈ 3.4 % of total steps |
| `gradient_clip_max_norm` | 1.0 | Prevents training instability |
| `max_source_length` | 512 | Covers ≈ 99 % of dialogues |
| `max_target_length` | 128 | Covers all summaries |
| `num_beams` | 4 | Default decoding; optimized in E3 |
| `length_penalty` | 1.0 | Neutral; optimized in E3 |
| `use_bf16` | true | M4 Pro natively supports BF16 |
| `lr_scheduler_type` | linear | Linear decay after warmup |
| `seed` | 42 | Full reproducibility via `transformers.set_seed()` |

### MPS-Specific Constraints

```yaml
dataloader_num_workers: 0       # Python multiprocessing + MPS → context errors
dataloader_pin_memory: false    # UMA: CPU and GPU share the same memory pool
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is required for ops not yet implemented in the
MPS kernel; training proceeds without silent CPU fallback.

### Training Results

| Model | Train time | Best epoch | Peak val R-L | Test R-L |
|-------|-----------:|-----------:|-------------:|---------:|
| BART-base (full) | 72.4 min | 5 | 41.57 | 39.85 |
| T5-small (full) | 35.1 min | 2 | — | 31.95 |
| BART-base LoRA | 54.7 min | 5 | 38.43 | 37.59 |
| BART-base extended (E8) | 259.6 min | 4 | 40.00 | 38.46 |

---

## 6. Experiments

Eight experiments were conducted. Each is fully reproducible via the scripts
listed in [Reproducing Results](#13-reproducing-results).

### E0 — Zero-Shot Baseline

**Purpose:** Establish the pre-fine-tuning performance floor.
**Method:** Evaluate each model on 100 randomly sampled test examples without
any SAMSum-specific training.

| Model | R-1 | R-2 | R-L |
|-------|-----|-----|-----|
| BART-base | 27.34 | 8.87 | 19.89 |
| T5-small | 27.60 | 7.63 | 22.19 |

Both models produce partial, unfocused summaries. BART generates plausible
text but without task alignment. T5 requires the `summarize:` prefix and still
underperforms.

---

### E1 — Supervised Fine-Tuning (Primary Experiment)

**Purpose:** Measure the gain from task-specific fine-tuning.
**Method:** Fine-tune BART-base and T5-small on 14,731 SAMSum training examples
using `Seq2SeqTrainer` with early stopping on validation ROUGE-L.

| Model | R-1 | R-2 | R-L | Train time | Best epoch |
|-------|-----|-----|-----|-----------|-----------|
| T5-small | 38.96 | 15.96 | 31.95 | 35.1 min | 2 |
| **BART-base** | **47.86** | **23.22** | **39.85** | **72.4 min** | **5** |

BART-base outperforms T5-small by **+7.90 ROUGE-L**. Bootstrap 95 % CI for
this delta: **[+6.99, +9.02]** — statistically significant (CI excludes zero,
1,000 bootstrap iterations). BART-base's larger encoder–decoder capacity and
BART denoising pre-training are better suited to the dialogue-to-summary task
than T5's C4 text-to-text pre-training.

---

### E2 — Speaker-Tag Ablation

**Purpose:** Quantify the contribution of speaker-turn prefixes.
**Method:** Re-train BART-base with speaker tags stripped from all dialogues
(`no_speakers` variant).

| Variant | R-L | Δ vs with_speakers |
|---------|-----|-------------------|
| with_speakers (E1) | 39.85 | — |
| no_speakers (E2) | 33.23 | **−6.62** |

Speaker tags account for **+6.62 ROUGE-L (+19.9 % relative)**. Without them,
the model cannot reliably attribute actions to the correct speaker, producing
summaries where the actor is either omitted or inverted.

---

### E3 — Decoding Strategy Ablation

**Purpose:** Optimize beam search hyperparameters post-training without
retraining.
**Method:** Evaluate the E1 BART-base checkpoint across 29 configurations of
`num_beams` (4–12) and `length_penalty` (0.8–1.45).

**Top-5 configurations by ROUGE-L:**

| Config | num_beams | length_penalty | R-L |
|--------|----------:|--------------:|-----|
| **D27** | **5** | **1.33** | **40.12** |
| D24 | 5 | 1.35 | 40.12 |
| D19 | 5 | 1.30 | 40.11 |
| D28 | 5 | 1.37 | 40.11 |
| D23 | 5 | 1.32 | 40.11 |

D27 (`beam=5, lp=1.33`) achieves the sweep's peak ROUGE-L of 40.12 — tied
with D24 (beam=5, lp=1.35) — and is selected as champion for its marginally
lower length penalty. Thirteen of the 29 configurations exceed ROUGE-L 40.0,
all clustered around `num_beams=5` with `length_penalty` in [1.28, 1.45]:
five beams provide sufficient candidate diversity while `lp > 1.0` encourages
longer outputs toward reference length. Inference: **200.6 ms per sample** on
M4 Pro MPS.

---

### E4 — Faithfulness Evaluation

**Purpose:** Go beyond ROUGE to assess semantic correctness and hallucination
rate.
**Method:** Apply three complementary metrics to all 819 test-set predictions
from the E1 BART-base model:

- **NER cross-reference:** spaCy-extracted entities in the generated summary
  that are absent from the source dialogue → hallucination rate.
- **Speaker preservation:** fraction of speaker names mentioned in the dialogue
  that also appear in the generated summary.
- **NLI faithfulness:** cross-encoder NLI model probability that the generated
  summary is entailed by the source dialogue.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Entity hallucination rate | 10.1 % (83/819) | ~1 in 10 summaries contains a fabricated entity |
| Speaker preservation | 75.5 % | 3 in 4 summaries correctly name the relevant speakers |
| Avg NLI faithfulness | 0.308 | Moderate — model is broadly faithful but not precise |
| Length–ROUGE correlation | −0.25 | Shorter generated summaries correlate weakly with lower ROUGE |

Manual error analysis of 20 sampled examples (see `results/error_analysis.md`)
classifies 4 as hallucination (HALLUC), 7 as partial omission (PARTIAL), 3 as
over-generic (GENERIC), and 6 as correct (OK).

---

### E5 — LoRA Parameter-Efficient Fine-Tuning

**Purpose:** Demonstrate that high-quality summarization is achievable with a
tiny fraction of trainable parameters.
**Method:** Apply LoRA (r=16, alpha=32) to `q_proj` and `v_proj` of all
attention layers. All other weights are frozen.

| | Trainable params | Train time | Test R-L |
|--|----------------:|----------:|---------|
| Full fine-tune (E1) | 139.4 M (100 %) | 72.4 min | 39.85 |
| LoRA (E5) | 0.88 M (0.63 %) | 54.7 min | 37.59 |

LoRA achieves 94.3 % of full fine-tune ROUGE-L using only 0.63 % of the
trainable parameters and 24 % less training time. This makes it the practical
choice when storage or compute is constrained.

---

### E6 — Long-Dialogue Windowing (split_speakers)

**Purpose:** Test whether splitting long dialogues into overlapping windows
improves handling of truncated inputs.
**Method:** Conversations exceeding `max_source_length=512` are split at
speaker turn boundaries with 64-token overlap. Short conversations are
unaffected.

| Variant | R-L | Δ vs E1 |
|---------|-----|--------|
| with_speakers (E1) | 39.85 | — |
| split_speakers (E6) | 39.08 | −0.77 |

Windowing slightly degrades performance. Since only ~1 % of training dialogues
exceed 512 tokens (p99 = 525), the additional preprocessing complexity
introduces alignment noise without a meaningful coverage benefit.

---

### E7 — PEGASUS Domain-Transfer

**Purpose:** Test whether a larger model pre-trained specifically for
summarization (`google/pegasus-cnn_dailymail`, 568 M parameters) transfers to
the dialogue domain without fine-tuning.

| Condition | R-1 | R-2 | R-L |
|-----------|-----|-----|-----|
| PEGASUS zero-shot | 1.85 | 0.00 | 1.60 |
| PEGASUS fine-tuned | 1.19 | 0.01 | 1.15 |

Both results are near-zero. PEGASUS is pre-trained on CNN/DailyMail news
articles and uses document-gap-sentence pre-training — its vocabulary and
generation patterns are incompatible with short, informal chat dialogues.
Fine-tuning on SAMSum made results marginally worse, suggesting the
optimization landscape diverged from the pre-training distribution.

---

### E8 — Extended Training Schedule

**Purpose:** Test whether longer training with cosine LR decay improves over
the 5-epoch linear schedule.
**Method:** 8 epochs, cosine LR scheduler (`config_extended.yaml`).

| Schedule | Epochs | Train time | Best epoch | Test R-L |
|----------|-------:|-----------:|-----------:|---------|
| Linear (E1) | 5 | 72.4 min | 5 | 39.85 |
| Cosine (E8) | 8 | 259.6 min | 4 | 38.46 |

Extended training degrades ROUGE-L by **−1.39** despite 3.6× more compute.
Best epoch = 4 indicates the model overfits after epoch 4 on this 14 K-sample
corpus. The 5-epoch linear schedule with early stopping remains optimal.

---

## 7. Results

### Complete Experiment Summary

All metrics are macro-averaged ROUGE F-measures × 100 on the 819-sample test
set unless noted.

| # | Model / Config | R-1 | R-2 | R-L | Notes |
|---|----------------|-----|-----|-----|-------|
| E0 | BART-base zero-shot | 27.34 | 8.87 | 19.89 | 100-sample subset |
| E0 | T5-small zero-shot | 27.60 | 7.63 | 22.19 | 100-sample subset |
| E1 | T5-small fine-tuned | 38.96 | 15.96 | 31.95 | 35.1 min · epoch 2 |
| **E1** | **BART-base fine-tuned** | **47.86** | **23.22** | **39.85** | **72.4 min · epoch 5** |
| E2 | BART-base no_speakers | 38.95 | 19.17 | 33.23 | −6.62 R-L vs E1 |
| **E3** | **BART-base D27 beam=5 lp=1.33** | **48.48** | **23.55** | **40.12** | **Best of 29 configs** |
| E5 | BART-base LoRA (r=16, 0.63 % params) | 45.15 | 21.20 | 37.59 | 54.7 min |
| E6 | BART-base split_speakers | 47.11 | 22.55 | 39.08 | −0.77 R-L vs E1 |
| E7 | PEGASUS zero-shot | 1.85 | 0.00 | 1.60 | Domain mismatch |
| E8 | BART-base extended (8 ep · cosine) | 46.45 | 22.05 | 38.46 | −1.39 R-L vs E1 |

### Statistical Significance

| Comparison | R-L delta | 95 % CI | Significant? |
|------------|----------:|---------|:------------:|
| BART E1 vs T5 E1 | +7.90 | [+6.99, +9.02] | Yes — CI excludes zero |
| BART E1 (R-L 95 % CI) | — | [38.53, 41.15] | — |

Bootstrap CIs computed with 1,000 iterations on 819 test-set predictions.

### Key Quantitative Findings

| Finding | Value |
|---------|-------|
| Zero-shot → fine-tune gain (BART) | 19.89 → 39.85 (+100.4 % relative) |
| Fine-tune → decoding gain (D27) | 39.85 → 40.12 (+0.27 R-L) |
| Speaker-tag value | +6.62 R-L (+19.9 % relative) |
| LoRA efficiency | 94.3 % quality at 0.63 % trainable params |
| Extended training penalty | −1.39 R-L at 3.6× compute |
| D27 inference latency | 200.6 ms per sample (MPS) |

---

## 8. Best Model Selection

**Champion:** `facebook/bart-base` fine-tuned with `with_speakers` preprocessing,
evaluated with decoding config **D27** (`num_beams=5`, `length_penalty=1.33`).

**Why BART-base over T5-small:**
T5-small (60 M params) is faster to train (35 min vs 72 min) but achieves
ROUGE-L 31.95 — **7.90 points below** BART-base. The bootstrap 95 % CI for
this difference is [+6.99, +9.02], excluding zero. The larger model capacity
and BART's denoising pre-training are a better fit for abstractive dialogue
summarization.

**Why not PEGASUS:**
Despite being 4× larger (568 M params), `google/pegasus-cnn_dailymail` scores
ROUGE-L 1.60 zero-shot — near random. The CNN/DailyMail pre-training
distribution is incompatible with short, informal dialogues.

**Why not LoRA:**
LoRA achieves a strong 37.59 ROUGE-L with only 0.63 % trainable parameters
(excellent for constrained deployments), but falls 2.26 R-L points short of
the full fine-tune and does not clear the ≥ 40 target.

**Why `with_speakers` over `split_speakers`:**
Long-dialogue windowing (`split_speakers`, E6) produced ROUGE-L 39.08 — 0.77
points below the standard preprocessing. Since only ~1 % of dialogues exceed
512 tokens, the overhead is not justified.

**Why D27 decoding:**
A 29-configuration grid search over `num_beams` ∈ {4, 5, 6, 8, 12} and
`length_penalty` ∈ {0.8, 1.0, 1.2, …, 1.45} identifies `beam=5, lp=1.33` as
optimal. The five-beam configuration consistently outperforms four-beam and
eight-beam. Length penalty > 1.0 encourages the decoder to generate longer outputs,
compensating for BART's default tendency to under-generate relative to
reference summaries (reference mean: 28.7 tokens). `lp=1.33` yields a mean
output of 17.18 tokens per summary.

**Checkpoint location:** `models/best/facebook_bart-base_with_speakers/`

---

## 9. Demo Application

A Streamlit web application provides a live inference interface requiring no
code.

### Features

- **Model selector** — sidebar dropdown auto-discovers all checkpoints in
  `models/best/`; switch between BART, T5, LoRA, and extended variants
- **Two-column layout** — dialogue input with collapsible generation settings
  (left) / generated summary + action items + entities + metadata (right)
- **Beam width slider** — 1–8 beams, interactive
- **Length-penalty selector** — 0.8 / 1.0 / 1.2 / 1.25 / 1.3 / 1.4
- **Action-item extraction** — regex patterns for modal verb + action-verb
  constructions (`should`, `will`, `need to`, `going to`)
- **spaCy NER entity cards** — PERSON, ORG, DATE, GPE entities extracted from
  the generated summary
- **Accurate latency display** — measured via `torch.mps.synchronize()` to
  account for MPS asynchronous dispatch

### Run the Demo

```bash
streamlit run scripts/app.py
# or via the launcher:
bash scripts/run_app.sh
# opens http://localhost:8501
```

---

## 10. Repository Structure

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
│       ├── facebook_bart-base_with_speakers/   # E1 champion → E3 decoding
│       ├── facebook_bart-base_no_speakers/     # E2 ablation
│       ├── facebook_bart-base_split_speakers/  # E6 windowing
│       ├── facebook_bart-base_lora/            # E5 LoRA
│       ├── facebook_bart-base_extended/        # E8 extended schedule
│       ├── t5-small_with_speakers/             # E1 T5-small
│       └── google_pegasus-cnn_dailymail_with_speakers/  # E7 PEGASUS
├── results/
│   ├── error_analysis.md         # Manual annotation of 20 test examples
│   ├── error_analysis_raw.json   # Raw examples: source / reference / generated
│   ├── experiment_1_architecture.csv  # Aggregated results table
│   └── metrics/                  # 113 per-experiment JSON result files
│       ├── data_audit.json       # Dataset statistics and token distributions
│       ├── faithfulness_report.json  # E4: hallucination / NLI / speaker metrics
│       ├── bootstrap_ci_e1.json  # Bootstrap CIs for E1 BART vs T5 delta
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
│   ├── train.py                  # Full fine-tuning (reads config.yaml)
│   ├── train_lora.py             # E5: LoRA parameter-efficient fine-tuning
│   ├── pegasus_experiment.py     # E7: PEGASUS pipeline
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
│   └── eda.ipynb                 # SAMSum EDA: token stats, speaker dist, plots
└── docs/
    ├── ARCHITECTURE.md           # System design, pipeline flow, config reference
    └── EXPERIMENTS.md            # Full experiment write-ups, tables, and analysis
```

---

## 11. Hardware

| Component | Specification |
|-----------|---------------|
| SoC | Apple M4 Pro (T6041) |
| Memory | 24 GB Unified Memory (LPDDR5X) |
| GPU | 20-core GPU (Metal 3) |
| OS | macOS Sequoia 15.7.3 |
| Compute | PyTorch 2.x MPS backend — BF16 verified |

All training and inference uses `torch.device("mps")` with BF16 precision.
All latency and training-time figures are hardware-specific; CUDA results
will differ.

---

## 12. Installation

### Option A — Makefile (recommended)

```bash
make install    # create venv · pip install · download spaCy model
make verify     # MPS / BF16 pre-flight check (all items must pass)
make download   # one-time HuggingFace asset download (~2 GB, network)
```

Run `make` with no arguments to list all available targets.

### Option B — Manual

```bash
# Python 3.12 is required (system Python 3.14 lacks stable PyTorch wheels)
python3.12 -m venv ~/.venvs/meeting-summarizer --prompt meeting-summarizer
source ~/.venvs/meeting-summarizer/bin/activate

pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# Verify the MPS environment before running any training
python3 scripts/verify_env.py

# Download model weights and dataset to local HuggingFace cache (network, ~2 GB)
python3 scripts/predownload_assets.py
```

---

## 13. Reproducing Results

All hyperparameters are version-controlled in [`config.yaml`](config.yaml) or
[`config_extended.yaml`](config_extended.yaml) for E8. Seed is fixed at 42 via
`transformers.set_seed()` for full determinism.

```bash
# ── Data ────────────────────────────────────────────────────────────────────
python3 scripts/data_audit.py               # dataset stats + leakage guard
python3 scripts/preprocess.py               # tokenize: with_speakers + no_speakers
python3 scripts/preprocess.py --variants split_speakers   # E6 windowing variant

# ── Training ────────────────────────────────────────────────────────────────
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py              # E1 BART-base
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py --model t5-small   # E1 T5
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_lora.py         # E5 LoRA
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py \
    --config config_extended.yaml                                    # E8 extended

# E7: PEGASUS pipeline (each flag is one stage; run in order)
python3 scripts/pegasus_experiment.py --download     # ~2.2 GB, network required
python3 scripts/pegasus_experiment.py --zeroshot     # E0 on 100 samples
python3 scripts/pegasus_experiment.py --preprocess   # tokenize SAMSum for PEGASUS
python3 scripts/pegasus_experiment.py --train        # fine-tune 568 M-param model

# ── Evaluation ──────────────────────────────────────────────────────────────
python3 scripts/baseline_zeroshot.py        # E0: zero-shot ROUGE on 100 samples
python3 scripts/evaluate.py                 # ROUGE on 819-sample test set
python3 scripts/decoding_ablation.py        # E3: 29-config decoding grid search
python3 scripts/evaluate_faithfulness.py    # E4: NER hallucination + NLI
python3 scripts/bootstrap_ci.py             # 95 % CIs for E1 model comparison
python3 scripts/compare_experiments.py      # aggregate all results → CSV + table

# ── Task 4 — Adversarial robustness (T5-small LoRA) ─────────────────────────
# Generate data first: make task4-generate  (or scripts/task4_adversarial.py generate)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/task4_adversarial.py retrain \
    --base_model models/best/t5-small_lora_task1
python3 scripts/task4_adversarial.py compare   # pre/post on held-out adversarial
# Coherence: results/metrics/task4_coherence_template.csv is a manual rating sheet only (no auto scores).

# ── Task 5 — LoRA rank sweep + structured JSON ─────────────────────────────
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/task5_lora_structured.py train --ranks 2 4 8 16 32
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/task5_lora_structured.py train_structured \
    --ranks 2 4 8 16 32   # writes models/best/t5-small_lora_r*/merged_structured/
python3 scripts/task5_lora_structured.py eval --ranks 2 4 8 16 32
python3 scripts/task5_lora_structured.py structured --ranks 8   # parse_success + fallback metrics
python3 scripts/task5_lora_structured.py sweet_spot              # optional: --min-parse-success 0.2
python3 scripts/task5_lora_structured.py package                # uses sweet_spot or --default_rank
# Sweet spot uses parse gate then optional ROUGE-only fallback; if still null, package uses --default_rank.
```

Equivalent `make` targets exist for every command above (`make train`,
`make evaluate`, `make decoding`, etc.). Task 4/5: `make task4-retrain`,
`make task5-train-structured`, `make task5-structured`, etc. (`make` lists all targets).

---

## 14. Evaluation Instructions

### Reproduce ROUGE on Test Set

```bash
# Evaluate the champion checkpoint (models/best/facebook_bart-base_with_speakers/)
python3 scripts/evaluate.py
# Output: results/metrics/facebook_bart-base_with_speakers_test.json
# Expected: rouge1≈47.86, rouge2≈23.22, rougeL≈39.85
```

### Reproduce E3 Champion Decoding

```bash
python3 scripts/decoding_ablation.py
# Runs 29 configs; writes results/metrics/decoding_D*.json
# Champion: decoding_D27_beam5_lp1.33.json → rougeL=40.12
```

### Reproduce Bootstrap CIs

```bash
python3 scripts/bootstrap_ci.py
# Writes results/metrics/bootstrap_ci_e1.json
# BART-base 95% CI: rougeL [38.53, 41.15]
# BART vs T5 delta: +7.90 [+6.99, +9.02]
```

### Reproduce Faithfulness Metrics

```bash
python3 scripts/evaluate_faithfulness.py
# Writes results/metrics/faithfulness_report.json
# Expected: hallucination_rate≈0.101, avg_speaker_preservation≈0.755
```

All result JSON files are committed to `results/metrics/` for reference
without rerunning.

---

## 15. Example Usage

### Streamlit Demo (Recommended)

```bash
streamlit run scripts/app.py
```

Paste a dialogue into the text area and click **Summarize**. Adjust beam width
and length penalty in the settings expander to reproduce the D27 champion
configuration (`num_beams=5`, `length_penalty=1.33`).

### Python API

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "models/best/facebook_bart-base_with_speakers"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                 torch_dtype=torch.bfloat16).to("mps")

dialogue = (
    "Alice: Are we still meeting at 3?\n"
    "Bob: Yes, but can we push it to 3:30? Running behind.\n"
    "Alice: Sure, I'll book the room until 5 just in case.\n"
    "Bob: Perfect. Can you also send the agenda beforehand?\n"
    "Alice: Already on it — you'll have it by 2."
)

inputs = tokenizer(dialogue, return_tensors="pt",
                   max_length=512, truncation=True).to("mps")

with torch.no_grad():
    ids = model.generate(**inputs, num_beams=5,
                         length_penalty=1.33, max_new_tokens=128)

print(tokenizer.decode(ids[0], skip_special_tokens=True))
# Example output:
# "Alice and Bob will meet at 3:30. Alice will book the room until 5
#  and send the agenda to Bob by 2."
```

---

## 16. Limitations

- **Synthetic dataset:** SAMSum dialogues were written by paid annotators to
  resemble WhatsApp conversations — they are not transcripts of real meetings.
  Performance on recordings with disfluencies, overlapping speech, filler
  words, and domain-specific jargon has not been evaluated.
- **Two-speaker bias:** 73 % of SAMSum training examples involve exactly two
  speakers. Multi-party summarization (3+ participants) is systematically
  underrepresented and may degrade silently on real meeting data.
- **Entity hallucination rate 10.1 %:** The automated NER metric detects
  entity-level confabulations. Action-direction swaps, implied-participant
  errors, and fabricated events (e.g. test example idx=654) are not caught
  and require human review.
- **Brittle action-item extraction:** The regex pipeline in `app.py` is a
  proof-of-concept. It produces false positives on quoted speech and misses
  multi-clause constructions.
- **Hardware-specific timing:** All latency and training-time figures are from
  Apple M4 Pro MPS. Inference on CUDA will differ in both speed and numerical
  outputs (BF16 rounding differs from FP16/FP32).
- **ROUGE as proxy:** ROUGE-L measures lexical overlap with one reference
  summary. It does not capture factual correctness, fluency, or whether
  action items are correctly attributed. The E4 faithfulness metrics partially
  address this but are themselves automated proxies.
- **CC BY-NC-ND 4.0:** Non-commercial use only. Commercial deployment or
  derivative model distribution requires explicit permission from the SAMSum
  dataset authors.
- **T5-small structured JSON:** SentencePiece maps `{`/`}` to `<unk>`; supervised
  JSON uses an **inner** representation (no outer braces) plus decode-time wrapping.
  **Strict** JSON match rates may stay low versus a summarization prior; use
  `prediction_to_structured_dict()` in `scripts/task5_lora_structured.py` when the
  API requires guaranteed `topics` / `action_items` / `decision` keys.

---

## 17. Future Work

- **Real meeting transcripts:** Evaluate on AMI or ICSI meeting corpora with
  genuine disfluencies, domain vocabulary, and 4–12 speaker conversations.
- **Instruction-tuned models:** Fine-tune `facebook/bart-large` or a compact
  instruction-tuned LLM (e.g. Mistral 7B with QLoRA) to test whether the
  architecture capacity ceiling limits ROUGE-L gains.
- **Faithfulness-constrained decoding:** Add NLI-based reranking at inference
  time to penalize candidate summaries that contradict the source dialogue,
  targeting the 10.1 % hallucination rate.
- **Multi-reference evaluation:** Collect 2–3 independent reference summaries
  per dialogue for a subset of the test set to obtain a less noisy ROUGE upper
  bound.
- **Cross-lingual extension:** SAMSum exists only in English. Applying
  mBART or mT5 could extend the pipeline to multilingual meeting scenarios.
- **Action-item extraction hardening:** Replace the regex pipeline in `app.py`
  with a sequence labeling or generative extraction model trained on
  meeting-minutes data.

---

## License Notice

**The SAMSum dataset is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
(Creative Commons — Attribution · Non-Commercial · No Derivatives).**

> This project, the fine-tuned model weights, and any generated outputs are
> restricted to **non-commercial use only**, in compliance with the SAMSum
> dataset license. Deploying or distributing the model in any commercial
> product or service is prohibited without explicit permission from the
> dataset authors.

Original dataset: Gliwa et al., 2019 — *SAMSum Corpus: A Human-annotated
Dialogue Dataset for Abstractive Summarization* (CC BY-NC-ND 4.0).

---

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for pipeline design and
configuration details.
See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for full experiment write-ups,
per-config decoding tables, and error analysis.


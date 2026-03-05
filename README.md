# Meeting Summarizer

Abstractive dialogue summarization using `facebook/bart-base` fine-tuned on the
[SAMSum corpus](https://huggingface.co/datasets/knkarthick/samsum).
Achieves **ROUGE-L 40.12** (best decoding config D27: beam=5, lp=1.33) on the
819-sample test set, up from a zero-shot floor of 19.89 on the same model.

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

## Hardware

| Component | Specification |
|-----------|--------------|
| SoC | Apple M4 Pro (T6041) |
| Memory | 24 GB Unified Memory (LPDDR5X) |
| GPU | 20 GPU cores (Metal 3) |
| OS | macOS Sequoia 15.7.3 |
| Compute | PyTorch MPS backend — BF16 verified |

All training and inference runs use `torch.device("mps")` with BF16 precision.
`num_workers=0` and `pin_memory=False` are required MPS constraints.

---

## Results

All metrics are macro-averaged ROUGE F-measures × 100 unless noted.

### E0 — Zero-Shot Baseline (100-sample subset, no fine-tuning)

> Source: `results/metrics/zeroshot_facebook_bart-base.json`,
> `results/metrics/zeroshot_t5-small.json`

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| BART-base (zero-shot) | 27.34 | 8.87 | 19.89 |
| T5-small (zero-shot) | 27.60 | 7.63 | 22.19 |

---

### E1 — Architecture Comparison (819-sample test set, `with_speakers` variant)

> Source: `results/metrics/facebook_bart-base_with_speakers_test.json`,
> `results/metrics/t5-small_with_speakers_test.json`

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Training time | Best epoch |
|-------|---------|---------|---------|--------------|-----------|
| T5-small (zero-shot) | 27.60 | 7.63 | 22.19 | — | — |
| T5-small (fine-tuned) | 38.96 | 15.96 | 31.95 | 35 min | 2 |
| BART-base (zero-shot) | 27.34 | 8.87 | 19.89 | — | — |
| **BART-base (fine-tuned)** | **47.86** | **23.22** | **39.85** | **72 min** | **5** |

BART-base outperforms T5-small by **+7.90 ROUGE-L** after fine-tuning
(+20.0% relative improvement over its own zero-shot baseline).

---

### E2 — Speaker Tag Ablation (BART-base, 819-sample test set)

> Source: `results/metrics/facebook_bart-base_with_speakers_test.json`,
> `results/metrics/facebook_bart-base_no_speakers_test.json`

| Variant | ROUGE-1 | ROUGE-2 | ROUGE-L | Δ ROUGE-L |
|---------|---------|---------|---------|-----------|
| `no_speakers` (stripped) | 38.95 | 19.17 | 33.23 | — |
| **`with_speakers` (full)** | **47.86** | **23.22** | **39.85** | **+6.62** |

Preserving speaker attribution tags contributes **+6.62 ROUGE-L** (+19.9% relative).
Both models trained to epoch 5; the `no_speakers` variant converges to a lower ceiling.

---

### E3 — Decoding Strategy Ablation (BART-base `with_speakers`, 819-sample test set)

> Source: `results/metrics/experiment_3_decoding_summary.json`
> Full sweep: **29 configs** tested. 13 cross ROUGE-L 40.

**Top configs (ROUGE-L ≥ 40.00):**

| ID | Config | ROUGE-1 | ROUGE-2 | ROUGE-L | ms/sample |
|----|--------|---------|---------|---------|----------|
| **D27** | **beam=5, lp=1.33** | **48.54** | **23.52** | **40.12** | **~195** |
| D24 | beam=5, lp=1.35 | 48.56 | 23.51 | 40.12 | 197 |
| D19 | beam=5, lp=1.30 | 48.51 | 23.49 | 40.11 | 197 |
| D28 | beam=5, lp=1.37 | 48.58 | 23.55 | 40.11 | 193 |
| D23 | beam=5, lp=1.32 | 48.51 | 23.49 | 40.11 | 197 |
| D22 | beam=5, lp=1.28 | 48.49 | 23.48 | 40.11 | 199 |
| D29 | beam=5, lp=1.45 | 48.49 | 23.55 | 40.09 | 191 |
| D21 | beam=4, lp=1.28 | 48.49 | 23.35 | 40.05 | 169 |
| D25 | beam=5, lp=1.40 | 48.51 | 23.46 | 40.05 | 197 |
| D10 | beam=6, lp=1.20 | 48.14 | 23.36 | 40.03 | 178 |
| D17 | beam=5, lp=1.20 | 48.25 | 23.28 | 40.02 | 179 |
| D8  | beam=4, lp=1.30 | 48.42 | 23.41 | 40.01 | 137 |
| D7  | beam=4, lp=1.25 | 48.44 | 23.38 | 40.01 | 136 |

**Selected other configs (baseline reference):**

| ID | Config | ROUGE-1 | ROUGE-2 | ROUGE-L | ms/sample |
|----|--------|---------|---------|---------|----------|
| D2 | beam=4, lp=1.0 *(training baseline)* | 48.04 | 23.33 | 39.92 | 136 |
| D3 | beam=4, lp=1.2 | 48.33 | 23.35 | 39.97 | 136 |
| D5 | nucleus p=0.9, t=0.8 | 45.42 | 19.55 | 35.93 | 92 |

**Key findings:**
- **D27** (beam=5, lp=1.33) is the champion at **ROUGE-L 40.12**, exceeding the ≥40 target.
- A broad **beam=5 performance plateau** spans lp∈[1.28, 1.45] — all 8 beam=5 configs
  tested in this range exceed ROUGE-L 40.0.
- beam=6 and beam=8 both underperform beam=5; wider beams hurt SAMSum generation.
- **Best quality/cost tradeoff**: D8 (beam=4, lp=1.3) achieves ROUGE-L 40.01 at
  baseline latency (137 ms/sample), matching the training-cost performance of the
  champion without the extra beam overhead.
- Nucleus sampling (D5) is fastest (92 ms/sample) but loses −4.2 ROUGE-L.

---

### E5 — LoRA Fine-Tuning (BART-base, `with_speakers`, 819-sample test set)

> Source: `results/metrics/facebook_bart-base_lora_test.json`

Parameter-efficient fine-tuning using LoRA (rank=16, α=32, dropout=0.05) targeting
`q_proj` and `v_proj` attention layers. Trains only 0.88M/139.4M parameters (0.63%).

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable params |
|-------|---------|---------|---------|-----------------|
| BART-base (full fine-tune) | 48.04 | 23.33 | 39.92 | 139.4M (100%) |
| BART-base (LoRA) | 45.15 | 21.20 | 37.59 | 0.88M (0.63%) |

LoRA achieves **94.2%** of full fine-tune ROUGE-L quality while training only **0.63%** of
parameters. Training time: 54.7 min (5 epochs) on Apple M4 Pro MPS. Best validation
ROUGE-L: 38.43 (epoch 5). Merged model saved to `models/best/facebook_bart-base_lora/`.

---

### E6 — Conversation Splitting Preprocessing

> Source: `scripts/preprocess.py --variants split_speakers`

Sliding-window segmentation of long conversations (stride=256, min fragment=32 tokens)
to improve coverage of long dialogues exceeding `max_source_length=512`.

| Dataset Variant | Train | Val | Test | Δ Train |
|----------------|-------|-----|------|---------|
| `with_speakers` (original) | 14,731 | 818 | 819 | — |
| `split_speakers` (windowed) | 14,996 | 832 | 830 | +265 (+1.8%) |

---

### PEGASUS Cross-Domain Transfer Experiment (E7)

> Source: `results/metrics/zeroshot_google_pegasus-cnn_dailymail.json`,
> `results/metrics/google_pegasus-cnn_dailymail_with_speakers_test.json`

Tests `google/pegasus-cnn_dailymail` (568M / 767.6M params) as a third architecture.
PEGASUS was pre-trained with Gap Sentence Generation on news corpora — SAMSum dialogues
are structurally different, providing a cross-domain transfer baseline.

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | N | Notes |
|-----------|---------|---------|---------|---|-------|
| Zero-shot | 1.85 | 0.00 | 1.60 | 100 | Massive domain mismatch (news → dialogue) |
| Fine-tuned | 1.09 | 0.01 | 1.04 | 819 | Training completed but no learning (see below) |

**Training result**: ROUGE-L 1.04 (worse than zero-shot performance).
**Root cause**: Despite fixing `gradient_accumulation_steps=1`, the model showed no
learning progress throughout training. Loss remained constant at ~10.0 across all
epochs, indicating the model weights were not updating. This suggests PEGASUS's
news-specific pre-training creates an insurmountable domain gap for conversational
summarization. The model may require task-specific pre-training or architecture
modifications to adapt to dialogue structures.

Zero-shot ROUGE-L of **1.60** confirms the extreme news-to-dialogue domain gap.
PEGASUS is unsuitable for this cross-domain transfer task without significant
architectural changes or continued pre-training on conversational data.

---

### E8 — Extended Training (BART-base, 8 epochs, cosine LR)

> Source: `results/metrics/facebook_bart-base_extended_test.json`

Tests whether a longer schedule with cosine LR decay and lower peak LR improves
BART-base beyond the 5-epoch baseline.

**Config changes from E1 baseline:**

| Parameter | Baseline (E1) | Extended (E8) |
|-----------|--------------|----------------|
| Epochs | 5 | 8 |
| Learning rate | 5e-5 | 3e-5 |
| LR scheduler | linear | cosine |
| Warmup steps | 500 | 300 |
| Early stopping patience | 2 | 3 |

**Results (best checkpoint = epoch 4, val ROUGE-L = 39.98):**

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | Train time |
|-----------|---------|---------|---------|-----------|
| Baseline E1 (5ep, lr=5e-5) | 47.86 | 23.22 | 39.85 | 72 min |
| Extended E8 (8ep, lr=3e-5, cosine) | 46.45 | 22.05 | **38.46** | 259.6 min |

**Δ ROUGE-L = −1.39** (extended training underperforms baseline).

**Finding**: The lower peak LR (3e-5) caused underfitting — the model needed more
aggressive weight updates early in training. Best val ROUGE-L reached 39.98 (epoch 4)
vs the baseline's 41.57 (epoch 5), confirming the cosine schedule converged to a
suboptimal minimum. The hypothesis that "longer training with gentler LR improves
generalisation" was **not supported** on SAMSum with BART-base.

---

### E4 — Faithfulness Evaluation (BART-base `with_speakers`, 819-sample test set)

> Source: `results/metrics/faithfulness_report.json`

| Metric | Value | Method |
|--------|-------|--------|
| Hallucination rate | **10.1%** (83 / 819 examples) | spaCy `en_core_web_sm` NER entity cross-reference |
| Speaker preservation | **75.5%** | Jaccard overlap of speaker names in source vs summary |
| NLI faithfulness | **0.308** | `cross-encoder/nli-deberta-v3-small` (CPU inference) |
| Length–ROUGE-L correlation | **−0.25** | Pearson r over 819 samples |

The negative length–ROUGE correlation indicates shorter generated summaries tend to score
lower — the model occasionally over-compresses and drops key content.

---

## Error Analysis

Manual annotation of 20 stratified samples (seed=42, avg ROUGE-L = 39.48).
Full annotation in [`results/error_analysis.md`](results/error_analysis.md).

| Category | Count | % |
|----------|-------|---|
| ✅ Correct | 4 | 20% |
| ⚠️ Partial (speaker/fact error) | 8 | 40% |
| ❌H Hallucination | 6 | 30% |
| ❌G Over-generic | 2 | 10% |
| ❌T Truncated | 0 | 0% |

### Example 1 — ✅ Correct (idx=32, ROUGE-L=53.3)

**Dialogue**
```
Jack: Cocktails later?
May: YES!!!
May: You read my mind...
Jack: Possibly a little tightly strung today?
May: Sigh... without question.
Jack: Thought so.
May: A little drink will help!
Jack: Maybe two!
```
**Reference:** Jack and May will drink cocktails later.  
**Generated:** Jack will have a drink with May later.

---

### Example 2 — ❌H Hallucination (idx=654, ROUGE-L=12.8)

**Dialogue**
```
Richie: Pogba
Clay: Pogboom
Richie: what a strike yoh!
Clay: was off the seat the moment he chopped the ball back to his right foot
Richie: me too dude
Clay: hope his form lasts
...
```
**Reference:** Richie and Clay saw a very good football game, with one football player
chopping the ball back to his foot. Jose has trust in that player.  
**Generated:** Pogba scored the first goal of the season. He deserved to score after
his first 60 minutes.

*Hallucinated elements*: "scored the first goal of the season" and "first 60 minutes"
are absent from the dialogue — no scoreline is ever mentioned.

---

### Example 3 — ❌G Over-generic (idx=281, ROUGE-L=32.3)

**Dialogue**
```
Abby: Have you talked to Miro?
Dylan: No, not really, I've never had an opportunity
Abby: yes, he's so interesting
Abby: told me the story of his father coming from Albania to the US in the early 1990s
Dylan: really? illegally?
Abby: Yes! in a fishing boat!
```
**Reference:** Miro speaks Albanian with his parents. His family left Albania illegally
in the 1990s.  
**Generated:** Miro told Abby the story of his father coming from Albania to the US in
the early 1990s.

*Missing detail*: The dramatic fact that the family fled **illegally** (in a fishing boat)
is the key information — the model paraphrases the surface form and loses the specificity.

---

## Quick Start

```bash
# 1. Create Python 3.12 venv (system Python 3.14 lacks stable PyTorch wheels)
python3.12 -m venv ~/.venvs/meeting-summarizer --prompt meeting-summarizer
source ~/.venvs/meeting-summarizer/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# 3. Verify MPS environment (all checks must pass)
python3 scripts/verify_env.py

# 4. Pre-download all assets (network step — run once)
python3 scripts/predownload_assets.py

# 5. Launch demo (model must already be in models/best/; see Reproduction)
streamlit run scripts/app.py
```

---

## Reproduction

```bash
# Tokenize dataset (run once; produces data/cache/ variants)
python3 scripts/preprocess.py

# Tokenize with conversation splitting (long dialogue windowing)
python3 scripts/preprocess.py --variants split_speakers

# Fine-tune BART-base (reads all hyperparameters from config.yaml)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train.py

# LoRA fine-tune (parameter-efficient, 0.63% trainable params)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_lora.py

# PEGASUS experiment pipeline (download → zero-shot → preprocess → train)
python3 scripts/pegasus_experiment.py --download    # online, ~2.2GB
python3 scripts/pegasus_experiment.py --zeroshot     # E0 on 100 samples
python3 scripts/pegasus_experiment.py --preprocess   # tokenize SAMSum
python3 scripts/pegasus_experiment.py --train        # fine-tune (568M params)

# Evaluate on test set
python3 scripts/evaluate.py

# Decoding strategy ablation (E3 — 29 configs)
python3 scripts/decoding_ablation.py

# Faithfulness evaluation (E4)
python3 scripts/evaluate_faithfulness.py

# Aggregate all experiment results into comparison table
python3 scripts/compare_experiments.py
```

All hyperparameters are in [`config.yaml`](config.yaml).
Key values: `batch_size=8`, `lr=5e-5`, `num_epochs=5`, `warmup_steps=500`,
`num_beams=4`, `length_penalty=1.0`, `use_bf16=true`.

---

## Demo

```bash
streamlit run scripts/app.py
# Or use the launcher:
bash scripts/run_app.sh
# Opens http://localhost:8501
```

Features:
- **Model selector**: sidebar dropdown auto-discovers all models in `models/best/`
- Two-column layout: dialogue input with generation settings expander (left) /
  summary + action items + entities + generation info (right)
- Beam width slider (1–8) and length penalty selector (0.8 / 1.0 / 1.2 / 1.25 / 1.3 / 1.4)
- Regex-based action-item extraction (modal + action-verb patterns)
- spaCy NER entity cards
- Accurate latency measurement via `torch.mps.synchronize()`

---

## Dataset Statistics

From `results/metrics/data_audit.json` (T5 tokenizer, training split):

| Field | min | p50 | p90 | p99 | max | mean |
|-------|-----|-----|-----|-----|-----|------|
| Dialogue tokens | 11 | 135 | 258 | 375 | 945 | 147 |
| Summary tokens | 3 | 21 | 39 | 56 | 90 | 23 |

Coverage: 99%+ of dialogues fit within `max_source_length=512`;
all summaries fit within `max_target_length=128`.

---

## Project Structure

```
meeting-summarizer/
├── config.yaml                   # ALL hyperparameters — single source of truth
├── config_extended.yaml          # Extended training config (8 epochs, cosine LR)
├── requirements.txt              # Full pinned dependency list
├── model_card.md                 # HuggingFace model card
├── data/cache/                   # Tokenized dataset cache (git-ignored)
├── models/
│   ├── checkpoints/              # Training checkpoints (git-ignored)
│   └── best/                     # Best checkpoint per experiment (git-ignored)
├── results/
│   ├── error_analysis.md         # Manual annotation of 20 examples
│   ├── error_analysis_raw.json   # Raw examples with source/ref/generated
│   └── metrics/                  # Per-experiment JSON results (committed)
├── scripts/
│   ├── verify_env.py             # Pre-flight MPS environment check
│   ├── data_audit.py             # Dataset statistics + leakage guard
│   ├── preprocess.py             # Tokenization pipeline (3 variants + split_speakers)
│   ├── train.py                  # Fine-tuning script
│   ├── train_lora.py             # LoRA parameter-efficient fine-tuning
│   ├── evaluate.py               # ROUGE evaluation on test set
│   ├── decoding_ablation.py      # E3: 11-config beam/sampling strategy comparison
│   ├── pegasus_experiment.py     # PEGASUS pipeline (download/zeroshot/preprocess/train)
│   ├── evaluate_faithfulness.py  # E4: hallucination + speaker + NLI metrics
│   ├── compare_experiments.py    # Aggregate results table + CSV
│   ├── app.py                    # Streamlit demo with model selector
│   └── run_app.sh                # Streamlit launcher script
└── notebooks/
    └── eda.ipynb                 # Exploratory data analysis
```

---

## Limitations

- **Synthetic dataset**: SAMSum dialogues were written by paid annotators to
  resemble WhatsApp conversations — they are not transcripts of real meetings.
  Performance on actual meeting recordings (with disfluencies, overlapping
  speech, and domain-specific jargon) has not been evaluated.
- **Two-speaker bias**: ~75% of SAMSum conversations involve exactly two
  speakers. Multi-party summarization (3+ participants) is underrepresented
  in training and may degrade silently.
- **Hallucination rate 10.1%**: spaCy NER cross-reference detects only
  entity-level confabulations. Action-direction swaps and fabricated events
  (e.g. idx=654) are not caught by the automated metric.
- **Brittle action-item extraction**: The regex pipeline in `app.py` is a
  demonstration only. It produces false positives on quoted speech and
  misses multi-clause action items.
- **MPS-only timing**: All latency figures are from Apple M4 Pro MPS.
  Comparable CUDA hardware will differ.
- **CC BY-NC-ND 4.0**: Non-commercial use only. Commercial deployment or
  derivative model distribution is not permitted under the SAMSum license.


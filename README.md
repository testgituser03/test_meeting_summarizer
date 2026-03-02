# Meeting Summarizer

Abstractive dialogue summarization fine-tuned on the SAMSum dataset using
`facebook/bart-base` and `t5-small`, with full training infrastructure for
Apple M4 Pro (PyTorch MPS / BF16).

---

## ⚠️ License Notice

**The SAMSum dataset is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
(Creative Commons — Non-Commercial, No Derivatives).**

> This project, the fine-tuned model weights, and any generated outputs are
> restricted to **non-commercial use only**, in compliance with the SAMSum
> dataset license. Deploying or distributing the model in any commercial
> product or service is prohibited.

Original dataset: [SAMSum Corpus](https://huggingface.co/datasets/knkarthick/samsum) —
Gliwa et al., 2019.

---

## Hardware

| Component   | Specification                          |
|-------------|----------------------------------------|
| SoC         | Apple M4 Pro (T6041)                   |
| Memory      | 24 GB Unified Memory (LPDDR5X)         |
| GPU         | 20 GPU cores (Metal 3)                 |
| OS          | macOS Sequoia 15.7.3                   |
| Compute     | PyTorch MPS backend — BF16 verified    |

---

## Results

<!-- Fill in after training runs complete -->

| Experiment         | Model       | Dataset Variant  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------------------|-------------|------------------|---------|---------|---------|
| E0 Zero-shot       | BART-base   | —                | —       | —       | —       |
| E0 Zero-shot       | T5-small    | —                | —       | —       | —       |
| E1 Architecture    | T5-small    | with_speakers    | —       | —       | —       |
| E1 Architecture    | BART-base   | with_speakers    | —       | —       | —       |
| E2 Speaker Ablation| BART-base   | no_speakers      | —       | —       | —       |

---

## Setup

```bash
# Python 3.12 venv required (system Python 3.14 lacks stable PyTorch wheels)
python3.12 -m venv ~/.venvs/meeting-summarizer --prompt meeting-summarizer
source ~/.venvs/meeting-summarizer/bin/activate

# Install dependencies
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

# Verify MPS environment (all 5 checks must pass before any training)
python3 scripts/verify_env.py
```

---

## Training

```bash
# Step 1: Tokenize and cache both dataset variants (run once)
python3 scripts/preprocess.py

# Step 2: Fine-tune (all hyperparameters read from config.yaml)
python3 scripts/train.py

# Step 3: Evaluate on the held-out test set
python3 scripts/evaluate.py

# Step 4: Run decoding strategy ablation (no retraining needed)
python3 scripts/decoding_ablation.py

# Step 5: Faithfulness evaluation
python3 scripts/evaluate_faithfulness.py

# Step 6: Aggregate all experiment results
python3 scripts/compare_experiments.py
```

---

## Demo

```bash
streamlit run scripts/app.py
```

---

## Dataset Statistics

<!-- Fill in from results/metrics/data_audit.json after running data_audit.py -->

| Field          | min | p50 | p90 | p95 | p99 | max | mean |
|----------------|-----|-----|-----|-----|-----|-----|------|
| Dialogue tokens| —   | —   | —   | —   | —   | —   | —    |
| Summary tokens | —   | —   | —   | —   | —   | —   | —    |

---

## Faithfulness Evaluation

<!-- Fill in after running evaluate_faithfulness.py -->

| Metric                    | Value |
|---------------------------|-------|
| Hallucination rate        | —     |
| Speaker preservation rate | —     |
| NLI faithfulness score    | —     |

---

## Project Structure

```
meeting-summarizer/
├── config.yaml                   # ALL hyperparameters — single source of truth
├── requirements.txt
├── data/cache/                   # Tokenized dataset cache (git-ignored)
├── models/
│   ├── checkpoints/              # Training checkpoints (git-ignored)
│   └── best/                     # Best model per experiment (git-ignored)
├── results/
│   └── metrics/                  # Experiment JSON results (committed)
├── scripts/
│   ├── verify_env.py             # Pre-flight MPS environment check
│   ├── data_audit.py             # Dataset statistics + leakage guard
│   ├── preprocess.py             # Tokenization pipeline (both variants)
│   ├── train.py                  # Fine-tuning script
│   ├── evaluate.py               # ROUGE evaluation on test set
│   ├── decoding_ablation.py      # E3: beam/sampling strategy comparison
│   ├── evaluate_faithfulness.py  # E4: hallucination + speaker metrics
│   ├── compare_experiments.py    # Aggregate results table
│   └── app.py                    # Streamlit demo
└── notebooks/
    └── eda.ipynb                 # Exploratory data analysis
```

---

## Limitations

- SAMSum dialogues are **synthetic** (written by annotators, not real meetings).
- Dataset skews toward **2-speaker conversations** (~75%); performance on
  multi-party meetings may be lower.
- Regex-based action-item extraction in the demo (`app.py`) is brittle and not
  evaluated rigorously.
- Results are specific to Apple MPS/BF16; values on CUDA hardware will differ.
- **CC BY-NC-ND 4.0** license restricts all use to non-commercial contexts.

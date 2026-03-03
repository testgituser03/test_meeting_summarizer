---
language: en
license: cc-by-nc-nd-4.0
datasets:
  - knkarthick/samsum
metrics:
  - rouge
tags:
  - summarization
  - abstractive-summarization
  - dialogue-summarization
  - bart
  - seq2seq
model-index:
  - name: bart-base-samsum-summarizer
    results:
      - task:
          type: summarization
        dataset:
          type: knkarthick/samsum
          name: SAMSum
          split: test
        metrics:
          - type: rouge1
            value: 47.86
            name: ROUGE-1
          - type: rouge2
            value: 23.22
            name: ROUGE-2
          - type: rougeL
            value: 39.85
            name: ROUGE-L
---

# bart-base-samsum-summarizer

`facebook/bart-base` fine-tuned on the [SAMSum](https://huggingface.co/datasets/knkarthick/samsum)
dialogue summarization corpus.

> **⚠️ License**: SAMSum is released under **CC BY-NC-ND 4.0** (non-commercial, no derivatives).
> This model card, the model weights, and any outputs produced with them are
> subject to the same terms. **Commercial use is prohibited.**

---

## Model Description

| Field | Value |
|-------|-------|
| Base model | `facebook/bart-base` (139M parameters) |
| Task | Abstractive dialogue summarization |
| Language | English |
| License | cc-by-nc-nd-4.0 |
| Dataset | SAMSum (`knkarthick/samsum`) |
| Hardware trained on | Apple M4 Pro, 24 GB UMA, MPS / BF16 |

---

## Intended Use

- **Intended use**: Summarizing short chat conversations (≤ 512 tokens) into
  1–3 sentence abstractive summaries.
- **Out-of-scope**: Real-time transcription, audio processing, multi-lingual
  dialogues, or any commercial product.
- **Not recommended for**: Mission-critical applications where hallucinations
  cannot be tolerated. The model hallucinates entity-level details in ~10% of
  test examples.

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_id = "your-hf-username/bart-base-samsum-summarizer"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.bfloat16)
model.eval()

dialogue = """
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
Jerry: Thanks! Do you know how to make the lemon ones?
Amanda: The biscuits? I'll send you the recipe. It's easy!
""".strip()

inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens = 128,
        num_beams      = 4,
        length_penalty = 1.2,   # D3 best config (ROUGE-L 39.97)
        early_stopping = True,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
# → "Amanda will bring Jerry some cookies tomorrow and send him the recipe."
```

---

## Performance

All metrics are macro-averaged ROUGE F-measures × 100 on the 819-sample SAMSum test set.

### Test-Set ROUGE

| Metric | Value |
|--------|-------|
| ROUGE-1 | 47.86 |
| ROUGE-2 | 23.22 |
| **ROUGE-L** | **39.85** *(training config: beam=4, lp=1.0)* |
| ROUGE-L (best decoding: D3 beam=4, lp=1.2) | **39.97** |

### Comparison: Fine-Tuned vs Zero-Shot

| | ROUGE-L |
|--|---------|
| BART-base zero-shot (100 samples) | 19.89 |
| BART-base fine-tuned (819 samples) | **39.85** (+19.96) |

### Decoding Strategy Ablation

| Config | ROUGE-L | Avg tokens | ms/sample |
|--------|---------|-----------|----------|
| beam=4, lp=0.8 | 39.49 | 15.2 | 138 |
| beam=4, lp=1.0 | 39.92 | 15.9 | 136 |
| **beam=4, lp=1.2** | **39.97** | **16.7** | **136** |
| beam=8, lp=1.0 | 39.74 | 15.8 | 220 |
| nucleus p=0.9, t=0.8 | 35.93 | 18.8 | 92 |

### Faithfulness Metrics

| Metric | Value |
|--------|-------|
| Hallucination rate (spaCy NER) | 10.1% (83 / 819) |
| Speaker preservation | 75.5% |
| NLI faithfulness (DeBERTa-v3) | 0.308 |
| Length–ROUGE-L Pearson r | −0.25 |

---

## Training Procedure

### Dataset

- **Train**: 14,731 examples
- **Validation**: 818 examples
- **Test**: 819 examples
- **Variant used**: `with_speakers` — speaker attribution tags (`Name: `) preserved.
  Ablation shows this contributes +6.62 ROUGE-L vs stripping tags.

### Preprocessing

Dialogues are tokenized with `AutoTokenizer` from `facebook/bart-base`.
`max_source_length=512`, `max_target_length=128` (covers 99%+ of SAMSum
examples at these lengths). No task prefix (BART does not require one;
T5 uses `"summarize: "`).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `facebook/bart-base` |
| Optimizer | AdamW |
| Learning rate | 5.0 × 10⁻⁵ |
| LR schedule | Linear decay |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Batch size | 8 |
| Max epochs | 5 |
| Early stopping patience | 2 |
| Gradient clip norm | 1.0 |
| Precision | BF16 |
| Best epoch | 5 |
| Best val ROUGE-L | 41.57 |
| Training time | 72.4 min (M4 Pro MPS) |

### Compute

Trained on Apple M4 Pro (T6041), 24 GB Unified Memory, 20 GPU cores.
PyTorch 2.10.0 MPS backend, BF16.

---

## Limitations

- **Synthetic training data**: SAMSum was constructed by human annotators
  writing fictional WhatsApp-style dialogues. The model has not been evaluated
  on real meeting transcripts or audio-derived text.
- **Two-speaker bias**: ~75% of SAMSum examples involve exactly 2 participants.
  Summarization quality for 3+ speaker conversations is likely lower.
- **Hallucination**: ~10.1% of test summaries contain at least one NER-detected
  hallucinated entity. The actual hallucination rate is higher for non-entity
  errors (e.g. fabricated scores, inverted speaker actions).
- **Speaker attribution errors**: ~25% of summaries have at least one
  speaker attribution mistake (e.g. "X will call Y" when it is Y who called).
- **Non-commercial only**: CC BY-NC-ND 4.0 applies to all outputs.

---

## Citation

```bibtex
@inproceedings{gliwa-etal-2019-samsum,
    title     = "{SAMS}um Corpus: A Human-annotated Dialogue Dataset
                 for Abstractive Summarization",
    author    = "Gliwa, Bogdan and Mochol, Iwona and Biesek, Maciej
                 and Wawer, Aleksander",
    booktitle = "Proceedings of the 2nd Workshop on New Frontiers in
                 Summarization",
    year      = "2019",
    publisher = "Association for Computational Linguistics",
    doi       = "10.18653/v1/D19-5409",
}
```

---

## How to Push to HuggingFace Hub

```bash
# 1. Log in
huggingface-cli login

# 2. Create the repository (replace <username>)
huggingface-cli repo create bart-base-samsum-summarizer --type model

# 3. Push model weights + tokenizer
python3 - <<'EOF'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "models/best/facebook_bart-base_with_speakers"
repo_id    = "your-hf-username/bart-base-samsum-summarizer"   # ← replace

tok = AutoTokenizer.from_pretrained(model_path)
mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=torch.bfloat16)

tok.push_to_hub(repo_id)
mdl.push_to_hub(repo_id)
print(f"✅ Pushed to https://huggingface.co/{repo_id}")
EOF

# 4. Push model card
huggingface-cli upload your-hf-username/bart-base-samsum-summarizer \
    model_card.md README.md

# 5. Verify
huggingface-cli whoami
# → Opens https://huggingface.co/your-hf-username/bart-base-samsum-summarizer
```

> **Note**: Do NOT push `models/best/` to GitHub — model weights belong on
> the HuggingFace Hub only. The `.gitignore` should already exclude `models/`.

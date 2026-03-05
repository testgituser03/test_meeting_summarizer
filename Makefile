# Makefile — Meeting Summarizer
# Usage: make <target>
# All targets assume the project venv is active:
#   source ~/.venvs/meeting-summarizer/bin/activate

PYTHON   := python3
VENV     := ~/.venvs/meeting-summarizer
CONFIG   := config.yaml
CONFIG_E := config_extended.yaml

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Meeting Summarizer — available targets"
	@echo "  ──────────────────────────────────────"
	@echo "  make install       Create venv + install all dependencies (first run)"
	@echo "  make verify        Pre-flight MPS/BF16 environment check"
	@echo "  make download      Download models + dataset to HF cache (network)"
	@echo "  make audit         Dataset statistics + leakage guard"
	@echo "  make preprocess    Tokenize SAMSum → data/cache/ (all variants)"
	@echo "  make zeroshot      E0: zero-shot ROUGE baseline (no fine-tuning)"
	@echo "  make train         Fine-tune BART-base (reads config.yaml)"
	@echo "  make train-lora    LoRA fine-tune BART-base (0.63% trainable params)"
	@echo "  make train-t5      Fine-tune T5-small"
	@echo "  make train-ext     Extended training schedule (config_extended.yaml)"
	@echo "  make evaluate      ROUGE evaluation on saved best checkpoint"
	@echo "  make decoding      E3: 29-config decoding ablation sweep"
	@echo "  make faithfulness  E4: NER hallucination + NLI faithfulness"
	@echo "  make bootstrap     Bootstrap 95% CIs for E1 models"
	@echo "  make compare       Aggregate all results → comparison table + CSV"
	@echo "  make demo          Launch Streamlit inference demo on :8501"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: install
install:
	python3.12 -m venv $(VENV) --prompt meeting-summarizer
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/python -m spacy download en_core_web_sm
	@echo "\n  ✅  Environment ready. Activate with:\n  source $(VENV)/bin/activate"

.PHONY: verify
verify:
	$(PYTHON) scripts/verify_env.py

.PHONY: download
download:
	$(PYTHON) scripts/predownload_assets.py

# ── Data ──────────────────────────────────────────────────────────────────────
.PHONY: audit
audit:
	$(PYTHON) scripts/data_audit.py

.PHONY: preprocess
preprocess:
	$(PYTHON) scripts/preprocess.py --variants with_speakers no_speakers
	$(PYTHON) scripts/preprocess.py --variants split_speakers

# ── Training ──────────────────────────────────────────────────────────────────
.PHONY: zeroshot
zeroshot:
	$(PYTHON) scripts/baseline_zeroshot.py

.PHONY: train
train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/train.py --config $(CONFIG)

.PHONY: train-lora
train-lora:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/train_lora.py

.PHONY: train-t5
train-t5:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/train.py --model t5-small

.PHONY: train-ext
train-ext:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/train.py --config $(CONFIG_E)

# ── Evaluation ────────────────────────────────────────────────────────────────
.PHONY: evaluate
evaluate:
	$(PYTHON) scripts/evaluate.py

.PHONY: decoding
decoding:
	$(PYTHON) scripts/decoding_ablation.py

.PHONY: faithfulness
faithfulness:
	$(PYTHON) scripts/evaluate_faithfulness.py

.PHONY: bootstrap
bootstrap:
	$(PYTHON) scripts/bootstrap_ci.py

.PHONY: compare
compare:
	$(PYTHON) scripts/compare_experiments.py

# ── Demo ──────────────────────────────────────────────────────────────────────
.PHONY: demo
demo:
	bash scripts/run_app.sh

# Makefile — Meeting Summarizer
# Usage: make <target>
# All targets assume the project venv is active:
#   source ~/.venvs/meeting-summarizer/bin/activate

# Prefer project .venv when present so datasets, peft, etc. are available
VENV     := $(or $(wildcard .venv),~/.venvs/meeting-summarizer)
PYTHON   := $(or $(wildcard .venv/bin/python),python3)
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
	@echo "  make task1-train   Task 1: train T5-small LoRA on SAMSum (5 epochs)"
	@echo "  make task1-analyze Task 1: extract attention + analyze 100 samples"
	@echo "  make task1         Task 1 end-to-end (train + analyze)"
	@echo "  make task2-quant   Task 2: quantize T5-small LoRA runtime artifacts"
	@echo "  make task2-bench   Task 2: benchmark quantized runtime + streaming + scaling"
	@echo "  make task2         Task 2 end-to-end (quantize + benchmark)"
	@echo "  make task3-extract    Task 3: extract decoder activations"
	@echo "  make task3-vector     Task 3: compute steering vectors"
	@echo "  make task3-infer      Task 3: run steered generation (mean_diff)"
	@echo "  make task3-eval       Task 3: evaluate steering + human template"
	@echo "  make task3-full-sweep Task 3: full sweep (all 3 methods)"
	@echo "  make task3            Task 3 end-to-end pipeline"
	@echo "  make task4-generate   Task 4: generate adversarial transcripts"
	@echo "  make task4-eval       Task 4: robustness eval (150+150)"
	@echo "  make task4-retrain   Task 4: retrain (70/30, LR 1e-5, ≤3 ep, early stop on adv ROUGE)"
	@echo "  make task4-compare   Task 4: pre/post ROUGE-L comparison"
	@echo "  make task4            Task 4 end-to-end"
	@echo "  make task5-train     Task 5: LoRA rank ablation (2,4,8,16,32)"
	@echo "  make task5-eval      Task 5: ROUGE-L, latency, size per rank"
	@echo "  make task5-train-structured Task 5: supervised inner-JSON LoRA → merged_structured/"
	@echo "  make task5-structured Task 5: JSON validity + structured ROUGE"
	@echo "  make task5-sweet     Task 5: identify sweet spot"
	@echo "  make task5-package   Task 5: package production baseline"
	@echo "  make task5           Task 5 end-to-end"
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

# ── Task 1: Attention patterns for speaker attribution ───────────────────────
.PHONY: task1-train
task1-train:
	$(PYTHON) scripts/task1_attention_patterns.py train \
		--output_dir models/best/t5-small_lora_task1

.PHONY: task1-analyze
task1-analyze:
	$(PYTHON) scripts/task1_attention_patterns.py analyze \
		--model_path models/best/t5-small_lora_task1 \
		--n_samples 100 --save_heatmaps --save_rollout

.PHONY: task1
task1: task1-train task1-analyze

# ── Task 2: Quantization & real-time benchmarking ───────────────────────────
.PHONY: task2-quant
task2-quant:
	$(PYTHON) scripts/task2_quantization.py --config $(CONFIG)

.PHONY: task2-bench
task2-bench:
	$(PYTHON) scripts/task2_benchmark.py --config $(CONFIG)

.PHONY: task2
task2: task2-quant task2-bench

# ── Task 3: Activation steering for focus control ──────────────────────────
.PHONY: task3-extract
task3-extract:
	$(PYTHON) scripts/extract_activations.py --config $(CONFIG)

.PHONY: task3-vector
task3-vector:
	$(PYTHON) scripts/compute_steering_vector.py \
		--activations results/activations/facebook_bart-base_with_speakers_train_layers-6.pt

.PHONY: task3-infer
task3-infer:
	$(PYTHON) scripts/steering_inference.py \
		--config $(CONFIG) \
		--steering results/steering/facebook_bart-base_with_speakers_train_steering.pt \
		--split test --method mean_diff --layers 6 --alphas 0 0.5 1.0 1.5

.PHONY: task3-eval
task3-eval:
	$(PYTHON) scripts/evaluate_steering.py \
		--input results/steering/facebook_bart-base_with_speakers_test_mean_diff_steering_generations.json

.PHONY: task3-full-sweep
task3-full-sweep: task3-extract task3-vector
	$(PYTHON) scripts/steering_inference.py --config $(CONFIG) \
		--steering results/steering/facebook_bart-base_with_speakers_train_steering.pt \
		--split test --method mean_diff --layers 6 --alphas 0 0.5 1.0 1.5
	$(PYTHON) scripts/steering_inference.py --config $(CONFIG) \
		--steering results/steering/facebook_bart-base_with_speakers_train_steering.pt \
		--split test --method pca_delta --layers 6 --alphas 0 0.5 1.0 1.5
	$(PYTHON) scripts/steering_inference.py --config $(CONFIG) \
		--steering results/steering/facebook_bart-base_with_speakers_train_steering.pt \
		--split test --method logistic --layers 6 --alphas 0 0.5 1.0 1.5
	$(PYTHON) scripts/evaluate_steering.py \
		--input results/steering/facebook_bart-base_with_speakers_test_mean_diff_steering_generations.json
	$(PYTHON) scripts/evaluate_steering.py \
		--input results/steering/facebook_bart-base_with_speakers_test_pca_delta_steering_generations.json
	$(PYTHON) scripts/evaluate_steering.py \
		--input results/steering/facebook_bart-base_with_speakers_test_logistic_steering_generations.json
	$(PYTHON) scripts/task3_summarize_results.py

.PHONY: task3
task3: task3-extract task3-vector task3-infer task3-eval

# ── Task 4: Adversarial robustness ──────────────────────────────────────────
.PHONY: task4-generate
task4-generate:
	$(PYTHON) scripts/task4_adversarial.py generate --n_original 150 --n_adversarial 150 --n_heldout 100

.PHONY: task4-eval
task4-eval:
	$(PYTHON) scripts/task4_adversarial.py eval --model_path models/best/t5-small_lora_task1

.PHONY: task4-retrain
task4-retrain:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/task4_adversarial.py retrain --base_model models/best/t5-small_lora_task1

.PHONY: task4-compare
task4-compare:
	$(PYTHON) scripts/task4_adversarial.py compare

.PHONY: task4
task4: task4-generate task4-eval task4-retrain task4-compare

# ── Task 5: LoRA rank ablation & structured output ───────────────────────────
.PHONY: task5-train
task5-train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/task5_lora_structured.py train --ranks 2 4 8 16 32

.PHONY: task5-train-structured
task5-train-structured:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTHON) scripts/task5_lora_structured.py train_structured --ranks 2 4 8 16 32

.PHONY: task5-eval
task5-eval:
	$(PYTHON) scripts/task5_lora_structured.py eval --ranks 2 4 8 16 32

.PHONY: task5-structured
task5-structured:
	$(PYTHON) scripts/task5_lora_structured.py structured --ranks 2 4 8 16 32

.PHONY: task5-sweet
task5-sweet:
	$(PYTHON) scripts/task5_lora_structured.py sweet_spot

.PHONY: task5-package
task5-package:
	$(PYTHON) scripts/task5_lora_structured.py package

.PHONY: task5
task5: task5-train task5-train-structured task5-eval task5-structured task5-sweet task5-package

.PHONY: compare
compare:
	$(PYTHON) scripts/compare_experiments.py

# ── Demo ──────────────────────────────────────────────────────────────────────
.PHONY: demo
demo:
	bash scripts/run_app.sh

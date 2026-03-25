#!/usr/bin/env python3
"""
Task 1 — Attention Patterns for Speaker Attribution & Key Moment Detection.

Implements:
1) Train T5-small + LoRA on SAMSum (with speaker tags), 5 epochs.
2) Extract encoder/decoder/cross attention for 100 test dialogues (all layers/heads).
3) Speaker-aware attention aggregation.
4) Key-moment detection from cross-attention.
5) Attention visualization heatmaps.
6) Speaker imbalance entropy metric.
7) Final per-sample JSON output.

Interpretation note: attention is extracted via a teacher-forced forward pass using the
full `generate()` output as `decoder_input_ids`. That is useful for exploratory
visualization but is **not** identical to step-by-step attention inside beam search
with KV-cache — do not treat as a strict causal account of decoding-time focus.

Usage:
  python3 scripts/task1_attention_patterns.py train \
      --output_dir models/best/t5-small_lora_task1

  python3 scripts/task1_attention_patterns.py analyze \
      --model_path models/best/t5-small_lora_task1 \
      --n_samples 100 \
      --save_heatmaps
"""

# MPS fallback and offline mode must be set before importing torch/transformers.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


MODEL_NAME = "t5-small"
TASK_PREFIX = "summarize: "
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_STEPS = 500
DEFAULT_NUM_EPOCHS = 5
DEFAULT_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokenize_for_training(tokenizer, batch: dict[str, list[str]]) -> dict[str, Any]:
    inputs = [TASK_PREFIX + d for d in batch["dialogue"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["summary"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_lora_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # T5 attention projections are named "q" and "v".
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    return tokenizer, lora_model, lora_config


def train_t5_lora(
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()

    ds_raw = load_dataset("knkarthick/samsum")
    tokenizer, model, lora_config = build_lora_model(MODEL_NAME)

    tokenized = ds_raw.map(
        lambda b: _tokenize_for_training(tokenizer, b),
        batched=True,
        remove_columns=ds_raw["train"].column_names,
        desc="Tokenizing SAMSum (with speakers)",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=DEFAULT_NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=4,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        bf16=(device.type in {"cuda", "mps"}),
        fp16=False,
        report_to="none",
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save LoRA adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Optional merged checkpoint for standalone inference
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    summary = {
        "model": MODEL_NAME,
        "epochs": DEFAULT_NUM_EPOCHS,
        "dataset": "knkarthick/samsum",
        "with_speaker_tags": True,
        "lora_config": {
            "target_modules": list(lora_config.target_modules),
            "r": lora_config.r,
            "alpha": lora_config.lora_alpha,
            "dropout": lora_config.lora_dropout,
        },
        "trainer_setup": {
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_source_length": MAX_SOURCE_LEN,
            "max_target_length": MAX_TARGET_LEN,
            "task_prefix": TASK_PREFIX,
        },
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_pct": round(trainable_params / max(total_params, 1) * 100, 4),
        "saved_adapter": str(output_dir),
        "saved_merged": str(merged_dir),
    }

    with open(output_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def parse_dialogue_turns(dialogue: str, prefix: str = TASK_PREFIX) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    offset = len(prefix)
    for idx, line in enumerate(dialogue.splitlines()):
        if not line.strip():
            offset += 1
            continue
        m = re.match(r"^([^:]{1,64}):\s*(.*)$", line)
        if m:
            speaker = m.group(1).strip()
            utterance = m.group(2)
        else:
            speaker = "UNKNOWN"
            utterance = line
        start = offset
        end = offset + len(line)
        turns.append(
            {
                "turn_id": idx,
                "speaker": speaker,
                "text": utterance,
                "line": line,
                "char_start": start,
                "char_end": end,
            }
        )
        offset = end + 1
    return turns


def token_to_turn_mapping(
    tokenizer,
    full_input_text: str,
    turns: list[dict[str, Any]],
) -> tuple[list[int], list[str], list[str]]:
    """
    Returns:
      token_to_turn: len=S, each entry is turn_id or -1.
      input_tokens: source token strings (len=S).
      token_to_speaker: len=S, each entry is speaker or "<SPECIAL>".
    """
    enc = tokenizer(
        full_input_text,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_to_turn: list[int] = []
    token_to_speaker: list[str] = []

    for (start, end) in offsets:
        if start == end:
            token_to_turn.append(-1)
            token_to_speaker.append("<SPECIAL>")
            continue

        mid = (start + end) // 2
        assigned_turn = -1
        assigned_speaker = "<UNK>"
        for t in turns:
            if t["char_start"] <= mid < t["char_end"]:
                assigned_turn = int(t["turn_id"])
                assigned_speaker = str(t["speaker"])
                break
        token_to_turn.append(assigned_turn)
        token_to_speaker.append(assigned_speaker)

    return token_to_turn, input_tokens, token_to_speaker


def stack_attention_tuple(attn_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
    # tuple[L] of [B,H,Q,K] -> [L,B,H,Q,K]
    return torch.stack(tuple(attn_tuple), dim=0)


def compute_speaker_attention(
    cross_attention_lhs: torch.Tensor,
    token_to_speaker: list[str],
) -> dict[str, float]:
    """
    cross_attention_lhs shape: [L,H,T,S]
    Output: {speaker: normalized attention share}
    """
    # Average over layers, heads, and generated positions -> [S]
    token_scores = cross_attention_lhs.float().mean(dim=(0, 1, 2)).cpu().numpy()

    speaker_totals: dict[str, float] = {}
    for i, score in enumerate(token_scores.tolist()):
        speaker = token_to_speaker[i] if i < len(token_to_speaker) else "<UNK>"
        if speaker in {"<SPECIAL>", "<UNK>"}:
            continue
        speaker_totals[speaker] = speaker_totals.get(speaker, 0.0) + float(score)

    total = sum(speaker_totals.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in sorted(speaker_totals.items(), key=lambda x: x[1], reverse=True)}


def extract_key_moments(
    cross_attention_lhs: torch.Tensor,
    input_tokens: list[str],
    token_to_turn: list[int],
    turns: list[dict[str, Any]],
    top_k_tokens: int = 12,
    top_k_turns: int = 3,
) -> dict[str, Any]:
    """
    cross_attention_lhs shape: [L,H,T,S]
    Returns top tokens and top turns by source-side cross-attention mass.
    """
    source_scores = cross_attention_lhs.float().mean(dim=(0, 1, 2)).cpu().numpy()  # [S]

    valid_idx = [i for i in range(len(source_scores)) if i < len(token_to_turn) and token_to_turn[i] >= 0]
    ranked_idx = sorted(valid_idx, key=lambda i: float(source_scores[i]), reverse=True)

    key_tokens: list[dict[str, Any]] = []
    for i in ranked_idx[:top_k_tokens]:
        turn_id = token_to_turn[i]
        turn = next((t for t in turns if t["turn_id"] == turn_id), None)
        key_tokens.append(
            {
                "token": input_tokens[i] if i < len(input_tokens) else "",
                "token_index": i,
                "score": float(source_scores[i]),
                "turn_id": int(turn_id),
                "speaker": turn["speaker"] if turn else "UNKNOWN",
            }
        )

    turn_scores: dict[int, float] = {}
    for i in valid_idx:
        t = token_to_turn[i]
        turn_scores[t] = turn_scores.get(t, 0.0) + float(source_scores[i])

    ranked_turns = sorted(turn_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_turns]

    top_turns: list[dict[str, Any]] = []
    for t_id, score in ranked_turns:
        turn = next((t for t in turns if t["turn_id"] == t_id), None)
        if turn is None:
            continue
        top_turns.append(
            {
                "turn_id": int(t_id),
                "speaker": str(turn["speaker"]),
                "line": str(turn["line"]),
                "score": float(score),
            }
        )

    return {"key_tokens": key_tokens, "top_turns": top_turns}


def compute_attention_entropy(speaker_scores: dict[str, float]) -> float:
    """H = -Σ p_i log(p_i), natural log."""
    if not speaker_scores:
        return 0.0
    entropy = 0.0
    for p in speaker_scores.values():
        if p > 0:
            entropy -= p * math.log(p)
    return float(entropy)


def compute_encoder_attention_rollout(encoder_attn_lhs: torch.Tensor) -> np.ndarray:
    """Compute attention rollout across encoder layers (Abnar & Zuidema style).

    encoder_attn_lhs shape: [L, H, S, S]
    Returns: [S, S] rollout matrix showing how input tokens influence each other.
    """
    # Average over heads first: [L, S, S]
    attn = encoder_attn_lhs.float().mean(dim=1).cpu().numpy()
    n_layers = attn.shape[0]
    # Rollout: R_1 = A_1, R_l = A_l @ R_{l-1} (normalized)
    rollout = attn[0].copy()
    for l in range(1, n_layers):
        rollout = attn[l] @ rollout
        # Normalize rows to sum to 1 (attention is already normalized per layer)
        row_sum = rollout.sum(axis=1, keepdims=True)
        rollout = np.where(row_sum > 1e-9, rollout / row_sum, rollout)
    return rollout.astype(np.float32)


def compute_cross_attention_rollout(cross_attn_lhs: torch.Tensor) -> np.ndarray:
    """Aggregate cross-attention across layers for dialogue -> summary mapping.

    cross_attn_lhs shape: [L, H, T, S]
    Returns: [T, S] aggregated attention (mean over layers/heads).
    """
    return cross_attn_lhs.float().mean(dim=(0, 1)).cpu().numpy()


def plot_attention_heatmap(
    cross_attention_lhs: torch.Tensor,
    input_tokens: list[str],
    summary_tokens: list[str],
    out_path: Path,
    max_src_tokens: int = 80,
    max_tgt_tokens: int = 60,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Cross-attention: head/layer averaged or already-rollout [T,S]
    mat = compute_cross_attention_rollout(cross_attention_lhs)

    s = min(mat.shape[1], max_src_tokens)
    t = min(mat.shape[0], max_tgt_tokens)
    mat = mat[:t, :s]

    xlabels = input_tokens[:s]
    ylabels = summary_tokens[:t]

    plt.figure(figsize=(max(10, s * 0.25), max(6, t * 0.25)))
    sns.heatmap(mat, cmap="magma", cbar=True)
    plt.xticks(np.arange(s) + 0.5, xlabels, rotation=90, fontsize=7)
    plt.yticks(np.arange(t) + 0.5, ylabels, rotation=0, fontsize=7)
    plt.xlabel("Input tokens (dialogue)")
    plt.ylabel("Summary tokens")
    if title:
        plt.title(title)
    else:
        plt.title("Cross-attention (dialogue → summary mapping)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_encoder_rollout_heatmap(
    encoder_attn_lhs: torch.Tensor,
    input_tokens: list[str],
    out_path: Path,
    max_tokens: int = 80,
    title: str | None = None,
) -> None:
    """Plot encoder attention rollout showing dialogue structure (token-to-token)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    rollout = compute_encoder_attention_rollout(encoder_attn_lhs)
    s = min(rollout.shape[0], max_tokens)
    mat = rollout[:s, :s]

    xlabels = input_tokens[:s]
    ylabels = input_tokens[:s]

    plt.figure(figsize=(max(10, s * 0.2), max(8, s * 0.2)))
    sns.heatmap(mat, cmap="viridis", cbar=True)
    plt.xticks(np.arange(s) + 0.5, xlabels, rotation=90, fontsize=6)
    plt.yticks(np.arange(s) + 0.5, ylabels, rotation=0, fontsize=6)
    plt.xlabel("Input tokens (dialogue)")
    plt.ylabel("Input tokens (dialogue)")
    plt.title(title or "Encoder attention rollout (dialogue structure)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_attention_arrays(
    out_npz: Path,
    encoder_attn_lhs: torch.Tensor,
    decoder_attn_lhs: torch.Tensor,
    cross_attn_lhs: torch.Tensor,
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        encoder=encoder_attn_lhs.cpu().numpy().astype(np.float16),
        decoder=decoder_attn_lhs.cpu().numpy().astype(np.float16),
        cross=cross_attn_lhs.cpu().numpy().astype(np.float16),
    )


def build_test_loader(tokenizer, n_samples: int, batch_size: int) -> DataLoader:
    ds = load_dataset("knkarthick/samsum")["test"].select(range(n_samples))

    def map_fn(batch: dict[str, list[str]], indices: list[int]) -> dict[str, Any]:
        prefixed = [TASK_PREFIX + d for d in batch["dialogue"]]
        enc = tokenizer(
            prefixed,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "dialogue": batch["dialogue"],
            "summary": batch["summary"],
            "dialogue_id": indices,
            "full_input": prefixed,
        }

    mapped = ds.map(map_fn, batched=True, with_indices=True)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    def _collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        text_fields = {
            "dialogue": [f["dialogue"] for f in features],
            "summary": [f["summary"] for f in features],
            "dialogue_id": [f["dialogue_id"] for f in features],
            "full_input": [f["full_input"] for f in features],
        }
        core = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = collator(core)
        batch.update(text_fields)
        return batch

    return DataLoader(mapped, batch_size=batch_size, shuffle=False, collate_fn=_collate)


def load_lora_or_merged(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # If model_path is adapter, PEFT loader is needed. If merged, base loader works.
    try:
        from peft import PeftModel

        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model = PeftModel.from_pretrained(base, str(model_path))
        model = model.merge_and_unload()
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))

    model.eval()
    model.config.output_attentions = True
    return tokenizer, model


def extract_attentions_and_analyze(
    model_path: Path,
    output_dir: Path,
    n_samples: int = 100,
    batch_size: int = 4,
    save_heatmaps: bool = False,
    save_rollout: bool = False,
) -> list[dict[str, Any]]:
    device = get_device()
    tokenizer, model = load_lora_or_merged(model_path)
    model = model.to(device)

    test_loader = build_test_loader(tokenizer, n_samples=n_samples, batch_size=batch_size)

    sample_outputs: list[dict[str, Any]] = []
    attn_dir = output_dir / "attention_tensors"
    plot_dir = output_dir / "heatmaps"
    rollout_dir = output_dir / "rollout_heatmaps" if save_rollout else None
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=4,
                return_dict_in_generate=False,
            )

            gen_attn_mask = generated.ne(tokenizer.pad_token_id).long()

            fwd = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated,
                output_attentions=True,
                return_dict=True,
            )

        enc_all = stack_attention_tuple(fwd.encoder_attentions)  # [L,B,H,S,S]
        dec_all = stack_attention_tuple(fwd.decoder_attentions)  # [L,B,H,T,T]
        cross_all = stack_attention_tuple(fwd.cross_attentions)  # [L,B,H,T,S]

        batch_size_now = input_ids.size(0)

        for b in range(batch_size_now):
            src_len = int(attention_mask[b].sum().item())
            tgt_len = int(gen_attn_mask[b].sum().item())

            enc_b = enc_all[:, b, :, :src_len, :src_len].detach().cpu()
            dec_b = dec_all[:, b, :, :tgt_len, :tgt_len].detach().cpu()
            cross_b = cross_all[:, b, :, :tgt_len, :src_len].detach().cpu()

            dialogue = batch["dialogue"][b]
            summary_ref = batch["summary"][b]
            dialogue_id = int(batch["dialogue_id"][b])
            full_input = batch["full_input"][b]

            turns = parse_dialogue_turns(dialogue)
            token_to_turn, input_tokens, token_to_speaker = token_to_turn_mapping(tokenizer, full_input, turns)

            gen_ids = generated[b, :tgt_len].detach().cpu().tolist()
            summary_tokens = tokenizer.convert_ids_to_tokens(gen_ids)

            speaker_scores = compute_speaker_attention(cross_b, token_to_speaker)
            key_info = extract_key_moments(
                cross_attention_lhs=cross_b,
                input_tokens=input_tokens,
                token_to_turn=token_to_turn,
                turns=turns,
                top_k_tokens=12,
                top_k_turns=3,
            )
            entropy = compute_attention_entropy(speaker_scores)

            save_attention_arrays(attn_dir / f"{dialogue_id}.npz", enc_b, dec_b, cross_b)

            if save_heatmaps:
                plot_attention_heatmap(
                    cross_attention_lhs=cross_b,
                    input_tokens=input_tokens,
                    summary_tokens=summary_tokens,
                    out_path=plot_dir / f"{dialogue_id}.png",
                    title=f"dialogue_id={dialogue_id}",
                )
            if save_rollout and rollout_dir:
                plot_encoder_rollout_heatmap(
                    encoder_attn_lhs=enc_b,
                    input_tokens=input_tokens,
                    out_path=rollout_dir / f"{dialogue_id}.png",
                    title=f"dialogue_id={dialogue_id}",
                )

            final_record = {
                "dialogue_id": dialogue_id,
                "top_3_turns": key_info["top_turns"],
                "speaker_scores": speaker_scores,
                "key_tokens": key_info["key_tokens"],
                "attention_entropy": entropy,
            }
            sample_outputs.append(final_record)

    out_json = output_dir / f"task1_attention_report_{n_samples}.json"
    with open(out_json, "w") as f:
        json.dump(sample_outputs, f, indent=2)

    return sample_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 1 implementation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train T5-small LoRA for 5 epochs")
    p_train.add_argument("--output_dir", type=Path, default=Path("models/best/t5-small_lora_task1"))
    p_train.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_analyze = sub.add_parser("analyze", help="Extract attentions and analyze 100 samples")
    p_analyze.add_argument("--model_path", type=Path, required=True)
    p_analyze.add_argument("--output_dir", type=Path, default=Path("results/metrics/task1_attention"))
    p_analyze.add_argument("--n_samples", type=int, default=100)
    p_analyze.add_argument("--batch_size", type=int, default=4)
    p_analyze.add_argument("--save_heatmaps", action="store_true")
    p_analyze.add_argument("--save_rollout", action="store_true",
                           help="Also save encoder attention rollout heatmaps (dialogue structure)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
        summary = train_t5_lora(
            output_dir=output_dir,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "analyze":
        model_path = args.model_path if args.model_path.is_absolute() else (PROJECT_ROOT / args.model_path)
        output_dir = args.output_dir if args.output_dir.is_absolute() else (PROJECT_ROOT / args.output_dir)
        rows = extract_attentions_and_analyze(
            model_path=model_path,
            output_dir=output_dir,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            save_heatmaps=args.save_heatmaps,
            save_rollout=args.save_rollout,
        )
        print(json.dumps({"n_records": len(rows), "output_dir": str(output_dir)}, indent=2))
        return


if __name__ == "__main__":
    main()

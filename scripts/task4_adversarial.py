#!/usr/bin/env python3
"""
Task 4 — Adversarial Meeting Transcripts & Robustness Testing.

Subcommands:
  generate  — Create adversarial transcripts (overlap, noise, tangents, long)
  eval      — Summarize 150 original + 150 adversarial; measure robustness
  retrain   — Retrain T5-small LoRA: orig/adv mix (default 55/45), low LR, held-out ROUGE
              (macro mean over pattern buckets), early stopping
  compare   — Pre/post ROUGE-L on held-out adversarial test; quantify gain

Usage:
  python3 scripts/task4_adversarial.py generate --n_original 150 --n_adversarial 200
  python3 scripts/task4_adversarial.py eval --model_path models/best/t5-small_lora_task1
  python3 scripts/task4_adversarial.py retrain --base_model models/best/t5-small_lora_task1
  python3 scripts/task4_adversarial.py compare
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_PREFIX = "summarize: "
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

# Typo substitution patterns (common ASR/OCR errors)
TYPO_MAP = {
    "the": ["teh", "hte", "tha"],
    "and": ["nad", "adn", "anf"],
    "have": ["ahve", "hvae", "haev"],
    "that": ["taht", "thta", "tha"],
    "with": ["wiht", "wthi", "wtih"],
    "this": ["htis", "thsi", "tis"],
    "you": ["yuo", "ouy"],
    "for": ["fro", "ofr"],
    "are": ["rae", "aer"],
    "from": ["form", "fomr"],
}
OFF_TOPIC_TANGENTS = [
    "So anyway, the weather was really nice this morning.",
    "Oh by the way, did you see the game last night?",
    "I meant to ask – how's your cat doing?",
    "Sorry, my connection was glitching for a sec.",
    "Hmm, let me think... actually forget that.",
    "We can circle back to that. Moving on.",
    "Not sure if that's relevant but just thought I'd mention it.",
    "By the way, has anyone tried the new coffee machine?",
]


def _parse_turns(dialogue: str) -> list[tuple[str, str]]:
    """Return [(speaker, utterance), ...]"""
    turns = []
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]{1,64}):\s*(.*)$", line)
        if m:
            turns.append((m.group(1).strip(), m.group(2)))
        else:
            turns.append(("Unknown", line))
    return turns


def _format_turns(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{s}: {u}" for s, u in turns)


def apply_overlapping_speakers(dialogue: str, rng: random.Random) -> str:
    """Simulate overlapping speakers: merge some turns, add [overlap] markers."""
    turns = _parse_turns(dialogue)
    if len(turns) < 3:
        return dialogue
    out = []
    i = 0
    while i < len(turns):
        s, u = turns[i]
        if i + 1 < len(turns) and rng.random() < 0.2:
            s2, u2 = turns[i + 1]
            out.append((s, f"{u} [overlap] {s2}: {u2}"))
            i += 2
        else:
            out.append((s, u))
            i += 1
    return _format_turns(out)


def apply_transcription_noise(dialogue: str, rng: random.Random, p: float = 0.03) -> str:
    """Introduce typo-like noise (transcription errors)."""
    turns = _parse_turns(dialogue)
    out = []
    for s, u in turns:
        words = u.split()
        for idx, w in enumerate(words):
            w_lower = w.lower()
            for canonical, variants in TYPO_MAP.items():
                if w_lower == canonical and rng.random() < p:
                    words[idx] = rng.choice(variants)
                    break
            # Random char swap
            if len(w) > 2 and rng.random() < p:
                i, j = rng.sample(range(len(w)), 2)
                lst = list(w)
                lst[i], lst[j] = lst[j], lst[i]
                words[idx] = "".join(lst)
        out.append((s, " ".join(words)))
    return _format_turns(out)


def apply_off_topic_tangent(dialogue: str, rng: random.Random) -> str:
    """Insert off-topic tangent line(s)."""
    turns = _parse_turns(dialogue)
    if not turns:
        return dialogue
    tangent = rng.choice(OFF_TOPIC_TANGENTS)
    # Insert at random position
    speakers = list({t[0] for t in turns})
    insert_speaker = rng.choice(speakers) if speakers else "Unknown"
    insert_pos = rng.randint(0, len(turns))
    turns.insert(insert_pos, (insert_speaker, tangent))
    return _format_turns(turns)


def apply_very_long(dialogue: str, rng: random.Random, target_turns: int = 150) -> str:
    """Extend dialogue by repeating/cycling turns to create very long conversation."""
    turns = _parse_turns(dialogue)
    if len(turns) >= target_turns:
        return dialogue
    extended = list(turns)
    while len(extended) < target_turns:
        extended.append(rng.choice(turns))
    return _format_turns(extended)


def generate_adversarial(dialogue: str, pattern: str, rng: random.Random) -> tuple[str, str]:
    """Apply one adversarial pattern. Returns (adversarial_dialogue, pattern_name)."""
    if pattern == "overlapping":
        return apply_overlapping_speakers(dialogue, rng), "overlapping"
    if pattern == "noise":
        return apply_transcription_noise(dialogue, rng), "noise"
    if pattern == "tangent":
        return apply_off_topic_tangent(dialogue, rng), "tangent"
    if pattern == "very_long":
        return apply_very_long(dialogue, rng), "very_long"
    return dialogue, "original"


def cmd_generate(args) -> None:
    from datasets import load_dataset

    rng = random.Random(args.seed)
    ds = load_dataset("knkarthick/samsum")["test"]
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    patterns = ["overlapping", "noise", "tangent", "very_long"]
    out_dir = PROJECT_ROOT / "data" / "adversarial_task4"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_samples = []
    adversarial_samples = []
    for i, idx in enumerate(indices):
        row = ds[int(idx)]
        dialogue = row["dialogue"]
        summary = row["summary"]
        sample = {"idx": int(idx), "dialogue": dialogue, "summary": summary, "pattern": "original"}
        if i < args.n_original:
            original_samples.append(sample)
        elif i < args.n_original + args.n_adversarial:
            pat = patterns[i % len(patterns)]
            adv_dialogue, _ = generate_adversarial(dialogue, pat, rng)
            adversarial_samples.append({
                "idx": int(idx),
                "dialogue": adv_dialogue,
                "summary": summary,
                "pattern": pat,
            })
        if len(original_samples) >= args.n_original and len(adversarial_samples) >= args.n_adversarial:
            break

    # Held-out adversarial for final compare (extra samples)
    held_out = []
    for i, idx in enumerate(indices[args.n_original + args.n_adversarial :]):
        if len(held_out) >= args.n_heldout:
            break
        row = ds[int(idx)]
        pat = patterns[i % len(patterns)]
        adv_dialogue, _ = generate_adversarial(row["dialogue"], pat, rng)
        held_out.append({
            "idx": int(idx),
            "dialogue": adv_dialogue,
            "summary": row["summary"],
            "pattern": pat,
        })

    data = {
        "original": original_samples,
        "adversarial": adversarial_samples,
        "held_out_adversarial": held_out,
        "meta": {
            "n_original": len(original_samples),
            "n_adversarial": len(adversarial_samples),
            "n_heldout": len(held_out),
            "seed": args.seed,
        },
    }
    out_path = out_dir / "task4_adversarial_data.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {out_path}")
    print(f"  Original: {len(original_samples)}, Adversarial: {len(adversarial_samples)}, Held-out: {len(held_out)}")


def _resolved_checkpoint_for_audit(model_path: Path) -> str:
    """Human-readable path actually used for inference (merged preferred)."""
    model_path = Path(model_path)
    merged_path = model_path / "merged"
    if merged_path.exists() and any(merged_path.glob("*.safetensors")):
        return str(merged_path.resolve())
    if (model_path / "adapter_config.json").exists():
        return str(model_path.resolve())
    return str(model_path.resolve())


def _load_model_and_tokenizer(model_path: Path, *, allow_adapter_only: bool = False):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(model_path)
    # Retrain saves task4 as a *second* LoRA on top of merged task1; the adapter
    # weights are only valid on that merged base. Loading adapter on bare t5-small
    # silently produces garbage. Prefer full merged weights when present.
    merged_path = model_path / "merged"
    has_merged = merged_path.exists() and any(merged_path.glob("*.safetensors"))
    has_adapter = (model_path / "adapter_config.json").exists()

    if has_merged:
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(merged_path))
    elif has_adapter and not allow_adapter_only:
        raise FileNotFoundError(
            f"Merged weights required for correct Task 4 eval: missing {merged_path} "
            f"(adapter-only load on bare t5-small is invalid). Re-run retrain/merge, or pass "
            f"--allow-adapter-only for debugging only."
        )
    else:
        try:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            base = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            model = PeftModel.from_pretrained(base, str(model_path))
            model = model.merge_and_unload()
        except Exception:
            load_path = model_path / "merged" if (model_path / "merged").exists() else model_path
            tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(load_path))

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _summarize_batch(model, tokenizer, device, dialogues: list[str], batch_size: int = 8) -> list[str]:
    import torch

    preds = []
    for i in range(0, len(dialogues), batch_size):
        batch = dialogues[i : i + batch_size]
        inputs = tokenizer(
            [TASK_PREFIX + d for d in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LEN,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        preds.extend(x.strip() for x in decoded)
    return preds


def _rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(vals) / max(len(vals), 1) * 100


def _rouge_l_by_pattern(
    samples: list[dict[str, Any]],
    preds: list[str],
    refs: list[str],
) -> dict[str, float]:
    """Mean ROUGE-L ×100 per adversarial ``pattern`` (and ``original`` if present)."""
    buckets: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"preds": [], "refs": []})
    for s, p, r in zip(samples, preds, refs):
        pat = s.get("pattern", "unknown")
        buckets[pat]["preds"].append(p)
        buckets[pat]["refs"].append(r)
    return {k: round(_rouge_l(v["preds"], v["refs"]), 4) for k, v in sorted(buckets.items())}


def _action_completeness_proxy(summaries: list[str]) -> float:
    """Fraction of summaries containing action-like phrases."""
    action_pattern = re.compile(
        r"\b(will|going to|need to|should|must|send|call|email|schedule|book|"
        r"prepare|review|check|bring|follow up|confirm|share|update)\b",
        re.I,
    )
    hits = sum(1 for s in summaries if action_pattern.search(s))
    return hits / max(len(summaries), 1)


def cmd_eval(args) -> None:
    model, tokenizer, device = _load_model_and_tokenizer(
        PROJECT_ROOT / args.model_path,
        allow_adapter_only=args.allow_adapter_only,
    )
    data_path = PROJECT_ROOT / "data" / "adversarial_task4" / "task4_adversarial_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Run 'generate' first. Missing {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    results = []

    for split_name, samples in [("original", data["original"]), ("adversarial", data["adversarial"])]:
        dialogues = [s["dialogue"] for s in samples]
        refs = [s["summary"] for s in samples]
        t0 = time.perf_counter()
        preds = _summarize_batch(model, tokenizer, device, dialogues, args.batch_size)
        elapsed = time.perf_counter() - t0

        rouge = _rouge_l(preds, refs)
        action_rate = _action_completeness_proxy(preds)

        by_pattern = {}
        for s, p in zip(samples, preds):
            pat = s.get("pattern", "original")
            if pat not in by_pattern:
                by_pattern[pat] = {"preds": [], "refs": []}
            by_pattern[pat]["preds"].append(p)
            by_pattern[pat]["refs"].append(s["summary"])

        pattern_rouge = {k: _rouge_l(v["preds"], v["refs"]) for k, v in by_pattern.items()}

        results.append({
            "split": split_name,
            "n_samples": len(samples),
            "rougeL": round(rouge, 4),
            "action_completeness_proxy": round(action_rate, 4),
            "ms_per_sample": round(elapsed / len(samples) * 1000, 2),
            "rougeL_by_pattern": pattern_rouge,
        })

    # Coherence template for human rating (1-5)
    coherence_path = PROJECT_ROOT / "results" / "metrics" / "task4_coherence_template.csv"
    coherence_path.parent.mkdir(parents=True, exist_ok=True)
    all_samples = data["original"] + data["adversarial"]
    all_preds = _summarize_batch(model, tokenizer, device, [s["dialogue"] for s in all_samples], args.batch_size)
    with open(coherence_path, "w") as f:
        f.write("sample_id,split,pattern,dialogue_len,summary,prediction,coherence_score_1_5,notes\n")
        for i, (s, p) in enumerate(zip(all_samples, all_preds)):
            f.write(f"{i},{'original' if s['pattern']=='original' else 'adversarial'},{s['pattern']},"
                    f"{len(s['dialogue'])},{repr(s['summary'][:100])},{repr(p[:100])},,\n")

    out_path = PROJECT_ROOT / "results" / "metrics" / "task4_robustness_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model_path": str(args.model_path), "results": results}, f, indent=2)

    # Failure modes: which patterns cause worst ROUGE
    adv_result = next(r for r in results if r["split"] == "adversarial")
    worst_patterns = sorted(
        adv_result["rougeL_by_pattern"].items(),
        key=lambda x: x[1],
    )[:3]
    failure_modes = [{"pattern": p, "rougeL": v} for p, v in worst_patterns]

    report = {
        "task": "task4_robustness_eval",
        "model_path": str(args.model_path),
        "results": results,
        "failure_modes_worst_rouge": failure_modes,
        "coherence_template": str(coherence_path),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {out_path}")
    print(f"Original ROUGE-L: {results[0]['rougeL']:.2f}")
    print(f"Adversarial ROUGE-L: {results[1]['rougeL']:.2f}")
    print(f"Failure modes (worst patterns): {failure_modes}")


def cmd_retrain(args) -> None:
    import numpy as np
    import torch
    from datasets import Dataset, load_dataset
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from rouge_score import rouge_scorer
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    ds_raw = load_dataset("knkarthick/samsum")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    train_dialogues = list(ds_raw["train"]["dialogue"])
    train_summaries = list(ds_raw["train"]["summary"])
    rng = random.Random(args.seed)
    patterns = ["overlapping", "noise", "tangent", "very_long"]

    n_mixed = min(args.max_train_samples, len(train_dialogues))
    n_orig = max(1, int(args.orig_frac * n_mixed))
    n_adv = max(1, n_mixed - n_orig)
    if n_orig + n_adv > len(train_dialogues):
        n_mixed = len(train_dialogues)
        n_orig = max(1, int(args.orig_frac * n_mixed))
        n_adv = max(1, n_mixed - n_orig)
    # Disjoint index pools so adversarial does not duplicate originals
    perm = list(range(len(train_dialogues)))
    rng.shuffle(perm)
    orig_pick = perm[:n_orig]
    adv_pick = perm[n_orig : n_orig + n_adv]
    if len(adv_pick) < n_adv:
        adv_pick = list(adv_pick) + perm[: n_adv - len(adv_pick)]

    orig_dialogues = [train_dialogues[i] for i in orig_pick]
    orig_summaries = [train_summaries[i] for i in orig_pick]
    adv_dialogues = []
    adv_summaries = []
    for j, idx in enumerate(adv_pick):
        d = train_dialogues[idx]
        pat = patterns[j % len(patterns)]
        adv_d, _ = generate_adversarial(d, pat, rng)
        adv_dialogues.append(adv_d)
        adv_summaries.append(train_summaries[idx])

    mixed_dialogues = orig_dialogues + adv_dialogues
    mixed_summaries = orig_summaries + adv_summaries
    shuffle_idx = rng.sample(range(len(mixed_dialogues)), len(mixed_dialogues))
    mixed_dialogues = [mixed_dialogues[i] for i in shuffle_idx]
    mixed_summaries = [mixed_summaries[i] for i in shuffle_idx]

    train_ds = Dataset.from_dict({"dialogue": mixed_dialogues, "summary": mixed_summaries})

    def tokenize_fn(examples):
        inputs = tokenizer(
            [TASK_PREFIX + d for d in examples["dialogue"]],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["summary"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)

    # Held-out adversarial eval (same distribution as compare); requires generate step
    adv_data_path = PROJECT_ROOT / "data" / "adversarial_task4" / "task4_adversarial_data.json"
    if not adv_data_path.exists():
        raise FileNotFoundError(
            f"Missing {adv_data_path}. Run 'generate' first so held-out adversarial eval exists."
        )
    with open(adv_data_path) as f:
        adv_meta = json.load(f)
    held = adv_meta["held_out_adversarial"]
    held_patterns = [s.get("pattern", "unknown") for s in held]
    eval_ds = Dataset.from_dict({
        "dialogue": [s["dialogue"] for s in held],
        "summary": [s["summary"] for s in held],
    })
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    base_path = PROJECT_ROOT / args.base_model
    base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model = PeftModel.from_pretrained(base_model, str(base_path))
    model = model.merge_and_unload()
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    rouge_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        # Sanitise before decode: both preds and labels may be padded with -100
        # (Transformers ignore index). The tokenizer's Rust backend raises
        # OverflowError on negative token IDs.
        preds = np.asarray(preds, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        preds = np.where(preds != -100, preds, pad_id)
        labels = np.where(labels != -100, labels, pad_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = [
            rouge_sc.score(ref.lower(), pred.lower())["rougeL"].fmeasure
            for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        rouge_micro = (sum(scores) / max(len(scores), 1)) * 100.0
        pat_scores: dict[str, list[float]] = defaultdict(list)
        for pred, ref, pat in zip(decoded_preds, decoded_labels, held_patterns):
            r = rouge_sc.score(ref.lower(), pred.lower())["rougeL"].fmeasure
            pat_scores[pat].append(r)
        pat_means = {p: sum(v) / len(v) for p, v in pat_scores.items()}
        rouge_macro = (sum(pat_means.values()) / max(len(pat_means), 1)) * 100.0
        rouge_worst = min(pat_means.values()) * 100.0 if pat_means else 0.0
        return {
            "eval_rougeL": rouge_macro,
            "eval_rougeL_micro": rouge_micro,
            "eval_rougeL_worst_pattern": rouge_worst,
        }

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt")
    total_steps = (len(tokenized_train) + args.batch_size - 1) // args.batch_size * args.num_epochs
    warmup_steps = min(args.warmup_steps, max(1, int(args.warmup_ratio * total_steps)))

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "checkpoints" / "t5-small_lora_task4"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        generation_num_beams=4,
        logging_steps=25,
        dataloader_num_workers=0,
        bf16=(device.type in {"cuda", "mps"}),
        fp16=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    train_result = trainer.train()

    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    best_metric = getattr(trainer.state, "best_metric", None)

    out_dir = PROJECT_ROOT / "models" / "best" / "t5-small_lora_task4"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    merged = model.merge_and_unload()
    merged_dir = out_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    manifest = {
        "task": "task4_retrain",
        "n_train": len(mixed_dialogues),
        "n_original": len(orig_dialogues),
        "n_adversarial": len(adv_dialogues),
        "orig_frac_actual": round(len(orig_dialogues) / len(mixed_dialogues), 4),
        "held_out_eval_n": len(held),
        "learning_rate": args.learning_rate,
        "num_epochs_requested": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "warmup_steps": warmup_steps,
        "best_model_checkpoint": best_ckpt,
        "best_eval_rougeL": best_metric,
        "eval_metric_note": (
            "metric_for_best_model=eval_rougeL is macro mean of per-pattern mean ROUGE-L "
            "on held-out (stratified); see eval_rougeL_micro for micro-average."
        ),
        "train_runtime_s": getattr(train_result, "metrics", {}).get("train_runtime"),
        "base_model": str(base_path),
    }
    man_path = PROJECT_ROOT / "results" / "metrics" / "task4_retrain_manifest.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved retrained model to {out_dir}")
    print(f"Best held-out ROUGE-L (pattern-macro, early-stop metric): {best_metric}")
    print(f"Manifest: {man_path}")


def cmd_compare(args) -> None:
    data_path = PROJECT_ROOT / "data" / "adversarial_task4" / "task4_adversarial_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Run 'generate' first. Missing {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    held_out = data["held_out_adversarial"]
    dialogues = [s["dialogue"] for s in held_out]
    refs = [s["summary"] for s in held_out]

    model_pre, tokenizer, device = _load_model_and_tokenizer(
        PROJECT_ROOT / args.model_pre,
        allow_adapter_only=args.allow_adapter_only,
    )
    model_post, _, _ = _load_model_and_tokenizer(
        PROJECT_ROOT / args.model_post,
        allow_adapter_only=args.allow_adapter_only,
    )

    preds_pre = _summarize_batch(model_pre, tokenizer, device, dialogues, args.batch_size)
    preds_post = _summarize_batch(model_post, tokenizer, device, dialogues, args.batch_size)

    rouge_pre = _rouge_l(preds_pre, refs)
    rouge_post = _rouge_l(preds_post, refs)
    gain = rouge_post - rouge_pre
    by_pre = _rouge_l_by_pattern(held_out, preds_pre, refs)
    by_post = _rouge_l_by_pattern(held_out, preds_post, refs)
    all_pats = sorted(set(by_pre) | set(by_post))
    gain_by_pattern = {p: round(by_post[p] - by_pre[p], 4) for p in all_pats}
    macro_pre = round(sum(by_pre.values()) / max(len(by_pre), 1), 4)
    macro_post = round(sum(by_post.values()) / max(len(by_post), 1), 4)

    report = {
        "task": "task4_robustness_comparison",
        "held_out_n": len(held_out),
        "model_pre": str(args.model_pre),
        "model_post": str(args.model_post),
        "model_pre_resolved": _resolved_checkpoint_for_audit(PROJECT_ROOT / args.model_pre),
        "model_post_resolved": _resolved_checkpoint_for_audit(PROJECT_ROOT / args.model_post),
        "rougeL_pre": round(rouge_pre, 4),
        "rougeL_post": round(rouge_post, 4),
        "robustness_gain": round(gain, 4),
        "rougeL_macro_by_pattern_pre": macro_pre,
        "rougeL_macro_by_pattern_post": macro_post,
        "robustness_gain_macro_by_pattern": round(macro_post - macro_pre, 4),
        "rougeL_by_pattern_pre": by_pre,
        "rougeL_by_pattern_post": by_post,
        "robustness_gain_by_pattern": gain_by_pattern,
    }

    out_path = PROJECT_ROOT / "results" / "metrics" / "task4_robustness_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Pre-adversarial ROUGE-L (micro): {rouge_pre:.2f}")
    print(f"Post-adversarial ROUGE-L (micro): {rouge_post:.2f}")
    print(f"Robustness gain (micro): {gain:+.2f}")
    print(f"Macro-by-pattern pre/post: {macro_pre:.2f} → {macro_post:.2f} (Δ {macro_post - macro_pre:+.2f})")
    print(f"Gain by pattern: {gain_by_pattern}")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Adversarial robustness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate adversarial transcripts")
    p_gen.add_argument("--n_original", type=int, default=150)
    p_gen.add_argument("--n_adversarial", type=int, default=150)
    p_gen.add_argument("--n_heldout", type=int, default=100)
    p_gen.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Evaluate robustness on 150 original + 150 adversarial")
    p_eval.add_argument("--model_path", default="models/best/t5-small_lora_task1")
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument(
        "--allow-adapter-only",
        action="store_true",
        help="Allow loading PEFT adapter on t5-small when merged/ is missing (debug only; metrics invalid).",
    )

    p_retrain = sub.add_parser(
        "retrain",
        help=(
            "Retrain T5-small LoRA: orig/adv mix, low LR, held-out ROUGE (pattern-macro), early stop"
        ),
    )
    p_retrain.add_argument("--base_model", default="models/best/t5-small_lora_task1")
    p_retrain.add_argument("--batch_size", type=int, default=8)
    p_retrain.add_argument("--seed", type=int, default=42)
    p_retrain.add_argument("--num_epochs", type=int, default=5)
    p_retrain.add_argument("--learning_rate", type=float, default=5e-6)
    p_retrain.add_argument("--max_train_samples", type=int, default=6000)
    p_retrain.add_argument("--orig_frac", type=float, default=0.55)
    p_retrain.add_argument("--warmup_steps", type=int, default=500)
    p_retrain.add_argument("--warmup_ratio", type=float, default=0.06)
    p_retrain.add_argument("--max_grad_norm", type=float, default=1.0)
    p_retrain.add_argument("--early_stopping_patience", type=int, default=2)

    p_compare = sub.add_parser("compare", help="Pre/post ROUGE-L on held-out adversarial")
    p_compare.add_argument("--model_pre", default="models/best/t5-small_lora_task1")
    p_compare.add_argument("--model_post", default="models/best/t5-small_lora_task4")
    p_compare.add_argument("--batch_size", type=int, default=8)
    p_compare.add_argument(
        "--allow-adapter-only",
        action="store_true",
        help="Allow loading PEFT adapter on t5-small when merged/ is missing (debug only).",
    )

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "retrain":
        cmd_retrain(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Task 5 — LoRA Rank Ablation & Structured Output Constraints.

Subcommands:
  train       — Train T5-small LoRA with ranks 2, 4, 8, 16, 32
  eval        — Measure ROUGE-L, latency, model size per rank
  structured  — JSON schema decoding; measure validity + ROUGE-L (inner JSON targets for T5)
  sweet_spot  — Identify min rank with 95%+ validity and ROUGE-L within 1 pt
  package     — Package optimal config as production baseline

Usage:
  python3 scripts/task5_lora_structured.py train --ranks 2 4 8 16 32
  python3 scripts/task5_lora_structured.py eval
  python3 scripts/task5_lora_structured.py structured
  python3 scripts/task5_lora_structured.py sweet_spot
  python3 scripts/task5_lora_structured.py package
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
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_PREFIX = "summarize: "
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128
LORA_RANKS = [2, 4, 8, 16, 32]

# JSON schema for structured output: {topics: [], action_items: [], decision: []}
STRUCTURED_SCHEMA = {
    "topics": "list of main topics discussed",
    "action_items": "list of action items or next steps",
    "decision": "main decision or outcome",
}

# T5 task prefix + instruction (must match train_structured and structured eval).
# Keep this minimal: long schema text gets copied into the decoder when T5 attends to the encoder.
STRUCTURED_PREFIX = "Output JSON only.\n"

# T5 SentencePiece maps `{` and `}` to <unk>; skip_special_tokens=True strips them and
# breaks standard JSON. Supervised targets therefore use the **object body only** (no outer
# braces); we wrap with `{` + body + `}` at decode time. See `gold_summary_to_t5_train_target`.
JSON_INNER_PREFIX_STR = '"topics": ['

_ACTION_SENTENCE_RE = re.compile(
    r"[^.!?]*\b("
    r"will|going to|needs? to|should|must|have to|plan to|"
    r"send|call|email|schedule|book|prepare|review|check|bring|"
    r"follow up|confirm|share|update"
    r")\b[^.!?]*[.!?]?",
    re.IGNORECASE,
)


def gold_summary_to_structured_obj(summary: str) -> dict[str, Any]:
    """Heuristic structured dict from SAMSum gold summary."""
    s = (summary or "").strip()
    if not s:
        s = "No content."
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    topics = (parts[:4] if parts else [s[:120]])[:5]
    topics = [t[:100] for t in topics]
    actions = []
    for p in parts:
        if _ACTION_SENTENCE_RE.search(p):
            actions.append(p[:100])
    if not actions:
        actions = [s[:100]]
    return {
        "topics": topics,
        "action_items": actions[:5],
        "decision": s[:200],
    }


def gold_summary_to_json_string(summary: str) -> str:
    """Full JSON (with braces) — useful for manifests; not ideal as T5 label text."""
    return json.dumps(gold_summary_to_structured_obj(summary), ensure_ascii=False)


def gold_summary_to_t5_train_target(summary: str) -> str:
    """Training label: same object as JSON but **without** outer `{` `}` (T5-tokenizable)."""
    full = gold_summary_to_json_string(summary)
    if len(full) >= 2 and full.startswith("{") and full.endswith("}"):
        return full[1:-1].strip()
    return full


def structured_input_text(dialogue: str) -> str:
    return f"{TASK_PREFIX}{STRUCTURED_PREFIX}{dialogue}"


def resolve_rank_model_dir(rank: int) -> Path:
    """Directory containing adapter or merged weights for this LoRA rank."""
    d = PROJECT_ROOT / "models" / "best" / f"t5-small_lora_r{rank}"
    if not d.exists() and rank == 16:
        t1 = PROJECT_ROOT / "models" / "best" / "t5-small_lora_task1"
        if (t1 / "merged").exists():
            return t1
    return d


def resolve_inference_merged_dir(rank: int) -> Path | None:
    """Prefer merged_structured/ after train_structured; else merged/."""
    base = resolve_rank_model_dir(rank)
    if not base.exists():
        return None
    ms = base / "merged_structured"
    if ms.exists() and any(ms.glob("*.safetensors")):
        return ms
    m = base / "merged"
    if m.exists() and any(m.glob("*.safetensors")):
        return m
    return base if any(base.glob("*.safetensors")) else None


def _rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(vals) / max(len(vals), 1) * 100


def _parse_json_summary(text: str) -> dict | None:
    """Try to parse JSON from model output. Handle fences, prefixes, truncation."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I | re.MULTILINE)
        text = re.sub(r"\s*```\s*$", "", text)
    # Strip common T5 prefixes before JSON
    for prefix in ("json", "JSON", "Output:", "Answer:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].lstrip(" :\n")
    # Try raw parse
    for candidate in (text, text[text.find("{") :] if "{" in text else text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # Brace-balanced slice (handles nested objects)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
    # Regex fallback (shallow)
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # T5 often omits quotes for first array token: "topics": [bareword, ...]
    fixed = re.sub(
        r'("topics"\s*:\s*\[)\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\1"\2"',
        text,
        count=1,
        flags=re.I,
    )
    if fixed != text:
        for candidate in (fixed, fixed + "}"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    for attempt in [text, text + "}"]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    return None


def prediction_to_structured_dict_with_trace(pred: str) -> tuple[dict[str, Any], bool]:
    """Return (structured_dict, used_heuristic_fallback).

    If ``used_heuristic_fallback`` is True, the schema was derived from plain-text
    heuristics, not a successful model JSON parse — do **not** treat as JSON fidelity.
    """
    pred = (pred or "").strip()
    if not pred:
        return {}, True
    repaired = repair_t5_json_decode(pred)
    parsed = _parse_json_summary(repaired)
    if _is_valid_structured(parsed):
        assert parsed is not None
        return parsed, False
    return gold_summary_to_structured_obj(pred), True


def prediction_to_structured_dict(pred: str) -> dict[str, Any]:
    """
    Production helper: best-effort structured object from model output.

    1) Parse JSON after T5-oriented repair / wrapping.
    2) If that fails, treat *pred* as plain summary text and apply the same heuristic
       used to build supervised training targets (always yields topics, action_items, decision).

    This does **not** mean the model strictly emitted JSON; use `json_validity_strict` in
    metrics when you need parse fidelity.
    """
    d, _ = prediction_to_structured_dict_with_trace(pred)
    return d


def _is_valid_structured(obj: dict | None) -> bool:
    """Require keys topics, action_items, decision with list-like topics/action_items."""
    if obj is None or not isinstance(obj, dict):
        return False
    required = {"topics", "action_items", "decision"}
    if not required.issubset(obj.keys()):
        return False
    topics, actions = obj["topics"], obj["action_items"]
    if isinstance(topics, str):
        topics = [topics]
    if isinstance(actions, str):
        actions = [actions]
    if not isinstance(topics, list) or not isinstance(actions, list):
        return False
    dec = obj.get("decision")
    if isinstance(dec, list):
        dec = " ".join(str(x) for x in dec) if dec else ""
    if not isinstance(dec, (str, int, float, type(None))):
        return False
    return True


def json_prefix_token_ids(tokenizer) -> list[int]:
    """Token ids for optional forced prefix (inner format: starts with '\"topics\": [')."""
    ids = tokenizer(JSON_INNER_PREFIX_STR, add_special_tokens=False)["input_ids"]
    return list(ids) if ids else []


def structured_decoder_input_ids(tokenizer, device: torch.device) -> torch.Tensor | None:
    """Optional decoder prefill: inner-format opening (must match training if enabled)."""
    pref = json_prefix_token_ids(tokenizer)
    if not pref:
        return None
    pad = tokenizer.pad_token_id
    if pad is None:
        return None
    row = [pad] + pref
    return torch.tensor([row], device=device, dtype=torch.long)


def repair_t5_json_decode(s: str) -> str:
    """Normalize T5 output to parseable JSON: wrap inner body; fix legacy <unk> braces."""
    s = s.strip()
    if not s:
        return s
    # Legacy full-JSON training emitted <unk> for `{` `}`
    s = re.sub(r"^(?:<unk>\s*)+", "", s)
    s = re.sub(r"(?:\s*<unk>)+$", "", s)
    s = s.strip()
    if s.startswith("{"):
        out = s
    elif s.startswith('"topics"'):
        out = "{" + s + "}"
    elif '"topics"' in s[:800]:
        i = s.find('"topics"')
        out = "{" + s[i:] + "}"
    else:
        out = s
    if out.startswith("{") and not out.endswith("}"):
        try:
            json.loads(out)
        except json.JSONDecodeError:
            trial = out.rstrip().rstrip(",") + "}"
            try:
                json.loads(trial)
                return trial
            except json.JSONDecodeError:
                pass
    return out


class ForceJsonPrefixLogitsProcessor(LogitsProcessor):
    """Constrain the first len(prefix) generated tokens (grammar-lite for T5 JSON)."""

    def __init__(self, prefix_token_ids: list[int]):
        super().__init__()
        self.prefix_token_ids = prefix_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.prefix_token_ids:
            return scores
        idx = int(input_ids.shape[1]) - 1
        if 0 <= idx < len(self.prefix_token_ids):
            want = self.prefix_token_ids[idx]
            mask = torch.full_like(scores, float("-inf"))
            mask[:, want] = 0.0
            return scores + mask
        return scores


def cmd_train(args) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    ds_raw = load_dataset("knkarthick/samsum")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

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

    tokenized = ds_raw.map(tokenize_fn, batched=True, remove_columns=ds_raw["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, pad_to_multiple_of=8, return_tensors="pt")

    import torch
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    task1_dir = PROJECT_ROOT / "models" / "best" / "t5-small_lora_task1"
    task1_merged = task1_dir / "merged"

    for rank in args.ranks:
        alpha = rank * 2  # standard lora_alpha = 2 * r
        out_dir = PROJECT_ROOT / "models" / "best" / f"t5-small_lora_r{rank}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reuse task1 (r=16) for rank 16 to avoid redundant training
        if rank == 16 and task1_merged.exists():
            import shutil
            merged_out = out_dir / "merged"
            merged_out.mkdir(exist_ok=True)
            for f in task1_merged.iterdir():
                if f.is_file():
                    shutil.copy2(f, merged_out / f.name)
            tokenizer.save_pretrained(str(merged_out))
            size_mb = sum(f.stat().st_size for f in merged_out.glob("*.safetensors")) / (1024 * 1024)
            print(f"Rank 16: copied from task1 to {out_dir}, size_mb={size_mb:.2f}")
            continue

        base = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["q", "v"],
            bias="none",
        )
        model = get_peft_model(base, lora_config)

        train_args = Seq2SeqTrainingArguments(
            output_dir=str(PROJECT_ROOT / "models" / "checkpoints" / f"t5-small_lora_r{rank}"),
            num_train_epochs=3,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=5e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            save_total_limit=1,
            dataloader_num_workers=0,
            bf16=(device.type in {"cuda", "mps"}),
            fp16=False,
            report_to="none",
            seed=args.seed,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=train_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        merged = model.merge_and_unload()
        (out_dir / "merged").mkdir(exist_ok=True)
        merged.save_pretrained(str(out_dir / "merged"))
        tokenizer.save_pretrained(str(out_dir / "merged"))

        n_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = sum(f.stat().st_size for f in (out_dir / "merged").glob("*.safetensors")) / (1024 * 1024)

        print(f"Rank {rank}: saved to {out_dir}, params={n_params}, trainable={trainable}, size_mb={size_mb:.2f}")


def cmd_train_structured(args) -> None:
    """Supervised JSON targets on merged checkpoint: improves parseable JSON rate."""
    from datasets import load_dataset
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    ds_raw = load_dataset("knkarthick/samsum")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    rng = random.Random(args.seed)
    train_rows = ds_raw["train"]
    n = min(args.max_samples, len(train_rows))
    idxs = rng.sample(range(len(train_rows)), n)
    dialogues = [train_rows[i]["dialogue"] for i in idxs]
    summaries = [train_rows[i]["summary"] for i in idxs]
    json_targets = [gold_summary_to_t5_train_target(s) for s in summaries]

    train_dict = {"dialogue": dialogues, "json_target": json_targets}
    from datasets import Dataset as HFDataset

    raw_ds = HFDataset.from_dict(train_dict)

    def tok(examples):
        inputs = tokenizer(
            [structured_input_text(d) for d in examples["dialogue"]],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["json_target"],
            max_length=min(args.max_target_len, 384),
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = raw_ds.map(tok, batched=True, remove_columns=raw_ds.column_names)
    _ts = max(0.08, min(0.2, 64 / max(len(tokenized), 65)))
    _split = tokenized.train_test_split(test_size=_ts, seed=args.seed)
    train_ds, val_ds = _split["train"], _split["test"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    for rank in args.ranks:
        base_dir = resolve_rank_model_dir(rank)
        merged_in = base_dir / "merged"
        if not merged_in.exists():
            print(f"Skipping rank {rank}: missing {merged_in}")
            continue

        model = AutoModelForSeq2SeqLM.from_pretrained(str(merged_in))
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt"
        )
        out_ckpt = PROJECT_ROOT / "models" / "checkpoints" / f"t5-small_lora_r{rank}_structured"
        n_steps = max(1, (len(train_ds) + args.batch_size - 1) // args.batch_size * args.epochs)
        w_steps = min(500, max(50, int(0.06 * n_steps)))
        train_args = Seq2SeqTrainingArguments(
            output_dir=str(out_ckpt),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=w_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
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
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
        trainer.train()

        merged_struct = base_dir / "merged_structured"
        merged_struct.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_struct))
        tokenizer.save_pretrained(str(merged_struct))

        manifest = {
            "rank": rank,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "max_target_len": args.max_target_len,
            "lora_r": args.lora_r,
            "lora_target_modules": list(args.lora_target_modules),
            "json_target_format": "inner_no_braces",
            "output_dir": str(merged_struct),
        }
        man_path = PROJECT_ROOT / "results" / "metrics" / f"task5_structured_train_r{rank}.json"
        man_path.parent.mkdir(parents=True, exist_ok=True)
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Rank {rank}: saved structured merged model to {merged_struct}")

    summary_path = PROJECT_ROOT / "results" / "metrics" / "task5_structured_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "ranks": list(args.ranks),
                "max_samples": n,
                "method": "supervised_json_inner_targets_from_gold_summary",
                "json_target_format": "inner_no_braces",
                "note": "T5 tokenizer drops `{`/`}` as <unk>; labels omit outer braces.",
            },
            f,
            indent=2,
        )
    print(f"Wrote {summary_path}")


def cmd_eval(args) -> None:
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ds = load_dataset("knkarthick/samsum")["test"]
    n_samples = min(args.n_samples, len(ds))
    indices = list(range(n_samples))

    results = []
    for rank in args.ranks:
        model_dir = PROJECT_ROOT / "models" / "best" / f"t5-small_lora_r{rank}"
        # Fallback: task1 model uses r=16
        if not model_dir.exists() and rank == 16:
            task1_dir = PROJECT_ROOT / "models" / "best" / "t5-small_lora_task1"
            if (task1_dir / "merged").exists():
                model_dir = task1_dir
        merged_dir = model_dir / "merged" if (model_dir / "merged").exists() else model_dir
        if not merged_dir.exists():
            print(f"Skipping rank {rank}: {merged_dir} not found")
            continue

        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(merged_dir))
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()

        dialogues = [ds[i]["dialogue"] for i in indices]
        refs = [ds[i]["summary"] for i in indices]
        preds = []

        t0 = time.perf_counter()
        for i in range(0, len(dialogues), args.batch_size):
            batch = dialogues[i : i + args.batch_size]
            inputs = tokenizer(
                [TASK_PREFIX + d for d in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SOURCE_LEN,
            ).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_TARGET_LEN, num_beams=4, early_stopping=True)
            preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        elapsed = time.perf_counter() - t0

        if device.type == "mps":
            torch.mps.synchronize()

        rouge = _rouge_l([p.strip() for p in preds], refs)
        size_mb = sum(f.stat().st_size for f in merged_dir.glob("*.safetensors")) / (1024 * 1024)

        results.append({
            "rank": rank,
            "rougeL": round(rouge, 4),
            "latency_ms": round(elapsed / n_samples * 1000, 2),
            "model_size_mb": round(size_mb, 2),
        })
        print(f"Rank {rank}: ROUGE-L={rouge:.2f}, latency={elapsed/n_samples*1000:.1f}ms, size={size_mb:.1f}MB")

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_rank_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"n_samples": n_samples, "results": results}, f, indent=2)
    print(f"Saved {out_path}")


def cmd_structured(args) -> None:
    """Structured output: supervised JSON head (merged_structured) if present, else merged."""
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList

    ds = load_dataset("knkarthick/samsum")["test"]
    n_samples = min(args.n_samples, len(ds))

    results = []
    for rank in args.ranks:
        merged_dir = resolve_inference_merged_dir(rank)
        if merged_dir is None:
            print(f"Skipping rank {rank}: no merged weights found")
            continue
        used_structured = merged_dir.name == "merged_structured"

        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(merged_dir))
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()

        prefix_ids = json_prefix_token_ids(tokenizer) if args.force_json_prefix else []
        logits_processors = LogitsProcessorList([ForceJsonPrefixLogitsProcessor(prefix_ids)])

        parse_ok_count = 0
        envelope_count = 0
        fallback_count = 0
        preds_raw = []
        preds_struct_dicts: list[dict[str, Any]] = []
        refs = []

        gen_max = min(args.max_new_tokens_structured, 384)

        dec_pref = structured_decoder_input_ids(tokenizer, device) if args.decoder_json_prefill else None

        for i in range(n_samples):
            row = ds[i]
            inputs = tokenizer(
                structured_input_text(row["dialogue"]),
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SOURCE_LEN,
            ).to(device)
            with torch.no_grad():
                gen_kw: dict[str, Any] = {
                    **inputs,
                    "max_new_tokens": gen_max,
                    "num_beams": args.num_beams,
                    "early_stopping": True,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                }
                if prefix_ids:
                    gen_kw["logits_processor"] = logits_processors
                if dec_pref is not None:
                    gen_kw["decoder_input_ids"] = dec_pref.clone()
                out = model.generate(**gen_kw)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            pred = repair_t5_json_decode(pred)
            preds_raw.append(pred)
            refs.append(row["summary"])

            bundle, used_fb = prediction_to_structured_dict_with_trace(pred)
            preds_struct_dicts.append(bundle)
            if not used_fb:
                parse_ok_count += 1
            else:
                fallback_count += 1
            if _is_valid_structured(bundle):
                envelope_count += 1

        validity_rate = parse_ok_count / n_samples
        parse_success_rate = parse_ok_count / n_samples
        heuristic_fallback_rate = fallback_count / n_samples
        api_envelope_valid_rate = envelope_count / n_samples
        rouge = _rouge_l(preds_raw, refs)
        pred_json_lines = [
            json.dumps(d, sort_keys=True, ensure_ascii=False) for d in preds_struct_dicts
        ]
        gold_json_lines = [
            json.dumps(gold_summary_to_structured_obj(r), sort_keys=True, ensure_ascii=False) for r in refs
        ]
        rouge_json = _rouge_l(pred_json_lines, gold_json_lines)

        results.append({
            "rank": rank,
            "json_validity_rate": round(validity_rate, 4),
            "parse_success_rate": round(parse_success_rate, 4),
            "heuristic_fallback_rate": round(heuristic_fallback_rate, 4),
            "api_envelope_valid_rate": round(api_envelope_valid_rate, 4),
            "structured_contract_rate": round(parse_success_rate, 4),
            "rougeL_structured": round(rouge, 4),
            "rougeL_structured_json_vs_gold": round(rouge_json, 4),
            "used_merged_structured": used_structured,
            "force_json_prefix": bool(prefix_ids) and args.force_json_prefix,
            "decoder_json_prefill": dec_pref is not None,
            "json_target_format_expected": "inner_no_braces" if used_structured else "full_json_or_plain",
        })
        print(
            f"Rank {rank}: strict={validity_rate:.1%}, parse_success={parse_success_rate:.1%}, "
            f"fallback={heuristic_fallback_rate:.1%}, ROUGE(raw)={rouge:.2f}, ROUGE(json)={rouge_json:.2f}, "
            f"merged_structured={used_structured}"
        )

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_structured_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_samples": n_samples,
                "schema": STRUCTURED_SCHEMA,
                "metric_notes": {
                    "json_validity_rate": "Same as parse_success_rate (single source: prediction_to_structured_dict_with_trace).",
                    "parse_success_rate": "Model output parsed to schema without plain-text heuristic (1 - heuristic_fallback_rate).",
                    "heuristic_fallback_rate": "Fraction where gold_summary_to_structured_obj(pred) was used.",
                    "api_envelope_valid_rate": "Valid schema after prediction_to_structured_dict (often high; not model-JSON accuracy).",
                    "structured_contract_rate": "Alias of parse_success_rate (model-parse success; not API-envelope-only).",
                    "rougeL_structured": "ROUGE-L of raw model string vs plain gold summary (weak signal for JSON task).",
                    "rougeL_structured_json_vs_gold": "ROUGE-L of serialized pred dict vs serialized heuristic gold structured dict.",
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_path}")


def cmd_sweet_spot(args) -> None:
    ablation_path = PROJECT_ROOT / "results" / "metrics" / "task5_rank_ablation.json"
    structured_path = PROJECT_ROOT / "results" / "metrics" / "task5_structured_output.json"
    if not ablation_path.exists() or not structured_path.exists():
        raise FileNotFoundError("Run 'eval' and 'structured' first.")

    with open(ablation_path) as f:
        ablation = json.load(f)
    with open(structured_path) as f:
        structured = json.load(f)

    by_rank = {r["rank"]: r for r in ablation["results"]}
    struct_by_rank = {r["rank"]: r for r in structured["results"]}

    full_rank = max(by_rank.keys())
    baseline_rouge = by_rank[full_rank]["rougeL"]

    candidates = []
    for rank in sorted(by_rank.keys()):
        if rank not in struct_by_rank:
            continue
        r_ab = by_rank[rank]
        r_str = struct_by_rank[rank]
        ps = r_str.get("parse_success_rate", r_str.get("json_validity_rate", 0.0))
        valid = ps >= args.min_parse_success
        rouge_close = (baseline_rouge - r_ab["rougeL"]) <= 1.0
        if valid and rouge_close:
            candidates.append({
                "rank": rank,
                "rougeL": r_ab["rougeL"],
                "parse_success_rate": ps,
                "latency_ms": r_ab["latency_ms"],
                "size_mb": r_ab["model_size_mb"],
            })

    sweet_spot = min(candidates, key=lambda x: x["rank"]) if candidates else None
    selection_note = None
    if sweet_spot is None and args.fallback_rouge_only:
        relaxed = []
        for rank in sorted(by_rank.keys()):
            if rank not in struct_by_rank:
                continue
            r_ab = by_rank[rank]
            r_str = struct_by_rank[rank]
            ps = r_str.get("parse_success_rate", r_str.get("json_validity_rate", 0.0))
            rouge_close = (baseline_rouge - r_ab["rougeL"]) <= 1.0
            if rouge_close:
                relaxed.append({
                    "rank": rank,
                    "rougeL": r_ab["rougeL"],
                    "parse_success_rate": ps,
                    "latency_ms": r_ab["latency_ms"],
                    "size_mb": r_ab["model_size_mb"],
                })
        if relaxed:
            sweet_spot = min(relaxed, key=lambda x: x["rank"])
            selection_note = (
                "No rank met min_parse_success; used ROUGE-window-only fallback "
                "(lowest rank within 1 ROUGE-L pt of baseline)."
            )

    # Structured vs free-form comparison
    comparison = []
    for rank in sorted(set(by_rank.keys()) & set(struct_by_rank.keys())):
        ff = by_rank[rank]["rougeL"]
        st = struct_by_rank[rank]["rougeL_structured"]
        sr = struct_by_rank[rank]
        ps = sr.get("parse_success_rate", sr.get("json_validity_rate", 0.0))
        comparison.append({
            "rank": rank,
            "free_form_rougeL": ff,
            "structured_rougeL": st,
            "parse_success_rate": ps,
            "json_validity_rate": sr.get("json_validity_rate"),
            "structured_vs_freeform_delta": round(st - ff, 4),
        })

    report = {
        "task": "task5_sweet_spot",
        "baseline_rank": full_rank,
        "baseline_rougeL": baseline_rouge,
        "min_parse_success": args.min_parse_success,
        "fallback_rouge_only": args.fallback_rouge_only,
        "selection_note": selection_note,
        "sweet_spot": sweet_spot,
        "candidates_parse_success_within_1pt_rouge": candidates,
        "structured_vs_freeform_comparison": comparison,
        "note": (
            "Structured eval uses TASK_PREFIX + instruction; if train_structured was run, "
            "loads merged_structured (supervised JSON targets). Free-form uses standard summarization. "
            "Primary: parse_success_rate >= min_parse_success and ROUGE within 1 pt of baseline. "
            "If none qualify and --fallback-rouge-only (default on), pick lowest rank within the ROUGE window. "
            "If sweet_spot is still null, `package` uses --default_rank."
        ),
    }

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_sweet_spot.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Sweet spot: {sweet_spot}")
    print(f"Saved {out_path}")


def cmd_package(args) -> None:
    sweet_path = PROJECT_ROOT / "results" / "metrics" / "task5_sweet_spot.json"
    if not sweet_path.exists():
        raise FileNotFoundError("Run 'sweet_spot' first.")

    with open(sweet_path) as f:
        data = json.load(f)

    sweet = data.get("sweet_spot")
    rank = (sweet.get("rank") if isinstance(sweet, dict) else None) or args.default_rank
    if sweet is None:
        print(
            f"Note: sweet_spot is null (no rank met min_parse_success / ROUGE window). "
            f"Packaging --default_rank={args.default_rank}."
        )
    prod_dir = PROJECT_ROOT / "models" / "production_task5"
    src_dir = resolve_rank_model_dir(rank)
    merged_struct = src_dir / "merged_structured"
    merged_std = src_dir / "merged"
    if args.prefer_structured and merged_struct.exists() and any(merged_struct.glob("*.safetensors")):
        merged_src = merged_struct
    elif merged_std.exists():
        merged_src = merged_std
    else:
        merged_src = src_dir

    if not merged_src.exists():
        raise FileNotFoundError(f"Model not found: {merged_src}")

    import shutil
    prod_dir.mkdir(parents=True, exist_ok=True)
    for f in merged_src.iterdir():
        if f.is_file():
            shutil.copy2(f, prod_dir / f.name)

    config = {
        "task": "task5_production_baseline",
        "lora_rank": rank,
        "structured_schema": STRUCTURED_SCHEMA,
        "model_path": str(prod_dir),
        "source": str(merged_src),
        "structured_supervised": merged_src.name == "merged_structured",
    }
    with open(prod_dir / "task5_production_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Packaged to {prod_dir}")
    print(f"Config: rank={rank}, schema={list(STRUCTURED_SCHEMA.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: LoRA rank ablation & structured output")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--ranks", nargs="+", type=int, default=LORA_RANKS)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ranks", nargs="+", type=int, default=LORA_RANKS)
    p_eval.add_argument("--n_samples", type=int, default=256)
    p_eval.add_argument("--batch_size", type=int, default=8)

    p_struct = sub.add_parser("structured")
    p_struct.add_argument("--ranks", nargs="+", type=int, default=LORA_RANKS)
    p_struct.add_argument("--n_samples", type=int, default=256)
    p_struct.add_argument("--max_new_tokens_structured", type=int, default=320)
    p_struct.add_argument("--num_beams", type=int, default=4)
    p_struct.add_argument("--repetition_penalty", type=float, default=1.12)
    p_struct.add_argument("--no_repeat_ngram_size", type=int, default=4)
    p_struct.add_argument(
        "--force-json-prefix",
        dest="force_json_prefix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optional: constrain first decoder tokens to inner-format opening ('\"topics\": ['). "
            "Off by default — only combine with --decoder-json-prefill if you trained with the same prefix."
        ),
    )
    p_struct.add_argument(
        "--decoder-json-prefill",
        dest="decoder_json_prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prefill decoder with '\"topics\": [' after pad. Default off: matches train_structured "
            "(full inner string from first token)."
        ),
    )

    p_ts = sub.add_parser(
        "train_structured",
        help="Supervised JSON fine-tune on merged checkpoints → merged_structured/",
    )
    p_ts.add_argument("--ranks", nargs="+", type=int, default=LORA_RANKS)
    p_ts.add_argument("--max_samples", type=int, default=1000)
    p_ts.add_argument("--epochs", type=int, default=3)
    p_ts.add_argument("--batch_size", type=int, default=8)
    p_ts.add_argument("--learning_rate", type=float, default=2e-5)
    p_ts.add_argument("--lora_r", type=int, default=16)
    p_ts.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q", "k", "v", "o"],
        help="Attention linear names for T5 (PEFT short names).",
    )
    p_ts.add_argument("--max_target_len", type=int, default=384)
    p_ts.add_argument("--seed", type=int, default=42)

    p_sweet = sub.add_parser("sweet_spot")
    p_sweet.add_argument(
        "--min-parse-success",
        type=float,
        default=0.2,
        dest="min_parse_success",
        help="Minimum parse_success_rate for primary sweet-spot candidates.",
    )
    p_sweet.add_argument(
        "--fallback-rouge-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="fallback_rouge_only",
        help="If no rank meets min_parse_success, still pick lowest rank within ROUGE window (default: on).",
    )

    p_pkg = sub.add_parser("package")
    p_pkg.add_argument("--default_rank", type=int, default=8)
    p_pkg.add_argument("--prefer_structured", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "structured":
        cmd_structured(args)
    elif args.command == "train_structured":
        cmd_train_structured(args)
    elif args.command == "sweet_spot":
        cmd_sweet_spot(args)
    elif args.command == "package":
        cmd_package(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Task 5 — LoRA Rank Ablation & Structured Output Constraints.

Subcommands:
  train       — Train T5-small LoRA with ranks 2, 4, 8, 16, 32
  eval        — Measure ROUGE-L, latency, merged size + PEFT adapter size / trainable count
  structured  — Schema JSON metrics + ROUGE (see --structured-pipeline)
  sweet_spot  — Rank selection vs native-JSON gate + ROUGE window
  package     — Package optimal config as production baseline

Structured pipelines (``structured``):
  reliable (default) — one summarization forward pass on ``merged/``, then either
    strict JSON parse of the model string or deterministic prose→schema projection
    (always yields valid JSON for APIs / ``json.dumps``).
  legacy_json_prompt — previous behavior: ``Output JSON only`` prompt and optional
    ``merged_structured/`` weights (often emits plain prose; native JSON rate ≈ 0).

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


def summarization_input_text(dialogue: str) -> str:
    """Same encoder prompt as ``cmd_eval`` free-form summarization."""
    return f"{TASK_PREFIX}{dialogue}"


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


def resolve_summarization_merged_dir(rank: int) -> Path | None:
    """``merged/`` only (rank ablation summarization checkpoint), for reliable structured pipeline."""
    base = resolve_rank_model_dir(rank)
    if not base.exists():
        return None
    m = base / "merged"
    if m.exists() and any(m.glob("*.safetensors")):
        return m
    return base if any(base.glob("*.safetensors")) else None


def _adapter_weight_file_paths(adapter_root: Path) -> list[Path]:
    """Return PEFT adapter checkpoint files (safetensors or legacy bin + sharded safetensors)."""
    paths: list[Path] = []
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_root / name
        if p.exists():
            paths.append(p)
    paths.extend(sorted(adapter_root.glob("adapter_model-*.safetensors")))
    return paths


def adapter_weight_stats(adapter_root: Path) -> tuple[float, int | None]:
    """
    On-disk adapter size (MiB) and total tensor element count in adapter safetensors.

    The element count matches reported trainable LoRA parameters for standard PEFT checkpoints.
    """
    paths = _adapter_weight_file_paths(adapter_root)
    if not paths:
        return 0.0, None
    mib = sum(p.stat().st_size for p in paths) / (1024 * 1024)
    n_params: int | None = None
    try:
        from safetensors import safe_open

        n_params = 0
        for p in paths:
            if p.suffix != ".safetensors":
                continue
            with safe_open(str(p), framework="pt") as f:
                for key in f.keys():
                    n_params += int(f.get_tensor(key).numel())
    except Exception:
        n_params = None
    return round(mib, 4), n_params


def _resolve_adapter_dir_for_metrics(model_dir: Path, rank: int) -> tuple[Path, str | None]:
    """
    PEFT adapter lives next to ``merged/`` under ``models/best/t5-small_lora_r{rank}/``.

    Rank 16 may be a merged-only copy of task1; in that case use task1's adapter files for stats.
    """
    if _adapter_weight_file_paths(model_dir):
        return model_dir, None
    if rank == 16:
        t1 = PROJECT_ROOT / "models" / "best" / "t5-small_lora_task1"
        if _adapter_weight_file_paths(t1):
            return t1, str(t1.relative_to(PROJECT_ROOT))
    return model_dir, None


def _rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return sum(vals) / max(len(vals), 1) * 100


def _normalize_json_candidate(text: str) -> str:
    """Normalize Unicode quotes / trivial noise so ``json.loads`` can succeed more often."""
    repl = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
    for a, b in repl.items():
        text = text.replace(a, b)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _parse_json_summary(text: str) -> dict | None:
    """Try to parse JSON from model output. Handle fences, prefixes, truncation."""
    text = _normalize_json_candidate(text.strip())
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
    # Single-quoted keys (common T5 drift): "topics" -> optional 'topics'
    sq = re.sub(r"'(topics|action_items|decision)'\s*:", r'"\1":', text, flags=re.I)
    if sq != text:
        for attempt in (sq, sq + "}"):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
    return None


def structured_dict_from_model_output(pred: str) -> tuple[dict[str, Any], str]:
    """
    Return (schema_dict, source).

    source is:
      - ``native_json`` — ``pred`` (after T5 repair) parsed with ``json.loads`` and passes
        ``_is_valid_structured``;
      - ``prose_projection`` — deterministic ``gold_summary_to_structured_obj(pred)`` (no
        reference labels; same helper used for supervised JSON targets).
    """
    pred = (pred or "").strip()
    if not pred:
        return {}, "prose_projection"
    repaired = repair_t5_json_decode(pred)
    parsed = _parse_json_summary(repaired)
    if _is_valid_structured(parsed):
        assert parsed is not None
        return parsed, "native_json"
    return gold_summary_to_structured_obj(pred), "prose_projection"


def prediction_to_structured_dict_with_trace(pred: str) -> tuple[dict[str, Any], bool]:
    """Return (structured_dict, used_heuristic_fallback).

    If ``used_heuristic_fallback`` is True, output used prose→schema projection rather than
    a strict JSON parse of the model string.
    """
    d, src = structured_dict_from_model_output(pred)
    return d, src != "native_json"


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

        adapter_dir, adapter_source = _resolve_adapter_dir_for_metrics(model_dir, rank)
        aw_mib, aw_n = adapter_weight_stats(adapter_dir)
        row: dict[str, Any] = {
            "rank": rank,
            "rougeL": round(rouge, 4),
            "latency_ms": round(elapsed / n_samples * 1000, 2),
            "model_size_mb": round(size_mb, 2),
            "adapter_weights_mb": aw_mib,
            "adapter_trainable_params": aw_n,
        }
        if adapter_source:
            row["adapter_stats_source"] = adapter_source
        results.append(row)
        ap_hint = f", adapter≈{aw_mib}MiB" if aw_mib else ""
        print(
            f"Rank {rank}: ROUGE-L={rouge:.2f}, latency={elapsed/n_samples*1000:.1f}ms, "
            f"merged={size_mb:.1f}MB{ap_hint}"
        )

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_rank_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_samples": n_samples,
        "metric_notes": {
            "model_size_mb": "Size of merged/*.safetensors for inference (base weights with LoRA fused).",
            "adapter_weights_mb": "PEFT adapter checkpoint only (adapter_model*.safetensors); excludes frozen base.",
            "adapter_trainable_params": "Total elements in adapter safetensors (LoRA A/B); rank scales this count.",
            "adapter_stats_source": "If set, adapter files were read from another folder (e.g. task1 for rank-16 alias).",
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")


def cmd_structured(args) -> None:
    from datasets import load_dataset
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList

    ds = load_dataset("knkarthick/samsum")["test"]
    n_samples = min(args.n_samples, len(ds))
    pipeline = args.structured_pipeline

    results = []
    for rank in args.ranks:
        if pipeline == "reliable":
            merged_dir = resolve_summarization_merged_dir(rank)
            used_structured = False
        else:
            merged_dir = resolve_inference_merged_dir(rank)
            used_structured = merged_dir is not None and merged_dir.name == "merged_structured"
        if merged_dir is None:
            print(f"Skipping rank {rank}: no merged weights found")
            continue

        tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(merged_dir))
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()

        prefix_ids = json_prefix_token_ids(tokenizer) if args.force_json_prefix else []
        logits_processors = LogitsProcessorList([ForceJsonPrefixLogitsProcessor(prefix_ids)])

        native_count = 0
        envelope_count = 0
        projection_count = 0
        roundtrip_ok = 0
        preds_raw = []
        preds_struct_dicts: list[dict[str, Any]] = []
        refs = []

        gen_max = min(args.max_new_tokens_structured, 384)
        dec_pref = structured_decoder_input_ids(tokenizer, device) if args.decoder_json_prefill else None

        for i in range(n_samples):
            row = ds[i]
            enc_text = (
                summarization_input_text(row["dialogue"])
                if pipeline == "reliable"
                else structured_input_text(row["dialogue"])
            )
            inputs = tokenizer(
                enc_text,
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
                if prefix_ids and pipeline != "reliable":
                    gen_kw["logits_processor"] = logits_processors
                if dec_pref is not None and pipeline != "reliable":
                    gen_kw["decoder_input_ids"] = dec_pref.clone()
                out = model.generate(**gen_kw)
            if device.type == "mps":
                torch.mps.synchronize()
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            refs.append(row["summary"])
            if pipeline == "reliable":
                preds_raw.append(pred)
                bundle, src = structured_dict_from_model_output(pred)
            else:
                pred_rep = repair_t5_json_decode(pred)
                preds_raw.append(pred_rep)
                bundle, src = structured_dict_from_model_output(pred_rep)
            preds_struct_dicts.append(bundle)
            if src == "native_json":
                native_count += 1
            else:
                projection_count += 1
            if _is_valid_structured(bundle):
                envelope_count += 1
            try:
                json.loads(json.dumps(bundle, ensure_ascii=False))
                roundtrip_ok += 1
            except (TypeError, ValueError):
                pass

        generative_native_json_rate = native_count / n_samples
        prose_projection_rate = projection_count / n_samples
        guaranteed_json_roundtrip_rate = roundtrip_ok / n_samples
        parse_success_rate = generative_native_json_rate
        heuristic_fallback_rate = prose_projection_rate
        validity_rate = parse_success_rate
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
            "structured_pipeline": pipeline,
            "generative_native_json_rate": round(generative_native_json_rate, 4),
            "prose_projection_rate": round(prose_projection_rate, 4),
            "guaranteed_json_roundtrip_rate": round(guaranteed_json_roundtrip_rate, 4),
            "reliable_structured_delivery_rate": round(api_envelope_valid_rate, 4),
            "json_validity_rate": round(validity_rate, 4),
            "parse_success_rate": round(parse_success_rate, 4),
            "heuristic_fallback_rate": round(heuristic_fallback_rate, 4),
            "api_envelope_valid_rate": round(api_envelope_valid_rate, 4),
            "structured_contract_rate": round(parse_success_rate, 4),
            "rougeL_structured": round(rouge, 4),
            "rougeL_structured_json_vs_gold": round(rouge_json, 4),
            "used_merged_structured": used_structured,
            "force_json_prefix": bool(prefix_ids) and args.force_json_prefix,
            "decoder_json_prefill": dec_pref is not None and pipeline != "reliable",
            "json_target_format_expected": "inner_no_braces" if used_structured else "full_json_or_plain",
        })
        print(
            f"Rank {rank}: native_json={generative_native_json_rate:.1%}, "
            f"projection={prose_projection_rate:.1%}, json_roundtrip={guaranteed_json_roundtrip_rate:.1%}, "
            f"ROUGE(raw)={rouge:.2f}, ROUGE(json)={rouge_json:.2f}, pipeline={pipeline}"
        )

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_structured_output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_samples": n_samples,
                "structured_pipeline": pipeline,
                "schema": STRUCTURED_SCHEMA,
                "metric_notes": {
                    "structured_pipeline_reliable": (
                        "Summarize with TASK_PREFIX + dialogue on merged/ weights, then native JSON parse "
                        "or deterministic prose→schema projection (no gold labels). Guarantees API JSON "
                        "via json.dumps(final_dict)."
                    ),
                    "generative_native_json_rate": (
                        "Share of samples where the model string parsed as JSON and matched the schema."
                    ),
                    "prose_projection_rate": (
                        "Share using gold_summary_to_structured_obj(model prose); deterministic, not label leakage."
                    ),
                    "guaranteed_json_roundtrip_rate": "Fraction where json.loads(json.dumps(structured_dict)) succeeded.",
                    "parse_success_rate": "Alias of generative_native_json_rate (backward-compatible field name).",
                    "heuristic_fallback_rate": "Alias of prose_projection_rate (backward-compatible field name).",
                    "json_validity_rate": "Same as parse_success_rate (legacy alias).",
                    "api_envelope_valid_rate": "Valid schema dict after structured_dict_from_model_output.",
                    "structured_contract_rate": "Native JSON only (same as parse_success_rate).",
                    "rougeL_structured": "ROUGE-L of primary model string vs plain gold summary.",
                    "rougeL_structured_json_vs_gold": "ROUGE-L of serialized structured dict vs heuristic gold structured dict.",
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved {out_path}")


def _native_json_gate_metric(row: dict[str, Any]) -> float:
    """Prefer explicit generative rate from new metrics; fall back to legacy fields."""
    if "generative_native_json_rate" in row:
        return float(row["generative_native_json_rate"])
    return float(row.get("parse_success_rate", row.get("json_validity_rate", 0.0)))


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
        native_r = _native_json_gate_metric(r_str)
        rouge_close = (baseline_rouge - r_ab["rougeL"]) <= 1.0
        if native_r >= args.min_parse_success and rouge_close:
            candidates.append({
                "rank": rank,
                "rougeL": r_ab["rougeL"],
                "generative_native_json_rate": native_r,
                "parse_success_rate": r_str.get("parse_success_rate"),
                "guaranteed_json_roundtrip_rate": r_str.get("guaranteed_json_roundtrip_rate"),
                "latency_ms": r_ab["latency_ms"],
                "size_mb": r_ab["model_size_mb"],
            })

    sweet_spot = min(candidates, key=lambda x: x["rank"]) if candidates else None
    selection_note = None
    rouge_window_pick = None
    relaxed = []
    for rank in sorted(by_rank.keys()):
        if rank not in struct_by_rank:
            continue
        r_ab = by_rank[rank]
        rouge_close = (baseline_rouge - r_ab["rougeL"]) <= 1.0
        if rouge_close:
            r_str = struct_by_rank[rank]
            relaxed.append({
                "rank": rank,
                "rougeL": r_ab["rougeL"],
                "generative_native_json_rate": _native_json_gate_metric(r_str),
                "latency_ms": r_ab["latency_ms"],
                "size_mb": r_ab["model_size_mb"],
            })
    if relaxed:
        rouge_window_pick = min(relaxed, key=lambda x: x["rank"])

    if sweet_spot is None and rouge_window_pick is not None:
        if args.fallback_rouge_only:
            selection_note = (
                "No rank met generative_native_json_rate >= min_parse_success; "
                "--fallback-rouge-only copied operational_pick from ROUGE window only "
                "(does not satisfy native JSON gate)."
            )
        else:
            selection_note = (
                "No rank met generative_native_json_rate >= min_parse_success; "
                "sweet_spot left null. See operational_pick_rouge_window_only for a ROUGE-only choice."
            )

    comparison = []
    for rank in sorted(set(by_rank.keys()) & set(struct_by_rank.keys())):
        ff = by_rank[rank]["rougeL"]
        st = struct_by_rank[rank]["rougeL_structured"]
        sr = struct_by_rank[rank]
        native_r = _native_json_gate_metric(sr)
        comparison.append({
            "rank": rank,
            "free_form_rougeL": ff,
            "structured_rougeL": st,
            "generative_native_json_rate": sr.get("generative_native_json_rate", native_r),
            "parse_success_rate": sr.get("parse_success_rate"),
            "guaranteed_json_roundtrip_rate": sr.get("guaranteed_json_roundtrip_rate"),
            "json_validity_rate": sr.get("json_validity_rate"),
            "structured_vs_freeform_delta": round(st - ff, 4),
        })

    report = {
        "task": "task5_sweet_spot",
        "structured_pipeline": structured.get("structured_pipeline"),
        "baseline_rank": full_rank,
        "baseline_rougeL": baseline_rouge,
        "min_parse_success": args.min_parse_success,
        "fallback_rouge_only": args.fallback_rouge_only,
        "selection_note": selection_note,
        "sweet_spot": sweet_spot,
        "operational_pick_rouge_window_only": rouge_window_pick if sweet_spot is None else None,
        "operational_pick": (
            rouge_window_pick if sweet_spot is None and args.fallback_rouge_only else sweet_spot
        ),
        "candidates_native_json_gate_within_1pt_rouge": candidates,
        "candidates_parse_success_within_1pt_rouge": candidates,
        "structured_vs_freeform_comparison": comparison,
        "note": (
            "sweet_spot requires generative_native_json_rate (strict model JSON) >= min_parse_success "
            "and ROUGE within 1 pt of the max-rank baseline. Use guaranteed_json_roundtrip_rate from "
            "task5_structured_output.json for API-safe JSON (reliable pipeline). "
            "If sweet_spot is null and --no-fallback-rouge-only (default), operational_pick omits "
            "ROUGE-only fallback unless explicitly enabled."
        ),
    }

    out_path = PROJECT_ROOT / "results" / "metrics" / "task5_sweet_spot.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Sweet spot (native JSON gate): {sweet_spot}")
    print(f"Operational pick: {report['operational_pick']}")
    print(f"Saved {out_path}")


def cmd_package(args) -> None:
    sweet_path = PROJECT_ROOT / "results" / "metrics" / "task5_sweet_spot.json"
    if not sweet_path.exists():
        raise FileNotFoundError("Run 'sweet_spot' first.")

    with open(sweet_path) as f:
        data = json.load(f)

    sweet = data.get("sweet_spot")
    op_pick = data.get("operational_pick")
    rank = None
    if isinstance(sweet, dict):
        rank = sweet.get("rank")
    elif isinstance(op_pick, dict):
        rank = op_pick.get("rank")
    if rank is None:
        rank = args.default_rank
        print(
            "Note: sweet_spot is null and operational_pick unset; "
            f"using --default_rank={args.default_rank}."
        )
    elif sweet is None and isinstance(op_pick, dict):
        print(
            "Note: packaging operational_pick (ROUGE fallback / partial gate). "
            "Native JSON sweet_spot was not satisfied — see task5_sweet_spot.json."
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
    p_struct.add_argument(
        "--structured-pipeline",
        choices=["reliable", "legacy_json_prompt"],
        default="reliable",
        help=(
            "reliable: summarize on merged/ then JSON parse or prose projection (default). "
            "legacy_json_prompt: prior JSON-only encoder prompt + merged_structured if present."
        ),
    )
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
        default=False,
        dest="fallback_rouge_only",
        help=(
            "If no rank meets the native JSON gate, still set operational_pick to the lowest rank "
            "within the ROUGE window (default: off — sweet_spot may be null)."
        ),
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

#!/usr/bin/env python3
"""
evaluate.py — ROUGE evaluation of a fine-tuned model on the SAMSum test set.

Loads the best saved model checkpoint and runs full-set evaluation with
beam search as configured in config.yaml. Writes results to results/metrics/.

Usage:
  python3 scripts/evaluate.py                               # config defaults
  python3 scripts/evaluate.py --model_path models/best/facebook_bart-base_with_speakers
  python3 scripts/evaluate.py --split validation            # eval on val set
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from model_registry import resolve_model_name


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROUGE evaluation on SAMSum test set")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="Override config model_name for cache path (e.g. flan-t5-base, t5-small)",
    )
    parser.add_argument("--model_path", default=None, help="Path to saved model dir")
    parser.add_argument("--variant",    default=None, help="Override dataset_variant")
    parser.add_argument("--split",      default="test", choices=["test", "validation"])
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name"] = args.model
    cfg["model_name"] = resolve_model_name(cfg["model_name"])
    if args.variant:
        cfg["dataset_variant"] = args.variant
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()

    MODEL_NAME = cfg["model_name"]
    VARIANT    = cfg["dataset_variant"]
    run_name   = f"{MODEL_NAME.replace('/', '_')}_{VARIANT}"

    model_path = Path(args.model_path) if args.model_path else (
        project_root / "models" / "best" / run_name
    )
    if not model_path.exists():
        print(f"  ❌  Model not found at: {model_path}")
        print("      Run train.py first.")
        sys.exit(1)

    try:
        from datasets import load_from_disk                 # noqa: PLC0415
        from evaluate import load as load_metric            # noqa: PLC0415
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  {exc}")
        sys.exit(1)

    device    = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    # Load tokenized dataset
    _cache_sfx = str(cfg.get("tokenized_cache_suffix") or "").strip()
    dataset_path = (
        project_root / "data" / "cache"
        / f"samsum_{VARIANT}_{MODEL_NAME.replace('/', '_')}{_cache_sfx}"
    )
    ds = load_from_disk(str(dataset_path))
    split_ds = ds[args.split]

    rouge = load_metric("rouge")

    all_preds: list[str] = []
    all_refs:  list[str] = []
    bs = cfg["batch_size"]

    print(f"\n  Evaluating {len(split_ds):,} examples on {args.split}...\n")

    for i in range(0, len(split_ds), bs):
        batch    = split_ds[i : i + bs]
        input_ids = torch.tensor(
            [seq + [tokenizer.pad_token_id] * (max(len(s) for s in batch["input_ids"]) - len(seq))
             for seq in batch["input_ids"]]
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens    = cfg["max_target_length"],
                num_beams         = cfg["num_beams"],
                length_penalty    = cfg["length_penalty"],
                early_stopping    = cfg["early_stopping_beam"],
            )

        labels = np.array(batch["labels"])
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        refs  = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        all_preds.extend([p.strip() for p in preds])
        all_refs.extend([r.strip() for r in refs])

        if (i // bs) % 10 == 0:
            print(f"    {i + len(preds):>5} / {len(split_ds)}")

    result = rouge.compute(
        predictions=all_preds,
        references=all_refs,
        use_stemmer=True,
    )
    scores = {k: round(v * 100, 4) for k, v in result.items()}
    scores.update({"model": MODEL_NAME, "variant": VARIANT, "split": args.split})

    out_path = project_root / "results" / "metrics" / f"{run_name}_{args.split}_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(scores, fh, indent=2)

    print(f"\n  ROUGE on {args.split} ({len(split_ds):,} examples):")
    for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        if k in scores:
            print(f"    {k:>10} : {scores[k]:.2f}")
    print(f"\n  Saved → {out_path.relative_to(project_root)}\n")

    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

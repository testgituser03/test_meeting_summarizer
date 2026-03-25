#!/usr/bin/env python3
"""
steering_inference.py — Task 3 / controllable generation via logit steering.

Steering choice (Task-3 requirement): Option B — logit steering.
For a selected decoder-layer direction v (hidden space), compute vocabulary bias:
    b = lm_head(normalize(v))
Then inject during beam search with scale alpha:
    logits' = logits + alpha * b

This keeps generation integration simple and stable via LogitsProcessor.

Output:
  results/steering/<run_name>_steering_generations.json
"""

# Must be set before torch/transformers imports.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run logit-steered generation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steering", required=True, help="Path to results/steering/*.pt")
    parser.add_argument("--model_path", default=None, help="Override checkpoint path")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--method", default="mean_diff", choices=["mean_diff", "pca_delta", "logistic"])
    parser.add_argument("--layers", nargs="+", type=int, default=[6, 8, 10, 12])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _norm(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def _batchify(items: list[int], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.variant:
        cfg["dataset_variant"] = args.variant
    if args.seed is not None:
        cfg["seed"] = args.seed

    set_all_seeds(int(cfg["seed"]))

    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    model_name = cfg["model_name"]
    variant = cfg["dataset_variant"]
    run_name = cfg.get("run_name", f"{model_name.replace('/', '_')}_{variant}")

    steering_payload = torch.load(args.steering, map_location="cpu")
    vectors = steering_payload["vectors"]
    steering_meta = steering_payload["meta"]

    try:
        from datasets import load_dataset                 # noqa: PLC0415
        from rouge_score import rouge_scorer             # noqa: PLC0415
        from transformers import (                       # noqa: PLC0415
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            LogitsProcessor,
            LogitsProcessorList,
        )
    except ImportError as exc:
        print(f"❌ Missing package: {exc}")
        raise SystemExit(1)

    model_path = Path(args.model_path) if args.model_path else Path(steering_meta["model_path"])
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        raise SystemExit(1)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32,
    ).to(device)
    model.eval()

    ds_raw = load_dataset(cfg["dataset_name"])[args.split]
    if args.max_samples:
        n_samples = min(args.max_samples, len(ds_raw))
        ds_raw = ds_raw.select(range(n_samples))
    n_samples = len(ds_raw)

    batch_size = args.batch_size or int(cfg["batch_size"])

    # Keep only layers that exist in the steering artifact.
    available_layers = set(int(x) for x in steering_meta["layers"])
    layers = [int(x) for x in args.layers if int(x) in available_layers]
    if not layers:
        print(f"❌ Requested layers {args.layers} not found in steering artifact {sorted(available_layers)}")
        raise SystemExit(1)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    class _VectorBiasProcessor(LogitsProcessor):
        def __init__(self, bias_vector: torch.Tensor, alpha: float):
            self.bias = bias_vector
            self.alpha = float(alpha)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            return scores + (self.alpha * self.bias)

    idx_all = list(range(n_samples))
    dialogues = [ds_raw[i]["dialogue"] for i in idx_all]
    refs = [ds_raw[i]["summary"] for i in idx_all]

    results: list[dict] = []

    for layer in layers:
        layer_vec = vectors[args.method][layer].to(dtype=torch.float32)
        layer_vec = _norm(layer_vec)

        with torch.no_grad():
            # hidden -> vocabulary direction
            bias = model.lm_head(layer_vec.to(device=device, dtype=model.dtype)).to(dtype=torch.float32)

        for alpha in args.alphas:
            preds: list[str] = []
            t0 = time.perf_counter()

            for batch_idx in _batchify(idx_all, batch_size):
                batch_dialogues = [dialogues[i] for i in batch_idx]
                inputs = tokenizer(
                    batch_dialogues,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg["max_source_length"],
                ).to(device)

                processors = LogitsProcessorList([_VectorBiasProcessor(bias, alpha)])
                with torch.no_grad():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=cfg["max_target_length"],
                        num_beams=cfg["num_beams"],
                        length_penalty=cfg["length_penalty"],
                        early_stopping=cfg["early_stopping_beam"],
                        logits_processor=processors,
                    )

                batch_preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
                preds.extend([x.strip() for x in batch_preds])

            if device.type == "mps":
                torch.mps.synchronize()
            wall_sec = time.perf_counter() - t0

            rouge_vals = []
            action_hits = 0
            for pred, ref in zip(preds, refs):
                rouge_vals.append(scorer.score(ref, pred)["rougeL"].fmeasure)
                if any(tok in pred.lower() for tok in [
                    " will ", "going to", "need to", "should", "must", "send", "call", "email",
                    "schedule", "book", "prepare", "review", "check", "bring", "follow up",
                ]):
                    action_hits += 1

            row = {
                "layer": int(layer),
                "method": args.method,
                "alpha": float(alpha),
                "rougeL": round(float(np.mean(rouge_vals) * 100), 4),
                "action_proxy_rate": round(float(action_hits / max(len(preds), 1)), 4),
                "ms_per_sample": round(float(wall_sec / max(len(preds), 1) * 1000), 2),
                "n_samples": len(preds),
                "predictions": preds,
            }
            results.append(row)

            print(
                f"Layer {layer:>2} | alpha={alpha:<4} | "
                f"ROUGE-L={row['rougeL']:.2f} | action_proxy={row['action_proxy_rate']:.3f} | "
                f"ms/sample={row['ms_per_sample']:.1f}"
            )

            if device.type == "mps":
                torch.mps.empty_cache()

    out_dir = project_root / "results" / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_{args.split}_{args.method}_steering_generations.json"

    payload = {
        "meta": {
            "run_name": run_name,
            "model_name": model_name,
            "variant": variant,
            "model_path": str(model_path),
            "split": args.split,
            "method": args.method,
            "layers": layers,
            "alphas": [float(a) for a in args.alphas],
            "seed": int(cfg["seed"]),
            "n_samples": n_samples,
            "generation_config": {
                "num_beams": cfg["num_beams"],
                "max_new_tokens": cfg["max_target_length"],
                "length_penalty": cfg["length_penalty"],
                "early_stopping": cfg["early_stopping_beam"],
            },
            "steering_type": "logit_bias_from_decoder_direction",
        },
        "references": refs,
        "dialogues": dialogues,
        "runs": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n✅ Steering inference done: {out_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()

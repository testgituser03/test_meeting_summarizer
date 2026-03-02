#!/usr/bin/env python3
"""
evaluate_faithfulness.py — Experiment 4: hallucination and speaker preservation.

Metrics computed:
  1. Hallucination rate — named entities in summary not found in source dialogue
     (via spaCy NER en_core_web_sm)
  2. Speaker preservation rate — speakers named in dialogue that appear in summary
     (regex extraction + substring match)
  3. NLI entailment score — fraction of summary sentences entailed by the
     source dialogue via cross-encoder/nli-deberta-v3-small

Output: results/metrics/faithfulness_report.json

Usage:
  python3 scripts/evaluate_faithfulness.py
  python3 scripts/evaluate_faithfulness.py --n_samples 100   # quick run
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_speakers(dialogue: str) -> set[str]:
    """Extract unique speaker names from 'Name: text' format."""
    return {
        m.group(1).strip().lower()
        for line in dialogue.strip().split("\n")
        if (m := re.match(r"^([^\n:]+):", line))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="E4: faithfulness evaluation")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Number of test examples (None = full 819)")
    parser.add_argument("--skip_nli",   action="store_true",
                        help="Skip NLI entailment scoring (saves ~2 min, ~440 MB)")
    args = parser.parse_args()

    cfg          = load_config(args.config)
    project_root = Path(args.config).parent if "/" in args.config else Path.cwd()
    MODEL_NAME   = cfg["model_name"]
    VARIANT      = cfg["dataset_variant"]
    run_name     = f"{MODEL_NAME.replace('/', '_')}_{VARIANT}"

    model_path = Path(args.model_path) if args.model_path else (
        project_root / "models" / "best" / run_name
    )
    if not model_path.exists():
        print(f"  ❌  Model not found: {model_path}\n  Run train.py first.")
        sys.exit(1)

    try:
        import spacy                                                        # noqa: PLC0415
        from datasets import load_dataset                                   # noqa: PLC0415
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM      # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  {exc}"); sys.exit(1)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  ❌  spaCy model missing. Run: python3 -m spacy download en_core_web_sm")
        sys.exit(1)

    device    = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,
    ).to(device)
    model.eval()

    ds = load_dataset("knkarthick/samsum", split="test")
    if args.n_samples:
        ds = ds.select(range(min(args.n_samples, len(ds))))

    print(f"\n  Faithfulness evaluation on {len(ds):,} examples...\n")

    hallucination_flags: list[bool] = []
    speaker_preservation: list[float] = []

    for ex in ds:
        dialogue = ex["dialogue"]
        inputs   = tokenizer(dialogue, return_tensors="pt",
                             max_length=cfg["max_source_length"], truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens  = cfg["max_target_length"],
                num_beams       = cfg["num_beams"],
                length_penalty  = cfg["length_penalty"],
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # ── Hallucination: NER entities in summary not present in dialogue ──
        dialogue_lower = dialogue.lower()
        doc = nlp(summary)
        ents = [ent.text.lower() for ent in doc.ents]
        if ents:
            hallucinated = sum(1 for e in ents if e not in dialogue_lower)
            hallucination_flags.append(hallucinated > 0)
        else:
            hallucination_flags.append(False)

        # ── Speaker preservation ─────────────────────────────────────────
        speakers = extract_speakers(dialogue)
        if speakers:
            summary_lower = summary.lower()
            preserved = sum(1 for s in speakers if s in summary_lower)
            speaker_preservation.append(preserved / len(speakers))

        if device.type == "mps":
            torch.mps.empty_cache()

    n = len(hallucination_flags)
    hall_rate      = sum(hallucination_flags) / n
    avg_speaker_pr = sum(speaker_preservation) / len(speaker_preservation) if speaker_preservation else 0.0

    report: dict = {
        "n_samples":               n,
        "hallucination_rate":      round(hall_rate, 4),
        "speaker_preservation_rate": round(avg_speaker_pr, 4),
        "model":                   MODEL_NAME,
        "variant":                 VARIANT,
    }

    # ── Optional NLI entailment ───────────────────────────────────────────
    if not args.skip_nli:
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

            scores_nli: list[float] = []
            for ex in ds:
                dialogue = ex["dialogue"]
                inputs   = tokenizer(dialogue, return_tensors="pt",
                                     max_length=cfg["max_source_length"], truncation=True).to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=cfg["max_target_length"],
                                        num_beams=cfg["num_beams"])
                summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()

                sentences = [s.strip() for s in re.split(r"[.!?]", summary) if s.strip()]
                if not sentences:
                    continue
                pairs     = [(dialogue, s) for s in sentences]
                logits    = nli_model.predict(pairs, apply_softmax=True)
                entailed  = sum(1 for l in logits if l.argmax() == 2)   # 2 = entailment
                scores_nli.append(entailed / len(sentences))

            report["nli_faithfulness_score"] = round(sum(scores_nli) / len(scores_nli), 4)
        except ImportError:
            report["nli_faithfulness_score"] = "skipped (sentence-transformers not installed)"
    else:
        report["nli_faithfulness_score"] = "skipped (--skip_nli)"

    out_path = project_root / "results" / "metrics" / "faithfulness_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"  Hallucination rate      : {hall_rate:.1%}")
    print(f"  Speaker preservation    : {avg_speaker_pr:.1%}")
    if "nli_faithfulness_score" in report and isinstance(report["nli_faithfulness_score"], float):
        print(f"  NLI faithfulness score  : {report['nli_faithfulness_score']:.4f}")
    print(f"\n  ✅  Saved → {out_path.relative_to(project_root)}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
evaluate_faithfulness.py — Experiment 4: faithfulness evaluation.

What ROUGE cannot measure — factual accuracy of generated summaries.

Four metrics, computed once over the full 819-example SAMSum test set:

  1. Hallucination rate (spaCy NER)
     Named entities that appear in the generated summary but are absent
     from the source dialogue (case-insensitive substring match).
     hallucination_rate = examples_with_≥1_hallucination / 819

  2. Speaker preservation rate (regex)
     Fraction of speakers named in the dialogue that are mentioned in the
     generated summary. Aggregated as avg_speaker_preservation across all
     examples that have at least one identified speaker.

  3. NLI faithfulness (cross-encoder/nli-deberta-v3-small, CPU only)
     For each summary sentence, check whether the source dialogue entails
     it (label index 2 = entailment, 0 = contradiction, 1 = neutral).
     avg_nli_faithfulness = mean(entailed_sentences / total_sentences)
     ⚠️  NLI model MUST run on CPU — MPS has unsupported ops for DeBERTa.
     torch.mps.empty_cache() is called after all 819 summaries are
     generated and BEFORE the NLI model is loaded.

  4. Length–ROUGE-L correlation (Pearson r)
     Pearson r between generated summary word count and per-example
     ROUGE-L score. r > 0.4 indicates length bias.

Execution order:
  Phase 1 — BART-base on MPS:  generate all 819 summaries in batches.
  torch.mps.empty_cache()       ← explicit MPS drain before NLI load
  Phase 2 — NLI on CPU:        score entailment sentence-by-sentence.
  Phase 3 — Write JSON report.

Output schema (results/metrics/faithfulness_report.json):
  {
    "model": "facebook/bart-base",
    "variant": "with_speakers",
    "n_samples": 819,
    "hallucination_rate": float,
    "avg_speaker_preservation": float,
    "avg_nli_faithfulness": float,
    "length_rouge_correlation": float,
    "evaluation_timestamp": "ISO8601"
  }

Usage:
  python3 scripts/evaluate_faithfulness.py
  python3 scripts/evaluate_faithfulness.py --model_path models/best/facebook_bart-base_with_speakers
  python3 scripts/evaluate_faithfulness.py --n_samples 50   # quick smoke-test
  python3 scripts/evaluate_faithfulness.py --skip_nli       # skip ~10 min NLI pass
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

# ── sys.path: prevent scripts/evaluate.py from shadowing HuggingFace evaluate ──
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _SCRIPT_DIR:
    sys.path.pop(0)

os_env_set = False
try:
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # NOTE: do NOT set TRANSFORMERS_OFFLINE=1 here.
    # The NLI cross-encoder is loaded from HF hub cache (not from local disk),
    # so Transformers must be allowed to resolve the cached snapshot path.
    # The BART checkpoint is loaded via from_pretrained(local_path), so it does
    # not need network access either.
    os_env_set = True
except Exception:
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_speakers(dialogue: str) -> list[str]:
    """Return unique speaker names (lowercased) from 'Name: utterance' lines."""
    seen: dict[str, None] = {}
    for line in dialogue.strip().split("\n"):
        m = re.match(r"^(\w+):", line)
        if m:
            seen[m.group(1).lower()] = None
    return list(seen)


def _pad_batch(sequences: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to max length; return (input_ids, attention_mask)."""
    max_len = max(len(s) for s in sequences)
    ids   = torch.tensor([s + [pad_id] * (max_len - len(s)) for s in sequences], dtype=torch.long)
    mask  = (ids != pad_id).long()
    return ids, mask


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="E4: faithfulness evaluation (819 examples)")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--model_path", default=None,
                        help="Override checkpoint path. Default: models/best/<model>_with_speakers")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Subsample test set (default: full 819)")
    parser.add_argument("--skip_nli",   action="store_true",
                        help="Skip NLI pass (~10 min). Useful for quick smoke-test.")
    args = parser.parse_args()

    cfg          = _load_config(args.config)
    project_root = Path.cwd()
    MODEL_NAME   = cfg["model_name"]

    # E4 ALWAYS uses with_speakers checkpoint — the best model from E1
    e4_run_name = f"{MODEL_NAME.replace('/', '_')}_with_speakers"
    model_path  = (
        Path(args.model_path) if args.model_path
        else project_root / "models" / "best" / e4_run_name
    )
    if not model_path.exists():
        print(f"  ❌  Checkpoint not found: {model_path}")
        print(f"       Run: python3 scripts/train.py")
        sys.exit(1)

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        import spacy                                                   # noqa: PLC0415
        from datasets import load_from_disk                            # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
        from rouge_score import rouge_scorer as rs                     # noqa: PLC0415
    except ImportError as exc:
        print(f"  ❌  Missing dependency: {exc}")
        sys.exit(1)

    # ── spaCy model ────────────────────────────────────────────────────────────
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  ❌  spaCy model missing.")
        print("       Run: python3 -m spacy download en_core_web_sm")
        sys.exit(1)

    # ── Device + generation model (MPS) ───────────────────────────────────────
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\n  ─── Experiment 4: Faithfulness Evaluation ───────────────────────────")
    print(f"  Device     : {device}  (generation)")
    print(f"  Checkpoint : {model_path}")
    print(f"  NLI model  : CPU only (MPS excluded — DeBERTa has unsupported ops)")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16 if cfg["use_bf16"] else torch.float32,  # Transformers 5.x API
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M  |  BF16: {cfg['use_bf16']}")

    # ── Dataset — always use with_speakers tokenized cache for E4 ─────────────
    dataset_cache = (
        project_root / "data" / "cache"
        / f"samsum_with_speakers_{MODEL_NAME.replace('/', '_')}"
    )
    if not dataset_cache.exists():
        print(f"  ❌  Tokenized dataset not found: {dataset_cache}")
        print(f"       Run: python3 scripts/preprocess.py")
        sys.exit(1)

    ds = load_from_disk(str(dataset_cache))["test"]
    if args.n_samples:
        ds = ds.select(range(min(args.n_samples, len(ds))))

    n          = len(ds)
    batch_size = cfg["batch_size"]
    max_tokens = cfg["max_target_length"]
    num_beams  = cfg["num_beams"]
    lp         = cfg["length_penalty"]
    scorer     = rs.RougeScorer(["rougeL"], use_stemmer=True)

    print(f"  Test set   : {n:,} examples")
    print(f"\n  ── Phase 1: Generating all {n:,} summaries on {device} ──────────────")

    # ── Storage vectors ────────────────────────────────────────────────────────
    all_dialogues:  list[str]   = []
    all_references: list[str]   = []
    all_summaries:  list[str]   = []
    all_rouge_l:    list[float] = []
    all_sum_words:  list[int]   = []

    hallucination_flags:  list[bool]  = []   # True = ≥1 hallucinated entity
    hall_counts:          list[int]   = []   # total hallucinated entities
    hall_gen_ent_counts:  list[int]   = []   # total generated entities
    hall_entities_lists:  list[list]  = []   # per-example hallucinated entity strings
    speaker_pres_rates:   list[float] = []   # per-example speaker preservation

    # ── Phase 1: batch generation ──────────────────────────────────────────────
    for i in range(0, n, batch_size):
        batch_slice = ds[i : i + batch_size]

        # dialogues come from the raw text field — retrieve from HF dataset
        # The tokenized cache stores input_ids; for NER we need raw text.
        # SAMSum raw text is stored in the `dialogue` column of the cache.
        raw_dialogues = batch_slice["dialogue"] if "dialogue" in ds.column_names else []
        raw_summaries = batch_slice["summary"]  if "summary"  in ds.column_names else []

        input_ids, attention_mask = _pad_batch(
            batch_slice["input_ids"], tokenizer.pad_token_id
        )
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                num_beams=num_beams,
                length_penalty=lp,
                early_stopping=True,
            )
            if device.type == "mps":
                torch.mps.synchronize()

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Reference decode from cached labels (variable-length, replace -100)
        refs = tokenizer.batch_decode(
            [
                [tokenizer.pad_token_id if t == -100 else t for t in seq]
                for seq in batch_slice["labels"]
            ],
            skip_special_tokens=True,
        )

        for j, (pred, ref) in enumerate(zip(preds, refs)):
            pred = pred.strip()
            ref  = ref.strip()

            # dialogue text: use raw column if available, else decode input_ids
            if raw_dialogues:
                dlg = raw_dialogues[j]
            else:
                dlg = tokenizer.decode(batch_slice["input_ids"][j], skip_special_tokens=True)

            ref_str = raw_summaries[j] if raw_summaries else ref

            all_dialogues.append(dlg)
            all_references.append(ref_str)
            all_summaries.append(pred)

            # Per-example ROUGE-L
            rl = scorer.score(ref, pred)["rougeL"].fmeasure * 100
            all_rouge_l.append(rl)
            all_sum_words.append(len(pred.split()))

            # ── Hallucination (NER) ────────────────────────────────────────────
            dlg_lower = dlg.lower()
            doc = nlp(pred)
            gen_ents  = [ent.text for ent in doc.ents]
            hall_ents = [e for e in gen_ents if e.lower() not in dlg_lower]
            hallucination_flags.append(len(hall_ents) > 0)
            hall_counts.append(len(hall_ents))
            hall_gen_ent_counts.append(len(gen_ents))
            hall_entities_lists.append(hall_ents)

            # ── Speaker preservation ───────────────────────────────────────────
            speakers   = _extract_speakers(dlg)
            if speakers:
                pred_lower = pred.lower()
                preserved  = sum(1 for s in speakers if s in pred_lower)
                speaker_pres_rates.append(preserved / len(speakers))

        if (i // batch_size + 1) % 10 == 0:
            done = min(i + batch_size, n)
            print(f"    {done:>4}/{n}  ({done/n:.0%})")

    print(f"    {n}/{n}  (100%)  — generation complete")

    # ── torch.mps.empty_cache() — free MPS memory BEFORE loading NLI model ────
    if device.type == "mps":
        torch.mps.empty_cache()
        print(f"\n  ✅  torch.mps.empty_cache() called — MPS memory freed before NLI load")

    # ── Aggregate generation-phase metrics ─────────────────────────────────────
    hall_rate     = sum(hallucination_flags) / n
    avg_spk_pres  = (
        sum(speaker_pres_rates) / len(speaker_pres_rates)
        if speaker_pres_rates else 0.0
    )

    # ── Length–ROUGE-L Pearson r ───────────────────────────────────────────────
    from scipy.stats import pearsonr  # noqa: PLC0415
    r_coef, _ = pearsonr(all_sum_words, all_rouge_l)
    length_rouge_r = round(float(r_coef), 4)

    print(f"\n  ── Phase 1 Results ───────────────────────────────────────────────────")
    print(f"  Hallucination rate      : {hall_rate:.1%}  ({sum(hallucination_flags)}/{n} examples with ≥1 hallucinated entity)")
    print(f"  Avg speaker preservation: {avg_spk_pres:.1%}")
    print(f"  Length–ROUGE-L Pearson r: {length_rouge_r:+.4f}")

    # ── Phase 2: NLI faithfulness on CPU ──────────────────────────────────────
    avg_nli = None
    if not args.skip_nli:
        print(f"\n  ── Phase 2: NLI faithfulness scoring on CPU ──────────────────────────")
        print(f"  Loading cross-encoder/nli-deberta-v3-small (~85 MB, CPU) ...")
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            # ⚠️  device=cpu is MANDATORY — DeBERTa-v3 uses ops unsupported on MPS
            nli_model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small",
                device="cpu",
                max_length=512,
            )
            print(f"  NLI model loaded on CPU")

            nli_scores: list[float] = []
            # Score in small sub-batches to avoid RAM spikes
            NLI_BATCH = 32

            for i, (dlg, summ) in enumerate(zip(all_dialogues, all_summaries)):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summ) if s.strip()]
                if not sentences:
                    nli_scores.append(0.0)
                    continue
                # Each pair: (premise=dialogue, hypothesis=summary_sentence)
                pairs   = [(dlg[:1024], s) for s in sentences]  # truncate long dialogues for NLI
                # predict returns array of shape (N, 3): [contradiction, neutral, entailment]
                probs   = nli_model.predict(pairs, batch_size=NLI_BATCH, apply_softmax=True)
                entailed = sum(1 for p in probs if p.argmax() == 2)  # index 2 = entailment
                nli_scores.append(entailed / len(sentences))

                if (i + 1) % 100 == 0:
                    print(f"    NLI: {i+1:>4}/{n}  ({(i+1)/n:.0%})  running avg={sum(nli_scores)/len(nli_scores):.3f}")

            avg_nli = round(sum(nli_scores) / len(nli_scores), 4)
            print(f"\n  NLI faithfulness score  : {avg_nli:.4f}  (1.0 = all sentences entailed)")

        except ImportError:
            print("  ⚠️  sentence-transformers not installed. Run: pip install sentence-transformers")
            avg_nli = None
    else:
        print("\n  ── Phase 2: NLI skipped (--skip_nli) ──────────────────────────────────")

    # ── Write faithfulness_report.json ────────────────────────────────────────
    report = {
        "model":                    MODEL_NAME,
        "variant":                  "with_speakers",
        "n_samples":                n,
        "hallucination_rate":       round(hall_rate, 4),
        "avg_speaker_preservation": round(avg_spk_pres, 4),
        "avg_nli_faithfulness":     avg_nli if avg_nli is not None else "skipped",
        "length_rouge_correlation": length_rouge_r,
        "evaluation_timestamp":     datetime.now(timezone.utc).isoformat(),
        # --- supplementary detail (not in required schema, but useful) ---
        "hallucinated_examples_count": int(sum(hallucination_flags)),
        "examples_with_speakers":      len(speaker_pres_rates),
    }

    out_dir  = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "faithfulness_report.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  ──────────────────────────────────────────────────────────────────────")
    print(f"  ✅  Report saved → {out_path.relative_to(project_root)}")
    print(f"  ──────────────────────────────────────────────────────────────────────\n")
    print(f"  Summary")
    print(f"  ───────────────────────────────────────────────────")
    print(f"  Hallucination rate        : {hall_rate:.1%}")
    print(f"  Avg speaker preservation  : {avg_spk_pres:.1%}")
    print(f"  Avg NLI faithfulness      : {avg_nli if avg_nli is not None else 'skipped'}")
    print(f"  Length–ROUGE-L Pearson r  : {length_rouge_r:+.4f}")
    print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
app.py — Production Streamlit demo for the Meeting Summarizer.

Features:
  - @st.cache_resource model loading — loaded once per session, never on click
  - BF16 inference on Apple MPS (or CPU fallback)
  - Two-column layout: left = dialogue + generation settings; right = results
  - Generation settings expander: beam width slider (1–8) + length penalty selectbox
  - Regex action-item extraction (modal + action-verb patterns), deduplicated, capped at 5
  - spaCy en_core_web_sm named entity recognition, shown as st.metric cards
  - st.json generation info block with latency (torch.mps.synchronize() before stop)
  - CC BY-NC-ND 4.0 disclaimer in footer
  - Graceful handling of empty / whitespace-only input

Usage:
  streamlit run scripts/app.py
  # Opens http://localhost:8501
"""

import re
import time
from pathlib import Path

import streamlit as st
import torch
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

# ── Available model checkpoints (discovered dynamically) ───────────────────────
_MODEL_REGISTRY: dict[str, str] = {}

def _discover_models() -> dict[str, str]:
    """Scan models/best/ for available checkpoints and return {label: path}."""
    best_dir = PROJECT_ROOT / "models" / "best"
    registry: dict[str, str] = {}
    if not best_dir.exists():
        return registry

    # Priority order of model checkpoints
    candidates = [
        ("BART-base (with_speakers)", "facebook_bart-base_with_speakers"),
        ("BART-base LoRA",            "facebook_bart-base_lora"),
        ("BART-base (no_speakers)",   "facebook_bart-base_no_speakers"),
        ("T5-small (with_speakers)",  "t5-small_with_speakers"),
        ("PEGASUS (with_speakers)",   "google_pegasus-cnn_dailymail_with_speakers"),
    ]
    for label, dirname in candidates:
        ckpt_path = best_dir / dirname
        if ckpt_path.exists():
            registry[label] = str(ckpt_path)
    return registry

# ── Regex patterns (per spec) ──────────────────────────────────────────────────
_MODAL_PATTERN  = r"\b(will|going to|needs? to|should|must|have to)\s+\w[\w\s]{4,40}"
_ACTION_PATTERN = r"\b(send|call|email|schedule|book|prepare|review|check|bring)\s+\w[\w\s]{3,35}"

# ── Sample dialogue pre-loaded into text area ──────────────────────────────────
_SAMPLE_DIALOGUE = """\
Amanda: I baked cookies. Do you want some?
Jerry: Sure!
Amanda: I'll bring you tomorrow :-)
Jerry: Thanks! Do you know how to make the lemon ones?
Amanda: The biscuits?
Jerry: Yeah.
Amanda: I'll send you the recipe. It's easy!\
"""


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"max_source_length": 512, "max_target_length": 128, "use_bf16": True}


# ── Cached resource loaders (one call per Streamlit session) ───────────────────

@st.cache_resource(show_spinner="⏳ Loading model weights (once per session)…")
def _load_model(model_path: str):
    """Load tokenizer + model onto MPS (or CPU). Called once per model; result is reused."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415

    cfg    = _load_config()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype  = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32

    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype).to(device)
    mdl.eval()
    return tok, mdl, device, dtype, cfg


@st.cache_resource(show_spinner="⏳ Loading spaCy model…")
def _load_nlp():
    """Load spaCy en_core_web_sm once per session."""
    import spacy  # noqa: PLC0415
    return spacy.load("en_core_web_sm")


# ── Inference ──────────────────────────────────────────────────────────────────

def _generate(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    cfg: dict,
    num_beams: int,
    length_penalty: float,
) -> tuple[str, float]:
    """Return (summary_text, latency_ms).

    torch.mps.synchronize() is called BEFORE stopping the timer so the
    latency figure includes all MPS kernel execution, not just dispatch.
    """
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=cfg["max_source_length"],
        truncation=True,
    ).to(device)

    t_start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = cfg["max_target_length"],
            num_beams      = num_beams,
            length_penalty = length_penalty,
            early_stopping = True,
        )
    # Synchronize before reading the clock — required for accurate MPS timing
    if device.type == "mps":
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    summary = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return summary, latency_ms


# ── Action-item extraction ─────────────────────────────────────────────────────

def _extract_action_items(text: str) -> list[str]:
    """Extract action items using modal-verb + action-verb patterns.

    Deduplicates (case-insensitive) and caps at 5 results.
    """
    raw: list[str] = []
    for pat in [_MODAL_PATTERN, _ACTION_PATTERN]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            phrase = m.group(0).strip().rstrip(".,;:")
            words  = phrase.split()
            if 3 <= len(words) <= 12:
                raw.append(phrase[0].upper() + phrase[1:])

    # Deduplicate preserving first-seen order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in raw:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped[:5]


# ── Main UI ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Meeting Summarizer",
        page_icon  ="📝",
        layout     ="wide",
    )

    st.title("📝 Meeting Summarizer")
    st.caption(
        "Fine-tuned seq2seq models on SAMSum · "
        "Apple M4 Pro · BF16 / MPS"
    )

    # ── Sidebar: model selector ───────────────────────────────────────────
    models = _discover_models()
    if not models:
        st.error(
            "No model checkpoints found in `models/best/`. "
            "Run `python3 scripts/train.py` first, then restart the app."
        )
        st.stop()

    with st.sidebar:
        st.header("🔧 Configuration")
        model_label = st.selectbox(
            "Model",
            options=list(models.keys()),
            index=0,
            help="Select a trained checkpoint to use for inference.",
        )
        model_path = models[model_label]

        st.divider()
        st.caption(f"**Checkpoint:** `{Path(model_path).name}`")
        st.caption(f"**Available models:** {len(models)}")

    # Load resources (cached per model_path; no-op on subsequent clicks)
    tokenizer, model, device, dtype, cfg = _load_model(model_path)
    nlp = _load_nlp()

    # ── Two equal columns ─────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    # ── Left column: input + settings ────────────────────────────────────
    with col1:
        st.subheader("💬 Dialogue Input")
        dialogue = st.text_area(
            label       = "Paste or type a conversation:",
            value       = _SAMPLE_DIALOGUE,
            height      = 300,
            placeholder = "Enter dialogue here…",
        )

        with st.expander("⚙️ Generation Settings", expanded=False):
            num_beams = st.slider(
                "Beam width",
                min_value = 1,
                max_value = 8,
                value     = 5,
                help      = "Higher → better quality, slower. E3 champion uses beam=5 (D27: ROUGE-L 40.12).",
            )
            length_penalty = st.slider(
                "Length penalty",
                min_value = 0.80,
                max_value = 1.50,
                value     = 1.33,
                step      = 0.01,
                help      = (
                    "< 1.0 favours shorter outputs; > 1.0 favours longer outputs. "
                    "D27 champion: lp=1.33. Plateau of ROUGE-L ≥ 40.0 spans lp∈[1.28, 1.45]."
                ),
            )

        summarize_btn = st.button(
            "▶  Summarize",
            type             = "primary",
            use_container_width = True,
        )

    # ── Right column: results ─────────────────────────────────────────────
    with col2:
        st.subheader("📋 Results")

        if summarize_btn:
            # Guard: empty / whitespace-only input
            if not dialogue or not dialogue.strip():
                st.warning("⚠️  Please enter a dialogue before clicking Summarize.")
                st.stop()

            with st.spinner("Generating summary…"):
                summary, latency_ms = _generate(
                    dialogue, tokenizer, model, device, cfg,
                    num_beams      = num_beams,
                    length_penalty = float(length_penalty),
                )

            # ── Summary ──────────────────────────────────────────────────
            st.markdown("**Summary**")
            st.success(summary)

            # ── Action items ──────────────────────────────────────────────
            # Extract from summary only (not dialogue) to avoid false positives
            # from modal verbs in quoted speech
            action_items = _extract_action_items(summary)
            st.markdown("**🗒️ Action Items**")
            if action_items:
                for item in action_items:
                    st.markdown(f"- {item}")
            else:
                st.caption("_No action items detected._")

            # ── Named entities (spaCy) ────────────────────────────────────
            doc      = nlp(summary)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                st.markdown("**🏷️ Named Entities**")
                # Up to 4 per row
                ncols    = min(len(entities), 4)
                ent_cols = st.columns(ncols)
                for i, (ent_text, ent_label) in enumerate(entities):
                    with ent_cols[i % ncols]:
                        st.metric(label=ent_label, value=ent_text)

            # ── Generation info ───────────────────────────────────────────
            st.markdown("**ℹ️ Generation Info**")
            st.json({
                "model"         : model_label,
                "checkpoint"    : Path(model_path).name,
                "device"        : str(device),
                "dtype"         : str(dtype),
                "num_beams"     : num_beams,
                "length_penalty": float(length_penalty),
                "latency_ms"    : round(latency_ms, 1),
            })

        else:
            st.info("Enter a dialogue on the left and click **▶  Summarize**.")

    # ── Footer disclaimer ─────────────────────────────────────────────────
    st.divider()
    st.caption(
        "📄 **Dataset**: SAMSum "
        "([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) "
        "— non-commercial use only). "
        "Model weights: `facebook/bart-base` fine-tuned on SAMSum. "
        "Hardware: Apple M4 Pro · MPS / BF16."
    )


if __name__ == "__main__":
    main()

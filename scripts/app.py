#!/usr/bin/env python3
"""
app.py — Streamlit demo for the Meeting Summarizer.

Features:
  - Paste any conversation → generate summary with configurable beam search
  - Speaker tag preservation toggle (E2 insight)
  - Regex-based action-item extraction
  - Inline model card with CC BY-NC-ND 4.0 license notice

Usage:
  streamlit run scripts/app.py
  # Opens http://localhost:8501 in your browser
"""

import re
import sys
from pathlib import Path

import streamlit as st
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best" / "facebook_bart-base_with_speakers"

ACTION_PATTERNS = [
    r"\b(will|going to|needs? to|should|must|have to)\s+(?:\w+\s){1,8}\w+",
    r"\b(send|call|email|schedule|book|prepare|review|check|bring|follow up|look into)\s+(?:\w+\s){0,6}\w+",
]

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"max_source_length": 512, "max_target_length": 128, "use_bf16": True}


# ── Model loading (cached so Streamlit doesn't reload on every interaction) ────

@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa: PLC0415
    cfg    = load_config()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    tok    = AutoTokenizer.from_pretrained(model_path)
    mdl    = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if cfg.get("use_bf16") else torch.float32,
    ).to(device)
    mdl.eval()
    return tok, mdl, device, cfg


# ── Inference ──────────────────────────────────────────────────────────────────

def summarize(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    cfg: dict,
    num_beams: int,
    length_penalty: float,
) -> str:
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=cfg["max_source_length"],
        truncation=True,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens  = cfg["max_target_length"],
            num_beams       = num_beams,
            length_penalty  = length_penalty,
            early_stopping  = True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def extract_action_items(text: str) -> list[str]:
    items: list[str] = []
    for pat in ACTION_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            item = m.group(0).strip().rstrip(".,;")
            if 4 <= len(item.split()) <= 12:
                items.append(item[0].upper() + item[1:])
    return list(dict.fromkeys(items))[:5]   # deduplicate, cap at 5


# ── UI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Meeting Summarizer",
        page_icon="📝",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        model_path = st.text_input(
            "Model path",
            value=str(DEFAULT_MODEL_PATH),
            help="Path to a saved model directory (models/best/…)",
        )
        num_beams      = st.slider("Beam width",      min_value=1, max_value=8, value=4)
        length_penalty = st.slider("Length penalty",  min_value=0.5, max_value=2.0, value=1.0, step=0.1)
        show_tokens    = st.checkbox("Show token count", value=False)

        st.divider()
        st.caption(
            "⚠️ **License**: This demo uses the SAMSum dataset "
            "([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)). "
            "Non-commercial use only."
        )
        st.caption("Hardware: Apple M4 Pro · MPS / BF16")

    # ── Header ────────────────────────────────────────────────────────────
    st.title("📝 Meeting Summarizer")
    st.caption("Fine-tuned BART-base on SAMSum · Apple M4 Pro · BF16 MPS")

    # ── Load model ────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        st.warning(
            f"Model not found at `{model_path}`. "
            "Run `python3 scripts/train.py` first, then restart the app."
        )
        st.stop()

    tokenizer, model, device, cfg = load_model(model_path)

    # ── Main layout ───────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    SAMPLE = (
        "Amanda: I baked cookies. Do you want some?\n"
        "Jerry: Sure!\n"
        "Amanda: I'll bring you tomorrow :-)\n"
        "Jerry: Thanks! Do you know how to make the lemon ones?\n"
        "Amanda: The biscuits?\n"
        "Jerry: Yeah.\n"
        "Amanda: I'll send you the recipe. It's easy!"
    )

    with col1:
        st.subheader("Conversation")
        dialogue = st.text_area("Paste dialogue here", value=SAMPLE, height=300)
        if show_tokens and dialogue:
            n_tok = len(tokenizer(dialogue)["input_ids"])
            st.caption(f"Input tokens: {n_tok} / {cfg['max_source_length']}")
        summarize_btn = st.button("Summarize →", type="primary", use_container_width=True)

    with col2:
        st.subheader("Summary")
        if summarize_btn and dialogue.strip():
            with st.spinner("Generating…"):
                summary = summarize(
                    dialogue, tokenizer, model, device, cfg,
                    num_beams=num_beams, length_penalty=length_penalty
                )
            st.success(summary)

            action_items = extract_action_items(summary + " " + dialogue)
            if action_items:
                st.subheader("🗒️ Action Items (extracted)")
                for item in action_items:
                    st.markdown(f"- {item}")
        elif summarize_btn:
            st.warning("Please enter a dialogue first.")


if __name__ == "__main__":
    main()

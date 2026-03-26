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

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import yaml

# ── Task 5 structured schema (same module as offline eval / packaging) ───────
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "task5_lora_structured",
    _SCRIPTS_DIR / "task5_lora_structured.py",
)
_task5 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_task5)
structured_dict_from_model_output = _task5.structured_dict_from_model_output
structured_input_text = _task5.structured_input_text
structured_decoder_input_ids = _task5.structured_decoder_input_ids
STRUCTURED_SCHEMA = _task5.STRUCTURED_SCHEMA
TASK_PREFIX_T5 = _task5.TASK_PREFIX

HF_TASK5_REPO_ID = os.getenv("HF_TASK5_REPO_ID", "saione/meeting-summarizer-dev")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

# ── Available model checkpoints (discovered dynamically) ───────────────────────
_MODEL_REGISTRY: dict[str, str] = {}


def _is_hf_repo_id(model_path: str) -> bool:
    """Best-effort check: hub IDs look like 'owner/repo' and are not local dirs."""
    return "/" in model_path and not Path(model_path).exists()


def _force_hf_online_mode() -> None:
    """Ensure hub downloads are allowed even if shell exported offline flags."""
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"


def _hf_token() -> str | None:
    """Read Hugging Face token from Streamlit secrets or env (optional for public repos)."""
    try:
        token = st.secrets.get("HF_TOKEN")
        if token:
            return str(token)
    except Exception:
        pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN")


def _discover_extra_hf_models() -> dict[str, str]:
    """Load additional HF model options from env or Streamlit secrets.

    Supported env formats:
      1) HF_MODEL_OPTIONS_JSON='{"Label": "owner/repo", ...}'
      2) HF_EXTRA_MODELS='Label=owner/repo;Other=owner/repo2'
    """
    out: dict[str, str] = {}

    # 1. First consult Streamlit's native secrets dictionary (Best Practice for Cloud)
    raw_json = ""
    try:
        raw_json = str(st.secrets.get("HF_MODEL_OPTIONS_JSON", "")).strip()
    except Exception:
        pass

    # 2. Fallback to OS environment variables
    if not raw_json:
        raw_json = (os.getenv("HF_MODEL_OPTIONS_JSON") or "").strip()

    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for label, repo_id in parsed.items():
                    if isinstance(label, str) and isinstance(repo_id, str) and "/" in repo_id:
                        out[label.strip()] = repo_id.strip()
        except Exception:
            pass

    raw_pairs = (os.getenv("HF_EXTRA_MODELS") or "").strip()
    if raw_pairs:
        for item in raw_pairs.split(";"):
            item = item.strip()
            if not item or "=" not in item:
                continue
            label, repo_id = item.split("=", 1)
            label = label.strip()
            repo_id = repo_id.strip()
            if label and "/" in repo_id:
                out[label] = repo_id

    return out


def _hf_error_diagnosis(exc: Exception, repo_id: str) -> list[str]:
    """Convert HF loading exception into actionable diagnostics for Streamlit UI."""
    msg = str(exc).lower()
    hints: list[str] = []

    if "couldn't connect" in msg or "connection" in msg or "name or service not known" in msg:
        hints.append("Network egress from this runtime to huggingface.co is blocked or failing.")
    if "localentrynotfounderror" in msg or "outgoing traffic has been disabled" in msg:
        hints.append("Runtime is effectively offline for HF downloads (no cached files available).")
    if "401" in msg or "403" in msg or "forbidden" in msg or "unauthorized" in msg or "gated" in msg:
        hints.append("Repository access denied. Add a valid HF token in Streamlit Secrets as HF_TOKEN.")
    if "404" in msg or "not found" in msg or "repository not found" in msg:
        hints.append(f"Repository id appears invalid or missing: '{repo_id}'. Use full owner/repo.")

    if not hints:
        hints.append("Model download failed for an unspecified HF Hub reason. Check Streamlit app logs.")
    return hints

def _discover_models() -> dict[str, str]:
    """Scan models/best/ for available checkpoints and return {label: path}."""
    best_dir = PROJECT_ROOT / "models" / "best"
    registry: dict[str, str] = {}
    if best_dir.exists():
        # Priority order of model checkpoints
        candidates = [
            ("BART-base (with_speakers)", "facebook_bart-base_with_speakers"),
            ("BART-base LoRA",            "facebook_bart-base_lora"),
            ("BART-base (no_speakers)",   "facebook_bart-base_no_speakers"),
            ("FLAN-T5-base (with_speakers)", "google_flan-t5-base_with_speakers"),
            ("T5-small (with_speakers)",  "t5-small_with_speakers"),
            ("PEGASUS (with_speakers)",   "google_pegasus-cnn_dailymail_with_speakers"),
        ]
        for label, dirname in candidates:
            ckpt_path = best_dir / dirname
            if ckpt_path.exists():
                registry[label] = str(ckpt_path)

    prod = PROJECT_ROOT / "models" / "production_task5"
    if prod.exists() and any(prod.glob("*.safetensors")):
        registry["T5 Task 5 (production_task5)"] = str(prod)

    # Keep HF option always available for cloud / lightweight repos.
    registry["T5 Task 5 (HF Hub)"] = HF_TASK5_REPO_ID

    # Optional extra HF-hosted checkpoints configured via environment.
    registry.update(_discover_extra_hf_models())
    return registry


def _first_local_model(models: dict[str, str]) -> tuple[str, str] | None:
    """Return first local model option from registry, if any."""
    for label, path in models.items():
        if not _is_hf_repo_id(path):
            return label, path
    return None

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

    if _is_hf_repo_id(model_path):
        _force_hf_online_mode()
        token = _hf_token()
        tok = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=False,
            token=token,
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=False,
            token=token,
        ).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, dtype=dtype).to(device)
    mdl.eval()
    return tok, mdl, device, dtype, cfg


def _load_task5_config(model_path: str) -> dict[str, Any] | None:
    local_cfg = Path(model_path).resolve() / "task5_production_config.json"
    if local_cfg.is_file():
        with open(local_cfg) as f:
            return json.load(f)

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError:
        return None

    _force_hf_online_mode()
    token = _hf_token()
    repo_id = model_path if _is_hf_repo_id(model_path) else HF_TASK5_REPO_ID
    try:
        remote_cfg = hf_hub_download(
            repo_id=repo_id,
            filename="task5_production_config.json",
            local_files_only=False,
            token=token,
        )
    except Exception:
        return None

    with open(remote_cfg) as f:
        return json.load(f)


@st.cache_resource(show_spinner="⏳ Loading spaCy model…")
def _load_nlp():
    """Load spaCy en_core_web_sm once per session."""
    import spacy  # noqa: PLC0415
    return spacy.load("en_core_web_sm")


# ── Inference ──────────────────────────────────────────────────────────────────

def _model_needs_summarize_prefix(model_path: str) -> bool:
    """T5 / FLAN-T5 / Task-5 production checkpoints use ``summarize: `` in preprocess."""
    p = Path(model_path)
    parts = " ".join(p.parts).lower()
    if "production_task5" in parts:
        return True
    if model_path == HF_TASK5_REPO_ID:
        return True
    name = p.name.lower()
    if "flan-t5" in name or "flan_t5" in parts:
        return True
    if name.startswith("t5-") or "t5_small" in name or "t5-small" in name:
        return True
    if "lora_task1" in parts or "lora_task4" in parts or "t5-small_lora" in parts:
        return True
    return False


def _generate(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    cfg: dict,
    num_beams: int,
    length_penalty: float,
    *,
    use_task_prefix: bool = False,
) -> tuple[str, float]:
    """Return (summary_text, latency_ms).

    torch.mps.synchronize() is called BEFORE stopping the timer so the
    latency figure includes all MPS kernel execution, not just dispatch.
    """
    enc_text = f"{TASK_PREFIX_T5}{dialogue}" if use_task_prefix else dialogue
    inputs = tokenizer(
        enc_text,
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


def _task5_packaging_config(model_path: str) -> dict[str, Any] | None:
    return _load_task5_config(model_path)


def _generate_json_supervised_t5(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    cfg: dict,
    num_beams: int,
    length_penalty: float,
) -> tuple[str, float]:
    """``train_structured``-aligned generation: JSON inner format + decoder prefill."""
    enc_text = structured_input_text(dialogue)
    inputs = tokenizer(
        enc_text,
        return_tensors="pt",
        max_length=cfg["max_source_length"],
        truncation=True,
    ).to(device)
    dec_pref = structured_decoder_input_ids(tokenizer, device)
    max_new = min(320, max(int(cfg.get("max_target_length", 128)) * 3, 256))

    t_start = time.perf_counter()
    with torch.no_grad():
        gen_kw: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": True,
            "repetition_penalty": 1.12,
        }
        if dec_pref is not None:
            gen_kw["decoder_input_ids"] = dec_pref
        out = model.generate(**gen_kw)
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


def _render_structured_schema(struct: dict[str, Any], source: str) -> None:
    """Display ``{topics, action_items, decision}`` plus provenance caption."""
    st.markdown("**Structured output (Task 5 schema)**")
    st.caption(
        "Strict JSON parse of model output"
        if source == "native_json"
        else (
            "Structured fields recovered from model JSON-shaped text (syntax repair; model-only)"
            if source == "salvaged_json"
            else (
                "Schema derived deterministically from the summary text "
                "(same path as ``task5_lora_structured`` reliable pipeline; not gold labels)."
            )
        )
    )
    topics = struct.get("topics") or []
    actions = struct.get("action_items") or []
    decision = struct.get("decision", "")
    if isinstance(decision, list):
        decision = " ".join(str(x) for x in decision) if decision else ""

    st.markdown("*Topics*")
    if topics:
        for t in topics:
            st.markdown(f"- {t}")
    else:
        st.caption("_None_")

    st.markdown("*Action items*")
    if actions:
        for a in actions:
            st.markdown(f"- {a}")
    else:
        st.caption("_None_")

    st.markdown("*Decision / outcome*")
    st.write(decision if decision else "_None_")

    with st.expander("Raw JSON (API-shaped)"):
        st.json(struct)
        st.caption(f"Expected keys: {list(STRUCTURED_SCHEMA.keys())}")


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

    # Heavy resources are loaded on-demand when user clicks Summarize.
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

            # Load selected model with robust fallback when HF is unreachable.
            loaded_label = model_label
            loaded_path = model_path
            try:
                tokenizer, model, device, dtype, cfg = _load_model(loaded_path)
            except Exception as exc:
                if _is_hf_repo_id(loaded_path):
                    fallback = _first_local_model(models)
                    if fallback is not None:
                        fb_label, fb_path = fallback
                        st.warning(
                            "HF Hub is not reachable from this runtime. "
                            f"Falling back to local checkpoint: {fb_label}."
                        )
                        tokenizer, model, device, dtype, cfg = _load_model(fb_path)
                        loaded_label, loaded_path = fb_label, fb_path
                    else:
                        st.error(
                            "Unable to load Hugging Face model. This runtime cannot reach huggingface.co "
                            "and no local checkpoint is available."
                        )
                        for hint in _hf_error_diagnosis(exc, loaded_path):
                            st.caption(f"• {hint}")
                        st.exception(exc)
                        st.stop()
                else:
                    st.error("Unable to load selected local checkpoint.")
                    st.exception(exc)
                    st.stop()

            use_prefix = _model_needs_summarize_prefix(loaded_path)
            t5_pack = _task5_packaging_config(loaded_path)
            json_supervised = bool(t5_pack and t5_pack.get("structured_supervised"))

            with st.spinner("Generating summary…"):
                if json_supervised and use_prefix:
                    summary, latency_ms = _generate_json_supervised_t5(
                        dialogue, tokenizer, model, device, cfg,
                        num_beams      = num_beams,
                        length_penalty = float(length_penalty),
                    )
                else:
                    summary, latency_ms = _generate(
                        dialogue, tokenizer, model, device, cfg,
                        num_beams      = num_beams,
                        length_penalty = float(length_penalty),
                        use_task_prefix = use_prefix,
                    )

            # ── Summary ──────────────────────────────────────────────────
            st.markdown("**Summary**")
            st.success(summary)

            struct, struct_src = structured_dict_from_model_output(summary)
            _render_structured_schema(struct, struct_src)

            # ── Regex action items (supplementary; schema above is the contract) ─
            action_items = _extract_action_items(summary)
            with st.expander("Regex action-item highlights (legacy heuristic)", expanded=False):
                if action_items:
                    for item in action_items:
                        st.markdown(f"- {item}")
                else:
                    st.caption("_No extra regex hits in the summary text._")

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
                "model"         : loaded_label,
                "checkpoint"    : Path(loaded_path).name,
                "device"        : str(device),
                "dtype"         : str(dtype),
                "num_beams"     : num_beams,
                "length_penalty": float(length_penalty),
                "latency_ms"    : round(latency_ms, 1),
                "t5_summarize_prefix": use_prefix,
                "task5_json_supervised": json_supervised,
                "structured_output_source": struct_src,
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

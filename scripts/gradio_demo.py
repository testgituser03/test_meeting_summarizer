#!/usr/bin/env python3
"""
gradio_demo.py — Gradio UI aligned with scripts/app.py (checkpoint list, prefixes, Task 5 JSON path).

Loads checkpoints from models/best/ + production_task5 the same way as Streamlit.

Usage:
  pip install -r requirements.txt   # includes gradio
  PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/gradio_demo.py
  # Default http://127.0.0.1:7860 — if busy, next free port is used (or set GRADIO_SERVER_PORT).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import gradio as gr
except ImportError as e:
    print("Install Gradio:  pip install gradio", file=sys.stderr)
    raise SystemExit(1) from e

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

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
TASK_PREFIX_T5 = _task5.TASK_PREFIX


def _discover_models() -> dict[str, str]:
    """Match scripts/app.py: scan models/best/ + production_task5."""
    best_dir = PROJECT_ROOT / "models" / "best"
    registry: dict[str, str] = {}
    if not best_dir.exists():
        pass
    else:
        candidates = [
            ("BART-base (with_speakers)", "facebook_bart-base_with_speakers"),
            ("BART-base LoRA", "facebook_bart-base_lora"),
            ("BART-base (no_speakers)", "facebook_bart-base_no_speakers"),
            ("FLAN-T5-base (with_speakers)", "google_flan-t5-base_with_speakers"),
            ("T5-small (with_speakers)", "t5-small_with_speakers"),
            ("PEGASUS (with_speakers)", "google_pegasus-cnn_dailymail_with_speakers"),
        ]
        for label, dirname in candidates:
            ckpt_path = best_dir / dirname
            if ckpt_path.exists():
                registry[label] = str(ckpt_path)

    prod = PROJECT_ROOT / "models" / "production_task5"
    if prod.exists() and any(prod.glob("*.safetensors")):
        registry["T5 Task 5 (production_task5)"] = str(prod)
    return registry


def _model_needs_summarize_prefix(model_path: str) -> bool:
    """Same rules as scripts/app.py ``_model_needs_summarize_prefix``."""
    p = Path(model_path).resolve()
    parts = " ".join(p.parts).lower()
    if "production_task5" in parts:
        return True
    name = p.name.lower()
    if "flan-t5" in name or "flan_t5" in parts:
        return True
    if name.startswith("t5-") or "t5_small" in parts or "t5-small" in parts:
        return True
    if "lora_task1" in parts or "lora_task4" in parts or "t5-small_lora" in parts:
        return True
    return False


def _task5_packaging_config(model_path: str) -> dict[str, Any] | None:
    cfg_path = Path(model_path).resolve() / "task5_production_config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path) as f:
        return json.load(f)


_tok = None
_mdl = None
_cur_path: str | None = None
_cfg: dict | None = None
_dev: torch.device | None = None


def _load_cfg() -> dict:
    global _cfg
    if _cfg is None:
        with open(CONFIG_PATH) as f:
            _cfg = yaml.safe_load(f)
    return _cfg


def _get_device() -> torch.device:
    global _dev
    if _dev is None:
        _dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return _dev


def _ensure_model(model_path: str):
    global _tok, _mdl, _cur_path
    if _cur_path == model_path and _tok is not None and _mdl is not None:
        return
    cfg = _load_cfg()
    device = _get_device()
    _tok = AutoTokenizer.from_pretrained(model_path)
    _mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if cfg.get("use_bf16") else torch.float32,
    ).to(device)
    _mdl.eval()
    _cur_path = model_path


def _generate_json_supervised_t5(
    dialogue: str,
    tokenizer,
    model,
    device: torch.device,
    cfg: dict,
    num_beams: int,
    length_penalty: float,
) -> tuple[str, float]:
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


def summarize(
    dialogue: str,
    model_choice: str,
    num_beams: float,
    length_penalty: float,
) -> tuple[str, dict[str, Any], str]:
    """Return (summary + meta, structured dict for JSON view, structured source caption)."""
    dialogue = (dialogue or "").strip()
    if not dialogue:
        return "Enter dialogue text.", {}, ""

    registry = _discover_models()
    if model_choice not in registry:
        return (
            f"No checkpoint for '{model_choice}'. Train or download weights first.",
            {},
            "",
        )
    path = registry[model_choice]
    _ensure_model(path)
    cfg = _load_cfg()
    device = _get_device()
    use_prefix = _model_needs_summarize_prefix(path)
    t5_pack = _task5_packaging_config(path)
    json_supervised = bool(t5_pack and t5_pack.get("structured_supervised"))

    nb = max(1, min(8, int(num_beams)))
    lp = float(length_penalty)

    if json_supervised and use_prefix:
        text, latency_ms = _generate_json_supervised_t5(
            dialogue, _tok, _mdl, device, cfg, nb, lp
        )
        mode = "task5_json_supervised"
    else:
        enc_text = f"{TASK_PREFIX_T5}{dialogue}" if use_prefix else dialogue
        enc = _tok(
            enc_text,
            return_tensors="pt",
            max_length=cfg["max_source_length"],
            truncation=True,
        ).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = _mdl.generate(
                **enc,
                max_new_tokens=cfg["max_target_length"],
                num_beams=nb,
                length_penalty=lp,
                early_stopping=True,
            )
        if device.type == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _tok.decode(out[0], skip_special_tokens=True).strip()
        mode = "standard"

    struct, struct_src = structured_dict_from_model_output(text)
    meta_lines = [
        f"latency_ms≈{latency_ms:.1f}  |  beams={nb}  lp={lp}",
        f"path={mode}  |  t5_summarize_prefix={use_prefix}  |  structured_source={struct_src}",
    ]
    return f"{text}\n\n---\n" + "\n".join(meta_lines), struct, struct_src


def _port_bindable(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _choose_server_port(preferred: int, host: str = "127.0.0.1", span: int = 64) -> int:
    """Use ``preferred`` if free; otherwise the next free port in ``[preferred, preferred+span)``."""
    if _port_bindable(preferred, host):
        return preferred
    for p in range(preferred + 1, preferred + span):
        if _port_bindable(p, host):
            print(
                f"Gradio: port {preferred} is in use; starting on {p} instead "
                f"(set GRADIO_SERVER_PORT or stop the other process).",
                file=sys.stderr,
            )
            return p
    raise OSError(
        f"No bindable TCP port in range {preferred}-{preferred + span - 1} on {host}. "
        "Free a port or set GRADIO_SERVER_PORT."
    )


def _ensure_event_loop_for_gradio_queue() -> None:
    """Register a default asyncio loop before ``gr.Blocks()`` is constructed.

    Gradio 5 builds ``Queue.pending_message_lock`` via ``utils.safe_get_lock()``,
    which does ``asyncio.get_event_loop()`` and returns ``None`` on failure.
    On Python 3.12+, ``get_event_loop()`` raises when no loop is set, so every
    button submit hits ``queue_join`` with a broken lock unless we prime a loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("event loop is closed")
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def main() -> None:
    _ensure_event_loop_for_gradio_queue()

    registry = _discover_models()
    if not registry:
        print("No models under models/best/ — run training first.", file=sys.stderr)
        sys.exit(1)

    choices = list(registry.keys())
    default = choices[0]
    cfg = _load_cfg()

    with gr.Blocks(title="Meeting Summarizer (Gradio)") as demo:
        gr.Markdown(
            "Course brief: **Gradio *or* Streamlit**. Parity UI: same checkpoints, T5 prefix rules, "
            "and Task 5 JSON-supervised decode when `task5_production_config.json` marks "
            "`structured_supervised`. Main full-feature UI: `streamlit run scripts/app.py`."
        )
        model_dd = gr.Dropdown(choices=choices, value=default, label="Checkpoint")
        dialogue = gr.Textbox(label="Dialogue", lines=8, placeholder="Paste multi-speaker chat…")
        beams = gr.Slider(1, 8, value=int(cfg.get("num_beams", 4)), step=1, label="num_beams")
        lp = gr.Slider(0.5, 1.5, value=float(cfg.get("length_penalty", 1.0)), step=0.05, label="length_penalty")
        out = gr.Textbox(label="Summary + meta", lines=8)
        struct_json = gr.JSON(label="Structured output (Task 5 schema)")
        struct_src = gr.Textbox(label="Structured provenance", lines=1)
        btn = gr.Button("Summarize")
        btn.click(
            summarize,
            inputs=[dialogue, model_dd, beams, lp],
            outputs=[out, struct_json, struct_src],
        )

    # Avoid demo.queue(): on some Python versions Gradio’s async queue leaves
    # pending_message_lock unset → TypeError on first request.
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    port = _choose_server_port(preferred, host=host)
    demo.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    main()

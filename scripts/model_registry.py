"""
Shared HuggingFace model id resolution and T5-class task prefixes.

CLI / Makefile shortcuts map to full hub IDs. When config.yaml leaves
task_prefix empty, T5 / FLAN-T5 runs use "summarize: " (same as
baseline_zeroshot.py and extension task scripts) — BART keeps "".
"""

from __future__ import annotations

# Short names → hub ids. For T5-class experiments, preferred try order (see README):
#   flan-t5-base → flan-t5-small → t5-small
MODEL_ALIASES: dict[str, str] = {
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-small": "google/flan-t5-small",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "bart-base": "facebook/bart-base",
}


def resolve_model_name(name: str) -> str:
    if not name:
        return name
    key = name.strip().lower()
    return MODEL_ALIASES.get(key, name)


def is_t5_family_model(model_id: str) -> bool:
    mid = model_id.lower().replace("_", "-")
    if "flan-t5" in mid:
        return True
    if mid.startswith("t5-") or "/t5-" in mid:
        return True
    return False


def infer_hf_id_from_run_name(run_dirname: str) -> str:
    """Best-effort hub id from ``models/best/<run>_with_speakers`` directory name."""
    stem = (
        run_dirname.replace("_with_speakers", "")
        .replace("_no_speakers", "")
        .replace("_split_speakers", "")
    )
    for hid in MODEL_ALIASES.values():
        if stem == hid.replace("/", "_"):
            return hid
    if stem == "t5-small":
        return "t5-small"
    if "_" in stem:
        org, rest = stem.split("_", 1)
        return f"{org}/{rest}"
    return stem


def effective_task_prefix(model_id: str, cfg_task_prefix: str | None) -> str:
    """Return task prefix for tokenization / generation.

    Non-empty cfg_task_prefix always wins. Otherwise T5/FLAN-T5 use
    ``summarize: ``; BART and PEGASUS use "".
    """
    if cfg_task_prefix is not None and str(cfg_task_prefix).strip() != "":
        return str(cfg_task_prefix)
    if is_t5_family_model(model_id):
        return "summarize: "
    return ""

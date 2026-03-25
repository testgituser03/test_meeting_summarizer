#!/usr/bin/env python3
"""
Task 2 — Benchmarking, streaming, and parallel scaling for quantized T5 summarization.

Reads quantized artifacts from models/quantized/task2/<Q4_K_M|Q5_K_M|Q8_0>
created by scripts/task2_quantization.py.

Benchmarks:
  1) Length sweep (10/50/100/200 utterances) with latency/throughput/memory/ROUGE-L.
  2) Streaming incremental summarization vs batch.
  3) Parallel inference scaling (1/2/4 processes) on Apple Silicon.
  4) Fair full-test ROUGE-L comparison with identical inputs.

Outputs:
  - results/metrics/task2_benchmark_inputs.jsonl
  - results/metrics/task2_benchmark_table.json
  - results/metrics/task2_streaming_vs_batch.json
  - results/metrics/task2_parallel_scaling.json
  - results/metrics/task2_eval_rougel.json  (object with benchmark_args + results[])

Use `--eval-only` to regenerate only `task2_eval_rougel.json` (same eval loop, skips length/streaming/parallel).
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


LENGTH_BUCKETS = [10, 50, 100, 200]
QUANT_LABELS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_turns(dialogue: str) -> list[str]:
    return [ln.strip() for ln in dialogue.splitlines() if ln.strip()]


def build_length_variant(dialogue: str, target_turns: int) -> str:
    """Deterministically clip/repeat turns to a fixed turn count.

    This produces identical stress-test inputs across all configurations.
    """
    turns = parse_turns(dialogue)
    if not turns:
        turns = ["Unknown: (empty conversation)"]

    if len(turns) >= target_turns:
        selected = turns[:target_turns]
    else:
        selected = []
        i = 0
        while len(selected) < target_turns:
            selected.append(turns[i % len(turns)])
            i += 1

    return "\n".join(selected)


@dataclass
class InferenceResult:
    text: str
    latency_s: float


class HFSeq2SeqRuntime:
    """Reference runtime: Transformers checkpoint (BF16 on MPS when available)."""

    def __init__(
        self,
        model_path: Path,
        max_source_len: int,
        max_target_len: int,
        num_beams: int,
        task_prefix: str = "summarize: ",
    ) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        self.torch = torch
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16 if self.device.type == "mps" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.num_beams = num_beams
        self.task_prefix = task_prefix

    def summarize(self, text: str) -> InferenceResult:
        inputs = self.tokenizer(
            f"{self.task_prefix}{text}",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_len,
        ).to(self.device)

        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_target_len,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        if self.device.type == "mps":
            self.torch.mps.synchronize()
        latency = time.perf_counter() - t0

        pred = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return InferenceResult(text=pred, latency_s=latency)


class CT2Seq2SeqRuntime:
    """Quantized runtime using CTranslate2 for T5 encoder-decoder inference."""

    def __init__(
        self,
        model_dir: Path,
        tokenizer_dir: Path,
        max_source_len: int,
        max_target_len: int,
        num_beams: int,
        task_prefix: str = "summarize: ",
    ) -> None:
        ctranslate2 = importlib.import_module("ctranslate2")
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.translator = ctranslate2.Translator(
            str(model_dir),
            device="cpu",
            inter_threads=1,
            intra_threads=max(1, (os.cpu_count() or 4) // 2),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.num_beams = num_beams
        self.task_prefix = task_prefix

    def _encode_source(self, text: str) -> list[str]:
        # T5 expects task prefix for summarization.
        prefixed = f"{self.task_prefix}{text}"
        enc = self.tokenizer(
            prefixed,
            return_attention_mask=False,
            return_tensors=None,
            truncation=True,
            max_length=self.max_source_len,
        )
        return self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

    def summarize(self, text: str) -> InferenceResult:
        source_tokens = self._encode_source(text)

        t0 = time.perf_counter()
        results = self.translator.translate_batch(
            [source_tokens],
            max_decoding_length=self.max_target_len,
            beam_size=self.num_beams,
            target_prefix=[["<pad>"]],
        )
        latency = time.perf_counter() - t0

        hypothesis_tokens = results[0].hypotheses[0]
        token_ids = self.tokenizer.convert_tokens_to_ids(hypothesis_tokens)
        pred = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return InferenceResult(text=pred, latency_s=latency)


def rouge_l(preds: list[str], refs: list[str]) -> float:
    from rouge_score import rouge_scorer  # noqa: PLC0415

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = []
    for p, r in zip(preds, refs):
        vals.append(scorer.score(r, p)["rougeL"].fmeasure)
    n = max(len(vals), 1)
    return round(sum(vals) / n * 100, 4)


def build_fixed_inputs(project_root: Path, seed: int, n_per_length: int) -> list[dict[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415

    rng = random.Random(seed)
    ds_test = load_dataset("knkarthick/samsum")["test"]
    indices = list(range(len(ds_test)))
    rng.shuffle(indices)

    inputs: list[dict[str, Any]] = []
    ptr = 0
    for length in LENGTH_BUCKETS:
        for _ in range(n_per_length):
            idx = indices[ptr % len(indices)]
            ptr += 1
            sample = ds_test[idx]
            dialogue = build_length_variant(sample["dialogue"], target_turns=length)
            inputs.append(
                {
                    "sample_id": f"L{length}_{idx}_{ptr}",
                    "length_bucket": length,
                    "dataset_index": int(idx),
                    "dialogue": dialogue,
                    "reference": sample["summary"],
                }
            )

    inputs_path = project_root / "results" / "metrics" / "task2_benchmark_inputs.jsonl"
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inputs_path, "w") as f:
        for row in inputs:
            f.write(json.dumps(row) + "\n")

    return inputs


def run_length_benchmark(runtime_name: str, runtime, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    by_length: dict[int, list[dict[str, Any]]] = {k: [] for k in LENGTH_BUCKETS}
    for c in cases:
        by_length[c["length_bucket"]].append(c)

    proc = psutil.Process(os.getpid())

    for length in LENGTH_BUCKETS:
        group = by_length[length]
        preds: list[str] = []
        refs: list[str] = []

        rss_before = proc.memory_info().rss
        peak_rss = rss_before
        total_s = 0.0

        for g in group:
            out = runtime.summarize(g["dialogue"])
            total_s += out.latency_s
            preds.append(out.text)
            refs.append(g["reference"])
            peak_rss = max(peak_rss, proc.memory_info().rss)

        n = len(group)
        latency_ms = (total_s / max(n, 1)) * 1000.0
        throughput = n / total_s if total_s > 0 else 0.0
        mem_mb = (peak_rss - rss_before) / (1024 * 1024)
        rl = rouge_l(preds, refs)

        rows.append(
            {
                "runtime": runtime_name,
                "length": length,
                "latency_ms": round(latency_ms, 3),
                "throughput_summaries_per_sec": round(throughput, 4),
                "memory_mb": round(mem_mb, 2),
                "rougeL": rl,
            }
        )

    return rows


class RollingStreamingSummarizer:
    """Hierarchical incremental summarization to reduce full-context recomputation.

    Design:
      - Keep rolling chunk of turns.
      - Summarize only new chunk when it reaches chunk_size.
      - Maintain chunk summaries.
      - Build current output by summarizing (chunk_summaries + active_chunk).

    This is an approximation and trades some quality for lower incremental cost.
    """

    def __init__(self, runtime, chunk_size: int = 25, max_chunk_summaries: int = 12) -> None:
        self.runtime = runtime
        self.chunk_size = chunk_size
        self.max_chunk_summaries = max_chunk_summaries
        self.active_turns: list[str] = []
        self.chunk_summaries: list[str] = []

    def _finalize_chunk(self) -> None:
        if not self.active_turns:
            return
        text = "\n".join(self.active_turns)
        chunk_summary = self.runtime.summarize(text).text
        self.chunk_summaries.append(chunk_summary)
        if len(self.chunk_summaries) > self.max_chunk_summaries:
            self.chunk_summaries = self.chunk_summaries[-self.max_chunk_summaries :]
        self.active_turns = []

    def update(self, new_turn: str) -> InferenceResult:
        self.active_turns.append(new_turn)
        if len(self.active_turns) >= self.chunk_size:
            self._finalize_chunk()

        stitched = []
        if self.chunk_summaries:
            stitched.append("Previous summary:\n" + " ".join(self.chunk_summaries))
        if self.active_turns:
            stitched.append("New turns:\n" + "\n".join(self.active_turns))

        composed = "\n\n".join(stitched) if stitched else new_turn
        return self.runtime.summarize(composed)


def compare_streaming_vs_batch(runtime_name: str, runtime, cases: list[dict[str, Any]]) -> dict[str, Any]:
    proc = psutil.Process(os.getpid())

    # Use the longest bucket for meaningful incremental behavior.
    longest = [c for c in cases if c["length_bucket"] == 200]
    if not longest:
        return {}

    batch_preds: list[str] = []
    batch_refs: list[str] = []
    stream_preds: list[str] = []
    stream_refs: list[str] = []

    batch_time = 0.0
    stream_time = 0.0

    batch_peak = proc.memory_info().rss
    stream_peak = proc.memory_info().rss

    for c in longest:
        # Batch mode
        b = runtime.summarize(c["dialogue"])
        batch_time += b.latency_s
        batch_preds.append(b.text)
        batch_refs.append(c["reference"])
        batch_peak = max(batch_peak, proc.memory_info().rss)

        # Streaming mode (turn-by-turn updates)
        streamer = RollingStreamingSummarizer(runtime=runtime, chunk_size=25)
        final = None
        turns = parse_turns(c["dialogue"])
        for t in turns:
            out = streamer.update(t)
            stream_time += out.latency_s
            final = out
            stream_peak = max(stream_peak, proc.memory_info().rss)

        stream_preds.append(final.text if final else "")
        stream_refs.append(c["reference"])

    n = len(longest)
    batch_rl = rouge_l(batch_preds, batch_refs)
    stream_rl = rouge_l(stream_preds, stream_refs)

    return {
        "runtime": runtime_name,
        "mode_comparison": [
            {
                "mode": "batch",
                "latency_ms": round((batch_time / max(n, 1)) * 1000.0, 3),
                "memory_mb": round(batch_peak / (1024 * 1024), 2),
                "rougeL": batch_rl,
            },
            {
                "mode": "streaming",
                "latency_ms": round((stream_time / max(n, 1)) * 1000.0, 3),
                "memory_mb": round(stream_peak / (1024 * 1024), 2),
                "rougeL": stream_rl,
            },
        ],
    }


def _parallel_worker(
    model_dir: str,
    tokenizer_dir: str,
    max_target_len: int,
    num_beams: int,
    task_prefix: str,
    max_source_len: int,
    items: list[dict[str, Any]],
    q: mp.Queue,
) -> None:
    runtime = CT2Seq2SeqRuntime(
        model_dir=Path(model_dir),
        tokenizer_dir=Path(tokenizer_dir),
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        num_beams=num_beams,
        task_prefix=task_prefix,
    )

    p = psutil.Process(os.getpid())
    peak = p.memory_info().rss

    t0 = time.perf_counter()
    count = 0
    for it in items:
        runtime.summarize(it["dialogue"])
        count += 1
        peak = max(peak, p.memory_info().rss)
    elapsed = time.perf_counter() - t0

    q.put(
        {
            "count": count,
            "elapsed_s": elapsed,
            "peak_rss_mb": round(peak / (1024 * 1024), 2),
        }
    )


def run_parallel_scaling(
    quant_dir: Path,
    tokenizer_dir: Path,
    max_target_len: int,
    num_beams: int,
    task_prefix: str,
    max_source_len: int,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    subset = [c for c in cases if c["length_bucket"] == 100][:64]
    if not subset:
        return []

    results: list[dict[str, Any]] = []

    for nproc in [1, 2, 4]:
        shards = [subset[i::nproc] for i in range(nproc)]
        q: mp.Queue = mp.Queue()

        procs: list[mp.Process] = []
        start = time.perf_counter()

        for shard in shards:
            p = mp.Process(
                target=_parallel_worker,
                args=(
                    str(quant_dir),
                    str(tokenizer_dir),
                    max_target_len,
                    num_beams,
                    task_prefix,
                    max_source_len,
                    shard,
                    q,
                ),
            )
            p.start()
            procs.append(p)

        cpu_samples: list[float] = []
        while any(p.is_alive() for p in procs):
            cpu_samples.append(psutil.cpu_percent(interval=0.2))

        payloads = [q.get() for _ in range(nproc)]
        for p in procs:
            p.join()

        wall = time.perf_counter() - start
        total_count = sum(x["count"] for x in payloads)
        throughput = total_count / wall if wall > 0 else 0.0

        results.append(
            {
                "processes": nproc,
                "wall_time_s": round(wall, 4),
                "throughput_summaries_per_sec": round(throughput, 4),
                "avg_cpu_util_percent": round(float(np.mean(cpu_samples)) if cpu_samples else 0.0, 2),
                "mps_driver_memory_mb": round(_mps_driver_memory_mb(), 2),
                "max_process_rss_mb": round(max(x["peak_rss_mb"] for x in payloads), 2),
                "total_process_rss_mb": round(sum(x["peak_rss_mb"] for x in payloads), 2),
            }
        )

    return results


def evaluate_on_real_testset(runtime_name: str, runtime, n_samples: int, seed: int) -> dict[str, Any]:
    from datasets import load_dataset  # noqa: PLC0415

    ds_test = load_dataset("knkarthick/samsum")["test"]
    idxs = list(range(len(ds_test)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n_samples]

    preds: list[str] = []
    refs: list[str] = []
    total_s = 0.0

    for i in idxs:
        row = ds_test[i]
        out = runtime.summarize(row["dialogue"])
        preds.append(out.text)
        refs.append(row["summary"])
        total_s += out.latency_s

    return {
        "runtime": runtime_name,
        "n_samples": len(idxs),
        "rougeL": rouge_l(preds, refs),
        "latency_ms": round((total_s / max(len(idxs), 1)) * 1000.0, 3),
        "throughput_summaries_per_sec": round(len(idxs) / total_s, 4) if total_s > 0 else 0.0,
    }


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _rel_to_root(project_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(p)


def _mps_driver_memory_mb() -> float:
    try:
        import torch  # noqa: PLC0415

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        return 0.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 benchmark runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quant_root", default="models/quantized/task2")
    parser.add_argument("--merged_model", default="models/best/t5-small_lora_task1/merged")
    parser.add_argument("--samples_per_length", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run SAMSum test-set ROUGE-L and write task2_eval_rougel.json (skip length/streaming/parallel).",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_config(args.config)

    max_source_len = int(cfg.get("max_source_length", 512))
    max_target_len = int(cfg.get("max_target_length", 128))
    num_beams = int(cfg.get("num_beams", 4))
    task_prefix = "summarize: "

    quant_root = project_root / args.quant_root
    merged_model = project_root / args.merged_model

    if not merged_model.exists():
        raise FileNotFoundError(f"Merged model not found: {merged_model}")

    cases = None
    if not args.eval_only:
        cases = build_fixed_inputs(project_root, seed=args.seed, n_per_length=args.samples_per_length)

    all_length_rows: list[dict[str, Any]] = []
    all_stream_rows: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []
    parallel_rows: dict[str, list[dict[str, Any]]] = {}

    # Baseline: non-quantized HF runtime (for quality/speed anchor)
    hf_runtime = HFSeq2SeqRuntime(
        model_path=merged_model,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
        num_beams=num_beams,
        task_prefix=task_prefix,
    )

    if args.eval_only:
        all_eval_rows.append(
            evaluate_on_real_testset("FP_BASELINE", hf_runtime, n_samples=args.eval_samples, seed=args.seed)
        )
    else:
        assert cases is not None
        all_length_rows.extend(run_length_benchmark("FP_BASELINE", hf_runtime, cases))
        all_stream_rows.append(compare_streaming_vs_batch("FP_BASELINE", hf_runtime, cases))
        all_eval_rows.append(evaluate_on_real_testset("FP_BASELINE", hf_runtime, n_samples=args.eval_samples, seed=args.seed))

    # Quantized runtimes
    for qlabel in QUANT_LABELS:
        qdir = quant_root / qlabel
        if not qdir.exists():
            print(f"⚠️  Skipping {qlabel}: not found at {qdir}")
            continue

        rt_name = qlabel
        rt = CT2Seq2SeqRuntime(
            model_dir=qdir,
            tokenizer_dir=merged_model,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
            num_beams=num_beams,
            task_prefix=task_prefix,
        )

        if args.eval_only:
            all_eval_rows.append(evaluate_on_real_testset(rt_name, rt, n_samples=args.eval_samples, seed=args.seed))
        else:
            assert cases is not None
            all_length_rows.extend(run_length_benchmark(rt_name, rt, cases))
            all_stream_rows.append(compare_streaming_vs_batch(rt_name, rt, cases))
            all_eval_rows.append(evaluate_on_real_testset(rt_name, rt, n_samples=args.eval_samples, seed=args.seed))

            parallel_rows[qlabel] = run_parallel_scaling(
                quant_dir=qdir,
                tokenizer_dir=merged_model,
                max_target_len=max_target_len,
                num_beams=num_beams,
                task_prefix=task_prefix,
                max_source_len=max_source_len,
                cases=cases,
            )

    metrics_dir = project_root / "results" / "metrics"
    if not args.eval_only:
        _save_json(metrics_dir / "task2_benchmark_table.json", all_length_rows)
        _save_json(metrics_dir / "task2_streaming_vs_batch.json", all_stream_rows)
        _save_json(metrics_dir / "task2_parallel_scaling.json", parallel_rows)
    eval_payload = {
        "benchmark_args": {
            "eval_samples": args.eval_samples,
            "samples_per_length": args.samples_per_length,
            "seed": args.seed,
            "merged_model": _rel_to_root(project_root, merged_model),
            "quant_root": _rel_to_root(project_root, quant_root),
            "config": args.config,
            "eval_only": args.eval_only,
        },
        "quant_labels_nominal": QUANT_LABELS,
        "quant_runtime_note": (
            "Folder names Q4_K_M / Q5_K_M / Q8_0 are nominal profiles; inference uses "
            "CTranslate2 (see task2_quantization_manifest.json), not GGUF K-quants."
        ),
        "results": all_eval_rows,
    }
    _save_json(metrics_dir / "task2_eval_rougel.json", eval_payload)

    print("\n✅  Task 2 benchmark complete")
    if not args.eval_only:
        print(f"  Inputs     : results/metrics/task2_benchmark_inputs.jsonl")
        print(f"  Length     : results/metrics/task2_benchmark_table.json")
        print(f"  Streaming  : results/metrics/task2_streaming_vs_batch.json")
        print(f"  Parallel   : results/metrics/task2_parallel_scaling.json")
    print(f"  Eval       : results/metrics/task2_eval_rougel.json\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

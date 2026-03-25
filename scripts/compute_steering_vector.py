#!/usr/bin/env python3
"""
compute_steering_vector.py — Task 3 / steering direction computation.

Loads activation artifact from extract_activations.py and computes per-layer
steering vectors with three methods:
  1) mean_diff:   mean(action) - mean(topic)
  2) pca_delta:   first principal component of paired action-topic deltas
  3) logistic:    linear separator direction from L2 logistic regression

Output:
  results/steering/<run_name>_<split>_steering.pt
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _l2_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(p=2) + eps)


def _mean_diff_direction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mu_action = x[y == 1].mean(dim=0)
    mu_topic = x[y == 0].mean(dim=0)
    return _l2_normalize(mu_action - mu_topic)


def _pca_delta_direction(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    """Build paired class deltas and return top PCA component."""
    from sklearn.decomposition import PCA  # noqa: PLC0415

    action = x[y == 1]
    topic = x[y == 0]
    m = min(action.size(0), topic.size(0))
    if m < 2:
        raise ValueError("Need at least 2 samples per class for PCA delta method.")

    g = torch.Generator().manual_seed(seed)
    a_idx = torch.randperm(action.size(0), generator=g)[:m]
    t_idx = torch.randperm(topic.size(0), generator=g)[:m]
    deltas = (action[a_idx] - topic[t_idx]).cpu().numpy()

    pca = PCA(n_components=1, random_state=seed)
    pca.fit(deltas)
    direction = torch.tensor(pca.components_[0], dtype=torch.float32)

    # Orient in action-positive direction.
    md = _mean_diff_direction(x, y)
    if torch.dot(direction, md) < 0:
        direction = -direction
    return _l2_normalize(direction)


def _logreg_direction(x: torch.Tensor, y: torch.Tensor, seed: int) -> torch.Tensor:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        random_state=seed,
        max_iter=400,
    )
    clf.fit(x.cpu().numpy(), y.cpu().numpy())

    direction = torch.tensor(clf.coef_[0], dtype=torch.float32)

    # Orient in action-positive direction.
    md = _mean_diff_direction(x, y)
    if torch.dot(direction, md) < 0:
        direction = -direction
    return _l2_normalize(direction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute steering vectors from activation artifact")
    parser.add_argument("--activations", required=True, help="Path to results/activations/*.pt")
    parser.add_argument("--out", default=None, help="Optional override output path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.activations)
    payload = torch.load(in_path, map_location="cpu")

    meta = payload["meta"]
    acts = payload["activations"].to(dtype=torch.float32)  # [N, L, D]
    labels = payload["labels"].to(dtype=torch.int64)

    n_samples, n_layers, hidden = acts.shape

    n_action = int((labels == 1).sum().item())
    n_topic = int((labels == 0).sum().item())
    if n_action < 10 or n_topic < 10:
        raise ValueError(f"Insufficient class coverage: action={n_action}, topic={n_topic}")

    layers = list(meta["layers"])

    vectors_mean: dict[int, torch.Tensor] = {}
    vectors_pca: dict[int, torch.Tensor] = {}
    vectors_logreg: dict[int, torch.Tensor] = {}

    diagnostics: list[dict[str, Any]] = []

    for li, layer in enumerate(layers):
        x = acts[:, li, :]  # [N, D]

        v_mean = _mean_diff_direction(x, labels)
        v_pca = _pca_delta_direction(x, labels, seed=args.seed)
        v_log = _logreg_direction(x, labels, seed=args.seed)

        vectors_mean[layer] = v_mean
        vectors_pca[layer] = v_pca
        vectors_logreg[layer] = v_log

        diagnostics.append(
            {
                "layer": int(layer),
                "cos_pca_vs_mean": round(float(torch.dot(v_pca, v_mean).item()), 6),
                "cos_logreg_vs_mean": round(float(torch.dot(v_log, v_mean).item()), 6),
            }
        )

    out_dir = in_path.parent.parent / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out) if args.out else out_dir / f"{meta['run_name']}_{meta['split']}_steering.pt"

    out_payload = {
        "meta": {
            "run_name": meta["run_name"],
            "model_name": meta["model_name"],
            "model_path": meta["model_path"],
            "variant": meta["variant"],
            "split": meta["split"],
            "n_samples": int(n_samples),
            "n_action": n_action,
            "n_topic": n_topic,
            "layers": layers,
            "hidden_size": int(hidden),
            "seed": int(args.seed),
            "methods": ["mean_diff", "pca_delta", "logistic"],
            "diagnostics": diagnostics,
        },
        "vectors": {
            "mean_diff": vectors_mean,
            "pca_delta": vectors_pca,
            "logistic": vectors_logreg,
        },
    }

    torch.save(out_payload, out_path)

    diag_json = out_path.with_suffix(".json")
    with open(diag_json, "w") as f:
        json.dump(out_payload["meta"], f, indent=2)

    print("\n✅ Steering vectors computed")
    print(f"   Input : {in_path}")
    print(f"   Output: {out_path}")
    print(f"   Layers: {layers}")
    print(f"   Methods: mean_diff, pca_delta, logistic")


if __name__ == "__main__":
    main()

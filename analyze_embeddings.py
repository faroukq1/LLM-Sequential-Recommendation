"""
PromptCraft-SeqRec: Intrinsic embedding quality metrics.

Computes isotropy and average pairwise distance for each strategy's raw embeddings.
Higher isotropy = more uniformly spread = better coverage of semantic space.
Higher avg distance = more discriminative item representations.

Usage:
  python analyze_embeddings.py --dataset beauty
  python analyze_embeddings.py --dataset beauty --emb-dir embeddings --results-dir results
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from prompt_strategies import STRATEGY_NAMES, STRATEGY_LABELS


def embedding_isotropy(emb: np.ndarray) -> float:
    """min(S) / max(S) of centered SVD. Range [0,1]. Higher = more isotropic."""
    centered = emb - emb.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        return float(S[-1] / S[0]) if S[0] > 0 else 0.0
    except Exception:
        return 0.0


def avg_pairwise_distance(emb: np.ndarray, sample_n: int = 2000) -> float:
    """Average cosine distance between random item pairs. Higher = more discriminative."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(emb), min(sample_n, len(emb)), replace=False)
    sims = cosine_similarity(emb[idx])
    np.fill_diagonal(sims, 0)
    mask = sims != 0
    return float(1 - sims[mask].mean()) if mask.any() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"Embedding Quality Analysis — Dataset: {args.dataset}")
    print(f"{'Strategy':<26} {'Isotropy':>10} {'AvgDist':>10}")
    print("-" * 50)

    quality = {}
    for strategy in STRATEGY_NAMES:
        raw_path = os.path.join(args.emb_dir, f"{args.dataset}_{strategy}_raw.npy")
        if not os.path.exists(raw_path):
            print(f"{strategy:<26} {'MISSING':>10}")
            continue

        emb = np.load(raw_path)
        iso = embedding_isotropy(emb)
        apd = avg_pairwise_distance(emb)
        quality[strategy] = {"isotropy": iso, "avg_pairwise_dist": apd}

        label = STRATEGY_LABELS.get(strategy, strategy)
        print(f"{label:<26} {iso:>10.4f} {apd:>10.4f}")

    out_path = os.path.join(args.results_dir, f"embedding_quality_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"\nQuality metrics saved -> {out_path}")


if __name__ == "__main__":
    main()

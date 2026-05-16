"""
PromptCraft-SeqRec: Generate BGE-M3 embeddings for all 6 prompt strategies.

Saves per-strategy:
  - embeddings/<dataset>_<strategy>_raw.npy    — raw 1024-dim BGE-M3 vectors (for analyze_embeddings.py)
  - embeddings/<dataset>_<strategy>_ids.json   — ordered list of item IDs matching .npy rows
  - embeddings/<dataset>_<strategy>.csv        — ItemId,embedding CSV for SASRecWithEmbeddings

Usage:
  python generate_embeddings.py --meta-gz data/beauty/meta_Beauty.json.gz --dataset beauty
  python generate_embeddings.py --meta-gz data/beauty/meta_Beauty.json.gz --dataset beauty --strategies type1_title_only type6_hybrid
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re

import numpy as np
import pandas as pd

from prompt_strategies import STRATEGY_NAMES, apply_strategy

try:
    from FlagEmbedding import BGEM3FlagModel
    _USE_BGE = True
except ImportError:
    from sentence_transformers import SentenceTransformer
    _USE_BGE = False


def _load_model():
    if _USE_BGE:
        print("Using FlagEmbedding BGE-M3")
        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    print("FlagEmbedding not found — using sentence-transformers BAAI/bge-m3")
    return SentenceTransformer("BAAI/bge-m3")


def _encode(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    if _USE_BGE:
        out = model.encode(
            texts,
            batch_size=batch_size,
            max_length=128,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"]
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def _parse_meta_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except Exception:
        pass
    try:
        return ast.literal_eval(line)
    except Exception:
        pass
    try:
        fixed = re.sub(r"(?<!\\)'", '"', line)
        fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
        return json.loads(fixed)
    except Exception:
        return None


def load_meta_gz(path: str) -> dict:
    """Load Amazon metadata gzip -> {asin: metadata_dict}."""
    meta = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = _parse_meta_line(line)
            if rec and rec.get("asin"):
                meta[rec["asin"]] = rec
    print(f"Loaded {len(meta):,} item metadata records from {path}")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-gz", required=True, help="Path to meta_<Dataset>.json.gz")
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--strategies", nargs="+", default=STRATEGY_NAMES)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="embeddings")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset}")
    print(f"Strategies: {args.strategies}")
    print(f"{'='*60}\n")

    items = load_meta_gz(args.meta_gz)

    sample_asin = next(iter(items))
    print(f"Sample item [{sample_asin}]: {list(items[sample_asin].keys())}\n")

    model = _load_model()

    for strategy in args.strategies:
        raw_path = os.path.join(args.output_dir, f"{args.dataset}_{strategy}_raw.npy")
        ids_path = os.path.join(args.output_dir, f"{args.dataset}_{strategy}_ids.json")
        csv_path = os.path.join(args.output_dir, f"{args.dataset}_{strategy}.csv")

        if os.path.exists(raw_path) and os.path.exists(csv_path):
            print(f"[SKIP] {strategy} — already exists")
            continue

        print(f"\n[ENCODING] {strategy}")
        item_ids, texts = apply_strategy(strategy, items, meta_lookup=items)

        print(f"  Sample[0]: {texts[0]}")
        print(f"  Sample[1]: {texts[1]}")

        emb = _encode(model, texts, batch_size=args.batch_size)

        np.save(raw_path, emb)
        with open(ids_path, "w") as f:
            json.dump(item_ids, f)

        rows = [
            {"ItemId": iid, "embedding": json.dumps([round(float(x), 6) for x in emb[i]])}
            for i, iid in enumerate(item_ids)
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        print(f"  Shape : {emb.shape}")
        print(f"  Saved : {raw_path}")
        print(f"  Saved : {csv_path}")

    print(f"\nDone. Files in {args.output_dir}/")


if __name__ == "__main__":
    main()

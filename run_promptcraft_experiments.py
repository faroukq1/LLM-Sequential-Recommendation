"""
PromptCraft-SeqRec: Run SASRec experiments for all 6 prompt strategies.

Requires:
  - Sessions CSV (SessionId,ItemId,Time,Reward) from preprocessing
  - Embedding CSVs from generate_embeddings.py

Usage:
  python run_promptcraft_experiments.py --sessions-csv data/beauty/sessions.csv --dataset beauty
  python run_promptcraft_experiments.py --sessions-csv data/beauty/sessions.csv --dataset beauty \\
      --strategies type1_title_only type6_hybrid
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main.data.session_dataset import SessionDataset
from main.data.temporal_split import TemporalSplit
from main.eval.evaluation import Evaluation
from main.eval.metrics.hitrate import HitRate
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from main.eval.metrics.metric import MetricDependency
from main.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings
from prompt_strategies import STRATEGY_NAMES


def load_dataset(sessions_csv: str, test_frac: float = 0.2):
    dataset = SessionDataset(sessions_csv, n_withheld=1)
    dataset.load_and_split(
        TemporalSplit(test_frac=test_frac, num_folds=0, filter_non_trained_test_items=True)
    )
    return (
        dataset.get_train_data(),
        dataset.get_test_prompts(),
        dataset.get_test_ground_truths(),
        dataset.get_unique_item_count(),
        dataset.get_item_counts(),
    )


def run_one(
    strategy: str,
    emb_csv: str,
    train_data,
    test_prompts,
    test_gts,
    num_items: int,
    item_counts: dict,
    cfg: dict,
) -> dict:
    # Reset class variable — required to reload embeddings for each strategy
    SASRecWithEmbeddings.product_embeddings = None

    model = SASRecWithEmbeddings(
        product_embeddings_location=emb_csv,
        red_method="PCA",
        red_params={"n_components": str(cfg["emb_dim"])},
        N=cfg["N"],
        L=cfg["L"],
        h=cfg["h"],
        emb_dim=cfg["emb_dim"],
        drop_rate=cfg["drop_rate"],
        optimizer_kwargs={
            "learning_rate": cfg["learning_rate"],
            "weight_decay": cfg["weight_decay"],
        },
        num_epochs=cfg["num_epochs"],
        fit_batch_size=cfg["fit_batch_size"],
        pred_batch_size=cfg["pred_batch_size"],
        train_val_fraction=cfg["train_val_fraction"],
        early_stopping_patience=cfg["early_stopping_patience"],
        activation=cfg["activation"],
        is_verbose=cfg.get("is_verbose", True),
        cores=cfg.get("cores", 1),
        transformer_layer_kwargs=cfg.get("transformer_layer_kwargs", {}),
    )

    t0 = time.time()
    model.train(train_data)

    preds = model.predict(test_prompts, top_k=max(cfg["top_ks"]))
    elapsed = time.time() - t0

    metric_list = [NormalizedDiscountedCumulativeGain(), HitRate()]
    deps = {
        MetricDependency.NUM_ITEMS: num_items,
        MetricDependency.ITEM_COUNT: item_counts,
    }

    metrics = {}
    for k in cfg["top_ks"]:
        report = Evaluation.eval(
            predictions=preds,
            ground_truths=test_gts,
            top_k=k,
            metrics=metric_list,
            metrics_per_sample=False,
            dependencies=deps,
            cores=1,
            model_name=f"LLM2SASRec_{strategy}",
        )
        metrics[f"NDCG@{k}"] = float(report.results.get(f"NDCG@{k}", 0.0))
        metrics[f"HR@{k}"] = float(report.results.get(f"HR@{k}", 0.0))

    del model
    gc.collect()

    return {"metrics": metrics, "elapsed_seconds": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-csv", required=True)
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--strategies", nargs="+", default=STRATEGY_NAMES)
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-frac", type=float, default=0.2)
    # SASRec hyperparameters
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[10, 20])
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.results_dir, f"promptcraft_{args.dataset}_{ts}.json")

    cfg = dict(
        emb_dim=args.emb_dim,
        N=args.N,
        L=args.L,
        h=args.h,
        drop_rate=0.0,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        fit_batch_size=args.batch_size,
        pred_batch_size=2048,
        train_val_fraction=0.1,
        early_stopping_patience=3,
        activation="relu",
        is_verbose=True,
        cores=1,
        transformer_layer_kwargs={"layout": "NFDR"},
        top_ks=args.top_ks,
    )

    print(f"\n{'='*60}")
    print(f"PromptCraft-SeqRec — Dataset: {args.dataset}")
    print(f"Strategies: {args.strategies}")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    train_data, test_prompts, test_gts, num_items, item_counts = load_dataset(
        args.sessions_csv, test_frac=args.test_frac
    )
    print(f"Train: {len(train_data):,} rows | Test: {len(test_prompts):,} cases | Items: {num_items:,}")

    all_results: dict = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)

    for strategy in args.strategies:
        if strategy in all_results:
            print(f"[SKIP] {strategy} — already in results")
            continue

        emb_csv = os.path.join(args.emb_dir, f"{args.dataset}_{strategy}.csv")
        if not os.path.exists(emb_csv):
            print(f"[MISSING] {emb_csv} — run generate_embeddings.py first")
            all_results[strategy] = {"error": f"embedding CSV not found: {emb_csv}"}
            continue

        print(f"\n[RUN] {strategy}")
        try:
            result = run_one(
                strategy=strategy,
                emb_csv=emb_csv,
                train_data=train_data,
                test_prompts=test_prompts,
                test_gts=test_gts,
                num_items=num_items,
                item_counts=item_counts,
                cfg=cfg,
            )
            all_results[strategy] = result
            m = result["metrics"]
            print(f"  NDCG@10={m.get('NDCG@10',0):.4f}  HR@10={m.get('HR@10',0):.4f}  ({result['elapsed_seconds']/60:.1f} min)")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            all_results[strategy] = {"error": str(e)}

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {args.dataset}")
    print(f"{'Strategy':<26} {'NDCG@10':>9} {'HR@10':>8} {'NDCG@20':>9} {'HR@20':>8}")
    print("-" * 65)

    baseline_ndcg = None
    for s in STRATEGY_NAMES:
        r = all_results.get(s, {})
        if "metrics" not in r:
            print(f"{s:<26} {'ERROR':>9}")
            continue
        m = r["metrics"]
        ndcg10 = m.get("NDCG@10", 0)
        hr10 = m.get("HR@10", 0)
        ndcg20 = m.get("NDCG@20", 0)
        hr20 = m.get("HR@20", 0)
        if s == "type1_title_only":
            baseline_ndcg = ndcg10
            tag = " [BASELINE]"
        elif baseline_ndcg and ndcg10 > baseline_ndcg:
            tag = f" (+{(ndcg10 - baseline_ndcg)/baseline_ndcg*100:.1f}%)"
        else:
            tag = ""
        print(f"{s:<26} {ndcg10:>9.4f} {hr10:>8.4f} {ndcg20:>9.4f} {hr20:>8.4f}{tag}")

    print(f"\nResults saved -> {results_path}")


if __name__ == "__main__":
    main()

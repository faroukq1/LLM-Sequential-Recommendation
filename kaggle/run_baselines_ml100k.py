#!/usr/bin/env python3
"""Run project baselines on MovieLens 100K in one command."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main.data.session_dataset import SessionDataset
from main.data.temporal_split import TemporalSplit
from main.eval.evaluation import Evaluation
from main.eval.metrics.catalog_coverage import CatalogCoverage
from main.eval.metrics.hitrate import HitRate
from main.eval.metrics.metric import MetricDependency
from main.eval.metrics.mrr import MeanReciprocalRank
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from main.grurec.grurec import GRURec
from main.popularity.session_popular import SessionBasedPopular
from main.sknn.sknn import SessionBasedCF
from main.transformer.bert.bert import BERT
from main.transformer.sasrec.sasrec import SASRec

DEFAULT_MODELS = [
    "Popular",
    "SKNN",
    "S-SKNN",
    "SF-SKNN",
    "V-SKNN",
    "GRU4Rec",
    "BERT4Rec",
    "SASRec",
]


def _parse_top_ks(top_ks: str) -> list[int]:
    values: list[int] = []
    for raw in top_ks.split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"top-k values must be positive, got: {value}")
        values.append(value)

    if not values:
        raise ValueError("At least one top-k value is required.")

    return sorted(set(values))


def _parse_models(raw_models: str, available: set[str]) -> list[str]:
    requested: list[str] = [m.strip() for m in raw_models.split(",") if m.strip()]
    if not requested:
        raise ValueError("No models were requested.")

    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(
            "Unknown models requested: "
            + ", ".join(unknown)
            + ". Available models: "
            + ", ".join(sorted(available))
        )

    # Keep order and remove duplicates.
    unique_requested: list[str] = []
    seen: set[str] = set()
    for model_name in requested:
        if model_name not in seen:
            unique_requested.append(model_name)
            seen.add(model_name)

    return unique_requested


def _download_movielens_100k(data_dir: Path, dataset_url: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    udata_path = data_dir / "ml-100k" / "u.data"
    if udata_path.exists():
        logging.info("Found existing MovieLens-100K at %s", udata_path)
        return udata_path

    archive_name = dataset_url.rsplit("/", 1)[-1]
    archive_path = data_dir / archive_name

    if not archive_path.exists():
        logging.info("Downloading %s", dataset_url)
        urllib.request.urlretrieve(dataset_url, archive_path)

    logging.info("Extracting %s", archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(data_dir)
    else:
        raise ValueError(
            f"Unsupported archive format for {archive_path}. Use a .zip or .tar.gz file."
        )

    if not udata_path.exists():
        raise FileNotFoundError(
            f"Expected MovieLens interactions file at {udata_path}, but it was not found."
        )
    return udata_path


def _load_movielens_interactions(udata_path: Path) -> pd.DataFrame:
    # u.data format: user_id, item_id, rating, timestamp separated by tabs.
    return pd.read_csv(
        udata_path,
        sep="\t",
        names=["UserId", "ItemId", "Rating", "Timestamp"],
        engine="python",
    )


def _sessionize(
    interactions: pd.DataFrame,
    session_gap_minutes: int,
    min_session_len: int,
) -> pd.DataFrame:
    if min_session_len < 2:
        raise ValueError("min_session_len must be at least 2 for next-item evaluation.")

    gap_seconds = session_gap_minutes * 60

    df = interactions.copy()
    df["Timestamp"] = df["Timestamp"].astype(int)
    df["ItemId"] = df["ItemId"].astype(int)
    df["Rating"] = df["Rating"].astype(float)

    df = df.sort_values(["UserId", "Timestamp", "ItemId"]).reset_index(drop=True)

    user_changed = df["UserId"].ne(df["UserId"].shift(1))
    ts_gap = df["Timestamp"].diff().fillna(0)
    new_session = user_changed | (ts_gap > gap_seconds)

    # Session ids are globally unique and stable for this conversion run.
    df["SessionId"] = new_session.cumsum().astype(int)

    session_sizes = df.groupby("SessionId").size()
    valid_sessions = session_sizes[session_sizes >= min_session_len].index
    df = df[df["SessionId"].isin(valid_sessions)].copy()

    # Re-map to consecutive ids for compactness.
    new_ids = {
        old_id: idx + 1 for idx, old_id in enumerate(sorted(df["SessionId"].unique()))
    }
    df["SessionId"] = df["SessionId"].map(new_ids).astype(int)

    df["Time"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True).dt.tz_localize(None)
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df["Reward"] = df["Rating"].astype(float)

    out_df = df[["SessionId", "ItemId", "Time", "Reward"]]
    out_df = out_df.sort_values(["SessionId", "Time", "ItemId"]).reset_index(drop=True)
    return out_df


def _build_dataset_pickle(
    sessions_csv_path: Path,
    dataset_pickle_path: Path,
    test_frac: float,
    num_folds: int,
    filter_non_trained_test_items: bool,
) -> SessionDataset:
    dataset = SessionDataset(str(sessions_csv_path), n_withheld=1, evolving=False)
    split_strategy = TemporalSplit(
        test_frac=test_frac,
        num_folds=num_folds,
        filter_non_trained_test_items=filter_non_trained_test_items,
        fold_strategy="chain",
    )
    dataset.load_and_split(split_strategy=split_strategy)
    dataset.to_pickle(str(dataset_pickle_path))
    return dataset


def _instantiate_models(args: argparse.Namespace) -> dict[str, object]:
    shared = {
        "cores": args.cores,
        "is_verbose": args.verbose,
    }
    neural_common = {
        "cores": args.cores,
        "is_verbose": args.verbose,
        "num_epochs": args.num_epochs,
        "fit_batch_size": args.fit_batch_size,
        "pred_batch_size": args.pred_batch_size,
        "train_val_fraction": args.train_val_fraction,
        "early_stopping_patience": args.early_stopping_patience,
        "pred_seen": False,
    }
    optimizer_kwargs = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }

    return {
        "Popular": SessionBasedPopular(**shared),
        "SKNN": SessionBasedCF(
            k=args.sknn_k,
            sample_size=args.sknn_sample_size,
            sampling="recent",
            similarity_measure="cosine",
            idf_weighting=True,
            filter_prompt_items=True,
            **shared,
        ),
        "S-SKNN": SessionBasedCF(
            k=args.sknn_k,
            sample_size=args.sknn_sample_size,
            sampling="recent",
            similarity_measure="cosine",
            idf_weighting=True,
            sequential_weighting=True,
            filter_prompt_items=True,
            **shared,
        ),
        "SF-SKNN": SessionBasedCF(
            k=args.sknn_k,
            sample_size=args.sknn_sample_size,
            sampling="recent",
            similarity_measure="cosine",
            idf_weighting=True,
            sequential_filter=True,
            filter_prompt_items=True,
            **shared,
        ),
        "V-SKNN": SessionBasedCF(
            k=args.sknn_k,
            sample_size=args.sknn_sample_size,
            sampling="recent",
            similarity_measure="dot",
            idf_weighting=True,
            decay="harmonic",
            filter_prompt_items=True,
            **shared,
        ),
        "GRU4Rec": GRURec(
            N=args.max_seq_len,
            emb_dim=args.neural_emb_dim,
            hidden_dim=args.hidden_dim,
            drop_rate=args.drop_rate,
            activation="relu",
            optimizer_kwargs=optimizer_kwargs,
            **neural_common,
        ),
        "BERT4Rec": BERT(
            N=args.max_seq_len,
            L=1,
            h=2,
            emb_dim=args.neural_emb_dim,
            drop_rate=args.drop_rate,
            mask_prob=args.mask_prob,
            activation="gelu",
            optimizer_kwargs=optimizer_kwargs,
            transformer_layer_kwargs={"layout": "FDRN"},
            **neural_common,
        ),
        "SASRec": SASRec(
            N=args.max_seq_len,
            L=1,
            h=2,
            emb_dim=args.neural_emb_dim,
            drop_rate=args.drop_rate,
            activation="relu",
            optimizer_kwargs=optimizer_kwargs,
            transformer_layer_kwargs={"layout": "NFDR"},
            **neural_common,
        ),
    }


def _run_single_model(
    model: object,
    dataset: SessionDataset,
    top_ks: list[int],
    eval_cores: int,
    recommendations_dir: Path,
) -> dict[str, float | str]:
    model_name = model.name()
    max_top_k = max(top_ks)

    start = time.perf_counter()
    model.train(dataset.get_train_data())
    recommendations = model.predict(dataset.get_test_prompts(), top_k=max_top_k)
    elapsed = time.perf_counter() - start

    recs_path = recommendations_dir / f"recs_{model_name}.pickle"
    with open(recs_path, mode="wb") as write_file:
        pickle.dump(recommendations, write_file)

    metric_list = [
        NormalizedDiscountedCumulativeGain(),
        HitRate(),
        MeanReciprocalRank(),
        CatalogCoverage(),
    ]
    dependencies = {
        MetricDependency.NUM_ITEMS: dataset.get_unique_item_count(),
    }

    row: dict[str, float | str] = {
        "Model": model_name,
        "TrainPredictSeconds": round(elapsed, 4),
    }
    for top_k in top_ks:
        report = Evaluation.eval(
            predictions=recommendations,
            ground_truths=dataset.get_test_ground_truths(),
            top_k=top_k,
            metrics=metric_list,
            metrics_per_sample=False,
            dependencies=dependencies,
            cores=eval_cores,
            model_name=model_name,
        )
        for metric_name, value in report.results.items():
            row[metric_name] = float(value)

    return row


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download MovieLens-100K, convert to session format, and run project "
            "baselines."
        )
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        default="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        help="MovieLens-100K archive URL.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(PROJECT_ROOT / "kaggle" / "artifacts"),
        help="Output folder for data, recommendations, and evaluation tables.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--top-ks",
        type=str,
        default="10,20",
        help="Comma-separated list of cutoff values for evaluation.",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Gap in minutes to start a new session for the same user.",
    )
    parser.add_argument(
        "--min-session-len",
        type=int,
        default=2,
        help="Drop sessions shorter than this length.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of sessions for temporal test split.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=0,
        help="Optional number of temporal folds for dataset object.",
    )
    parser.add_argument(
        "--filter-non-trained-test-items",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter test items that are unseen in train.",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Cores used by models that support multiprocessing.",
    )
    parser.add_argument(
        "--eval-cores",
        type=int,
        default=1,
        help="Cores used during metric computation.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=6,
        help="Max epochs for neural baselines.",
    )
    parser.add_argument(
        "--fit-batch-size",
        type=int,
        default=256,
        help="Training batch size for neural baselines.",
    )
    parser.add_argument(
        "--pred-batch-size",
        type=int,
        default=4096,
        help="Prediction batch size for neural baselines.",
    )
    parser.add_argument(
        "--train-val-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for neural early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Early-stopping patience for neural models.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=20,
        help="Session truncation length for neural models.",
    )
    parser.add_argument(
        "--neural-emb-dim",
        type=int,
        default=64,
        help="Embedding dimension for neural models.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for GRU4Rec.",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.2,
        help="Dropout for neural models.",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.2,
        help="Masking probability for BERT4Rec.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for neural models.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay for neural models.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one model fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose model logs.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    work_dir = Path(args.work_dir).resolve()
    data_dir = work_dir / "data"
    recommendations_dir = work_dir / "recommendations"
    results_dir = work_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    recommendations_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    udata_path = _download_movielens_100k(data_dir=data_dir, dataset_url=args.dataset_url)
    raw_interactions = _load_movielens_interactions(udata_path)
    sessions_df = _sessionize(
        raw_interactions,
        session_gap_minutes=args.session_gap_minutes,
        min_session_len=args.min_session_len,
    )

    sessions_csv_path = data_dir / "ml100k_sessions.csv"
    sessions_df.to_csv(sessions_csv_path, index=False)

    dataset_pickle_path = data_dir / "ml100k_session_dataset.pickle"
    dataset = _build_dataset_pickle(
        sessions_csv_path=sessions_csv_path,
        dataset_pickle_path=dataset_pickle_path,
        test_frac=args.test_frac,
        num_folds=args.num_folds,
        filter_non_trained_test_items=args.filter_non_trained_test_items,
    )

    all_models = _instantiate_models(args)
    model_names = _parse_models(args.models, available=set(all_models.keys()))
    top_ks = _parse_top_ks(args.top_ks)

    logging.info("Running models: %s", ", ".join(model_names))
    logging.info("Top-k cutoffs: %s", ", ".join(str(k) for k in top_ks))
    logging.info(
        "Dataset stats: %s interactions, %s sessions, %s unique items",
        dataset.get_num_interactions(),
        dataset.get_unique_sample_count(),
        dataset.get_unique_item_count(),
    )

    rows: list[dict[str, float | str]] = []
    failures: list[dict[str, str]] = []

    for model_name in model_names:
        model = all_models[model_name]
        logging.info("Starting model %s", model_name)
        try:
            row = _run_single_model(
                model=model,
                dataset=dataset,
                top_ks=top_ks,
                eval_cores=args.eval_cores,
                recommendations_dir=recommendations_dir,
            )
            rows.append(row)
            logging.info("Finished model %s", model_name)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            failures.append({"Model": model_name, "Error": error_message})
            logging.exception("Model %s failed", model_name)
            if args.fail_fast:
                raise

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        primary_col = f"NDCG@{max(top_ks)}"
        if primary_col in results_df.columns:
            results_df = results_df.sort_values(primary_col, ascending=False)

    results_csv = results_dir / "baseline_results_ml100k.csv"
    results_df.to_csv(results_csv, index=False)

    failures_csv = results_dir / "baseline_failures_ml100k.csv"
    pd.DataFrame(failures).to_csv(failures_csv, index=False)

    print("\n=== Baseline Results ===")
    if results_df.empty:
        print("No successful model runs.")
    else:
        print(results_df.to_string(index=False))

    print(f"\nSaved results: {results_csv}")
    print(f"Saved failures: {failures_csv}")
    print(f"Saved recommendations: {recommendations_dir}")
    print(f"Saved dataset pickle: {dataset_pickle_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run PopDebias + ColdBridge novelty experiments on MovieLens-100K.

This script is designed for local runs and Kaggle GPU notebooks. It:
1. Downloads MovieLens-100K and sessionizes interactions.
2. Builds BGE-M3 item embeddings from movie title + genres.
3. Trains core baselines (MostPopular, SASRec, BGE2SASRec).
4. Runs novelty ablations (alpha sweep, tau sweep, combined variants).
5. Evaluates NDCG/HR/Coverage/Serendipity + LongTail_HR@10 + ILD@10.
6. Saves results tables, best model summary, and analysis figures.
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main.eval.evaluation import Evaluation
from main.eval.metrics.catalog_coverage import CatalogCoverage
from main.eval.metrics.hitrate import HitRate
from main.eval.metrics.metric import MetricDependency
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from main.eval.metrics.serendipity import Serendipity
from main.popularity.session_popular import SessionBasedPopular
from main.transformer.bert.bert_with_embeddings import BERTWithEmbeddings
from main.transformer.sasrec.sasrec import SASRec
from main.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieLens-100K novelty experiment runner")

    parser.add_argument(
        "--dataset-url",
        type=str,
        default="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        help="MovieLens-100K archive URL.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="kaggle/artifacts/novelty_ml100k",
        help="Output directory for data, embeddings, and results.",
    )
    parser.add_argument(
        "--session-gap-minutes",
        type=int,
        default=30,
        help="Start a new session if same-user gap exceeds this value.",
    )
    parser.add_argument(
        "--min-session-len",
        type=int,
        default=2,
        help="Drop sessions shorter than this length.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Validation split fraction (temporal, by sessions).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Test split fraction (temporal, by sessions).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Max recommendation cutoff used for evaluation/export.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=300,
        help="Candidate list size before reranking.",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.1,0.3,0.5,0.7",
        help="Comma-separated alpha sweep values for PopDebias.",
    )
    parser.add_argument(
        "--tau-values",
        type=str,
        default="2,3,5,10,15",
        help="Comma-separated tau sweep values for ColdBridge.",
    )
    parser.add_argument(
        "--decay-values",
        type=str,
        default="0.5,0.7,0.8,0.9,1.0",
        help="Comma-separated decay sweep values for weighted cold branch.",
    )
    parser.add_argument(
        "--long-tail-threshold",
        type=int,
        default=500,
        help="Items with train-count < threshold are considered long-tail.",
    )
    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=256,
        help="Batch size for BGE-M3 embedding generation.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=256,
        help="Max token length for BGE-M3 encoder.",
    )
    parser.add_argument(
        "--force-regenerate-embeddings",
        action="store_true",
        help="Regenerate BGE embeddings even if cached .npy file exists.",
    )
    parser.add_argument(
        "--include-bert-backbone",
        action="store_true",
        help="Run optional BERT4Rec embedding-initialized backbone and combined variant.",
    )

    # Neural model settings (kept close to existing baselines).
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--fit-batch-size", type=int, default=256)
    parser.add_argument("--pred-batch-size", type=int, default=4096)
    parser.add_argument("--train-val-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=20)
    parser.add_argument("--neural-emb-dim", type=int, default=64)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _parse_csv_values(raw: str, cast: Any) -> list[Any]:
    values: list[Any] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        values.append(cast(item))
    return values


def download_movielens_100k(data_dir: Path, dataset_url: str) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    udata_path = data_dir / "ml-100k" / "u.data"
    uitem_path = data_dir / "ml-100k" / "u.item"

    if udata_path.exists() and uitem_path.exists():
        logging.info("Found cached MovieLens-100K files.")
        return udata_path, uitem_path

    archive_name = dataset_url.rsplit("/", 1)[-1]
    archive_path = data_dir / archive_name
    if not archive_path.exists():
        logging.info("Downloading dataset from %s", dataset_url)
        urllib.request.urlretrieve(dataset_url, archive_path)

    logging.info("Extracting dataset archive: %s", archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    elif archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(data_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    if not (udata_path.exists() and uitem_path.exists()):
        raise FileNotFoundError("MovieLens-100K files not found after extraction.")

    return udata_path, uitem_path


def load_movielens_interactions(udata_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        udata_path,
        sep="\t",
        names=["UserId", "ItemId", "Rating", "Timestamp"],
        engine="python",
    )


def sessionize(
    interactions: pd.DataFrame,
    session_gap_minutes: int,
    min_session_len: int,
) -> pd.DataFrame:
    if min_session_len < 2:
        raise ValueError("min_session_len must be at least 2.")

    gap_seconds = session_gap_minutes * 60

    df = interactions.copy()
    df["Timestamp"] = df["Timestamp"].astype(int)
    df["ItemId"] = df["ItemId"].astype(int)
    df["Rating"] = df["Rating"].astype(float)

    df = df.sort_values(["UserId", "Timestamp", "ItemId"]).reset_index(drop=True)

    user_changed = df["UserId"].ne(df["UserId"].shift(1))
    ts_gap = df["Timestamp"].diff().fillna(0)
    new_session = user_changed | (ts_gap > gap_seconds)
    df["SessionId"] = new_session.cumsum().astype(int)

    session_sizes = df.groupby("SessionId").size()
    valid_sessions = session_sizes[session_sizes >= min_session_len].index
    df = df[df["SessionId"].isin(valid_sessions)].copy()

    # Re-map to consecutive IDs for compactness.
    remap = {old_id: idx + 1 for idx, old_id in enumerate(sorted(df["SessionId"].unique()))}
    df["SessionId"] = df["SessionId"].map(remap).astype(int)

    df["Time"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True).dt.tz_localize(None)
    df["Time"] = df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df["Reward"] = df["Rating"].astype(float)

    out = df[["SessionId", "ItemId", "Time", "Reward"]].copy()
    out = out.sort_values(["SessionId", "Time", "ItemId"]).reset_index(drop=True)
    return out


def temporal_train_val_test_split(
    sessions_df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1:
        raise ValueError("Require 0 < val_frac, test_frac and val_frac + test_frac < 1.")

    df = sessions_df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    session_end = df.groupby("SessionId")["Time"].max().sort_values()
    session_ids = session_end.index.to_numpy()
    n_sessions = len(session_ids)

    n_test = int(round(n_sessions * test_frac))
    n_val = int(round(n_sessions * val_frac))
    n_train = n_sessions - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Temporal split produced an empty partition.")

    train_ids = set(session_ids[:n_train])
    val_ids = set(session_ids[n_train : n_train + n_val])
    test_ids = set(session_ids[n_train + n_val :])

    train_df = df[df["SessionId"].isin(train_ids)].copy()
    val_df = df[df["SessionId"].isin(val_ids)].copy()
    test_df = df[df["SessionId"].isin(test_ids)].copy()

    train_df = train_df.sort_values(["SessionId", "Time", "ItemId"]).reset_index(drop=True)
    val_df = val_df.sort_values(["SessionId", "Time", "ItemId"]).reset_index(drop=True)
    test_df = test_df.sort_values(["SessionId", "Time", "ItemId"]).reset_index(drop=True)

    logging.info(
        "Temporal split sessions -> train=%d, val=%d, test=%d",
        train_df["SessionId"].nunique(),
        val_df["SessionId"].nunique(),
        test_df["SessionId"].nunique(),
    )

    return train_df, val_df, test_df


def build_eval_cases(
    split_df: pd.DataFrame,
    n_withheld: int = 1,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    prompts: dict[int, np.ndarray] = {}
    ground_truths: dict[int, np.ndarray] = {}
    lengths: dict[int, int] = {}

    for session_id, cur in split_df.groupby("SessionId"):
        items = cur.sort_values("Time")["ItemId"].to_numpy(dtype=int)
        if len(items) <= n_withheld:
            continue

        prompt = items[:-n_withheld]
        gt = items[-n_withheld:]

        if len(prompt) == 0:
            continue

        sid = int(session_id)
        prompts[sid] = prompt
        ground_truths[sid] = gt
        lengths[sid] = int(len(prompt))

    return prompts, ground_truths, lengths


def load_movielens_item_metadata(uitem_path: Path) -> pd.DataFrame:
    genre_cols = [
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western",
    ]
    cols = [
        "ItemId",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *genre_cols,
    ]

    df = pd.read_csv(
        uitem_path,
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=list(range(len(cols))),
        engine="python",
    )

    def make_text(row: pd.Series) -> str:
        genres = [g.replace("_", " ") for g in genre_cols if int(row[g]) == 1]
        genre_text = ", ".join(genres) if genres else "unknown"
        return f"{row['title']}. Genres: {genre_text}."

    df["item_text"] = df.apply(make_text, axis=1)
    return df[["ItemId", "title", "item_text"]]


def generate_bge_m3_embeddings(
    item_text_df: pd.DataFrame,
    all_item_ids: np.ndarray,
    output_npy_path: Path,
    batch_size: int,
    max_text_length: int,
    force_regenerate: bool,
) -> tuple[np.ndarray, np.ndarray]:
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)

    item_text_df = item_text_df.copy()
    item_text_df["ItemId"] = item_text_df["ItemId"].astype(int)

    all_item_ids = np.array(sorted(set(int(i) for i in all_item_ids)), dtype=int)

    metadata_lookup = item_text_df.set_index("ItemId")["item_text"].to_dict()
    texts = [metadata_lookup.get(int(i), f"Movie item {int(i)}.") for i in all_item_ids]

    if output_npy_path.exists() and not force_regenerate:
        embeddings = np.load(output_npy_path)
        if embeddings.shape[0] == len(all_item_ids):
            logging.info("Loaded cached BGE embeddings from %s", output_npy_path)
            return all_item_ids, embeddings
        logging.warning("Cached embedding shape mismatch, regenerating: %s", output_npy_path)

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise ImportError(
            "FlagEmbedding is required. Install with: pip install -U FlagEmbedding"
        ) from exc

    logging.info("Generating BGE-M3 embeddings for %d items...", len(texts))
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        cur = texts[start : start + batch_size]
        out = model.encode(
            cur,
            batch_size=batch_size,
            max_length=max_text_length,
        )["dense_vecs"]
        chunks.append(np.asarray(out, dtype=np.float32))

    embeddings = np.vstack(chunks).astype(np.float32)
    np.save(output_npy_path, embeddings)
    logging.info("Saved BGE embeddings to %s", output_npy_path)

    return all_item_ids, embeddings


def build_item_embedding_dataframes(
    item_ids: np.ndarray,
    embeddings: np.ndarray,
    embeddings_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb_df = pd.DataFrame(
        {
            "ItemId": item_ids.astype(int),
            "embedding": [embeddings[i].astype(np.float32) for i in range(len(item_ids))],
            "class": 0,
        }
    )

    csv_df = emb_df[["ItemId", "embedding", "class"]].copy()
    csv_df["embedding"] = csv_df["embedding"].apply(
        lambda arr: json.dumps([float(x) for x in arr])
    )

    embeddings_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(embeddings_csv_path, index=False)
    logging.info("Saved embedding CSV for LLM2SASRec/BERT: %s", embeddings_csv_path)

    return emb_df, csv_df


def instantiate_sasrec(args: argparse.Namespace) -> SASRec:
    return SASRec(
        N=args.max_seq_len,
        L=1,
        h=2,
        emb_dim=args.neural_emb_dim,
        drop_rate=args.drop_rate,
        activation="relu",
        optimizer_kwargs={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        transformer_layer_kwargs={"layout": "NFDR"},
        cores=args.cores,
        is_verbose=args.verbose,
        num_epochs=args.num_epochs,
        fit_batch_size=args.fit_batch_size,
        pred_batch_size=args.pred_batch_size,
        train_val_fraction=args.train_val_fraction,
        early_stopping_patience=args.early_stopping_patience,
        pred_seen=False,
    )


def instantiate_llm2sasrec(
    args: argparse.Namespace,
    embedding_csv_path: Path,
) -> SASRecWithEmbeddings:
    return SASRecWithEmbeddings(
        product_embeddings_location=str(embedding_csv_path),
        red_method="PCA",
        red_params={},
        N=args.max_seq_len,
        L=1,
        h=2,
        emb_dim=args.neural_emb_dim,
        drop_rate=args.drop_rate,
        activation="relu",
        optimizer_kwargs={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        transformer_layer_kwargs={"layout": "NFDR"},
        cores=args.cores,
        is_verbose=args.verbose,
        num_epochs=args.num_epochs,
        fit_batch_size=args.fit_batch_size,
        pred_batch_size=args.pred_batch_size,
        train_val_fraction=args.train_val_fraction,
        early_stopping_patience=args.early_stopping_patience,
        pred_seen=False,
    )


def instantiate_llm2bert(
    args: argparse.Namespace,
    embedding_csv_path: Path,
) -> BERTWithEmbeddings:
    return BERTWithEmbeddings(
        product_embeddings_location=str(embedding_csv_path),
        red_method="PCA",
        red_params={},
        N=args.max_seq_len,
        L=1,
        h=2,
        emb_dim=args.neural_emb_dim,
        drop_rate=args.drop_rate,
        activation="gelu",
        mask_prob=args.mask_prob,
        optimizer_kwargs={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        transformer_layer_kwargs={"layout": "FDRN"},
        cores=args.cores,
        is_verbose=args.verbose,
        num_epochs=args.num_epochs,
        fit_batch_size=args.fit_batch_size,
        pred_batch_size=args.pred_batch_size,
        train_val_fraction=args.train_val_fraction,
        early_stopping_patience=args.early_stopping_patience,
        pred_seen=False,
    )


def fit_model(model: Any, train_df: pd.DataFrame) -> float:
    start = time.perf_counter()
    model.train(train_df)
    end = time.perf_counter()
    return end - start


def predict_model(
    model: Any,
    prompts: dict[int, np.ndarray],
    top_k: int,
) -> tuple[dict[int, np.ndarray], float]:
    start = time.perf_counter()
    preds = model.predict(prompts, top_k=top_k)
    end = time.perf_counter()
    return preds, end - start


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def get_session_embedding(
    prompt_items: np.ndarray,
    item_to_index: dict[int, int],
    item_emb_norm: np.ndarray,
    decay: float = 1.0,
) -> np.ndarray | None:
    idxs = [item_to_index[int(i)] for i in prompt_items if int(i) in item_to_index]
    if not idxs:
        return None

    vecs = item_emb_norm[idxs]
    if len(vecs) == 1:
        return vecs[0]

    if decay == 1.0:
        emb = vecs.mean(axis=0)
    else:
        n = len(vecs)
        weights = np.array([decay ** (n - i - 1) for i in range(n)], dtype=np.float32)
        weights = weights / np.sum(weights)
        emb = np.average(vecs, axis=0, weights=weights)

    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm


def _debias_weight(
    counts: np.ndarray,
    alpha: float,
    variant: str,
    ranks: np.ndarray,
    median_count: float,
) -> np.ndarray:
    c = np.maximum(counts.astype(np.float64), 1.0)

    if variant == "inverse":
        return 1.0 / np.power(c, alpha)

    if variant == "log_norm":
        return 1.0 / np.power(1.0 + np.log(c), alpha)

    if variant == "rank":
        return 1.0 / np.power(np.maximum(ranks.astype(np.float64), 1.0), alpha)

    if variant == "sigmoid":
        # sigmoid(-alpha * (count/median - 1))
        x = -alpha * ((c / max(median_count, 1.0)) - 1.0)
        return 1.0 / (1.0 + np.exp(-x))

    raise ValueError(f"Unknown debias variant: {variant}")


def rerank_popdebias(
    base_candidates: dict[int, np.ndarray],
    prompts: dict[int, np.ndarray],
    item_to_index: dict[int, int],
    item_emb_norm: np.ndarray,
    item_counts: dict[int, int],
    alpha: float,
    top_k: int,
    variant: str = "inverse",
) -> dict[int, np.ndarray]:
    reranked: dict[int, np.ndarray] = {}

    # Popularity ranks (1 = most popular).
    sorted_items = sorted(item_counts.items(), key=lambda kv: kv[1], reverse=True)
    pop_rank_map = {int(item): idx + 1 for idx, (item, _) in enumerate(sorted_items)}
    median_count = float(np.median(np.array(list(item_counts.values()), dtype=np.float64)))

    for sid, candidates in base_candidates.items():
        prompt = prompts[sid]
        query = get_session_embedding(prompt, item_to_index, item_emb_norm, decay=1.0)
        if query is None:
            reranked[sid] = np.array(candidates[:top_k], dtype=int)
            continue

        cand = np.array([int(i) for i in candidates], dtype=int)
        cand = np.array([i for i in cand if i in item_to_index], dtype=int)
        if cand.size == 0:
            reranked[sid] = np.array([], dtype=int)
            continue

        cand_idxs = np.array([item_to_index[int(i)] for i in cand], dtype=int)
        sims = item_emb_norm[cand_idxs] @ query

        counts = np.array([item_counts.get(int(i), 1) for i in cand], dtype=np.float64)
        ranks = np.array([pop_rank_map.get(int(i), len(pop_rank_map) + 1) for i in cand], dtype=np.float64)

        weights = _debias_weight(counts, alpha, variant, ranks, median_count)
        scores = sims * weights

        # Exclude prompt items.
        seen = set(int(i) for i in prompt)
        if seen:
            mask = np.array([i not in seen for i in cand], dtype=bool)
            cand = cand[mask]
            scores = scores[mask]

        if cand.size == 0:
            reranked[sid] = np.array([], dtype=int)
            continue

        order = np.argsort(scores)[::-1]
        reranked[sid] = cand[order][:top_k].astype(int)

    return reranked


def predict_cold_branch(
    prompts: dict[int, np.ndarray],
    all_item_ids: np.ndarray,
    item_to_index: dict[int, int],
    item_emb_norm: np.ndarray,
    top_k: int,
    decay: float,
) -> dict[int, np.ndarray]:
    predictions: dict[int, np.ndarray] = {}

    for sid, prompt in prompts.items():
        query = get_session_embedding(prompt, item_to_index, item_emb_norm, decay=decay)
        if query is None:
            predictions[sid] = np.array(all_item_ids[:top_k], dtype=int)
            continue

        sims = item_emb_norm @ query
        scores = sims.copy()

        # Remove seen items.
        seen_idxs = [item_to_index[int(i)] for i in prompt if int(i) in item_to_index]
        if seen_idxs:
            scores[np.array(seen_idxs, dtype=int)] = -1e12

        order = np.argsort(scores)[::-1][:top_k]
        predictions[sid] = all_item_ids[order].astype(int)

    return predictions


def route_coldbridge(
    warm_predictions: dict[int, np.ndarray],
    cold_predictions: dict[int, np.ndarray],
    prompts: dict[int, np.ndarray],
    tau: int,
    top_k: int,
) -> dict[int, np.ndarray]:
    routed: dict[int, np.ndarray] = {}
    for sid, prompt in prompts.items():
        if len(prompt) <= tau:
            routed[sid] = np.array(cold_predictions[sid][:top_k], dtype=int)
        else:
            routed[sid] = np.array(warm_predictions[sid][:top_k], dtype=int)
    return routed


def long_tail_hitrate(
    predictions: dict[int, np.ndarray],
    ground_truths: dict[int, np.ndarray],
    item_counts: dict[int, int],
    threshold: int,
    top_k: int,
) -> float:
    tail_items = {int(i) for i, c in item_counts.items() if int(c) < threshold}

    hits = 0
    total = 0
    for sid, gt in ground_truths.items():
        gt_tail = [int(i) for i in gt if int(i) in tail_items]
        if not gt_tail:
            continue

        total += 1
        pred = [int(i) for i in predictions.get(sid, np.array([], dtype=int))[:top_k]]
        if any(i in gt_tail for i in pred):
            hits += 1

    if total == 0:
        return 0.0
    return hits / total


def intra_list_diversity(
    predictions: dict[int, np.ndarray],
    item_to_index: dict[int, int],
    item_emb_norm: np.ndarray,
    top_k: int,
) -> float:
    ild_vals: list[float] = []

    for recs in predictions.values():
        rec_ids = [int(i) for i in recs[:top_k] if int(i) in item_to_index]
        if len(rec_ids) < 2:
            continue

        idxs = np.array([item_to_index[i] for i in rec_ids], dtype=int)
        vecs = item_emb_norm[idxs]
        sims = vecs @ vecs.T

        iu = np.triu_indices(len(rec_ids), k=1)
        if len(iu[0]) == 0:
            continue
        distances = 1.0 - sims[iu]
        ild_vals.append(float(np.mean(distances)))

    if not ild_vals:
        return 0.0
    return float(np.mean(ild_vals))


def segmented_hitrate(
    predictions: dict[int, np.ndarray],
    ground_truths: dict[int, np.ndarray],
    prompts: dict[int, np.ndarray],
    segment_threshold: int,
    top_k: int,
) -> tuple[float, float]:
    cold_hits = 0
    cold_total = 0
    warm_hits = 0
    warm_total = 0

    for sid, gt in ground_truths.items():
        pred = predictions.get(sid, np.array([], dtype=int))[:top_k]
        hit = int(len(np.intersect1d(pred, gt, assume_unique=False)) > 0)

        if len(prompts[sid]) <= segment_threshold:
            cold_total += 1
            cold_hits += hit
        else:
            warm_total += 1
            warm_hits += hit

    cold_hr = cold_hits / cold_total if cold_total > 0 else 0.0
    warm_hr = warm_hits / warm_total if warm_total > 0 else 0.0
    return cold_hr, warm_hr


def evaluate_predictions(
    model_name: str,
    predictions: dict[int, np.ndarray],
    ground_truths: dict[int, np.ndarray],
    item_counts: dict[int, int],
    num_items: int,
    item_to_index: dict[int, int],
    item_emb_norm: np.ndarray,
    long_tail_threshold: int,
    alpha: float | None,
    tau: int | None,
    training_time_sec: float,
    inference_time_sec: float,
) -> dict[str, Any]:
    metric_list = [
        NormalizedDiscountedCumulativeGain(),
        HitRate(),
        CatalogCoverage(),
        Serendipity(),
    ]

    deps = {
        MetricDependency.NUM_ITEMS: num_items,
        MetricDependency.ITEM_COUNT: item_counts,
    }

    report10 = Evaluation.eval(
        predictions=predictions,
        ground_truths=ground_truths,
        top_k=10,
        metrics=metric_list,
        metrics_per_sample=False,
        dependencies=deps,
        cores=1,
        model_name=model_name,
    )
    report20 = Evaluation.eval(
        predictions=predictions,
        ground_truths=ground_truths,
        top_k=20,
        metrics=metric_list,
        metrics_per_sample=False,
        dependencies=deps,
        cores=1,
        model_name=model_name,
    )

    lt_hr10 = long_tail_hitrate(
        predictions,
        ground_truths,
        item_counts=item_counts,
        threshold=long_tail_threshold,
        top_k=10,
    )
    ild10 = intra_list_diversity(
        predictions,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        top_k=10,
    )

    row = {
        "model_name": model_name,
        "alpha": alpha,
        "tau": tau,
        "NDCG@10": float(report10.results.get("NDCG@10", 0.0)),
        "NDCG@20": float(report20.results.get("NDCG@20", 0.0)),
        "HR@10": float(report10.results.get("HitRate@10", 0.0)),
        "HR@20": float(report20.results.get("HitRate@20", 0.0)),
        "LongTail_HR@10": float(lt_hr10),
        "CatalogCoverage": float(report10.results.get("Catalog coverage@10", 0.0)),
        "Serendipity": float(report10.results.get("Serendipity@10", 0.0)),
        "ILD@10": float(ild10),
        "training_time_sec": float(training_time_sec),
        "inference_time_sec": float(inference_time_sec),
    }
    return row


def choose_best_by_joint_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows passed for model selection.")

    scored = []
    for row in rows:
        joint = 0.7 * float(row["NDCG@10"]) + 0.3 * float(row["LongTail_HR@10"])
        scored.append((joint, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def save_best_model_txt(
    best_row: dict[str, Any],
    output_path: Path,
    baseline_row: dict[str, Any] | None,
) -> None:
    baseline_ndcg = None
    if baseline_row is not None:
        baseline_ndcg = float(baseline_row.get("NDCG@10", 0.0))

    best_ndcg = float(best_row["NDCG@10"])
    if baseline_ndcg and baseline_ndcg > 0:
        delta_pct = ((best_ndcg - baseline_ndcg) / baseline_ndcg) * 100.0
        delta_text = f"{delta_pct:+.2f}%"
        baseline_text = f"{baseline_ndcg:.6f}"
    else:
        delta_text = "N/A"
        baseline_text = "N/A"

    lines = [
        f"BEST MODEL: {best_row['model_name']}",
        f"Best Alpha: {best_row.get('alpha')}",
        f"Best Tau:   {best_row.get('tau')}",
        f"NDCG@10:    {best_ndcg:.6f} (vs baseline: {baseline_text}, delta: {delta_text})",
        f"HR@10:      {float(best_row['HR@10']):.6f}",
        f"LT-HR@10:   {float(best_row['LongTail_HR@10']):.6f}",
        f"ILD@10:     {float(best_row['ILD@10']):.6f}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_alpha_sweep(alpha_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not alpha_rows:
        return

    df = pd.DataFrame(alpha_rows).sort_values("alpha")

    plt.figure(figsize=(7, 5))
    plt.plot(df["alpha"], df["NDCG@10"], marker="o", label="NDCG@10")
    plt.plot(df["alpha"], df["LongTail_HR@10"], marker="s", label="LongTail_HR@10")
    plt.xlabel("alpha")
    plt.ylabel("Metric")
    plt.title("Alpha Sweep (MovieLens-100K)")
    plt.grid(alpha=0.25)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_tau_sweep(
    tau_stats: list[dict[str, float]],
    output_path: Path,
) -> None:
    if not tau_stats:
        return

    df = pd.DataFrame(tau_stats).sort_values("tau")

    plt.figure(figsize=(7, 5))
    plt.plot(df["tau"], df["cold_hr10"], marker="o", label="Cold users HR@10")
    plt.plot(df["tau"], df["warm_hr10"], marker="s", label="Warm users HR@10")
    plt.xlabel("tau")
    plt.ylabel("HR@10")
    plt.title("Tau Sweep (MovieLens-100K)")
    plt.grid(alpha=0.25)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_cold_warm_gap(
    baseline_cold_hr: float,
    baseline_warm_hr: float,
    bridge_cold_hr: float,
    bridge_warm_hr: float,
    output_path: Path,
) -> None:
    labels = ["Cold", "Warm"]
    base_vals = [baseline_cold_hr, baseline_warm_hr]
    bridge_vals = [bridge_cold_hr, bridge_warm_hr]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, base_vals, width, label="BGE2SASRec")
    plt.bar(x + width / 2, bridge_vals, width, label="PopDebias-ColdBridge")
    plt.xticks(x, labels)
    plt.ylabel("HR@10")
    plt.title("Cold vs Warm HR@10 Gap")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run() -> None:
    args = parse_args()
    set_seed(args.seed)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    alpha_values = _parse_csv_values(args.alpha_values, float)
    tau_values = _parse_csv_values(args.tau_values, int)
    decay_values = _parse_csv_values(args.decay_values, float)

    work_dir = Path(args.work_dir).resolve()
    data_dir = work_dir / "data"
    embeddings_dir = work_dir / "embeddings" / "movielens_100k"
    results_dir = work_dir / "results" / "movielens_100k"
    figures_dir = work_dir / "results" / "figures"

    for d in [data_dir, embeddings_dir, results_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    udata_path, uitem_path = download_movielens_100k(data_dir, args.dataset_url)
    interactions = load_movielens_interactions(udata_path)
    sessions = sessionize(
        interactions,
        session_gap_minutes=args.session_gap_minutes,
        min_session_len=args.min_session_len,
    )

    sessions_csv = data_dir / "ml100k_sessions.csv"
    sessions.to_csv(sessions_csv, index=False)

    train_df, val_df, test_df = temporal_train_val_test_split(
        sessions,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    val_prompts, val_gts, val_lengths = build_eval_cases(val_df)
    test_prompts, test_gts, test_lengths = build_eval_cases(test_df)

    all_item_ids = np.array(sorted(sessions["ItemId"].unique().tolist()), dtype=int)
    item_metadata = load_movielens_item_metadata(uitem_path)

    embedding_npy_path = embeddings_dir / "item_embeddings_bge_m3.npy"
    item_ids_for_emb, bge_embeddings = generate_bge_m3_embeddings(
        item_text_df=item_metadata,
        all_item_ids=all_item_ids,
        output_npy_path=embedding_npy_path,
        batch_size=args.bge_batch_size,
        max_text_length=args.max_text_length,
        force_regenerate=args.force_regenerate_embeddings,
    )

    embedding_csv_path = embeddings_dir / "ml100k_item_embeddings_bge_m3.csv"
    item_data_df, _ = build_item_embedding_dataframes(
        item_ids=item_ids_for_emb,
        embeddings=bge_embeddings,
        embeddings_csv_path=embedding_csv_path,
    )

    # Embedding lookup utilities for reranking/ILD.
    item_ids_for_emb = np.array(item_ids_for_emb, dtype=int)
    item_emb_norm = normalize_rows(bge_embeddings.astype(np.float32))
    item_to_index = {int(item_id): idx for idx, item_id in enumerate(item_ids_for_emb.tolist())}

    item_counts = train_df["ItemId"].value_counts().to_dict()
    num_items = int(len(item_ids_for_emb))

    max_candidate_k = min(args.candidate_k, num_items)
    top_k = min(args.top_k, num_items)

    rows_test: list[dict[str, Any]] = []

    # 1) MostPopular baseline
    pop_model = SessionBasedPopular(is_verbose=args.verbose, cores=args.cores)
    pop_train_t = fit_model(pop_model, train_df)
    pop_test_pred, pop_test_t = predict_model(pop_model, test_prompts, top_k=top_k)
    row_pop = evaluate_predictions(
        model_name="MostPopular",
        predictions=pop_test_pred,
        ground_truths=test_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=None,
        tau=None,
        training_time_sec=pop_train_t,
        inference_time_sec=pop_test_t,
    )
    rows_test.append(row_pop)

    # 2) SASRec baseline
    sasrec_model = instantiate_sasrec(args)
    sas_train_t = fit_model(sasrec_model, train_df)
    sas_test_pred, sas_test_t = predict_model(sasrec_model, test_prompts, top_k=top_k)
    row_sas = evaluate_predictions(
        model_name="SASRec",
        predictions=sas_test_pred,
        ground_truths=test_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=None,
        tau=None,
        training_time_sec=sas_train_t,
        inference_time_sec=sas_test_t,
    )
    rows_test.append(row_sas)

    # 3) BGE2SASRec baseline (LLM2SASRec with BGE-M3 embeddings)
    llm2sas_model = instantiate_llm2sasrec(args, embedding_csv_path)
    llm2sas_train_t = fit_model(llm2sas_model, train_df)

    base_val_candidates, base_val_t = predict_model(
        llm2sas_model,
        val_prompts,
        top_k=max_candidate_k,
    )
    base_test_candidates, base_test_t = predict_model(
        llm2sas_model,
        test_prompts,
        top_k=max_candidate_k,
    )

    bge2sas_val_top = {sid: recs[:top_k] for sid, recs in base_val_candidates.items()}
    bge2sas_test_top = {sid: recs[:top_k] for sid, recs in base_test_candidates.items()}

    row_bge2sas = evaluate_predictions(
        model_name="BGE2SASRec",
        predictions=bge2sas_test_top,
        ground_truths=test_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=None,
        tau=None,
        training_time_sec=llm2sas_train_t,
        inference_time_sec=base_test_t,
    )
    rows_test.append(row_bge2sas)

    # 4) PopDebias alpha sweep on top of BGE2SASRec candidates.
    alpha_rows_val: list[dict[str, Any]] = []
    alpha_rows_test: list[dict[str, Any]] = []

    for alpha in alpha_values:
        val_pred = rerank_popdebias(
            base_candidates=base_val_candidates,
            prompts=val_prompts,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            item_counts=item_counts,
            alpha=alpha,
            top_k=top_k,
            variant="inverse",
        )
        test_pred = rerank_popdebias(
            base_candidates=base_test_candidates,
            prompts=test_prompts,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            item_counts=item_counts,
            alpha=alpha,
            top_k=top_k,
            variant="inverse",
        )

        row_val = evaluate_predictions(
            model_name="PopDebias-BGE2SASRec-val",
            predictions=val_pred,
            ground_truths=val_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=alpha,
            tau=None,
            training_time_sec=0.0,
            inference_time_sec=0.0,
        )
        alpha_rows_val.append(row_val)

        row_test = evaluate_predictions(
            model_name="PopDebias-BGE2SASRec",
            predictions=test_pred,
            ground_truths=test_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=alpha,
            tau=None,
            training_time_sec=llm2sas_train_t,
            inference_time_sec=base_test_t,
        )
        alpha_rows_test.append(row_test)
        rows_test.append(row_test)

    best_alpha_row_val = choose_best_by_joint_metric(alpha_rows_val)
    best_alpha = float(best_alpha_row_val["alpha"])
    logging.info("Best alpha from validation: %.3f", best_alpha)

    # 5) ColdBridge tau sweep (uniform cold branch = decay 1.0).
    cold_val_uniform = predict_cold_branch(
        prompts=val_prompts,
        all_item_ids=item_ids_for_emb,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        top_k=top_k,
        decay=1.0,
    )
    cold_test_uniform = predict_cold_branch(
        prompts=test_prompts,
        all_item_ids=item_ids_for_emb,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        top_k=top_k,
        decay=1.0,
    )

    tau_rows_val: list[dict[str, Any]] = []
    tau_rows_test: list[dict[str, Any]] = []
    tau_sweep_stats: list[dict[str, float]] = []

    warm_val = {sid: recs[:top_k] for sid, recs in base_val_candidates.items()}
    warm_test = {sid: recs[:top_k] for sid, recs in base_test_candidates.items()}

    for tau in tau_values:
        val_pred = route_coldbridge(
            warm_predictions=warm_val,
            cold_predictions=cold_val_uniform,
            prompts=val_prompts,
            tau=tau,
            top_k=top_k,
        )
        test_pred = route_coldbridge(
            warm_predictions=warm_test,
            cold_predictions=cold_test_uniform,
            prompts=test_prompts,
            tau=tau,
            top_k=top_k,
        )

        row_val = evaluate_predictions(
            model_name="ColdBridge-BGE2SASRec-val",
            predictions=val_pred,
            ground_truths=val_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=None,
            tau=tau,
            training_time_sec=0.0,
            inference_time_sec=0.0,
        )
        tau_rows_val.append(row_val)

        row_test = evaluate_predictions(
            model_name="ColdBridge-BGE2SASRec",
            predictions=test_pred,
            ground_truths=test_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=None,
            tau=tau,
            training_time_sec=llm2sas_train_t,
            inference_time_sec=base_test_t,
        )
        tau_rows_test.append(row_test)
        rows_test.append(row_test)

        cold_hr, warm_hr = segmented_hitrate(
            predictions=test_pred,
            ground_truths=test_gts,
            prompts=test_prompts,
            segment_threshold=5,
            top_k=10,
        )
        tau_sweep_stats.append({"tau": float(tau), "cold_hr10": cold_hr, "warm_hr10": warm_hr})

    best_tau_row_val = choose_best_by_joint_metric(tau_rows_val)
    best_tau = int(best_tau_row_val["tau"])
    logging.info("Best tau from validation: %d", best_tau)

    # 6) Combined PopDebias + ColdBridge (best alpha + best tau).
    best_warm_val = rerank_popdebias(
        base_candidates=base_val_candidates,
        prompts=val_prompts,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        item_counts=item_counts,
        alpha=best_alpha,
        top_k=top_k,
        variant="inverse",
    )
    best_warm_test = rerank_popdebias(
        base_candidates=base_test_candidates,
        prompts=test_prompts,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        item_counts=item_counts,
        alpha=best_alpha,
        top_k=top_k,
        variant="inverse",
    )

    combined_val = route_coldbridge(
        warm_predictions=best_warm_val,
        cold_predictions=cold_val_uniform,
        prompts=val_prompts,
        tau=best_tau,
        top_k=top_k,
    )
    combined_test = route_coldbridge(
        warm_predictions=best_warm_test,
        cold_predictions=cold_test_uniform,
        prompts=test_prompts,
        tau=best_tau,
        top_k=top_k,
    )

    row_combined_val = evaluate_predictions(
        model_name="PopDebias-ColdBridge-BGE2SASRec-val",
        predictions=combined_val,
        ground_truths=val_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=best_alpha,
        tau=best_tau,
        training_time_sec=0.0,
        inference_time_sec=0.0,
    )

    row_combined_test = evaluate_predictions(
        model_name="PopDebias-ColdBridge-BGE2SASRec",
        predictions=combined_test,
        ground_truths=test_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=best_alpha,
        tau=best_tau,
        training_time_sec=llm2sas_train_t,
        inference_time_sec=base_test_t,
    )
    rows_test.append(row_combined_test)

    # 7) Improvement 2: Debias formula variants.
    variant_name_map = {
        "log_norm": "A_log_norm",
        "rank": "B_rank",
        "sigmoid": "C_sigmoid",
    }

    variant_eval_rows_val: list[dict[str, Any]] = []
    variant_eval_rows_test: list[dict[str, Any]] = []

    for variant in ["log_norm", "rank", "sigmoid"]:
        for alpha in [0.1, 0.3, 0.5]:
            warm_val_variant = rerank_popdebias(
                base_candidates=base_val_candidates,
                prompts=val_prompts,
                item_to_index=item_to_index,
                item_emb_norm=item_emb_norm,
                item_counts=item_counts,
                alpha=alpha,
                top_k=top_k,
                variant=variant,
            )
            warm_test_variant = rerank_popdebias(
                base_candidates=base_test_candidates,
                prompts=test_prompts,
                item_to_index=item_to_index,
                item_emb_norm=item_emb_norm,
                item_counts=item_counts,
                alpha=alpha,
                top_k=top_k,
                variant=variant,
            )

            val_pred = route_coldbridge(
                warm_predictions=warm_val_variant,
                cold_predictions=cold_val_uniform,
                prompts=val_prompts,
                tau=best_tau,
                top_k=top_k,
            )
            test_pred = route_coldbridge(
                warm_predictions=warm_test_variant,
                cold_predictions=cold_test_uniform,
                prompts=test_prompts,
                tau=best_tau,
                top_k=top_k,
            )

            row_val = evaluate_predictions(
                model_name=f"PopDebias-ColdBridge-{variant_name_map[variant]}-val",
                predictions=val_pred,
                ground_truths=val_gts,
                item_counts=item_counts,
                num_items=num_items,
                item_to_index=item_to_index,
                item_emb_norm=item_emb_norm,
                long_tail_threshold=args.long_tail_threshold,
                alpha=alpha,
                tau=best_tau,
                training_time_sec=0.0,
                inference_time_sec=0.0,
            )
            row_val["debias_variant"] = variant
            variant_eval_rows_val.append(row_val)

            row_test = evaluate_predictions(
                model_name=f"PopDebias-ColdBridge-{variant_name_map[variant]}",
                predictions=test_pred,
                ground_truths=test_gts,
                item_counts=item_counts,
                num_items=num_items,
                item_to_index=item_to_index,
                item_emb_norm=item_emb_norm,
                long_tail_threshold=args.long_tail_threshold,
                alpha=alpha,
                tau=best_tau,
                training_time_sec=llm2sas_train_t,
                inference_time_sec=base_test_t,
            )
            row_test["debias_variant"] = variant
            variant_eval_rows_test.append(row_test)
            rows_test.append(row_test)

    best_variant_val = choose_best_by_joint_metric(variant_eval_rows_val)
    best_variant = str(best_variant_val.get("debias_variant", "inverse"))
    best_variant_alpha = float(best_variant_val["alpha"])
    logging.info(
        "Best debias variant from validation: %s (alpha=%.3f)",
        best_variant,
        best_variant_alpha,
    )

    # 8) Improvement 3: weighted cold-branch sweep.
    decay_rows_val: list[dict[str, Any]] = []
    decay_rows_test: list[dict[str, Any]] = []

    warm_val_best_variant = rerank_popdebias(
        base_candidates=base_val_candidates,
        prompts=val_prompts,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        item_counts=item_counts,
        alpha=best_variant_alpha,
        top_k=top_k,
        variant=best_variant,
    )
    warm_test_best_variant = rerank_popdebias(
        base_candidates=base_test_candidates,
        prompts=test_prompts,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        item_counts=item_counts,
        alpha=best_variant_alpha,
        top_k=top_k,
        variant=best_variant,
    )

    for decay in decay_values:
        cold_val_weighted = predict_cold_branch(
            prompts=val_prompts,
            all_item_ids=item_ids_for_emb,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            top_k=top_k,
            decay=decay,
        )
        cold_test_weighted = predict_cold_branch(
            prompts=test_prompts,
            all_item_ids=item_ids_for_emb,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            top_k=top_k,
            decay=decay,
        )

        val_pred = route_coldbridge(
            warm_predictions=warm_val_best_variant,
            cold_predictions=cold_val_weighted,
            prompts=val_prompts,
            tau=best_tau,
            top_k=top_k,
        )
        test_pred = route_coldbridge(
            warm_predictions=warm_test_best_variant,
            cold_predictions=cold_test_weighted,
            prompts=test_prompts,
            tau=best_tau,
            top_k=top_k,
        )

        row_val = evaluate_predictions(
            model_name="PopDebias-ColdBridge-WeightedCold-val",
            predictions=val_pred,
            ground_truths=val_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=best_variant_alpha,
            tau=best_tau,
            training_time_sec=0.0,
            inference_time_sec=0.0,
        )
        row_val["cold_decay"] = decay
        decay_rows_val.append(row_val)

        row_test = evaluate_predictions(
            model_name="PopDebias-ColdBridge-WeightedCold",
            predictions=test_pred,
            ground_truths=test_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=best_variant_alpha,
            tau=best_tau,
            training_time_sec=llm2sas_train_t,
            inference_time_sec=base_test_t,
        )
        row_test["cold_decay"] = decay
        decay_rows_test.append(row_test)
        rows_test.append(row_test)

    best_decay_val = choose_best_by_joint_metric(decay_rows_val)
    best_decay = float(best_decay_val.get("cold_decay", 1.0))
    logging.info("Best weighted cold decay from validation: %.3f", best_decay)

    # 9) Optional BERT backbone.
    if args.include_bert_backbone:
        llm2bert_model = instantiate_llm2bert(args, embedding_csv_path)
        llm2bert_train_t = fit_model(llm2bert_model, train_df)

        bert_val_candidates, _ = predict_model(llm2bert_model, val_prompts, top_k=max_candidate_k)
        bert_test_candidates, bert_test_t = predict_model(llm2bert_model, test_prompts, top_k=max_candidate_k)

        bert_warm_val = rerank_popdebias(
            base_candidates=bert_val_candidates,
            prompts=val_prompts,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            item_counts=item_counts,
            alpha=best_variant_alpha,
            top_k=top_k,
            variant=best_variant,
        )
        bert_warm_test = rerank_popdebias(
            base_candidates=bert_test_candidates,
            prompts=test_prompts,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            item_counts=item_counts,
            alpha=best_variant_alpha,
            top_k=top_k,
            variant=best_variant,
        )

        cold_val_weighted = predict_cold_branch(
            prompts=val_prompts,
            all_item_ids=item_ids_for_emb,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            top_k=top_k,
            decay=best_decay,
        )
        cold_test_weighted = predict_cold_branch(
            prompts=test_prompts,
            all_item_ids=item_ids_for_emb,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            top_k=top_k,
            decay=best_decay,
        )

        bert_combined_test = route_coldbridge(
            warm_predictions=bert_warm_test,
            cold_predictions=cold_test_weighted,
            prompts=test_prompts,
            tau=best_tau,
            top_k=top_k,
        )

        row_bert_combined = evaluate_predictions(
            model_name="PopDebias-ColdBridge-BERT4Rec",
            predictions=bert_combined_test,
            ground_truths=test_gts,
            item_counts=item_counts,
            num_items=num_items,
            item_to_index=item_to_index,
            item_emb_norm=item_emb_norm,
            long_tail_threshold=args.long_tail_threshold,
            alpha=best_variant_alpha,
            tau=best_tau,
            training_time_sec=llm2bert_train_t,
            inference_time_sec=bert_test_t,
        )
        rows_test.append(row_bert_combined)

    # Final full system row (best variant + weighted cold branch).
    cold_test_best = predict_cold_branch(
        prompts=test_prompts,
        all_item_ids=item_ids_for_emb,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        top_k=top_k,
        decay=best_decay,
    )
    full_system_test = route_coldbridge(
        warm_predictions=warm_test_best_variant,
        cold_predictions=cold_test_best,
        prompts=test_prompts,
        tau=best_tau,
        top_k=top_k,
    )
    row_full_system = evaluate_predictions(
        model_name="FULL_SYSTEM",
        predictions=full_system_test,
        ground_truths=test_gts,
        item_counts=item_counts,
        num_items=num_items,
        item_to_index=item_to_index,
        item_emb_norm=item_emb_norm,
        long_tail_threshold=args.long_tail_threshold,
        alpha=best_variant_alpha,
        tau=best_tau,
        training_time_sec=llm2sas_train_t,
        inference_time_sec=base_test_t,
    )
    row_full_system["debias_variant"] = best_variant
    row_full_system["cold_decay"] = best_decay
    rows_test.append(row_full_system)

    # Compose outputs.
    results_df = pd.DataFrame(rows_test)
    cols_in_order = [
        "model_name",
        "alpha",
        "tau",
        "NDCG@10",
        "NDCG@20",
        "HR@10",
        "HR@20",
        "LongTail_HR@10",
        "CatalogCoverage",
        "Serendipity",
        "ILD@10",
        "training_time_sec",
        "inference_time_sec",
    ]
    optional_cols = ["debias_variant", "cold_decay"]
    final_cols = cols_in_order + [c for c in optional_cols if c in results_df.columns]
    results_df = results_df[final_cols]

    results_df = results_df.sort_values(["NDCG@10", "LongTail_HR@10"], ascending=False)

    full_results_csv = results_dir / "full_results.csv"
    results_df.to_csv(full_results_csv, index=False)

    # Pick best model by joint metric.
    best_row = choose_best_by_joint_metric(results_df.to_dict("records"))
    baseline_ref = next((r for r in rows_test if r["model_name"] == "BGE2SASRec"), None)

    best_model_txt = results_dir / "best_model.txt"
    save_best_model_txt(best_row, best_model_txt, baseline_ref)

    # Figures
    plot_alpha_sweep(
        alpha_rows_test,
        figures_dir / "alpha_sweep_movielens_100k.png",
    )
    plot_tau_sweep(
        tau_sweep_stats,
        figures_dir / "tau_sweep_movielens_100k.png",
    )

    # Cold/warm gap chart comparing BGE2SASRec and combined model.
    baseline_cold, baseline_warm = segmented_hitrate(
        predictions=bge2sas_test_top,
        ground_truths=test_gts,
        prompts=test_prompts,
        segment_threshold=5,
        top_k=10,
    )
    combined_cold, combined_warm = segmented_hitrate(
        predictions=full_system_test,
        ground_truths=test_gts,
        prompts=test_prompts,
        segment_threshold=5,
        top_k=10,
    )
    plot_cold_warm_gap(
        baseline_cold_hr=baseline_cold,
        baseline_warm_hr=baseline_warm,
        bridge_cold_hr=combined_cold,
        bridge_warm_hr=combined_warm,
        output_path=figures_dir / "cold_warm_gap_movielens_100k.png",
    )

    print("\n=== Novelty Experiment Complete ===")
    print(f"Results CSV: {full_results_csv}")
    print(f"Best model:  {best_model_txt}")
    print(f"Figures dir: {figures_dir}")
    print(f"Embeddings:  {embedding_npy_path}")
    print("\nTop rows:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    run()

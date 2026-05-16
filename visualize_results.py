"""
PromptCraft-SeqRec: Generate publication-quality figures.

Reads result JSON files from results/ and embedding quality metrics,
produces bar charts, strategy×metric heatmaps, and isotropy scatter plots.

Usage:
  python visualize_results.py
  python visualize_results.py --results-dir results --figures-dir figures --dataset beauty
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from prompt_strategies import STRATEGY_NAMES, STRATEGY_LABELS


def load_results(results_dir: str, dataset: str) -> dict:
    """Load the most recent result file for the given dataset."""
    pattern = os.path.join(results_dir, f"promptcraft_{dataset}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    path = files[-1]
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded results: {path}")
    return data


def load_quality(results_dir: str, dataset: str) -> dict:
    path = os.path.join(results_dir, f"embedding_quality_{dataset}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def plot_bar(results: dict, dataset: str, figures_dir: str) -> None:
    strategies_with_data = [s for s in STRATEGY_NAMES if "metrics" in results.get(s, {})]
    if not strategies_with_data:
        print("No results to plot.")
        return

    metric_cols = ["NDCG@10", "HR@10"]
    colors = ["#2196F3", "#FF9800"]
    x = np.arange(len(strategies_with_data))
    width = 0.35

    baseline_row = results.get("type1_title_only", {}).get("metrics", {})

    fig, ax = plt.subplots(figsize=(14, 5))
    for k, (metric, color) in enumerate(zip(metric_cols, colors)):
        vals = [results[s]["metrics"].get(metric, 0) for s in strategies_with_data]
        baseline_val = baseline_row.get(metric, 0)
        bars = ax.bar(x + k * width, vals, width, label=metric, color=color, alpha=0.85)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if strategies_with_data[i] == "type1_title_only":
                bar.set_edgecolor("red")
                bar.set_linewidth(2)
            if val == max(vals) and val > baseline_val:
                bar.set_edgecolor("darkgreen")
                bar.set_linewidth(2)
                delta = (val - baseline_val) / max(baseline_val, 1e-9) * 100
                ax.annotate(
                    f"+{delta:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="darkgreen",
                    fontweight="bold",
                )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(s, s) for s in strategies_with_data],
        rotation=15,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Score")
    ax.set_title(
        f"PromptCraft-SeqRec — {dataset.title()} Dataset\n"
        "(red border = T1 baseline, green border + % = best improvement)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, f"bar_{dataset}.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Bar chart saved -> {path}")
    plt.close()


def plot_heatmap(results: dict, dataset: str, figures_dir: str) -> None:
    strategies_with_data = [s for s in STRATEGY_NAMES if "metrics" in results.get(s, {})]
    if not strategies_with_data:
        return

    metric_cols = [c for c in ["NDCG@10", "NDCG@20", "HR@10", "HR@20"] if any(
        c in results[s]["metrics"] for s in strategies_with_data
    )]
    if not metric_cols:
        return

    mat = np.array([
        [results[s]["metrics"].get(m, 0) for m in metric_cols]
        for s in strategies_with_data
    ])
    baseline_row = mat[strategies_with_data.index("type1_title_only")] if "type1_title_only" in strategies_with_data else np.zeros(len(metric_cols))
    delta_mat = (mat - baseline_row) / (np.abs(baseline_row) + 1e-9) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    im1 = ax1.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(metric_cols)))
    ax1.set_xticklabels(metric_cols)
    ax1.set_yticks(range(len(strategies_with_data)))
    ax1.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_with_data])
    ax1.set_title("Absolute Scores")
    plt.colorbar(im1, ax=ax1)
    for i in range(len(strategies_with_data)):
        for j in range(len(metric_cols)):
            ax1.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)

    vmin, vmax = delta_mat.min(), delta_mat.max()
    if vmin == vmax:
        vmin, vmax = -1, 1
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im2 = ax2.imshow(delta_mat, cmap="RdYlGn", aspect="auto", norm=norm)
    ax2.set_xticks(range(len(metric_cols)))
    ax2.set_xticklabels(metric_cols)
    ax2.set_yticks(range(len(strategies_with_data)))
    ax2.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_with_data])
    ax2.set_title("% Change vs T1 Baseline")
    plt.colorbar(im2, ax=ax2)
    for i in range(len(strategies_with_data)):
        for j in range(len(metric_cols)):
            col = "white" if abs(delta_mat[i, j]) > 10 else "black"
            ax2.text(j, i, f"{delta_mat[i,j]:+.1f}%", ha="center", va="center", fontsize=7, color=col)

    plt.suptitle(f"PromptCraft-SeqRec: Strategy × Metric — {dataset.title()}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"heatmap_{dataset}.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Heatmap saved -> {path}")
    plt.close()


def plot_scatter(results: dict, quality: dict, dataset: str, figures_dir: str) -> None:
    if not quality:
        return
    strategies = [s for s in STRATEGY_NAMES if "metrics" in results.get(s, {}) and s in quality]
    if len(strategies) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in strategies:
        x_val = quality[s]["avg_pairwise_dist"]
        y_val = results[s]["metrics"].get("NDCG@10", 0)
        color = "red" if s == "type1_title_only" else "steelblue"
        ax.scatter(x_val, y_val, s=100, color=color, zorder=5)
        ax.annotate(STRATEGY_LABELS.get(s, s), (x_val, y_val), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Avg Pairwise Distance (higher = more discriminative)")
    ax.set_ylabel("NDCG@10")
    ax.set_title(f"Embedding Discriminability vs Downstream Accuracy — {dataset.title()}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"scatter_{dataset}.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Scatter plot saved -> {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    results = load_results(args.results_dir, args.dataset)
    quality = load_quality(args.results_dir, args.dataset)

    if not results:
        print(f"No result files found for dataset '{args.dataset}' in {args.results_dir}/")
        return

    plot_bar(results, args.dataset, args.figures_dir)
    plot_heatmap(results, args.dataset, args.figures_dir)
    plot_scatter(results, quality, args.dataset, args.figures_dir)

    print(f"\nAll figures in {args.figures_dir}/")


if __name__ == "__main__":
    main()

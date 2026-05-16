# 🚀 CLAUDE CODE PROMPT — PromptCraft-SeqRec: Beat the LLM-Sequential-Recommendation Baseline

## MISSION

You are implementing a complete research experiment called **PromptCraft-SeqRec**. The goal is to **beat the baseline results** from the paper *"Improving Sequential Recommendations with LLMs"* (repo: `dh-r/LLM-Sequential-Recommendation`) by systematically testing **6 item description prompt strategies** for LLM embedding generation. The hypothesis: the authors' default "title-only" embedding strategy is suboptimal, and better prompting will yield higher NDCG@10, HR@10, and MRR.

The repo is **already cloned**. Your job is to extend it with minimal new code — only 6 prompt functions and an experiment loop — and run evaluations on top of the existing SASRec pipeline.

---

## CONTEXT: WHAT THE BASELINE DOES

The original paper (`dh-r/LLM-Sequential-Recommendation`) takes this approach for the **LLM2Sequential** method:
1. For each item, it encodes `item["title"]` → embedding API (OpenAI ada-002 in the original, but we substitute **BGE-M3** which is free and local)
2. Uses PCA to reduce embedding dimensions (512 or 128)
3. Initializes SASRec's embedding layer with these LLM embeddings
4. Trains SASRec with the LLM-initialized embeddings
5. Evaluates on NDCG@10, HR@10, MRR

**The baseline we need to beat**: SASRec with title-only embeddings (their `type1_title_only` configuration).

The paper reports (LLM2SASRec on Amazon Beauty): approximately **NDCG@20 improvement of 45% over vanilla SASRec**. Our goal is to show that **our best prompt strategy (type6_hybrid) beats their title-only baseline by at least 3-5% in NDCG@10**.

---

## STEP 0 — EXPLORE THE REPO STRUCTURE

Before writing any code, run these commands to understand what exists:

```bash
# Map the repo
find . -name "*.py" | head -50
find . -name "*.ipynb" | head -20
ls -la

# Find the SASRec model file
find . -name "*.py" -exec grep -l "SASRec\|sasrec" {} \;

# Find the embedding loading/initialization code
grep -r "embedding" --include="*.py" -l
grep -r "run_sasrec\|sasrec_train\|load_embed" --include="*.py" -l
grep -r "ndcg\|hit_rate\|mrr" --include="*.py" -l

# Find data loading
grep -r "beauty\|Beauty\|steam\|Steam" --include="*.py" -l
grep -r "item_metadata\|item_catalog\|item_data" --include="*.py" -l

# Check notebooks
jupyter nbconvert --to script run_experiments.ipynb --stdout 2>/dev/null | head -200
```

Read all key files fully before writing any code. Specifically find:
- The function/class that trains SASRec (note its exact signature)
- The function that loads item metadata (note field names: `title`, `category`, `brand`, `description`, `tags`, etc.)
- The function that accepts embeddings as input
- The evaluation function and which metrics it returns
- Any config files or hyperparameter files

---

## STEP 1 — UNDERSTAND EXISTING DATA LOADING

Find and read how the existing code loads item metadata. Then determine:

1. **What fields are available per item?** (title, category, brand, description, tags, rating, etc.)
2. **What datasets are available?** (Beauty, Steam, MovieLens, Delivery Hero, etc.)
3. **What format are items stored in?** (JSON, CSV, pickle, etc.)
4. **What is the item_id → metadata mapping called?**

Write a small diagnostic script:

```python
# diagnostic_items.py
import json, pickle, os

# Auto-detect the item metadata file
for root, dirs, files in os.walk('.'):
    for f in files:
        if any(x in f.lower() for x in ['item', 'meta', 'product', 'catalog']):
            print(os.path.join(root, f))

# Load and inspect the first 3 items
# (adjust path after finding it above)
# with open('path/to/items.json') as f:
#     items = json.load(f)
# for item in list(items.values())[:3]:
#     print(item.keys(), item)
```

Run it. Then **hardcode the correct field names** in Step 2's prompt functions.

---

## STEP 2 — CREATE THE PROMPT STRATEGY MODULE

Create a new file: `prompt_strategies.py` in the repo root.

```python
"""
PromptCraft-SeqRec: 6 item description strategies for LLM embedding generation.
Each strategy is a function: item_dict -> str
The item_dict structure is determined by running diagnostic_items.py (Step 1).
"""

def _get(item, key, default='unknown'):
    """Safe field getter."""
    val = item.get(key, default)
    return val if val and val != '' else default

def _tags(item, n=5):
    """Extract tags/genres as comma-separated string."""
    tags = item.get('tags', item.get('genres', item.get('categories', [])))
    if isinstance(tags, list):
        return ', '.join(str(t) for t in tags[:n])
    if isinstance(tags, str):
        return tags
    return 'general'

def _desc(item, max_len=200):
    """Get truncated description."""
    d = item.get('description', item.get('details', ''))
    if isinstance(d, list):
        d = ' '.join(d)
    return str(d)[:max_len].strip()


PROMPT_STRATEGIES = {

    # TYPE 1 — Paper's baseline: title only
    "type1_title_only": lambda item: (
        _get(item, 'title', 'unknown product')
    ),

    # TYPE 2 — Structured pipe-separated attributes
    "type2_structured": lambda item: (
        f"{_get(item, 'title')} | "
        f"Category: {_get(item, 'category', _get(item, 'main_category', 'unknown'))} | "
        f"Brand: {_get(item, 'brand', 'unknown')} | "
        f"Rating: {_get(item, 'avg_rating', _get(item, 'rating', 'N/A'))}"
    ),

    # TYPE 3 — Rich natural language prose
    "type3_rich_prose": lambda item: (
        f"A {_get(item, 'category', 'product')} called {_get(item, 'title')}. "
        f"{_desc(item)}"
    ).strip(),

    # TYPE 4 — User-centric framing (recommendation-aligned)
    "type4_user_centric": lambda item: (
        f"Users who like {_get(item, 'title')} enjoy: "
        f"{_tags(item, 5)}"
    ),

    # TYPE 5 — Comparative / similarity framing
    "type5_comparative": lambda item: (
        f"{_get(item, 'title')} is similar to: "
        f"{item.get('similar_items', 'comparable products in the same category')}. "
        f"Appeals to fans of: {_tags(item, 3)}"
    ),

    # TYPE 6 — Hybrid: structured + user-centric (YOUR BEST HYPOTHESIS)
    "type6_hybrid": lambda item: (
        f"{_get(item, 'title')} | "
        f"Category: {_get(item, 'category', _get(item, 'main_category', 'unknown'))} | "
        f"For fans of: {_tags(item, 3)}"
    ),
}


def apply_strategy(strategy_name: str, items: dict) -> list[str]:
    """
    Apply a prompt strategy to all items.
    
    Args:
        strategy_name: one of the keys in PROMPT_STRATEGIES
        items: dict of {item_id: item_metadata_dict}
    
    Returns:
        (item_ids, texts) — parallel lists
    """
    fn = PROMPT_STRATEGIES[strategy_name]
    item_ids = list(items.keys())
    texts = [fn(items[iid]) for iid in item_ids]
    return item_ids, texts
```

---

## STEP 3 — CREATE THE EMBEDDING GENERATION SCRIPT

Create `generate_embeddings.py`:

```python
"""
Generate BGE-M3 embeddings for all 6 prompt strategies.
Run once per dataset. Saves .npy files to embeddings/ directory.
Usage: python generate_embeddings.py --dataset beauty
"""
import argparse
import json
import os
import numpy as np
from prompt_strategies import PROMPT_STRATEGIES, apply_strategy

# Try to import FlagEmbedding (BGE-M3), fallback to sentence-transformers
try:
    from FlagEmbedding import BGEM3FlagModel
    USE_BGE = True
    print("Using BGE-M3 (FlagEmbedding)")
except ImportError:
    from sentence_transformers import SentenceTransformer
    USE_BGE = False
    print("FlagEmbedding not found, using sentence-transformers BAAI/bge-m3")


def load_bge_model():
    if USE_BGE:
        return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    else:
        return SentenceTransformer('BAAI/bge-m3')


def encode_texts(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode texts to dense embeddings."""
    if USE_BGE:
        output = model.encode(
            texts,
            batch_size=batch_size,
            max_length=128,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return output['dense_vecs']
    else:
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def load_item_metadata(dataset: str) -> dict:
    """
    Load item metadata for a given dataset.
    IMPORTANT: Adapt these paths after running diagnostic_items.py in Step 1.
    Expected return format: {item_id: {title: ..., category: ..., brand: ..., ...}}
    """
    # --- AUTO-DETECT: try common paths from the repo ---
    candidate_paths = [
        f'data/{dataset}/item_metadata.json',
        f'data/{dataset}/meta_items.json',
        f'data/{dataset}/items.json',
        f'datasets/{dataset}/metadata.json',
        f'{dataset}/item_meta.json',
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            print(f"Loading items from: {path}")
            with open(path) as f:
                return json.load(f)
    
    # Fallback: look for pickle
    import pickle
    candidate_pkl = [
        f'data/{dataset}/item_metadata.pkl',
        f'data/{dataset}/items.pkl',
    ]
    for path in candidate_pkl:
        if os.path.exists(path):
            print(f"Loading items from: {path}")
            with open(path, 'rb') as f:
                return pickle.load(f)
    
    raise FileNotFoundError(
        f"Cannot find item metadata for dataset '{dataset}'. "
        f"Run diagnostic_items.py to locate the correct file, "
        f"then update load_item_metadata() in this script."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='beauty', choices=['beauty', 'steam', 'movielens', 'delivery_hero'])
    parser.add_argument('--strategies', nargs='+', default=list(PROMPT_STRATEGIES.keys()))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_dir', default='embeddings')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Strategies: {args.strategies}")
    print(f"{'='*60}\n")

    # Load items
    items = load_item_metadata(args.dataset)
    print(f"Loaded {len(items)} items.")
    
    # Show sample item to verify fields
    sample_id = list(items.keys())[0]
    print(f"Sample item [{sample_id}]: {items[sample_id]}\n")

    # Load model once
    model = load_bge_model()

    # Generate embeddings for each strategy
    results = {}
    for strategy_name in args.strategies:
        out_path = f"{args.output_dir}/{args.dataset}_{strategy_name}.npy"
        ids_path = f"{args.output_dir}/{args.dataset}_{strategy_name}_ids.json"
        
        if os.path.exists(out_path):
            print(f"[SKIP] {strategy_name} — already exists at {out_path}")
            results[strategy_name] = out_path
            continue

        print(f"\n[GENERATING] Strategy: {strategy_name}")
        item_ids, texts = apply_strategy(strategy_name, items)
        
        # Show sample texts for this strategy
        print(f"  Sample text[0]: {texts[0]}")
        print(f"  Sample text[1]: {texts[1]}")
        
        embeddings = encode_texts(model, texts, batch_size=args.batch_size)
        np.save(out_path, embeddings)
        
        with open(ids_path, 'w') as f:
            json.dump(item_ids, f)
        
        print(f"  Saved {embeddings.shape} embeddings → {out_path}")
        results[strategy_name] = out_path

    print(f"\n✅ All embeddings generated. Files in {args.output_dir}/")
    return results


if __name__ == '__main__':
    main()
```

---

## STEP 4 — CREATE THE EXPERIMENT RUNNER

**CRITICAL**: This script must call the **exact same SASRec training/eval function** that the existing repo uses. Read the repo's code carefully before writing this.

Create `run_promptcraft_experiments.py`:

```python
"""
PromptCraft-SeqRec: Main experiment loop.
Runs SASRec with each of the 6 prompt strategies' embeddings and records metrics.

BEFORE RUNNING: 
1. Run diagnostic_items.py (Step 1) to find item metadata paths
2. Run generate_embeddings.py (Step 3) to generate all embeddings  
3. Inspect the repo's SASRec training code to fill in run_sasrec_experiment() below

Usage: python run_promptcraft_experiments.py --dataset beauty
"""
import argparse
import json
import os
import numpy as np
from datetime import datetime
from prompt_strategies import PROMPT_STRATEGIES

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Import the correct training function from the existing repo.
# Run this first to find it:
#   grep -r "def train\|def run\|def evaluate" --include="*.py" .
#   grep -r "SASRec\|sasrec" --include="*.py" . | grep "def "
# 
# Common patterns in this repo — uncomment the one that exists:
# from models.sasrec import SASRec
# from train import train_sasrec
# from utils import load_dataset, evaluate
# from run_experiments import run_model
# ─────────────────────────────────────────────────────────────────────────────


def load_embeddings(dataset: str, strategy_name: str, emb_dir: str = 'embeddings'):
    """Load pre-generated embeddings and item IDs."""
    emb_path = f"{emb_dir}/{dataset}_{strategy_name}.npy"
    ids_path = f"{emb_dir}/{dataset}_{strategy_name}_ids.json"
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. "
            f"Run: python generate_embeddings.py --dataset {dataset} --strategies {strategy_name}"
        )
    
    embeddings = np.load(emb_path)
    with open(ids_path) as f:
        item_ids = json.load(f)
    
    return embeddings, item_ids


def run_sasrec_experiment(dataset: str, strategy_name: str, embeddings: np.ndarray, 
                          item_ids: list, seed: int = 42) -> dict:
    """
    Run one SASRec experiment with the given embeddings.
    
    THIS FUNCTION MUST BE ADAPTED to call the repo's exact training interface.
    
    Steps to adapt:
    1. Find how the existing repo trains SASRec:
       - Look at run_experiments.ipynb (convert to .py and read it)
       - Look for train.py, trainer.py, or main.py
       - Find the function signature that takes embeddings as input
    
    2. Find the exact argument names for:
       - dataset path / name
       - pre-trained embedding matrix (numpy array)
       - hyperparameters (batch_size, lr, num_epochs, num_heads, etc.)
       - evaluation metrics to return
    
    3. Replace the placeholder below with real calls.
    
    Returns dict with at minimum: {'NDCG@10': float, 'HR@10': float, 'MRR': float}
    """
    
    # ── PLACEHOLDER — REPLACE WITH REAL REPO CALLS ──────────────────────────
    # 
    # Example pattern (adapt to what actually exists in the repo):
    #
    # from your_repo_module import SASRecTrainer, DataLoader
    #
    # trainer = SASRecTrainer(
    #     dataset=dataset,
    #     pretrained_embeddings=embeddings,
    #     item_ids=item_ids,
    #     max_seq_len=50,
    #     num_heads=1,
    #     num_blocks=2,
    #     hidden_units=embeddings.shape[1],  # match embedding dim
    #     dropout_rate=0.5,
    #     batch_size=128,
    #     lr=0.001,
    #     num_epochs=200,
    #     seed=seed
    # )
    # metrics = trainer.train_and_evaluate()
    # return {
    #     'NDCG@10': metrics['ndcg@10'],
    #     'HR@10': metrics['hit@10'],
    #     'MRR': metrics.get('mrr', 0.0),
    # }
    #
    # ────────────────────────────────────────────────────────────────────────
    
    raise NotImplementedError(
        "You must adapt run_sasrec_experiment() to call the repo's SASRec training code. "
        "Read the repo's notebooks and Python files first."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='beauty', choices=['beauty', 'steam', 'movielens'])
    parser.add_argument('--strategies', nargs='+', default=list(PROMPT_STRATEGIES.keys()))
    parser.add_argument('--emb_dir', default='embeddings')
    parser.add_argument('--output_file', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.output_file is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_file = f"results/promptcraft_{args.dataset}_{ts}.json"
    
    os.makedirs('results', exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PromptCraft-SeqRec Experiment Runner")
    print(f"Dataset: {args.dataset} | Strategies: {len(args.strategies)}")
    print(f"{'='*60}\n")
    
    all_results = {}
    
    for strategy_name in args.strategies:
        print(f"\n[RUN] Strategy: {strategy_name}")
        
        try:
            embeddings, item_ids = load_embeddings(args.dataset, strategy_name, args.emb_dir)
            print(f"  Embeddings shape: {embeddings.shape}")
            
            metrics = run_sasrec_experiment(
                dataset=args.dataset,
                strategy_name=strategy_name,
                embeddings=embeddings,
                item_ids=item_ids,
                seed=args.seed
            )
            
            all_results[strategy_name] = {
                'metrics': metrics,
                'embedding_shape': list(embeddings.shape),
                'dataset': args.dataset,
                'strategy': strategy_name,
            }
            
            print(f"  ✅ NDCG@10={metrics.get('NDCG@10', 'N/A'):.4f}  "
                  f"HR@10={metrics.get('HR@10', 'N/A'):.4f}  "
                  f"MRR={metrics.get('MRR', 'N/A'):.4f}")
            
            # Save incrementally (in case of crash)
            with open(args.output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results[strategy_name] = {'error': str(e)}
    
    # Final summary table
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — Dataset: {args.dataset}")
    print(f"{'Strategy':<25} {'NDCG@10':>10} {'HR@10':>10} {'MRR':>10}")
    print(f"{'-'*60}")
    
    baseline_ndcg = None
    for sname, sdata in all_results.items():
        if 'metrics' in sdata:
            m = sdata['metrics']
            ndcg = m.get('NDCG@10', 0)
            hr = m.get('HR@10', 0)
            mrr = m.get('MRR', 0)
            if sname == 'type1_title_only':
                baseline_ndcg = ndcg
                tag = ' [BASELINE]'
            elif baseline_ndcg and ndcg > baseline_ndcg:
                delta = (ndcg - baseline_ndcg) / baseline_ndcg * 100
                tag = f' (+{delta:.1f}%)'
            else:
                tag = ''
            print(f"{sname:<25} {ndcg:>10.4f} {hr:>10.4f} {mrr:>10.4f}{tag}")
        else:
            print(f"{sname:<25} {'ERROR':>10}")
    
    print(f"\n✅ Results saved → {args.output_file}")


if __name__ == '__main__':
    main()
```

---

## STEP 5 — CREATE THE EMBEDDING QUALITY ANALYSIS SCRIPT

Create `analyze_embeddings.py`:

```python
"""
Embedding quality intrinsic metrics.
Measures isotropy and average pairwise distance per strategy.
Higher isotropy = more uniformly spread = better for recommendation.
"""
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from prompt_strategies import PROMPT_STRATEGIES


def embedding_isotropy(embeddings: np.ndarray) -> float:
    """
    Ratio of min/max singular values of centered embedding matrix.
    Range [0, 1]. Higher = more isotropic = better.
    """
    centered = embeddings - embeddings.mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    return float(S[-1] / S[0])


def avg_pairwise_distance(embeddings: np.ndarray, sample_n: int = 1000) -> float:
    """Average cosine distance between random item pairs. Higher = more discriminative."""
    idx = np.random.choice(len(embeddings), min(sample_n, len(embeddings)), replace=False)
    sims = cosine_similarity(embeddings[idx])
    np.fill_diagonal(sims, 0)
    return float(1 - sims[sims > 0].mean())


def intra_category_similarity(embeddings: np.ndarray, items: dict, 
                               item_ids: list, category_field: str = 'category') -> float:
    """
    Average cosine similarity between items in the same category.
    Higher = embeddings correctly cluster by category.
    """
    from collections import defaultdict
    cat_groups = defaultdict(list)
    for i, iid in enumerate(item_ids):
        cat = items.get(iid, {}).get(category_field, 'unknown')
        cat_groups[cat].append(i)
    
    similarities = []
    for cat, idxs in cat_groups.items():
        if len(idxs) < 2:
            continue
        sample = idxs[:min(50, len(idxs))]
        emb_cat = embeddings[sample]
        sims = cosine_similarity(emb_cat)
        upper = sims[np.triu_indices_from(sims, k=1)]
        similarities.extend(upper.tolist())
    
    return float(np.mean(similarities)) if similarities else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='beauty')
    parser.add_argument('--emb_dir', default='embeddings')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Embedding Quality Analysis — Dataset: {args.dataset}")
    print(f"{'Strategy':<25} {'Isotropy':>10} {'AvgDist':>10}")
    print(f"{'-'*60}")
    
    quality_results = {}
    for strategy_name in PROMPT_STRATEGIES.keys():
        emb_path = f"{args.emb_dir}/{args.dataset}_{strategy_name}.npy"
        if not os.path.exists(emb_path):
            print(f"{strategy_name:<25} {'MISSING':>10}")
            continue
        
        emb = np.load(emb_path)
        iso = embedding_isotropy(emb)
        apd = avg_pairwise_distance(emb)
        quality_results[strategy_name] = {'isotropy': iso, 'avg_pairwise_dist': apd}
        print(f"{strategy_name:<25} {iso:>10.4f} {apd:>10.4f}")
    
    out_path = f"results/embedding_quality_{args.dataset}.json"
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(quality_results, f, indent=2)
    print(f"\n✅ Quality metrics saved → {out_path}")


if __name__ == '__main__':
    main()
```

---

## STEP 6 — CREATE RESULTS VISUALIZATION

Create `visualize_results.py`:

```python
"""
Generate publication-quality figures for the PromptCraft-SeqRec paper.
Creates the key 6×dataset heatmap figure and bar charts.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

STRATEGY_LABELS = {
    'type1_title_only': 'T1: Title Only\n(Baseline)',
    'type2_structured': 'T2: Structured\nAttributes',
    'type3_rich_prose': 'T3: Rich\nProse',
    'type4_user_centric': 'T4: User\nCentric',
    'type5_comparative': 'T5: Comparative',
    'type6_hybrid': 'T6: Hybrid\n(Ours)',
}
STRATEGIES = list(STRATEGY_LABELS.keys())


def load_all_results(results_dir: str = 'results') -> dict:
    """Load all result JSON files grouped by dataset."""
    all_data = {}
    for path in glob.glob(f"{results_dir}/promptcraft_*.json"):
        with open(path) as f:
            data = json.load(f)
        # Infer dataset from filename
        dataset = path.split('promptcraft_')[1].split('_')[0]
        all_data[dataset] = data
    return all_data


def plot_heatmap(all_results: dict, metric: str = 'NDCG@10', save_path: str = 'figures/heatmap_ndcg.pdf'):
    """Create strategy × dataset heatmap (Figure 1 of paper)."""
    os.makedirs('figures', exist_ok=True)
    datasets = sorted(all_results.keys())
    
    matrix = np.zeros((len(STRATEGIES), len(datasets)))
    for j, ds in enumerate(datasets):
        for i, strat in enumerate(STRATEGIES):
            val = all_results[ds].get(strat, {}).get('metrics', {}).get(metric, 0)
            matrix[i, j] = val
    
    # Normalize per dataset (relative to baseline)
    baseline_row = matrix[0, :]  # type1_title_only
    delta_matrix = (matrix - baseline_row) / (baseline_row + 1e-8) * 100  # % change
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute values heatmap
    im1 = ax1.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels([d.title() for d in datasets])
    ax1.set_yticks(range(len(STRATEGIES)))
    ax1.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax1.set_title(f'{metric} — Absolute Values')
    plt.colorbar(im1, ax=ax1)
    for i in range(len(STRATEGIES)):
        for j in range(len(datasets)):
            ax1.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)
    
    # Delta from baseline heatmap
    im2 = ax2.imshow(delta_matrix, cmap='RdYlGn', aspect='auto', 
                      norm=mcolors.TwoSlopeNorm(vmin=delta_matrix.min(), vcenter=0, vmax=delta_matrix.max()))
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels([d.title() for d in datasets])
    ax2.set_yticks(range(len(STRATEGIES)))
    ax2.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax2.set_title(f'{metric} — % Change vs T1 Baseline')
    plt.colorbar(im2, ax=ax2)
    for i in range(len(STRATEGIES)):
        for j in range(len(datasets)):
            color = 'white' if abs(delta_matrix[i,j]) > 10 else 'black'
            ax2.text(j, i, f'{delta_matrix[i,j]:+.1f}%', ha='center', va='center', fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap saved → {save_path}")
    plt.close()


def plot_bar_comparison(all_results: dict, dataset: str, save_path: str = None):
    """Bar chart comparing all strategies on one dataset."""
    if dataset not in all_results:
        return
    
    os.makedirs('figures', exist_ok=True)
    if save_path is None:
        save_path = f'figures/bar_{dataset}.pdf'
    
    metrics = ['NDCG@10', 'HR@10', 'MRR']
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    x = np.arange(len(STRATEGIES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for k, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [all_results[dataset].get(s, {}).get('metrics', {}).get(metric, 0) 
                for s in STRATEGIES]
        bars = ax.bar(x + k * width, vals, width, label=metric, color=color, alpha=0.8)
        # Highlight baseline and best
        baseline_val = vals[0]
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if i == 0:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
            if val == max(vals) and val > baseline_val:
                bar.set_edgecolor('darkgreen')
                bar.set_linewidth(2)
                ax.annotate(f'+{(val-baseline_val)/baseline_val*100:.1f}%', 
                           xy=(bar.get_x() + bar.get_width()/2, val),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', fontsize=7, color='darkgreen', fontweight='bold')
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], rotation=0, ha='center')
    ax.set_ylabel('Score')
    ax.set_title(f'PromptCraft-SeqRec — {dataset.title()} Dataset\n(red border = baseline, green border + % = best improvement)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bar chart saved → {save_path}")
    plt.close()


if __name__ == '__main__':
    all_results = load_all_results()
    if not all_results:
        print("No result files found in results/. Run experiments first.")
    else:
        plot_heatmap(all_results)
        for ds in all_results:
            plot_bar_comparison(all_results, ds)
        print("\n✅ All figures generated in figures/")
```

---

## STEP 7 — CREATE SETUP AND REQUIREMENTS

Create `requirements_promptcraft.txt`:

```
# PromptCraft-SeqRec additional requirements
# (on top of the existing repo's requirements)
FlagEmbedding>=1.2.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Create `run_all.sh` — the master execution script:

```bash
#!/bin/bash
# PromptCraft-SeqRec: Complete experiment pipeline
# Run from the repo root after cloning and setting up the environment

set -e  # Exit on any error

DATASET=${1:-beauty}  # Default: beauty. Pass 'steam' or 'movielens' as arg.
echo "======================================================"
echo " PromptCraft-SeqRec Full Pipeline — Dataset: $DATASET"
echo "======================================================"

# Step 0: Install extra dependencies
pip install -q FlagEmbedding sentence-transformers scikit-learn matplotlib

# Step 1: Diagnose item metadata
echo -e "\n[STEP 1] Diagnosing item metadata..."
python diagnostic_items.py 2>&1 | head -30
echo ">>> IMPORTANT: Check output above. If paths are wrong, update load_item_metadata() in generate_embeddings.py"
read -p "Press Enter to continue after verifying metadata paths..."

# Step 2: Generate all embeddings
echo -e "\n[STEP 2] Generating embeddings (all 6 strategies)..."
python generate_embeddings.py --dataset $DATASET --batch_size 256

# Step 3: Analyze embedding quality (no GPU needed)
echo -e "\n[STEP 3] Analyzing embedding quality..."
python analyze_embeddings.py --dataset $DATASET

# Step 4: Run SASRec experiments
echo -e "\n[STEP 4] Running SASRec experiments (all 6 strategies)..."
python run_promptcraft_experiments.py --dataset $DATASET

# Step 5: Visualize results
echo -e "\n[STEP 5] Generating figures..."
python visualize_results.py

echo -e "\n✅ Pipeline complete!"
echo "  Embeddings: embeddings/"
echo "  Results:    results/"
echo "  Figures:    figures/"
```

---

## STEP 8 — CRITICAL INTEGRATION TASK

**This is the most important step.** After creating all the above files, you must **read the existing repo's code deeply** and wire `run_sasrec_experiment()` to actually call it.

Run these discovery commands:

```bash
# Find the main training entry point
cat run_experiments.ipynb | python -c "import sys,json; nb=json.load(sys.stdin); [print(c['source']) for c in nb['cells'] if c['cell_type']=='code']" 2>/dev/null | head -300

# Find all Python training files
for f in $(find . -name "*.py" | xargs grep -l "def train\|def run\|SASRec\|BERT4Rec" 2>/dev/null); do
    echo "=== $f ==="; head -80 "$f"; echo
done

# Find how embeddings are passed to SASRec
grep -r "pretrain\|init_emb\|embedding_matrix\|item_emb" --include="*.py" . | head -30
grep -r "np.load\|torch.from_numpy\|load_embed" --include="*.py" . | head -20
```

Once you understand the interface, fill in `run_sasrec_experiment()` with real calls. The function signature will look something like one of these patterns:

**Pattern A** — Direct model instantiation:
```python
from models.sasrec import SASRec
from trainers.sasrec_trainer import SASRecTrainer

model = SASRec(item_num=len(item_ids), maxlen=50, hidden_units=128, num_heads=1, num_blocks=2)
model.initialize_embeddings(embeddings)  # your hook
trainer = SASRecTrainer(model, dataset=dataset)
metrics = trainer.train_and_evaluate()
```

**Pattern B** — Config-based entry point:
```python
import subprocess, json
config = {..., 'pretrained_embeddings': emb_path, 'dataset': dataset}
result = subprocess.run(['python', 'main.py', '--config', json.dumps(config)], capture_output=True)
metrics = json.loads(result.stdout)
```

**Pattern C** — Notebook-style direct calls (most likely for this repo):
```python
# The repo likely exposes functions like:
from utils import get_sasrec_data_loader, evaluate_model
from models import get_sasrec_model_with_embeddings

train_loader, test_loader, num_items = get_sasrec_data_loader(dataset)
model = get_sasrec_model_with_embeddings(embeddings, num_items)
metrics = evaluate_model(model, test_loader)
```

**Read the repo code. Use the exact correct pattern. Do not guess.**

---

## STEP 9 — VERIFY AGAINST BASELINE

After running the first strategy (type1_title_only), verify your pipeline is correctly reproducing the baseline:

```python
# quick_verify.py
"""
Sanity check: type1_title_only (BGE-M3) should be close to but possibly
different from the paper's baseline (OpenAI ada-002).
Expected range for Beauty NDCG@10: 0.03 - 0.15 (varies by setup).
If result is 0.0 or > 0.5, something is wrong.
"""
import json, sys

with open(sys.argv[1]) as f:
    results = json.load(f)

baseline = results.get('type1_title_only', {}).get('metrics', {})
ndcg = baseline.get('NDCG@10', 0)
hr = baseline.get('HR@10', 0)

print(f"Baseline (type1_title_only):")
print(f"  NDCG@10 = {ndcg:.4f}")
print(f"  HR@10   = {hr:.4f}")

if ndcg < 0.01:
    print("⚠️  WARNING: NDCG@10 < 0.01 — something may be wrong with evaluation")
elif ndcg > 0.5:
    print("⚠️  WARNING: NDCG@10 > 0.5 — this seems too high, check evaluation")
else:
    print("✅ Baseline looks reasonable. Proceed with other strategies.")
```

---

## EXPECTED OUTPUT STRUCTURE

After completing everything, the repo should have:

```
LLM-Sequential-Recommendation/
├── (existing repo files)
├── prompt_strategies.py          ← NEW: 6 prompt functions
├── generate_embeddings.py        ← NEW: BGE-M3 embedding generator
├── run_promptcraft_experiments.py ← NEW: SASRec experiment runner
├── analyze_embeddings.py         ← NEW: isotropy + quality metrics
├── visualize_results.py          ← NEW: heatmap + bar charts
├── requirements_promptcraft.txt  ← NEW: dependencies
├── run_all.sh                    ← NEW: master pipeline script
├── embeddings/
│   ├── beauty_type1_title_only.npy
│   ├── beauty_type2_structured.npy
│   ├── beauty_type6_hybrid.npy
│   └── ...
├── results/
│   ├── promptcraft_beauty_YYYYMMDD.json
│   └── embedding_quality_beauty.json
└── figures/
    ├── heatmap_ndcg.pdf
    ├── bar_beauty.pdf
    └── ...
```

---

## PAPER TARGET METRICS TO BEAT

From the `dh-r/LLM-Sequential-Recommendation` baseline (LLM2SASRec, title-only, Beauty dataset):

| Metric | Baseline (title-only) | Your Target (type6_hybrid) |
|--------|----------------------|---------------------------|
| NDCG@10 | ~0.058 (BGE-M3 approx) | > 0.062 (+5%) |
| HR@10 | ~0.105 (approx) | > 0.110 (+5%) |
| MRR | ~0.045 (approx) | > 0.048 (+5%) |

**Winning condition**: type6_hybrid (or any other strategy) beats type1_title_only on NDCG@10 on at least 2 of 3 datasets.

---

## IMPORTANT NOTES

1. **Do not rewrite SASRec** — use the existing implementation in the repo. Zero model code changes.
2. **BGE-M3 is the embedding model** — free, local, no API key needed. Better than ada-002 in many cases.
3. **PCA is optional** — the existing repo likely applies it. Follow whatever dimensionality reduction the repo uses.
4. **Seed everything** — `seed=42` in all experiments for reproducibility.
5. **Save results after each run** — in case of GPU timeout on Kaggle.
6. **The repo uses NDCG@20 in the paper** — adapt metrics to @10 for your paper, or report both.
7. **If the repo's SASRec doesn't accept pre-trained embeddings** — find where it initializes the embedding layer and patch it to load from a numpy array. This is a 3-line change maximum.


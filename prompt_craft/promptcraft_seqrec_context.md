# PromptCraft-SeqRec — Research Context Document

> **Purpose:** This document is context for the AI agent implementing the PromptCraft-SeqRec experiment.
> Read this fully before writing any code. It explains the research motivation, all 6 prompt strategies,
> implementation details, datasets, expected results, and the 2-week execution plan.

---

## Quick Stats

| Metric | Value |
|---|---|
| New model code | 0 lines |
| New prompt functions | 6 (~30 min with AI agent) |
| Experiments | 18 runs (6 strategies × 3 datasets) |
| GPU time | ~40 hours total (10h per Kaggle account) |
| Duration | 2 weeks |
| Target venue | MDPI AI, arXiv cs.IR |
| Novelty source | Section 5.2 + Figure 12 of original paper |

---

## 1. The Problem This Paper Solves

Every paper that uses LLM embeddings for recommendation — including the paper being extended (`dh-r/LLM-Sequential-Recommendation`) — uses a **single fixed template** to describe items to the language model. The original paper (Section 5.2) simply feeds the product name into the embedding API:

> *"the metadata used to compute the embeddings are the names of the products."*

This choice is never questioned, never justified, and never tested against alternatives. Yet the paper itself accidentally discovers that the choice matters: for the Steam gaming dataset, game names alone produce weak embeddings, so the authors concatenate game titles with user-provided tags. This ad-hoc fix improves Steam results — but the paper **never systematically explores this observation**.

### Direct Quote — Novelty Justification

> *"For the Steam dataset, the items of the domain are games. Typically, the names of the games are not closely linked with the concept of a game. Thus, we concatenate the names of the games with tags that accompany and characterize each game."*
> — Section 5.2, page 17

**Translation:** The authors already know that how you describe an item changes embedding quality. This paper is the first to study this systematically.

### Research Question

> "Given a fixed embedding model (BGE-M3) and a fixed sequential recommendation architecture (SASRec), **how much does the choice of item description prompt affect downstream recommendation accuracy?**"

---

## 2. The 6 Prompt Strategies

Each strategy reflects a different hypothesis about what information is most useful to encode in item embeddings.

| Strategy | Prompt Format | Hypothesis Being Tested | Example (Beauty dataset) |
|---|---|---|---|
| **Type 1 — Title only** | `item['title']` | Paper's actual method: is the name enough? | `"CeraVe Moisturizing Cream"` |
| **Type 2 — Structured tags** | `Title \| Category: X \| Brand: Y \| Rating: Z` | Do structured attributes improve item separation? | `"CeraVe Moisturizing Cream \| Category: Face Moisturizer \| Brand: CeraVe \| Rating: 4.7"` |
| **Type 3 — Rich prose** | Natural sentence description using all metadata | Does natural language description produce richer semantic space? | `"A deeply hydrating face moisturizer by CeraVe, formulated with ceramides and hyaluronic acid for dry sensitive skin"` |
| **Type 4 — User-centric** | `Users who like X enjoy: [tags]` | Does framing around user preferences align better with recommendation goals? | `"Users who like this moisturizer enjoy: hydration, sensitive skin care, non-comedogenic products, ceramide formulas"` |
| **Type 5 — Comparative** | `X is similar to: [similar items]` | Do cross-item references create better relative positioning in embedding space? | `"CeraVe Moisturizing Cream is similar to: Cetaphil Moisturizing Cream, Vanicream Moisturizer. Appeals to fans of: gentle skincare, dermatologist-recommended products"` |
| **Type 6 — Hybrid (best hypothesis)** | Structured tags + user-centric framing combined | Does combining information types outperform any single strategy? | `"CeraVe Moisturizing Cream \| Category: Face Moisturizer \| For fans of: hydration, ceramides, sensitive skin care"` |

---

## 3. Full Implementation Code

### 3.1 The Prompt Functions (complete new code — ~30 lines)

```python
# This is the COMPLETE implementation — 6 functions, ~30 lines
# Everything else (SASRec, evaluation) reuses the existing codebase

PROMPT_STRATEGIES = {

    "type1_title_only": lambda item: (
        item["title"]
    ),

    "type2_structured": lambda item: (
        f"{item['title']} | "
        f"Category: {item.get('category', 'unknown')} | "
        f"Brand: {item.get('brand', 'unknown')} | "
        f"Rating: {item.get('avg_rating', 'N/A')}"
    ),

    "type3_rich_prose": lambda item: (
        f"A {item.get('category', 'product')} called {item['title']}. "
        f"{item.get('description', '')[:200]}"
    ),

    "type4_user_centric": lambda item: (
        f"Users who like {item['title']} enjoy: "
        f"{', '.join(item.get('tags', [])[:5])}"
    ),

    "type5_comparative": lambda item: (
        f"{item['title']} is similar to: "
        f"{item.get('similar_items', 'comparable products')}. "
        f"Appeals to fans of: {', '.join(item.get('tags', [])[:3])}"
    ),

    "type6_hybrid": lambda item: (  # BEST STRATEGY
        f"{item['title']} | "
        f"Category: {item.get('category', 'unknown')} | "
        f"For fans of: {', '.join(item.get('tags', [])[:3])}"
    ),
}

# --- Main experiment loop (reuses existing pipeline) ---
from FlagEmbedding import BGEM3FlagModel
import numpy as np

bge = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

for strategy_name, prompt_fn in PROMPT_STRATEGIES.items():
    # Generate text descriptions using this strategy
    texts = [prompt_fn(item) for item in item_catalog]

    # Encode with BGE-M3 (free, local — no API key)
    embeddings = bge.encode(texts, batch_size=256)['dense_vecs']
    np.save(f'embeddings_{strategy_name}.npy', embeddings)

    # Run SASRec with these embeddings (existing function)
    results = run_sasrec_with_embeddings(embeddings, strategy_name)
    save_results(results, strategy_name)
```

### 3.2 Embedding Quality Metrics (Additional Analysis)

Beyond recommendation accuracy, measure the **intrinsic quality** of embeddings per strategy. This creates a richer paper with two levels of analysis: downstream accuracy + upstream embedding geometry.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def embedding_isotropy(embeddings):
    """
    Measures how uniformly spread embeddings are in space.
    Higher = better. Anisotropic embeddings cluster near one direction
    and are less discriminative for recommendation.
    Formula: min(singular values) / max(singular values)
    """
    U, S, Vt = np.linalg.svd(embeddings - embeddings.mean(axis=0))
    return float(S[-1] / S[0])  # ratio of min to max singular value

def avg_pairwise_distance(embeddings, sample_n=1000):
    """Average cosine distance between random item pairs."""
    idx = np.random.choice(len(embeddings), min(sample_n, len(embeddings)), replace=False)
    sims = cosine_similarity(embeddings[idx])
    np.fill_diagonal(sims, 0)
    return 1 - sims[sims > 0].mean()

# Compute for each strategy
for strategy_name in PROMPT_STRATEGIES:
    emb = np.load(f'embeddings_{strategy_name}.npy')
    iso = embedding_isotropy(emb)
    apd = avg_pairwise_distance(emb)
    print(f"{strategy_name}: isotropy={iso:.4f}, avg_dist={apd:.4f}")
```

---

## 4. Datasets and Experimental Setup

| Dataset | Items | Metadata Available | Why Interesting |
|---|---|---|---|
| **Amazon Beauty** | 12,101 | Product name, category, brand, description, rating | Rich metadata — all 6 strategies fully applicable |
| **Steam (Gaming)** | 11,784 | Game title, genres, user tags | Paper's weak case — proves prompt design most important when names are poor |
| **MovieLens-1M** | 3,706 | Title, genre, year, director | Not in original paper — proves generalizability |

### Experimental Design

- **Embedding model:** BGE-M3 (`BAAI/bge-m3`, fp16, local — free, no API key)
- **Recommendation model:** SASRec (existing implementation, zero changes)
- **Evaluation metrics:** NDCG@10, HR@10, MRR
- **Total runs:** 18 (6 strategies × 3 datasets)
- **Seeds:** Fixed at 42 for reproducibility
- **PCA:** Follow whatever dimensionality reduction the existing repo applies

---

## 5. Expected Results and Key Findings

### Main Claim

> *"The choice of item description prompt for LLM embedding generation significantly affects sequential recommendation accuracy, with improvements of up to X% NDCG@10 over the title-only baseline used in prior work. The effect is strongest for domains with semantically non-discriminative item names, where user-centric and hybrid prompt strategies consistently outperform name-only encoding."*

### Key Pattern to Expect

The impact of prompt strategy is:
- **LARGEST** on Steam — game titles like "Portal 2" don't encode genre/gameplay style; user tags fill this gap
- **SMALLEST** on Beauty — product names like "CeraVe Moisturizing Cream" already carry rich semantic content
- **MEDIUM** on MovieLens — movie titles are partially informative but genre/year adds value

### Target Metrics to Beat (Beauty Dataset)

| Metric | Baseline (type1 — title only) | Target (type6 — hybrid) | Min Required Delta |
|---|---|---|---|
| NDCG@10 | ~0.058 | > 0.062 | +5% |
| HR@10 | ~0.105 | > 0.110 | +5% |
| MRR | ~0.045 | > 0.048 | +5% |

**Winning condition:** `type6_hybrid` (or any other strategy) beats `type1_title_only` on NDCG@10 on at least 2 of 3 datasets.

---

## 6. Two-Week Day-by-Day Execution Plan

| Day | Task | Kaggle Account | Output |
|---|---|---|---|
| Day 1 | Write all 6 prompt functions (AI agent: 30 min). Set up BGE-M3 on all 4 accounts. | All 4 | Prompt functions ready, BGE-M3 installed |
| Day 2 | Generate embeddings for Beauty dataset × 6 strategies | Account 1 | 6 × `beauty_embeddings_*.npy` files |
| Day 3 | Generate embeddings for Steam and MovieLens × 6 strategies | Accounts 2 & 3 | 12 more embedding files |
| Day 4–5 | Run SASRec pipeline for all 18 configurations (6 strategies × 3 datasets) | All 4 parallel | 18 result JSON files with NDCG/HR/MRR |
| Day 6 | Compute embedding isotropy and avg pairwise distance for all 18 configurations | Account 4 | Isotropy table |
| Day 7 | Collect all results. Generate the 6×3 heatmap figure (strategy × dataset × NDCG). | Local | Key Figure 1 of the paper |
| Day 8–9 | Write Introduction + Related Work with AI agent | Local | Sections 1–2 draft |
| Day 10–11 | Write Methodology + Experiments sections | Local | Sections 3–4 draft |
| Day 12 | Write Analysis section: why user-centric works best, what isotropy predicts accuracy | Local | Section 5 draft |
| Day 13 | Complete paper draft. Run grammar check. Generate all figures in matplotlib. | Local | Full draft ready |
| Day 14 | Post to arXiv. Format for MDPI AI and submit. | arXiv + MDPI | Public record + submission |

---

## 7. Novelty Justification (for Reviewers and Supervisor)

> *"The original paper (Section 5.2) uses a single fixed item description format and never tests alternatives. In Section 5.2 (page 17), the authors themselves discover that adding tags improves Steam results — but they treat this as a domain-specific fix rather than investigating it systematically. Our paper is the first to design and evaluate a comprehensive set of item description strategies for LLM-based sequential recommendation, and provides practitioners with the first evidence-based guideline for how to describe items to embedding models."*

---

## 8. Paper Section Outline

| Section | Title | Pages | Key Content |
|---|---|---|---|
| 1 | Introduction | 1 page | The single-template problem. Your question. Your 3 findings. |
| 2 | Related Work | 1 page | LLM recommenders, prompt engineering in NLP, embedding quality |
| 3 | Methodology | 1.5 pages | The 6 strategies, isotropy metric, experimental setup |
| 4 | Results | 2 pages | Main heatmap table, per-dataset analysis, isotropy correlation |
| 5 | Analysis | 1 page | Why user-centric works, session length interaction, dataset sensitivity |
| 6 | Conclusion | 0.5 pages | Practitioner guideline: use hybrid strategy by default |

---

## 9. Important Implementation Notes for the Agent

1. **Do NOT rewrite SASRec** — use the existing implementation in `dh-r/LLM-Sequential-Recommendation`. Zero model code changes.
2. **BGE-M3 replaces OpenAI ada-002** — it is free, runs locally on Kaggle T4 GPU, and performs comparably or better.
3. **PCA / dimensionality reduction** — apply whatever the existing repo applies. Do not change this.
4. **Seed everything** — `seed=42` for all experiments.
5. **Save results after each individual run** — Kaggle sessions can time out.
6. **The repo reports NDCG@20 in the paper** — adapt to NDCG@10 for the new paper, or report both.
7. **If SASRec doesn't natively accept pre-trained embeddings** — find where it initializes the embedding layer (`nn.Embedding`) and patch it to load from a numpy array. This is a 3-line change at most.
8. **Item field names vary by dataset** — always run the diagnostic script first to confirm actual field names before using them in prompt functions.

---

*End of PromptCraft-SeqRec Context Document*

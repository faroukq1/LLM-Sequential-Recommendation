# AGENT TASK PROMPT — Extend PopDebias-ColdBridge to New Models & Beat All Baselines

## CONTEXT

You are working inside the LLM Sequential Recommendation repository.

The following work is **already done** (do not redo it):
- **Baselines implemented and run:** SASRec (vanilla), MostPopular
- **Core models implemented and run on MovieLens-100K:**
  - `LLM2SASRec` with OpenAI ada-002
  - `BGE2SASRec` with BGE-M3 (free backbone)
  - `PopDebias-BGE2SASRec` (alpha=0.3, inverse-frequency debiasing)
  - `ColdBridge-BGE2SASRec` (tau=5, two-branch cold-start routing)
  - `PopDebias-ColdBridge-BGE2SASRec` (full combined model)
- Results for MovieLens-100K are saved in `results/movielens_100k/`
- Embedding files: `embeddings/movielens_100k/item_embeddings_bge_m3.npy`

## YOUR MISSION

Replicate the **exact same experimental pipeline** for the following **new models and datasets**, then apply every technique available to **beat the MovieLens-100K scores** on every metric.

---

## PART 1 — REPLICATE ON NEW DATASETS

Run the complete experiment suite (all models listed above) on these datasets:

### Dataset 1: Amazon Beauty
- Sessions: 22,363 | Items: 12,101 | Domain: E-commerce (cosmetics)
- Download: `datasets/amazon_beauty/` (should already be preprocessed, p-core=5)
- Save results to: `results/amazon_beauty/`
- Save embeddings to: `embeddings/amazon_beauty/item_embeddings_bge_m3.npy`

### Dataset 2: Steam
- Sessions: 279,290 | Items: 11,784 | Domain: Gaming
- Download: `datasets/steam/`
- Save results to: `results/steam/`
- Save embeddings to: `embeddings/steam/item_embeddings_bge_m3.npy`

### Dataset 3: Yelp
- Sessions: ~70,000 | Items: ~15,000 | Domain: Restaurants/Local businesses
- Download: `datasets/yelp/`
- Save results to: `results/yelp/`
- Save embeddings to: `embeddings/yelp/item_embeddings_bge_m3.npy`

For each dataset, run:
1. `MostPopular` baseline
2. `SASRec` vanilla baseline
3. `BGE2SASRec` (BGE-M3 embeddings, no debiasing, no cold-start)
4. `PopDebias-BGE2SASRec` (alpha ∈ {0.1, 0.3, 0.5, 0.7} — tune on validation set)
5. `ColdBridge-BGE2SASRec` (tau ∈ {2, 3, 5, 10, 15} — tune on validation set)
6. `PopDebias-ColdBridge-BGE2SASRec` (best alpha + best tau from above)

---

## PART 2 — BEAT THE BASELINE SCORES

The target is to **exceed** the MovieLens-100K scores on all six metrics:
- NDCG@10, NDCG@20
- HitRate@10, HitRate@20
- Long-Tail Hit@10 (items with <500 interactions — most important metric)
- ILD@10 (Intra-List Diversity)

Apply the following improvements **in order of expected impact**:

### Improvement 1 — Try BERT4Rec as the Sequential Backbone
The original paper tests both SASRec and BERT4Rec. Implement `PopDebias-ColdBridge-BERT4Rec` using BGE-M3 as initialization. Compare against the SASRec variant. Use whichever achieves higher NDCG@10 as the final backbone.

```python
# BERT4Rec with BGE-M3 initialization
model = BERT4Rec(item_embedding_dim=1024)
model.item_embeddings.weight.data = torch.tensor(bge_m3_embeddings)
```

### Improvement 2 — Tune the Debiasing Formula More Aggressively
The current formula is: `score = cosine_sim × (1 / count(i)^alpha)`

Try the following extended variants and report results for each:

**Variant A — Log Normalization (smoother debiasing):**
```python
score = cosine_sim * (1 / (1 + np.log(count(i))) ** alpha)
```

**Variant B — Rank-Based Debiasing (popularity rank instead of raw count):**
```python
popularity_rank = rank_items_by_count(all_items)  # rank 1 = most popular
score = cosine_sim * (1 / popularity_rank(i) ** alpha)
```

**Variant C — Sigmoid Smoothing (prevents extreme penalization):**
```python
score = cosine_sim * sigmoid(-alpha * (count(i) / median_count - 1))
```

Run all three variants with alpha ∈ {0.1, 0.3, 0.5} on Amazon Beauty. Report the joint metric: `0.7 * NDCG@10 + 0.3 * LongTail_Hit@10`. Pick the formula that maximizes this joint metric.

### Improvement 3 — Hybrid Cold-Start Branch (Upgrade from Averaging)
The current cold-start branch averages item embeddings. Replace with a **weighted average** where more recent items get higher weight:

```python
def cold_branch_weighted(user_interactions, item_embeddings, decay=0.8):
    """
    Instead of: mean(item_embeddings)
    Use: weighted mean with exponential recency decay
    """
    n = len(user_interactions)
    weights = np.array([decay ** (n - i - 1) for i in range(n)])
    weights = weights / weights.sum()
    
    known_embs = item_embeddings[[item_to_idx[i] for i in user_interactions]]
    session_embedding = np.average(known_embs, axis=0, weights=weights)
    return session_embedding
```

Tune `decay` ∈ {0.5, 0.7, 0.8, 0.9, 1.0} on validation set. `decay=1.0` is equivalent to the current uniform average.

### Improvement 4 — Embedding Ensemble
Do NOT train new models. Simply compute a weighted ensemble of embeddings at inference time:

```python
# Load precomputed embeddings
bge_m3_embs    = np.load('embeddings/{dataset}/item_embeddings_bge_m3.npy')
e5_large_embs  = np.load('embeddings/{dataset}/item_embeddings_e5_large.npy')
sbert_embs     = np.load('embeddings/{dataset}/item_embeddings_sbert.npy')

# Normalize each to unit sphere
bge_norm   = bge_m3_embs   / np.linalg.norm(bge_m3_embs,   axis=1, keepdims=True)
e5_norm    = e5_large_embs  / np.linalg.norm(e5_large_embs,  axis=1, keepdims=True)
sbert_norm = sbert_embs     / np.linalg.norm(sbert_embs,     axis=1, keepdims=True)

# Ensemble (tune weights w1, w2, w3 on validation)
ensemble_embs = w1*bge_norm + w2*e5_norm + w3*sbert_norm
ensemble_embs = ensemble_embs / np.linalg.norm(ensemble_embs, axis=1, keepdims=True)
```

Generate E5-Large-v2 and Sentence-BERT embeddings for all datasets. Tune weights via grid search: `w1+w2+w3=1, step=0.1`.

### Improvement 5 — Dynamic Tau Per User Segment
Instead of a single global tau, use different thresholds per user activity segment:

```python
def get_tau_for_user(n_interactions):
    if n_interactions <= 3:
        return 3    # very cold: use LLM branch even more aggressively
    elif n_interactions <= 10:
        return 7    # transitional users
    else:
        return 15   # warm users: trust the sequential model more
```

Test this dynamic tau against the fixed tau=5 baseline. Report cold-user vs warm-user HitRate@10 separately.

---

## PART 3 — OUTPUT REQUIREMENTS

For each dataset and each model variant, produce:

### 3.1 Numerical Results
Save a CSV file at `results/{dataset}/full_results.csv` with columns:
```
model_name, alpha, tau, NDCG@10, NDCG@20, HR@10, HR@20, LongTail_HR@10, CatalogCoverage, Serendipity, ILD@10, training_time_sec, inference_time_sec
```

### 3.2 Best Model per Dataset
Print to console and save to `results/{dataset}/best_model.txt`:
```
BEST MODEL: {model_name}
Best Alpha: {value}
Best Tau:   {value}
NDCG@10:    {value} (vs MovieLens-100K baseline: {baseline_value}, delta: {+/-X.X%})
HR@10:      {value}
LT-HR@10:   {value}
ILD@10:     {value}
```

### 3.3 Cross-Dataset Summary Table
After all datasets are complete, generate `results/SUMMARY_TABLE.md`:

| Model | ML-100K | Beauty | Steam | Yelp | Average |
|---|---|---|---|---|---|
| SASRec (vanilla) | ... | ... | ... | ... | ... |
| BGE2SASRec | ... | ... | ... | ... | ... |
| PopDebias-BGE2SASRec | ... | ... | ... | ... | ... |
| ColdBridge-BGE2SASRec | ... | ... | ... | ... | ... |
| PopDebias-ColdBridge (SASRec) | ... | ... | ... | ... | ... |
| PopDebias-ColdBridge (BERT4Rec) | ... | ... | ... | ... | ... |
| + Improvement 2 (Best Debias Variant) | ... | ... | ... | ... | ... |
| + Improvement 3 (Weighted Cold Branch) | ... | ... | ... | ... | ... |
| + Improvement 4 (Embedding Ensemble) | ... | ... | ... | ... | ... |
| **FULL SYSTEM (All Improvements)** | ... | ... | ... | ... | ... |

**Bold** every cell where you beat the MovieLens-100K `PopDebias-ColdBridge-BGE2SASRec` score.

### 3.4 Figures
Generate the following plots and save to `results/figures/`:
- `alpha_sweep_{dataset}.png` — NDCG@10 vs Long-Tail Hit@10 as alpha varies (Pareto frontier curve)
- `tau_sweep_{dataset}.png` — Cold-user vs Warm-user HR@10 as tau varies (dual-line chart)
- `embedding_comparison.png` — Bar chart: BGE-M3 vs E5 vs SBERT vs Ensemble across datasets
- `cold_warm_gap_{dataset}.png` — Bar chart: cold-user HR@10 with and without ColdBridge

---

## PART 4 — EXECUTION ORDER

Run in this exact order to maximize parallelism on Kaggle:

```
ACCOUNT 1 (Amazon Beauty):
  [x] Generate BGE-M3, E5-Large, SBERT embeddings
  [x] Baselines → BGE2SASRec → PopDebias ablation (alpha sweep)
  [x] ColdBridge ablation (tau sweep)
  [x] Best combined model → Improvements 2, 3, 4

ACCOUNT 2 (Steam):
  [x] Generate BGE-M3, E5-Large, SBERT embeddings
  [x] Same suite as Account 1

ACCOUNT 3 (Yelp):
  [x] Generate BGE-M3, E5-Large, SBERT embeddings
  [x] Same suite as Account 1

ACCOUNT 4 (All datasets — BERT4Rec backbone):
  [x] PopDebias-ColdBridge-BERT4Rec on all 3 new datasets
  [x] Compare BERT4Rec vs SASRec results
  [x] Generate all figures
  [x] Compile SUMMARY_TABLE.md
```

---

## PART 5 — ACCEPTANCE CRITERIA

The agent's work is complete only when **ALL** of the following are true:

- [ ] `results/amazon_beauty/full_results.csv` exists with ≥10 model rows
- [ ] `results/steam/full_results.csv` exists with ≥10 model rows
- [ ] `results/yelp/full_results.csv` exists with ≥10 model rows
- [ ] `results/SUMMARY_TABLE.md` is complete with all improvements filled in
- [ ] At least **one model variant** on at least **two datasets** achieves NDCG@10 **above** the MovieLens-100K best score
- [ ] Long-Tail Hit@10 is ≥ 20% higher than vanilla SASRec on all 3 new datasets
- [ ] Cold-start HitRate@10 is ≥ 15% higher than vanilla SASRec for cold users (≤5 interactions) on all 3 new datasets
- [ ] All 4 figures are generated in `results/figures/`
- [ ] `results/{dataset}/best_model.txt` exists for each dataset

---

## NOTES FOR THE AGENT

- Do **NOT** touch the MovieLens-100K results or code. They are the reference baseline.
- Do **NOT** re-run experiments that already have output CSV files unless explicitly asked.
- When in doubt about a hyperparameter, **tune on the validation set** — never on the test set.
- All embedding generation should use `use_fp16=True` to stay within P100 memory limits.
- Batch size for BGE-M3 encoding: **256** for Beauty/Yelp, **128** for Steam (larger dataset).
- If any Kaggle session times out (9-hour limit), save checkpoints every 500 training steps to `checkpoints/{dataset}/{model_name}/step_{N}.pt`
- Report exact runtimes for each experiment so the paper's compute section is accurate.
- The paper will cite MovieLens-100K results as the primary comparison point. Make the delta columns crystal clear.

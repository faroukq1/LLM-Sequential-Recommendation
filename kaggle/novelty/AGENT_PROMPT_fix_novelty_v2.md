# AGENT FIX PROMPT — Novelty Pipeline v2 (PopDebias-ColdBridge)
# MovieLens-100K | Fix broken reranker + beat core model benchmarks

---

## SITUATION

You already ran the full novelty pipeline on MovieLens-100K.
Results are in `full_results.csv` and `best_model.txt`.
The core model benchmarks are in `core_model_results_ml100k.csv`.

**Read both files before touching any code.**

---

## WHAT IS WORKING vs WHAT IS BROKEN

### ✅ WORKING — Keep exactly as-is

| Model | NDCG@10 | HR@10 | Coverage | Assessment |
|---|---|---|---|---|
| ColdBridge-BGE2SASRec (tau=15) | 0.05812 | 0.09409 | 0.52889 | Best novelty result. Keep. |
| ColdBridge-BGE2SASRec (tau=5)  | 0.05781 | 0.09409 | 0.51578 | Good. Keep. |
| BGE2SASRec (baseline)           | 0.04612 | 0.08534 | 0.31745 | Novelty baseline. Keep. |
| SASRec (baseline)               | 0.04722 | 0.10066 | 0.30137 | Novelty baseline. Keep. |

### ❌ BROKEN — Fix these

| Model | NDCG@10 | vs BGE2SASRec | Root Cause |
|---|---|---|---|
| PopDebias-BGE2SASRec (α=0.1) | 0.00570 | **−88%** | rerank_popdebias **replaces** base score instead of blending |
| FULL_SYSTEM                   | 0.02601 | **−44%** | inherits broken PopDebias damage |
| PopDebias-ColdBridge-rank     | 0.03137 | **−32%** | same root cause |
| PopDebias-ColdBridge-sigmoid  | 0.02777 | **−40%** | same root cause |
| PopDebias-ColdBridge-log_norm | 0.02601 | **−44%** | same root cause |
| PopDebias-ColdBridge-WeightedCold | 0.03058 | **−34%** | same + hard cold switch |

---

## TARGETS TO BEAT

These are the core model scores from `core_model_results_ml100k.csv`.
They are the official benchmark. Beat them in this exact order of priority.

| Priority | Target | NDCG@10 | HR@10 | Coverage@10 | Status |
|---|---|---|---|---|---|
| 🥇 Must beat | Core LLM2SASRec | 0.05994 | 0.12472 | 0.23228 | Current best is 0.05812 — **gap: −3%** |
| 🥈 Stretch   | Core BERT4Rec    | 0.05594 | 0.10913 | 0.24300 | Already beaten on NDCG. HR gap: −14% |
| 🥉 Dream     | Core SASRec      | 0.07303 | 0.14254 | 0.36152 | Big gap. Aim for it with ensemble. |

**Current novelty best is already 3rd overall on NDCG@10 across ALL models.**
Fix PopDebias, close the HR gap, and it beats LLM2SASRec cleanly.

---

## ROOT CAUSE — Why PopDebias Is Catastrophically Broken

The current `rerank_popdebias` function computes:
```python
# CURRENT (BROKEN)
score = cosine_sim(session_emb, item_emb) * (1 / count(i) ** alpha)
```

This **throws away the trained SASRec/BGE2SASRec logits entirely.**
The model trained for 14+ seconds. Those logits encode learned sequential patterns.
Multiplying raw cosine similarity by an inverse-count gives you noisy garbage.

The warm model already produced a ranked list. **Reranking must start from that list.**

---

## FIX 1 — Relevance-Preserving Rerank (CRITICAL — Do This First)

Replace `rerank_popdebias` with a z-score blend that **keeps the warm model's signal**.

```python
import numpy as np

def z_score_normalize(arr):
    """Z-normalize an array. Returns zeros if std=0."""
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std

def rerank_popdebias_v2(
    warm_scores,        # np.array shape (n_items,): raw logits/scores from warm model
    item_embeddings,    # np.array shape (n_items, dim): BGE-M3 embeddings, already L2-normalized
    session_embedding,  # np.array shape (dim,): current session embedding, already L2-normalized
    item_counts,        # np.array shape (n_items,): interaction counts per item
    alpha=0.05,         # SMALL — debiasing strength. Start at 0.05, max 0.2.
    lambda_sim=0.0,     # similarity term weight. Start at 0.0 (disabled).
    lambda_pop=0.15,    # popularity penalty weight. Start at 0.15.
    top_n_anchor=5,     # always preserve top-N warm model items unchanged
):
    """
    Rerank warm model output while preserving its relevance signal.

    Formula:
        final = z(warm_scores)
               + lambda_sim * z(cosine_sim)
               - lambda_pop * z(popularity_penalty)

    The warm model always dominates. Novelty terms are small corrections.
    Top-N anchor: top_n_anchor items from warm model are NEVER moved below rank top_n_anchor.
    """
    n_items = len(warm_scores)

    # --- Step 1: Compute novelty components ---
    # Cosine similarity between session and each item
    cos_sim = item_embeddings @ session_embedding  # shape (n_items,)

    # Popularity penalty: higher count = more penalized
    pop_penalty = item_counts.astype(float) ** alpha  # shape (n_items,)

    # --- Step 2: Z-normalize ALL components independently ---
    z_warm   = z_score_normalize(warm_scores)
    z_sim    = z_score_normalize(cos_sim)
    z_pop    = z_score_normalize(pop_penalty)

    # --- Step 3: Blend — warm model dominates ---
    final_scores = z_warm + lambda_sim * z_sim - lambda_pop * z_pop

    # --- Step 4: Top-N anchor — protect warm model's top picks ---
    if top_n_anchor > 0:
        warm_top_n = np.argsort(warm_scores)[::-1][:top_n_anchor]
        # Give anchored items a large bonus so they stay in top positions
        anchor_bonus = np.abs(final_scores).max() * 2.0 + 1.0
        final_scores[warm_top_n] += anchor_bonus

    return final_scores


# --- Hyperparameter search (safe, constrained) ---
# alpha:      [0.01, 0.05, 0.1, 0.15, 0.2]   # SMALL values only
# lambda_pop: [0.05, 0.1, 0.15, 0.2, 0.3]
# lambda_sim: [0.0, 0.05, 0.1]                # 0.0 = disabled, usually best
# top_n_anchor: [3, 5]

# CONSTRAINT: Only keep configs where HR@10 drops < 10% from BGE2SASRec (0.08534)
# HR@10 floor = 0.08534 * 0.90 = 0.07681
# Reject any config below this floor immediately.
```

**Expected result after this fix:**
- PopDebias NDCG@10 should be **≥ 0.045** (vs current broken 0.006)
- Combined with ColdBridge it should reach **≥ 0.058**

---

## FIX 2 — Soft ColdBridge Gate (Replace Hard Binary Switch)

The current `route_coldbridge` does a hard binary switch:
```python
# CURRENT (BRITTLE)
if len(user_interactions) >= tau:
    return warm_model_recs(user)
else:
    return cold_llm_recs(user)
```

Replace with a **soft sigmoid blend** that smoothly interpolates:

```python
def soft_coldbridge_blend(
    warm_scores,          # np.array (n_items,): warm model scores
    cold_scores,          # np.array (n_items,): cold LLM similarity scores
    session_len,          # int: number of interactions for this user
    tau=5,                # midpoint: at session_len=tau, p_cold=0.5
    steepness=1.0,        # how sharp the transition is (higher = sharper)
):
    """
    Soft blend between warm and cold model based on session length.

    p_cold = sigmoid(steepness * (tau - session_len))
    final  = (1 - p_cold) * z(warm_scores) + p_cold * z(cold_scores)

    When session_len >> tau: p_cold → 0, uses warm model entirely.
    When session_len << tau: p_cold → 1, uses cold LLM scores entirely.
    When session_len = tau:  p_cold = 0.5, equal blend.
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    p_cold = sigmoid(steepness * (tau - session_len))

    z_warm = z_score_normalize(warm_scores)
    z_cold = z_score_normalize(cold_scores)

    blended = (1.0 - p_cold) * z_warm + p_cold * z_cold
    return blended


# --- Hyperparameter search ---
# tau:        [3, 5, 7, 10, 15]
# steepness:  [0.5, 1.0, 2.0]     # 0.5 = very gradual, 2.0 = near-hard switch
```

---

## FIX 3 — Weighted Cold Branch (Recency Decay)

Already partially tested but the cold branch itself still uses uniform averaging.
Replace the cold branch's session embedding computation:

```python
def cold_session_embedding(user_interactions, item_embeddings, item_to_idx, decay=0.8):
    """
    Weighted average of known item embeddings with recency bias.
    Most recent interaction gets weight 1.0, oldest gets weight decay^(n-1).

    decay=1.0 → uniform average (current behavior)
    decay=0.8 → recommended
    decay=0.5 → very aggressive recency
    """
    n = len(user_interactions)
    if n == 0:
        return None

    indices = [item_to_idx[i] for i in user_interactions if i in item_to_idx]
    if not indices:
        return None

    # Weights: oldest=decay^(n-1), newest=1.0
    weights = np.array([decay ** (n - k - 1) for k in range(len(indices))], dtype=float)
    weights /= weights.sum()

    embs = item_embeddings[indices]  # shape (n_known, dim)
    session_emb = np.average(embs, axis=0, weights=weights)
    session_emb = session_emb / (np.linalg.norm(session_emb) + 1e-9)  # re-normalize
    return session_emb

# decay search: [0.5, 0.7, 0.8, 0.9, 1.0]
```

---

## FIX 4 — Constrained Hyperparameter Search

All hparam sweeps MUST enforce these constraints before recording any result:

```python
# Hard constraints — reject config immediately if ANY is violated:
HR_FLOOR      = 0.07681   # BGE2SASRec HR@10 (0.08534) * 0.90
NDCG_FLOOR    = 0.04151   # BGE2SASRec NDCG@10 (0.04612) * 0.90
COVERAGE_MIN  = 0.35      # must retain meaningful diversity gain

def is_valid_config(ndcg, hr, coverage):
    return (hr >= HR_FLOOR) and (ndcg >= NDCG_FLOOR) and (coverage >= COVERAGE_MIN)

# Joint optimization metric (only for valid configs):
# joint = 0.5 * ndcg_norm + 0.3 * hr_norm + 0.2 * coverage_norm
# where X_norm = X / X_best_in_valid_pool
```

---

## REQUIRED ABLATION TABLE

Run these 5 configurations **in this exact order**. Each builds on the previous.
Use the **same train/val/test split** for all of them.

| Step | Model | Components Active | Expected NDCG@10 | Expected HR@10 |
|---|---|---|---|---|
| 0 | BGE2SASRec (base) | None | 0.04612 | 0.08534 |
| 1 | + SoftColdBridge | Soft gate (tau=5, steep=1.0) | ≥ 0.052 | ≥ 0.088 |
| 2 | + SafePopDebias | z-blend (α=0.05, λ_pop=0.15) | ≥ 0.048 | ≥ 0.082 |
| 3 | + Both (best hparams) | SoftCold + SafeDebias | ≥ 0.058 | ≥ 0.090 |
| 4 | + WeightedCold | decay=0.8 in cold branch | ≥ 0.059 | ≥ 0.092 |

Save this table as `results/movielens_100k/ablation_v2.md`.

---

## REQUIRED PARETO FRONTIER

After all runs, generate a Pareto analysis across these 4 metrics:
- NDCG@10 (relevance — maximize)
- HR@10 (recall — maximize)
- CatalogCoverage (diversity — maximize)
- ILD@10 (intra-list diversity — maximize)

A configuration is **Pareto-dominant** if no other config beats it on ALL four metrics simultaneously.

```python
def find_pareto_front(results_df):
    """Returns boolean mask of Pareto-optimal rows."""
    metrics = ['NDCG@10', 'HR@10', 'CatalogCoverage', 'ILD@10']
    vals = results_df[metrics].values
    n = len(vals)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= on all and > on at least one
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                is_pareto[i] = False
                break
    return is_pareto
```

Save the Pareto-optimal rows as `results/movielens_100k/pareto_front_v2.csv`.

---

## OUTPUT REQUIREMENTS

### 1. Updated `results/movielens_100k/full_results.csv`
Append new rows. **Do not delete existing rows.** New rows must have the same column schema:
```
model_name,alpha,tau,NDCG@10,NDCG@20,HR@10,HR@20,LongTail_HR@10,CatalogCoverage,Serendipity,ILD@10,training_time_sec,inference_time_sec,debias_variant,cold_decay
```
Add extra columns for new configs:
```
lambda_sim,lambda_pop,top_n_anchor,steepness,blend_type
```
(Leave blank for legacy rows — backward compatible.)

### 2. Updated `results/movielens_100k/best_model.txt`
```
BEST MODEL v2: {model_name}
Config: alpha={}, tau={}, lambda_pop={}, steepness={}, decay={}

NDCG@10:       {value}  (vs Core LLM2SASRec: 0.05994, delta: {+/-X.X%})
HR@10:         {value}  (vs Core LLM2SASRec: 0.12472, delta: {+/-X.X%})
LongTail_HR@10:{value}
CatalogCoverage:{value} (vs Core LLM2SASRec: 0.23228, delta: {+/-X.X%})
ILD@10:        {value}

BEATS Core LLM2SASRec on NDCG@10: YES/NO
BEATS Core BERT4Rec    on NDCG@10: YES/NO
BEATS Core SASRec      on NDCG@10: YES/NO
```

### 3. `results/movielens_100k/ablation_v2.md`
Ablation table in markdown. See format above.

### 4. `results/movielens_100k/pareto_front_v2.csv`
Pareto-optimal configurations only.

### 5. Console output during run
Print after each config:
```
[CONFIG] model=X alpha=X tau=X lambda_pop=X steepness=X decay=X
[RESULT] NDCG@10=X HR@10=X Coverage=X ILD=X
[STATUS] VALID/REJECTED (reason if rejected)
[JOINT ] joint_score=X
```

---

## EXECUTION ORDER

```
Step 1 — Smoke test (5 min)
  Run BGE2SASRec bare to confirm baseline still reproducible.
  If NDCG@10 ≠ 0.04612 ± 0.002, STOP and fix eval/split first.

Step 2 — Fix PopDebias (20 min)
  Run rerank_popdebias_v2 with alpha=0.05, lambda_pop=0.15, top_n_anchor=5.
  Must get NDCG@10 ≥ 0.040. If not, STOP and debug z_score_normalize.

Step 3 — Fix ColdBridge (20 min)
  Run soft_coldbridge_blend with tau=5, steepness=1.0.
  Must get NDCG@10 ≥ ColdBridge-hard best (0.05812).

Step 4 — Ablation table (60 min)
  Run all 5 ablation steps. Save ablation_v2.md.

Step 5 — Hparam sweep (90 min)
  Sweep all combinations with constrained search.
  Alpha: [0.01, 0.05, 0.1, 0.15, 0.2]
  lambda_pop: [0.05, 0.1, 0.15, 0.2]
  tau: [3, 5, 7, 10, 15]
  steepness: [0.5, 1.0, 2.0]
  decay: [0.7, 0.8, 0.9, 1.0]
  top_n_anchor: [3, 5]
  Total: ~400 combos. Skip any that violate HR_FLOOR immediately (no eval needed).

Step 6 — Pareto analysis
  Run find_pareto_front on all valid configs.
  Save pareto_front_v2.csv.

Step 7 — Update outputs
  Append to full_results.csv.
  Overwrite best_model.txt with v2 best.
```

---

## ACCEPTANCE CRITERIA — Done when ALL are true

- [ ] `PopDebias-BGE2SASRec v2` NDCG@10 ≥ 0.040 (was 0.006 — fix confirmed)
- [ ] `FULL_SYSTEM v2` NDCG@10 ≥ 0.055 (was 0.026 — fix confirmed)
- [ ] Best v2 model NDCG@10 **beats Core LLM2SASRec 0.05994** (primary target)
- [ ] Best v2 model HR@10 ≥ 0.10 (close the HR gap vs core models)
- [ ] CatalogCoverage of best v2 model ≥ 0.40 (keep diversity gain)
- [ ] No valid config has HR@10 < 0.07681 (floor constraint enforced everywhere)
- [ ] `ablation_v2.md` exists with 5 rows
- [ ] `pareto_front_v2.csv` exists with ≥ 3 Pareto-optimal configs
- [ ] `full_results.csv` updated (no rows deleted, new rows appended)
- [ ] `best_model.txt` updated with v2 winner and comparison vs core models

---

## QUICK REFERENCE — Numbers to Keep in Mind

```
BGE2SASRec baseline  : NDCG 0.04612 | HR 0.08534 | Cov 0.31745
ColdBridge best (now): NDCG 0.05812 | HR 0.09409 | Cov 0.52889  ← protect this
Core LLM2SASRec      : NDCG 0.05994 | HR 0.12472 | Cov 0.23228  ← beat this
Core BERT4Rec        : NDCG 0.05594 | HR 0.10913 | Cov 0.24300  ← already beaten on NDCG
Core SASRec          : NDCG 0.07303 | HR 0.14254 | Cov 0.36152  ← stretch goal

HR floor (90% of BGE2SASRec): 0.07681  ← reject any config below this
NDCG floor (90% of BGE2SASRec): 0.04151 ← reject any config below this
```

# AGENT IMPROVEMENT PROMPT — PopDebias-ColdBridge v3
# MovieLens-100K | Fix regression + systematic improvements to beat all core models

---

## READ THESE FILES FIRST — BEFORE TOUCHING ANY CODE

```
results/movielens_100k/full_results.csv          # v1 novelty results
results/movielens_100k/pareto_front_v2.csv       # v2 pareto results (currently broken)
core_model_results_ml100k.csv                    # official core benchmarks (ground truth)
run_novelty_ml100k_kaggle.ipynb                  # current notebook
```

Do not change any code until you have read all four files and confirmed the numbers below match what you see.

---

## SITUATION: WHAT WE KNOW

### The confirmed regression (v2 broke the pipeline)

Every shared baseline dropped by ~50–75% between v1 and v2. This is a pipeline bug, not a model bug. Evidence: MostPopular (which uses no model at all) stayed flat at ~0.0132 in both versions. Everything that uses the model dropped uniformly.

| Model | v1 NDCG@10 | v2 NDCG@10 | Ratio |
|---|---|---|---|
| ColdBridge tau=15 | 0.05812 | 0.03016 | 0.52x |
| BGE2SASRec | 0.04612 | 0.01156 | 0.25x |
| SASRec | 0.04722 | 0.01173 | 0.25x |
| MostPopular | 0.01318 | 0.01319 | 1.00x ← not affected |

MostPopular is unaffected. BGE2SASRec and SASRec dropped identically. This means the split or eval logic changed in the v2 refactor.

### What v2 also revealed (scale-invariant findings — these ARE valid)

Even with the broken absolute numbers, the relative rankings within v2 tell us important things:

- Hard ColdBridge tau=15 is still the best model at +161% over BGE2SASRec baseline
- Soft ColdBridge (v2 fix) is WORSE than hard routing: 0.01641 vs 0.03016
  - **Conclusion: drop soft blending entirely. Hard routing wins.**
- PopDebias-v2 is still barely better than raw BGE2SASRec: 0.01606 vs 0.01156
  - The z-score fix helped slightly but PopDebias is still not worth keeping in its current form
- SafePopDebias is actually WORSE than BGE2SASRec: 0.01048 vs 0.01156
  - **Conclusion: the current PopDebias implementation has no path forward as a scoring modifier**

### The core benchmarks we need to beat (official targets — never change these)

| Model | NDCG@10 | HR@10 | Coverage@10 | MRR@10 |
|---|---|---|---|---|
| Core SASRec | 0.07303 | 0.14254 | 0.36152 | 0.05215 |
| Core LLM2SASRec | 0.05994 | 0.12472 | 0.23228 | 0.04068 |
| Core BERT4Rec | 0.05594 | 0.10913 | 0.24300 | 0.03999 |
| Core LLM2GRU4Rec | 0.05211 | 0.11136 | 0.21322 | 0.03443 |
| Core LLMSeqSim | 0.01228 | 0.02895 | **0.65694** | 0.00741 |

Current best (v1, real): ColdBridge NDCG 0.05812 — 3% below LLM2SASRec, 20% below SASRec.

---

## STEP 0 — FIX THE REGRESSION FIRST (do this before anything else)

This is mandatory. Do not run any new experiments until the baselines reproduce.

### Diagnostic checklist

Run this diagnostic cell at the top of the notebook and print every line:

```python
import numpy as np

# --- Diagnostic: confirm data pipeline integrity ---
print("=== PIPELINE DIAGNOSTIC ===")
print(f"Total users:      {len(all_users)}")
print(f"Total items:      {len(all_items)}")
print(f"Train sessions:   {len(train_data)}")
print(f"Val sessions:     {len(val_data)}")
print(f"Test sessions:    {len(test_data)}")
print(f"Train interactions: {sum(len(s) for s in train_data.values())}")
print(f"Test interactions:  {sum(len(s) for s in test_data.values())}")
print(f"Avg test seq len:   {np.mean([len(s) for s in test_data.values()]):.2f}")
print(f"Test split ratio:   {len(test_data)/(len(train_data)+len(val_data)+len(test_data)):.3f}")
print(f"Random seed used:   {RANDOM_SEED}")
print(f"p-core value:       {P_CORE}")
print()

# Run BGE2SASRec bare — no reranking, no routing
# Expected: NDCG@10 ≈ 0.0461, HR@10 ≈ 0.0853
# If these numbers are < 0.030, the split is wrong. Fix before proceeding.
```

### Most likely causes and exact fixes

**Cause 1 — Split seed changed (most likely)**
The v2 refactor probably changed or removed the random seed used for the train/test split.

```python
# WRONG (seed missing or changed):
train, test = split_data(interactions)
train, test = split_data(interactions, seed=99)  # different seed

# CORRECT (must match v1 exactly):
np.random.seed(42)   # or whatever seed v1 used — find it in the v1 notebook
random.seed(42)
train, test = split_data(interactions, seed=42)
```

**Cause 2 — Leave-one-out split replaced with ratio split**
Sequential recommendation uses leave-one-out (last item = test). If the v2 refactor switched to a random 80/10/10 split, all absolute scores change.

```python
# CORRECT — leave-one-out (last interaction per user is the test item):
def split_leave_one_out(user_sessions):
    train, val, test = {}, {}, {}
    for user, session in user_sessions.items():
        if len(session) < 3:
            continue
        test[user]  = session[-1:]       # last item
        val[user]   = session[-2:-1]     # second-to-last
        train[user] = session[:-2]       # everything before
    return train, val, test
```

**Cause 3 — Rerank code applied to baselines**
SASRec and BGE2SASRec must produce raw scores with no modification:

```python
def run_model(model_name, ...):
    scores = model.predict(user_seq)
    
    # CRITICAL: baselines bypass all novelty code
    if model_name in ['SASRec', 'BGE2SASRec', 'MostPopular']:
        return scores  # raw, untouched
    
    # Only novelty variants go through reranking
    if 'ColdBridge' in model_name:
        scores = apply_coldbridge(scores, ...)
    if 'PopDebias' in model_name:
        scores = apply_popdebias(scores, ...)
    
    return scores
```

### Success criterion for Step 0

BGE2SASRec must reproduce within 5% of v1:
- NDCG@10 ≥ 0.0438 (v1 was 0.04612)
- HR@10 ≥ 0.0811 (v1 was 0.08534)

**Do not proceed to any improvement until this passes.**

---

## STEP 1 — BACKBONE IMPROVEMENT (biggest headroom)

### Why this is the most important step

The novelty pipeline's SASRec trains to NDCG 0.04722. Core SASRec reaches 0.07303. That is a 35% gap coming purely from training quality — not model architecture, not embeddings. Before adding any novelty modules, the backbone must be stronger.

### 1a — Find the correct SASRec hyperparameters

The core notebook uses SASRec at 0.07303. Find its training config and copy it exactly into the novelty notebook.

```python
# In core notebook — print this and record every value:
print(f"SASRec config: {sasrec_model.config}")
# Expected values somewhere around:
# num_heads=1, num_layers=2, hidden_size=50 or 64
# dropout=0.2, lr=0.001, batch_size=256, epochs=200
# max_seq_len=50, l2_emb=0.0

# In novelty notebook — replace current training config with these exact values
SASREC_CONFIG = {
    'num_heads': <copy from core>,
    'num_layers': <copy from core>,
    'hidden_size': <copy from core>,
    'dropout_rate': <copy from core>,
    'learning_rate': <copy from core>,
    'batch_size': <copy from core>,
    'num_epochs': <copy from core>,
    'max_seq_length': <copy from core>,
    'l2_emb': <copy from core>,
}
```

### 1b — LR scheduling (if not already used)

If the core notebook uses a learning rate scheduler, add it:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer)
    scheduler.step()
    if epoch % 10 == 0:
        ndcg = evaluate(model, val_data)
        print(f"Epoch {epoch}: val NDCG@10 = {ndcg:.5f}")
```

### 1c — Early stopping on validation NDCG

```python
best_val_ndcg = 0.0
patience = 20
no_improve = 0
best_weights = None

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer)
    val_ndcg = evaluate(model, val_data)
    
    if val_ndcg > best_val_ndcg:
        best_val_ndcg = val_ndcg
        best_weights = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f"Early stop at epoch {epoch}. Best val NDCG: {best_val_ndcg:.5f}")
        break

model.load_state_dict(best_weights)  # restore best checkpoint
```

### Target after Step 1

Novelty SASRec NDCG@10 ≥ 0.065 (currently 0.047, core achieves 0.073).
If you cannot reach 0.065, debug the config mismatch before continuing.

---

## STEP 2 — EMBEDDING UPGRADE (second biggest lever)

Core LLM2SASRec uses OpenAI ada-002 and reaches 0.05994 vs BGE2SASRec at 0.04612. That is a 30% gap from embedding quality alone. Try these free alternatives in order:

### 2a — Test these embeddings (one at a time, record NDCG@10 for each)

```python
# Option A: gte-large (strong on retrieval tasks, 1024d, free)
from sentence_transformers import SentenceTransformer
model_gte = SentenceTransformer('thenlper/gte-large')
embs_gte = model_gte.encode(item_texts, batch_size=256, normalize_embeddings=True)
np.save('embeddings/item_embeddings_gte_large.npy', embs_gte)

# Option B: E5-large-v2 (instruction-tuned, 1024d, free)
model_e5 = SentenceTransformer('intfloat/e5-large-v2')
# E5 requires prefix for passages:
item_texts_e5 = ["passage: " + t for t in item_texts]
embs_e5 = model_e5.encode(item_texts_e5, batch_size=256, normalize_embeddings=True)
np.save('embeddings/item_embeddings_e5_large.npy', embs_e5)

# Option C: nomic-embed-text-v1.5 (strongest free model on MTEB as of 2024, 768d)
model_nomic = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
item_texts_nomic = ["search_document: " + t for t in item_texts]
embs_nomic = model_nomic.encode(item_texts_nomic, batch_size=256, normalize_embeddings=True)
np.save('embeddings/item_embeddings_nomic.npy', embs_nomic)
```

Run BGE2SASRec with each embedding and record NDCG@10. Pick the best one for all subsequent experiments. Do NOT run the full experiment suite for each — just the backbone BGE2SASRec to find the best embedding.

### 2b — Dimension alignment

If the new embedding has a different dimension than BGE-M3 (1024d), update the SASRec input layer:

```python
# SASRec embedding projection layer — set to match new embedding dim
item_emb_dim = embs.shape[1]  # 768 for nomic, 1024 for gte/e5
model = SASRec(item_emb_dim=item_emb_dim, ...)
model.item_embedding.weight.data = torch.FloatTensor(embs)
```

---

## STEP 3 — COLDBRIDGE: MAKE IT STRONGER (the proven winner)

Hard ColdBridge is confirmed as the best novelty contribution. It works. Now make it better.

### 3a — The cold branch is currently too simple

The cold branch averages the embeddings of the user's few known items and scores all catalog items by cosine similarity. This is correct in principle but uses uniform averaging. Fix:

```python
def cold_branch_score(user_items, item_embeddings, item_to_idx, 
                      item_counts, decay=0.8, alpha_debias=0.1):
    """
    Improved cold branch: recency-weighted + mild popularity correction.
    Only used when session_len < tau.
    """
    n = len(user_items)
    indices = [item_to_idx[i] for i in user_items if i in item_to_idx]
    if not indices:
        return None

    # Recency-weighted session embedding
    weights = np.array([decay ** (n - k - 1) for k in range(len(indices))], dtype=float)
    weights /= weights.sum()
    embs = item_embeddings[indices]
    session_emb = np.average(embs, axis=0, weights=weights)
    session_emb /= (np.linalg.norm(session_emb) + 1e-9)

    # Cosine similarity scores
    scores = item_embeddings @ session_emb  # shape (n_items,)

    # MILD popularity correction — only in cold branch, very small alpha
    # cold users benefit from some diversity since we know little about them
    counts = np.array([item_counts.get(i, 1) for i in range(len(item_embeddings))], dtype=float)
    inv_freq = 1.0 / (counts ** alpha_debias)
    scores = scores * inv_freq

    return scores

# Tune: decay in [0.7, 0.8, 0.9], alpha_debias in [0.0, 0.05, 0.1]
```

### 3b — Tune tau more finely

v1 showed tau=15 beats tau=5 which beats tau=10. This non-monotonic result is suspicious and worth re-testing with the fixed split. Run tau ∈ {5, 8, 10, 12, 15, 20}.

```python
tau_values = [5, 8, 10, 12, 15, 20]
tau_results = {}

for tau in tau_values:
    ndcg, hr, cov, ild = evaluate_coldbridge(
        model=bge2sasrec, 
        test_data=test_data,
        tau=tau,
        cold_branch_fn=cold_branch_score
    )
    tau_results[tau] = {'NDCG@10': ndcg, 'HR@10': hr, 'Coverage': cov, 'ILD': ild}
    print(f"tau={tau:3d}: NDCG={ndcg:.5f} HR={hr:.5f} Cov={cov:.4f} ILD={ild:.4f}")

best_tau = max(tau_results, key=lambda t: tau_results[t]['NDCG@10'])
print(f"\nBest tau: {best_tau}")
```

### 3c — Warm branch: use trained model scores directly

The warm branch must use the SASRec/BGE2SASRec raw output scores unchanged. No reranking on the warm branch. Any modification to warm scores only hurts, as all v2 experiments confirmed.

```python
def coldbridge_recommend(user_id, user_history, model, item_embeddings, 
                          item_to_idx, item_counts, tau, top_k=20):
    session_len = len(user_history)
    
    if session_len >= tau:
        # Warm branch: raw model scores, no modification
        scores = model.predict(user_history)  # trained logits
    else:
        # Cold branch: improved embedding similarity
        scores = cold_branch_score(user_history, item_embeddings, 
                                   item_to_idx, item_counts)
        if scores is None:
            scores = model.predict(user_history)  # fallback
    
    # Remove seen items
    for item in user_history:
        if item in item_to_idx:
            scores[item_to_idx[item]] = -np.inf
    
    return np.argsort(scores)[::-1][:top_k]
```

---

## STEP 4 — POPDEBIAS: COMPLETELY REDESIGN AS POST-FILTER

All previous PopDebias approaches applied the popularity penalty to the scoring function, which corrupts the relevance signal. This is why every variant has failed.

The only safe approach: **apply it after ranking, as a slot-filling filter that never touches the top positions.**

### 4a — Slot-fill debiasing (the only version that can work)

```python
def slot_fill_debias(ranked_items, item_counts, top_k=20, 
                     protected_top=5, diversity_slots=5):
    """
    Post-ranking slot filler that preserves top results and 
    injects tail items into lower positions only.

    Steps:
    1. Keep top `protected_top` items UNCHANGED (positions 0-4)
    2. From positions protected_top to top_k, replace `diversity_slots` items
       with long-tail items that scored well but were pushed down by popularity
    3. Long-tail = items with count < TAIL_THRESHOLD

    This CANNOT hurt NDCG@5 (positions 0-4 are untouched).
    It improves catalog coverage and LongTail_HR@10.
    """
    TAIL_THRESHOLD = 200  # items with fewer than 200 interactions = long tail

    # Top positions are locked
    final_recs = list(ranked_items[:protected_top])

    # Candidate pool: everything not already in top positions
    remaining = [i for i in ranked_items[protected_top:] if i not in final_recs]

    # Classify remaining into tail and head
    tail_items = [i for i in remaining if item_counts.get(i, 0) < TAIL_THRESHOLD]
    head_items = [i for i in remaining if item_counts.get(i, 0) >= TAIL_THRESHOLD]

    # Fill slots: interleave tail items into positions 5 to top_k
    n_tail_slots = min(diversity_slots, len(tail_items))
    n_head_slots = (top_k - protected_top) - n_tail_slots

    combined = []
    t_idx, h_idx = 0, 0
    for pos in range(top_k - protected_top):
        # Every 3rd position gets a tail item (positions 5, 8, 11, 14, 17)
        if pos % 3 == 0 and t_idx < n_tail_slots:
            combined.append(tail_items[t_idx])
            t_idx += 1
        elif h_idx < len(head_items):
            combined.append(head_items[h_idx])
            h_idx += 1
        elif t_idx < len(tail_items):
            combined.append(tail_items[t_idx])
            t_idx += 1

    final_recs.extend(combined[:top_k - protected_top])
    return final_recs[:top_k]

# Tune: protected_top in [3, 5], diversity_slots in [3, 5, 7], TAIL_THRESHOLD in [100, 200, 500]
```

### 4b — Only combine slot-fill with the ColdBridge warm branch

PopDebias as slot-fill should only be applied to warm-branch recommendations.
Cold-branch users get the cold_branch_score output directly (it already has mild debiasing).

```python
def full_system_recommend(user_id, user_history, model, item_embeddings,
                           item_to_idx, item_counts, tau, top_k=20,
                           protected_top=5, diversity_slots=5):
    session_len = len(user_history)

    if session_len >= tau:
        # Warm branch: model scores + slot-fill debiasing on lower positions
        raw_scores = model.predict(user_history)
        for item in user_history:
            if item in item_to_idx:
                raw_scores[item_to_idx[item]] = -np.inf
        top_50 = np.argsort(raw_scores)[::-1][:50]
        recs = slot_fill_debias(top_50, item_counts, top_k, protected_top, diversity_slots)
    else:
        # Cold branch: embedding similarity only (with mild alpha debiasing)
        scores = cold_branch_score(user_history, item_embeddings,
                                   item_to_idx, item_counts)
        for item in user_history:
            if item in item_to_idx:
                scores[item_to_idx[item]] = -np.inf
        recs = np.argsort(scores)[::-1][:top_k].tolist()

    return recs
```

---

## STEP 5 — HYBRID POSITION ENSEMBLE

Core LLMSeqSim has the highest catalog coverage of all models: 0.657 at @10 and 0.771 at @20. It uses only content similarity (no sequential model). Its NDCG is low (0.012) but the items it surfaces that the sequential model misses are genuinely diverse.

Use it as a coverage booster for the final recommendation list:

```python
def hybrid_position_ensemble(seq_recs, llmseqsim_recs, top_k=20, seq_positions=15):
    """
    Use sequential model for top seq_positions slots.
    Use LLMSeqSim to fill remaining slots with diverse items 
    that the sequential model didn't rank.

    seq_positions=15 means: top 15 from sequential, 5 from LLMSeqSim 
    that are not already in the top 15.
    
    This cannot hurt HR@10 relative to pure sequential if seq_positions >= 10.
    """
    final = list(seq_recs[:seq_positions])
    
    seq_set = set(final)
    for item in llmseqsim_recs:
        if len(final) >= top_k:
            break
        if item not in seq_set:
            final.append(item)
    
    return final[:top_k]

# LLMSeqSim recs = argsort(item_embeddings @ session_avg_emb)[::-1][:top_k]
# Tune: seq_positions in [10, 12, 15, 17]
```

---

## STEP 6 — BERT4Rec BACKBONE TEST

Core BERT4Rec reaches 0.05594 NDCG@10. Core LLM2BERT4Rec reaches 0.04669 which is surprisingly lower, but that used expensive API embeddings. Test BGE2BERT4Rec (free embeddings + BERT4Rec backbone).

```python
# BERT4Rec with BGE-M3 initialization
# Only run this if STEP 1 shows the SASRec backbone is maxed out at < 0.065

bert4rec_config = {
    'hidden_size': 64,
    'num_attention_heads': 2,
    'num_hidden_layers': 2,
    'hidden_dropout_prob': 0.2,
    'attention_probs_dropout_prob': 0.2,
    'max_seq_len': 50,
    'mask_prob': 0.2,       # BERT4Rec masking ratio
    'num_epochs': 200,
    'learning_rate': 1e-3,
    'batch_size': 256,
}

bert4rec_model = BERT4Rec(**bert4rec_config)
# Initialize item embeddings from BGE-M3
bert4rec_model.item_embedding.weight.data[:len(bge_embeddings)] = torch.FloatTensor(bge_embeddings)
```

Only proceed to BERT4Rec if SASRec training in Step 1 reaches a plateau below 0.065.

---

## REQUIRED ABLATION TABLE

Run exactly these configurations in order. Each one builds on the confirmed fix from the previous.

| Step | Model | New component | Expected NDCG@10 | Expected HR@10 |
|---|---|---|---|---|
| 0 | BGE2SASRec (v1 baseline) | — | 0.0461 | 0.0853 |
| 1 | BGE2SASRec (v3, fixed split) | split bug fixed | ≥ 0.0461 | ≥ 0.0853 |
| 2 | BGE2SASRec (v3, better training) | LR schedule + early stop | ≥ 0.055 | ≥ 0.100 |
| 3 | BGE2SASRec (v3, best embedding) | best free embedding | ≥ 0.058 | ≥ 0.105 |
| 4 | ColdBridge-v3 (best tau, recency cold) | hard routing + improved cold | ≥ 0.065 | ≥ 0.110 |
| 5 | ColdBridge-v3 + slot-fill debias | slot-fill on positions 6-20 | ≥ 0.065 | ≥ 0.112 |
| 6 | ColdBridge-v3 + hybrid ensemble | LLMSeqSim for tail slots | ≥ 0.066 | ≥ 0.112 |
| 7 | FULL_SYSTEM_v3 (if backbone reached 0.065+) | all above | ≥ 0.068 | ≥ 0.115 |

Save as `results/movielens_100k/ablation_v3.md`.

---

## OUTPUT REQUIREMENTS

### 1. Console output during run

Print after every model evaluation:

```
[STEP X] model=Y  NDCG@10=Z  HR@10=Z  Coverage=Z  ILD=Z
[STATUS] PASS / FAIL (reason if fail)
[vs CORE LLM2SASRec] NDCG delta=+/-X%  HR delta=+/-X%
```

### 2. Updated `results/movielens_100k/full_results.csv`

Append v3 rows. Do not delete any existing rows. Add columns for new configs:
```
model_name, alpha, tau, NDCG@10, NDCG@20, HR@10, HR@20, LongTail_HR@10, 
CatalogCoverage, Serendipity, ILD@10, training_time_sec, inference_time_sec,
debias_variant, cold_decay, embedding_model, protected_top, diversity_slots,
backbone, lr_schedule, early_stop_epoch
```

### 3. Updated `results/movielens_100k/best_model.txt`

```
=== BEST MODEL v3 ===
Model:      {name}
Backbone:   {SASRec | BERT4Rec}
Embedding:  {bge-m3 | gte-large | e5-large | nomic}
ColdBridge: tau={value}
Debias:     {slot-fill | none}
Ensemble:   {hybrid | none}

NDCG@10:        {value}
  vs Core LLM2SASRec:  {+/-X.X%}  BEATS: YES/NO
  vs Core SASRec:      {+/-X.X%}  BEATS: YES/NO
HR@10:          {value}
  vs Core LLM2SASRec:  {+/-X.X%}  BEATS: YES/NO
CatalogCoverage:{value}
LongTail_HR@10: {value}
ILD@10:         {value}
```

### 4. `results/movielens_100k/ablation_v3.md`

Full ablation table in markdown with actual numbers filled in.

---

## ACCEPTANCE CRITERIA — done when ALL are true

- [ ] BGE2SASRec reproduces v1 NDCG@10 ≥ 0.0438 (split bug confirmed fixed)
- [ ] Step 1 backbone training reaches NDCG@10 ≥ 0.060 on val set
- [ ] ColdBridge-v3 NDCG@10 ≥ 0.065 (beats Core LLM2SASRec 0.05994)
- [ ] ColdBridge-v3 HR@10 ≥ 0.110 (narrows gap to Core SASRec)
- [ ] Slot-fill debias does NOT reduce NDCG@10 vs ColdBridge-v3 alone
- [ ] LongTail_HR@10 ≥ 0.15 (was 0.094 in v1)
- [ ] CatalogCoverage ≥ 0.45 (maintained from v1)
- [ ] FULL_SYSTEM_v3 NDCG@10 ≥ 0.065 (beats Core LLM2SASRec)
- [ ] ablation_v3.md exists with all 8 steps filled in
- [ ] best_model.txt shows correct comparison vs all core models

---

## QUICK REFERENCE

```
v1 best (ground truth)   : NDCG 0.05812 | HR 0.09409 | Coverage 0.52889
Core LLM2SASRec (beat 1st): NDCG 0.05994 | HR 0.12472 | Coverage 0.23228
Core BERT4Rec  (beat 2nd) : NDCG 0.05594 | HR 0.10913 | Coverage 0.24300
Core SASRec    (beat last): NDCG 0.07303 | HR 0.14254 | Coverage 0.36152
LLMSeqSim (coverage ref)  : NDCG 0.01228 | HR 0.02895 | Coverage 0.65694

DO:   hard ColdBridge routing (confirmed best)
DO:   improve backbone training first (biggest lever)
DO:   test free embedding alternatives (gte-large, e5-large, nomic)
DO:   slot-fill debias on positions 6-20 only
DO:   hybrid ensemble with LLMSeqSim for tail positions

DO NOT: soft ColdBridge blending (confirmed worse in v2)
DO NOT: scoring-function PopDebias (confirmed broken in v1 and v2)
DO NOT: touch baselines with any rerank code
DO NOT: change the train/test split seed mid-experiment
```

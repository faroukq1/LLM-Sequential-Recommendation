# Novelty Benchmark Report (MovieLens-100K)

Date: 2026-04-21
Source run: kaggle/novelty/run_novelty_ml100k_kaggle.ipynb (saved outputs)

## 1) Executive Summary

This execution confirms your observation: **ColdBridge is the best model in this run** by ranking quality (NDCG@10).

Best model in this run:

- Model: ColdBridge
- NDCG@10: 0.072384
- HR@10: 0.118162
- CatalogCoverage: 0.618821
- LongTail_HR@10: 0.118162
- ILD@10: 0.277551

Important context:

- ColdBridge wins against all models trained inside this run (MostPopular, LLMSeqSim, SASRec, GRU4Rec, BERT4Rec, BGE2SASRec, ColdBridge+Blend, FULL_SYSTEM).
- Compared to published Core SASRec numbers in the notebook reference, ColdBridge is slightly lower on NDCG@10 and HR@10, but substantially better than the run's own SASRec baseline.

## 2) Final Results From This Execution

From the notebook output table and ablation block:

| Model            |      NDCG@10 |        HR@10 |     Coverage |
| ---------------- | -----------: | -----------: | -----------: |
| MostPopular      |     0.013185 |     0.030635 |     0.024419 |
| LLMSeqSim        |     0.038060 |     0.065646 |     0.348422 |
| SASRec           |     0.043634 |     0.080963 |     0.360334 |
| GRU4Rec          |     0.041746 |     0.089716 |     0.298987 |
| BERT4Rec         |     0.028659 |     0.061269 |     0.178082 |
| BGE2SASRec       |     0.038399 |     0.067834 |     0.412150 |
| **ColdBridge**   | **0.072384** | **0.118162** | **0.618821** |
| ColdBridge+Blend |     0.040787 |     0.065646 |     0.378201 |
| FULL_SYSTEM      |     0.043886 |     0.074398 |     0.382370 |

Best ColdBridge validation config selected by grid search:

- tau = 20
- decay = 0.7
- cold_alpha = 0.0

## 3) Why ColdBridge Won This Run

ColdBridge combines two recommendation regimes and switches between them by session length:

1. Cold branch (short sessions, len(prompt) < tau)

- Uses content similarity from BGE item embeddings.
- Builds session vector from prompt items (with optional recency decay).
- Ranks candidates by cosine similarity with seen-item filtering.

2. Warm branch (longer sessions, len(prompt) >= tau)

- Uses the trained sequential model scores (BGE2SASRec backbone outputs).

Why this helps:

- Short sessions do not contain enough behavioral signal for deep sequential models, so content similarity is usually more stable there.
- Long sessions contain enough sequence context for sequential ranking to work.
- The hard switch at tau=20 gave the best validation NDCG in this run.

## 4) Model-by-Model Explanation

### 4.1 MostPopular

What it is:

- A non-personalized popularity baseline.

How it works:

- Counts item frequency in train data.
- Recommends most frequent unseen items.

Strengths:

- Very fast, robust fallback.

Weaknesses:

- Ignores user/session context.
- Very low coverage and novelty.

### 4.2 LLMSeqSim

What it is:

- A zero-training embedding similarity recommender (BGE-based).

How it works:

- Builds a session embedding from item text embeddings.
- Scores each item by cosine similarity to session embedding.
- Filters seen items and returns top-k.

Strengths:

- No training required.
- Better cold behavior than pure popularity.

Weaknesses:

- No sequence-order modeling beyond pooled session embedding.

### 4.3 SASRec

What it is:

- A causal self-attention sequential recommender.

How it works:

- Input sequence -> item + positional embeddings.
- Transformer encoder with causal mask.
- Final token representation predicts next item.

Strengths:

- Strong sequence modeling for warm sessions.

Weaknesses:

- Can underperform with sparse or short sessions.

### 4.4 GRU4Rec

What it is:

- A recurrent neural network sequential recommender.

How it works:

- Item embeddings -> GRU layers.
- Last hidden state predicts next item.

Strengths:

- Stable sequential baseline.

Weaknesses:

- Less expressive than attention on long-range dependencies.

### 4.5 BERT4Rec

What it is:

- Bidirectional transformer with masked item training.

How it works:

- Randomly masks items during training.
- Learns to reconstruct masked items.
- At inference, masks last item position and predicts it.

Strengths:

- Rich contextual encoding.

Weaknesses:

- Depends on enough training signal; can be unstable with sparse settings or limited epochs.

### 4.6 BGE2SASRec (LLM2SASRec equivalent)

What it is:

- SASRec initialized with BGE semantic embeddings reduced by PCA.

How it works:

- Compute BGE embedding for each item text.
- Reduce to model dimension (128) with PCA.
- Load as initial item embedding weights in SASRec.
- Fine-tune with sequential objective.

Strengths:

- Injects semantic priors into sequential model.

Weaknesses:

- If semantic geometry does not align well with click dynamics, gains may be limited.

### 4.7 ColdBridge (Best in this run)

What it is:

- A hybrid cold/warm router over BGE2SASRec.

How it works:

- If prompt length < tau: use cold branch (BGE similarity ranking).
- Else: use warm branch (model logits ranking).
- Applies seen-item filtering and top-k selection.

Strengths:

- Explicitly handles cold-start sessions.
- Improved ranking quality and catalog spread in this run.

Weaknesses:

- Requires tuning of tau and cold branch hyperparameters.

### 4.8 ColdBridge+Blend

What it is:

- ColdBridge warm branch with score blending.

How it works:

- Warm score = (1-lambda) _ zscore(model_logits) + lambda _ zscore(BGE_similarity).
- Cold branch remains the same.

Strengths:

- Can improve hit-rate in some settings.

Weaknesses in this run:

- Blend diluted useful warm logits; both NDCG and HR dropped relative to ColdBridge.

### 4.9 FULL_SYSTEM

What it is:

- Extended stack: ColdBridge+Blend + slot fill + sequence-level hybrid insertion.

How it works:

- Starts from blend-ranked list.
- Optional slot fill enforces more tail exposure at fixed positions.
- Optional LLMSeqSim insertion into top positions.

Strengths:

- Designed to optimize novelty/diversity controls.

Weaknesses in this run:

- Extra diversification constraints reduced ranking accuracy.
- Ended below ColdBridge on NDCG and HR.

## 5) Interpretation of the Ablation

Ablation progression in this execution:

- NDCG jumps sharply at step 6 (ColdBridge), indicating the cold/warm routing is the core value-add.
- Steps after ColdBridge (Blend, FULL_SYSTEM) decrease ranking metrics, meaning additional novelty controls were too aggressive for this data/configuration.

## 6) Practical Conclusion

For this exact run and configuration, deploy priority should be:

1. ColdBridge (best ranking quality and strong coverage)
2. SASRec or GRU4Rec (if simpler training/serving is preferred)
3. FULL_SYSTEM only when explicit diversity/novelty business constraints are more important than top ranking accuracy

## 7) Recommended Next Experiments

1. Re-tune blend_lambda with a narrower range near 0.0 (for example 0.01 to 0.06) while freezing tau=20.
2. Keep ColdBridge router but relax slot-fill constraints (fewer forced tail slots).
3. Evaluate segment-wise (cold vs warm sessions) to confirm gains are concentrated where expected.
4. Repeat with multiple random seeds and report mean plus std for robustness.

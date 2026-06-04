# PromptCraft-SeqRec: A Memoir

*On what happens when you stop treating item descriptions as a fixed constant.*

---

## 1. Opening

The project started with a small observation that felt like it should have been obvious. I was reading through the code and paper for *Improving Sequential Recommendations with LLMs* — a solid piece of work that initializes SASRec's embedding layer with pre-computed language model vectors instead of learning embeddings from scratch. The idea is sound: a language model already knows something about what products mean, so why throw that knowledge away? The numbers agreed. LLM-initialized SASRec consistently outperformed vanilla SASRec across the datasets the authors tested.

But I kept coming back to one line in Section 5.2 of the paper:

> *"The metadata used to compute the embeddings are the names of the products."*

Just the names. No category. No brand. No description. No context about what kind of user might want this product. Just the title, passed raw into the embedding API, and whatever the model made of it was accepted as the item's semantic representation.

That seemed like a choice that had never been questioned — not because it was clearly correct, but because it had simply never been treated as a variable. The whole paper was about what you do *after* you have the embeddings. Nobody had asked whether the embeddings themselves could be made better by describing items differently.

That question became this project.

---

## 2. The Overlooked Assumption

The original paper uses BGE-M3 (in my reproduction; the paper itself used OpenAI's ada-002) to encode items. The encoding process is simple: take the item's title string, pass it through the model, get a 1024-dimensional vector, reduce with PCA to 256 dimensions, and initialize SASRec's embedding table. Repeat for every item. Train. Evaluate.

The assumption baked into this process is that the item's *name* is the best possible summary of what that item means to a language model. For a beauty product like "CeraVe Moisturizing Cream," this seems defensible — the name carries real semantic content. But what about a product called "B00123XYZ"? Or a Steam game called "Portal 2," which is just a proper noun that tells you almost nothing about what the game feels like to play?

The paper actually noticed this problem for Steam. Game titles alone produced weak embeddings, so the authors concatenated titles with user-provided tags. That ad-hoc fix improved Steam results noticeably. They documented it in a footnote and moved on.

What struck me was that the paper had already found the evidence that prompt design matters — it was sitting right there in their own ablation — but they framed it as a domain-specific patch rather than a general research question. No systematic exploration. No comparison of strategies. No principle to take forward.

The gap was clear: the field had produced sophisticated methods for what to do with item embeddings, but left the question of how to describe items to the embedding model entirely unexamined. PromptCraft-SeqRec is an attempt to fill that gap.

---

## 3. Designing the Six Prompts

The experiment was structured around six item description strategies, each representing a distinct hypothesis about what information is most useful for a language model to encode about a product.

**Type 1 — Title Only** is the paper's original approach, kept as the baseline. It encodes nothing beyond the product name: `"CeraVe Moisturizing Cream"`. Whatever the language model infers from those words alone is what SASRec gets to work with.

**Type 2 — Structured Attributes** adds factual metadata in a pipe-separated format: `"CeraVe Moisturizing Cream | Brand: CeraVe | Category: Face Moisturizer | Price: $14.99"`. The hypothesis is that structuring attributes explicitly forces the model to represent each dimension of the item cleanly, rather than leaving it to infer brand from context or guess category from product name.

**Type 3 — Rich Prose** constructs a natural language sentence using all available metadata: `"A face moisturizer by CeraVe, formulated for dry sensitive skin. Deeply hydrating with ceramides and hyaluronic acid."` The hypothesis is that fluent, descriptive language activates more of the model's training knowledge and produces richer semantic representations.

**Type 4 — User-Centric** reframes the item description from the perspective of a potential user: `"Users who like this moisturizer enjoy: face moisturizers, sensitive skin care, ceramide formulas."` The intuition is that recommendation-aligned language — describing who wants the item rather than what the item is — might produce embeddings that sit closer to how users actually think about products.

**Type 5 — Comparative** anchors the item in a relational context: `"CeraVe Moisturizing Cream is similar to: Cetaphil Moisturizing Cream, Vanicream Moisturizer. Appeals to fans of: gentle skincare, dermatologist-recommended products."` The hypothesis is that relative positioning — describing what an item resembles — gives the model richer geometric information about where in semantic space the item belongs.

**Type 6 — Hybrid** combines structured attributes with category information: `"CeraVe Moisturizing Cream | Category: Face Moisturizer | Brand: CeraVe"`. Cleaner than Type 2 (no price, no noise), designed to be the best practical balance between completeness and signal quality.

After running the initial six, I added a seventh: **Type 7 — Structured-Comparative**, which combines the attribute structure of Type 2 with the relational framing of Type 5: `"CeraVe Moisturizing Cream | Category: Face Moisturizer | Brand: CeraVe | Similar to: Cetaphil Moisturizing Cream, Vanicream Moisturizer"`. The rationale was that if Types 2 and 5 both independently improved over the baseline, combining them might compound the gains.

Each of these strategies is a hypothesis. The experiment's job was to test them.

---

## 4. Implementation

One of the constraints I imposed on myself from the start was to change as little of the existing codebase as possible. The SASRec implementation in the original repository works. The evaluation pipeline works. The data loading infrastructure works. The point of this project is not to improve the model — it is to study the input to the embedding model. Any code I add should be exactly as large as that question requires, and no larger.

In practice, this meant writing one new module — a set of seven prompt-building functions, each taking an item metadata dictionary and returning a text string. The entire logic for all seven strategies fits in under 100 lines of Python. The embedding generation script calls BGE-M3 once per strategy, saving raw 1024-dimensional vectors to disk. A separate PCA step reduces these to 256 dimensions and writes them to CSV files in the format the model expects.

The downstream training is completely untouched. Each strategy produces a CSV of item embeddings, and SASRec is initialized from that CSV. Everything else — the attention mechanism, the training loop, the loss function, the evaluation protocol — stays identical across all runs. This is the clean experimental design: one variable changes, everything else holds constant.

BGE-M3 was chosen over the original paper's ada-002 for two reasons: it runs locally on a Kaggle T4 GPU at no cost, and it performs comparably to ada-002 on semantic similarity benchmarks. Using it avoids API rate limits and makes the experiment fully reproducible without credentials.

I also included a vanilla SASRec run — no LLM embeddings, random initialization — to establish the floor: what does the model achieve when it has no semantic prior at all?

---

## 5. Experimental Setup

The dataset for this initial experiment is Amazon Beauty, a standard benchmark in sequential recommendation research. After 5-core filtering — requiring every user and every item to appear at least five times — it contains approximately 22,000 users and 12,000 items. Interactions are sorted by timestamp and split temporally: the last 20% of sessions form the test set, the first 80% are used for training.

Each strategy's embeddings were generated once and cached. SASRec was trained for 15 epochs with early stopping patience of 3, using AdamW with a learning rate of 3×10⁻⁴. The hidden dimension and embedding dimension were both set to 256, matching the PCA output size. Every strategy was trained with an identical configuration under a fixed random seed (42), so differences in outcome can only be attributed to the embedding strategy.

The primary evaluation metrics are NDCG@10 and HR@10, which measure ranking quality and hit rate within the top-10 recommendations. MRR captures how highly the correct item is ranked on average. Results are reported for @10 and @20 cutoffs.

---

## 6. What Happened

The most important result was the one that established what we are comparing against. Vanilla SASRec — trained with random embeddings, no semantic initialization — achieved an NDCG@10 of 0.0275 and an HR@10 of 0.0472. The title-only baseline, which is the original paper's method, scored 0.0351 NDCG@10 and 0.0642 HR@10. That is a 27.5% improvement in NDCG@10 and a 36% improvement in HR@10 from semantic initialization alone, before any prompt engineering. The case for using LLM embeddings at all is confirmed.

Among the seven prompt strategies, the results were more nuanced.

Type 5 — Comparative — achieved the highest NDCG@10 at 0.0360, a 2.62% improvement over the title-only baseline. Type 7 — Structured-Comparative — achieved the highest HR@10 at 0.0671, a 4.53% improvement. Type 2 — Structured Attributes — produced the largest MRR gain at 0.0297, exceeding the 5% improvement threshold for that metric specifically.

The most instructive result was Type 3 — Rich Prose. It performed worse than the title-only baseline across all three primary metrics, with a 3.04% drop in NDCG@10. This is not a failure of the hypothesis — it is the hypothesis failing, which is itself informative. Long, fluent descriptions drawn from product metadata introduce noise. The description fields in Amazon Beauty are inconsistently written, sometimes promotional, sometimes technical, sometimes empty. When they are fed wholesale into BGE-M3, the resulting embeddings reflect that noise. Structured, compact prompts outperform verbose natural language descriptions on this dataset.

The embedding quality analysis added another layer of interpretation. Isotropy scores were effectively zero across all strategies — a known artifact of BGE-M3's high-dimensional output space, which tends to cluster along a dominant principal component. The average pairwise distance between embeddings told a more interesting story: the title-only strategy produced the most spread-out embedding space (0.580), while the user-centric strategy produced the most compact (0.423). Yet the strategies with intermediate pairwise distances — comparative at 0.481 and structured at 0.487 — performed better downstream. This suggests that raw embedding spread does not directly predict recommendation quality. What matters is not how far apart the embeddings are, but whether the distances are semantically meaningful along dimensions that matter for user preferences.

The overall picture is consistent: adding structured, relational context to item descriptions improves recommendation performance. Adding unstructured prose degrades it. The best single metric improvement — 5.39% in MRR for Type 2 — crosses the threshold that was set as a success criterion before the experiments ran.

---

## 7. What I Learned

The clearest lesson from this project is that the text you hand to a language model is a design decision, not a preprocessing step. In most LLM-for-recommendation work, the item description is treated as a fixed input — something you read from the database and pass along. This project demonstrates that varying that input, systematically, produces measurable differences in downstream performance on a standard benchmark.

The second lesson is about what kind of text works. BGE-M3 was trained on a broad corpus that includes structured and semi-structured documents, not just prose. When you feed it a structured pipe-delimited string — title, brand, category — it processes it coherently. When you feed it a noisy 200-character product description scraped from an e-commerce database, it processes that too, but the noise becomes part of the embedding. The model does not know which parts of the text are signal and which are boilerplate. The prompt strategy is the mechanism by which you make that decision on its behalf.

The third lesson is about the interaction between dataset and strategy. Amazon Beauty is a domain where product names already carry substantial semantic content. "CeraVe Moisturizing Cream" is already informative to a language model that has seen CeraVe in training data, read reviews of it, and encountered the phrase "moisturizing cream" thousands of times. The baseline starts strong, which compresses the room for improvement. Steam is the opposite case: game titles like "Portal 2" or "Dota 2" are proper nouns that carry almost no intrinsic semantic signal about gameplay, genre, or user preferences. That is where prompt strategy differences are expected to be largest — where the title-only baseline is weakest.

There is also something worth saying about research design. This project adds no new model architecture, no new training procedure, no new loss function. It adds seven text-formatting functions. The fact that those seven functions produce statistically distinguishable results is both a finding about sequential recommendation and a statement about how much variation can be produced by decisions that are usually not studied at all.

---

## 8. Closing

PromptCraft-SeqRec is a small project in terms of code. It is, I think, a meaningful project in terms of what it points at.

Sequential recommendation research has invested heavily in model architecture — attention mechanisms, graph neural networks, contrastive objectives — and more recently in using language models to provide semantic priors. But between those two investments, there is a gap: the question of how to describe items to the language model in the first place. That question determines the quality of the semantic prior. The quality of the semantic prior determines how much the model can benefit from it. The whole chain depends on a design decision that has been made, until now, by default.

What this experiment shows is that the default is not optimal, that the choice is tractable, and that systematic study of it produces actionable conclusions. Structured, relational descriptions outperform prose on the Amazon Beauty benchmark. The comparative strategy — grounding an item in the context of similar items — captures something about relative position in preference space that a title alone does not. A hybrid of structure and comparison captures even more of that, at least for hit-rate metrics.

Whether these patterns hold across domains is an open question, and one worth pursuing. The expectation, grounded in the original paper's own observation about Steam, is that the gap between strategies will widen in domains where item names are semantically weak. That is the next experiment.

For now, the conclusion is this: prompt design is a research variable. It should be treated as one.

---

*Experiment conducted on Amazon Beauty dataset. Models: BGE-M3 (embedding), SASRec (recommendation). All experiments run on Kaggle T4 GPU. Seed: 42.*

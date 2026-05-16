#!/bin/bash
# PromptCraft-SeqRec: Complete local experiment pipeline
# Run from the repo root.
#
# Usage:
#   bash run_all.sh <meta_gz_path> <sessions_csv_path> [dataset_name]
#
# Example:
#   bash run_all.sh data/beauty/meta_Beauty.json.gz data/beauty/sessions.csv beauty

set -e

META_GZ=${1:?"Usage: $0 <meta_gz_path> <sessions_csv_path> [dataset]"}
SESSIONS_CSV=${2:?"Usage: $0 <meta_gz_path> <sessions_csv_path> [dataset]"}
DATASET=${3:-beauty}

echo "======================================================"
echo " PromptCraft-SeqRec Pipeline — Dataset: $DATASET"
echo " Meta GZ      : $META_GZ"
echo " Sessions CSV : $SESSIONS_CSV"
echo "======================================================"

# Step 0: Install extra dependencies
echo -e "\n[STEP 0] Installing dependencies..."
pip install -q FlagEmbedding sentence-transformers scikit-learn matplotlib

# Step 1: Generate BGE-M3 embeddings for all 6 strategies
echo -e "\n[STEP 1] Generating embeddings (all 6 strategies)..."
python generate_embeddings.py \
    --meta-gz "$META_GZ" \
    --dataset "$DATASET" \
    --batch-size 256

# Step 2: Analyze embedding quality (CPU-only, fast)
echo -e "\n[STEP 2] Analyzing embedding quality..."
python analyze_embeddings.py --dataset "$DATASET"

# Step 3: Run SASRec experiments for all 6 strategies
echo -e "\n[STEP 3] Running SASRec experiments..."
python run_promptcraft_experiments.py \
    --sessions-csv "$SESSIONS_CSV" \
    --dataset "$DATASET"

# Step 4: Generate figures
echo -e "\n[STEP 4] Generating figures..."
python visualize_results.py --dataset "$DATASET"

echo -e "\n======================================================"
echo " Pipeline complete!"
echo "  Embeddings : embeddings/"
echo "  Results    : results/"
echo "  Figures    : figures/"
echo "======================================================"

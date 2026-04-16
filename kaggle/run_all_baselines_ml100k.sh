#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/kaggle/artifacts}"
MODELS="${MODELS:-Popular,SKNN,S-SKNN,SF-SKNN,V-SKNN,GRU4Rec,BERT4Rec,SASRec}"
TOP_KS="${TOP_KS:-10,20}"
CORES="${CORES:-2}"
EVAL_CORES="${EVAL_CORES:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
SESSION_GAP_MINUTES="${SESSION_GAP_MINUTES:-30}"
MIN_SESSION_LEN="${MIN_SESSION_LEN:-2}"
TEST_FRAC="${TEST_FRAC:-0.2}"
FAIL_FAST="${FAIL_FAST:-0}"
VERBOSE="${VERBOSE:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install -q --upgrade pip
  python -m pip install -q pandas numpy scipy scikit-learn tqdm multiprocess matplotlib

  if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("tensorflow") else 1)
PY
  then
    python -m pip install -q tensorflow==2.13.0
  fi
fi

EXTRA_FLAGS=()
if [[ "${FAIL_FAST}" == "1" ]]; then
  EXTRA_FLAGS+=(--fail-fast)
fi
if [[ "${VERBOSE}" == "1" ]]; then
  EXTRA_FLAGS+=(--verbose)
fi

python kaggle/run_baselines_ml100k.py \
  --work-dir "${WORK_DIR}" \
  --models "${MODELS}" \
  --top-ks "${TOP_KS}" \
  --cores "${CORES}" \
  --eval-cores "${EVAL_CORES}" \
  --num-epochs "${NUM_EPOCHS}" \
  --session-gap-minutes "${SESSION_GAP_MINUTES}" \
  --min-session-len "${MIN_SESSION_LEN}" \
  --test-frac "${TEST_FRAC}" \
  "${EXTRA_FLAGS[@]}"

echo ""
echo "Finished baseline run."
echo "Results: ${WORK_DIR}/results/baseline_results_ml100k.csv"
echo "Failures: ${WORK_DIR}/results/baseline_failures_ml100k.csv"

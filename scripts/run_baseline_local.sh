#!/usr/bin/env bash
set -euo pipefail

PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/data/raw}"
RESULTS_DIR="${HAR_BASELINE_RESULTS_DIR:-$PACK_ROOT/outputs/results_baseline}"
CHECKPOINTS_DIR="${HAR_BASELINE_CHECKPOINTS_DIR:-$PACK_ROOT/outputs/checkpoints_baseline}"

mkdir -p "$PACK_ROOT/logs" "$PACK_ROOT/outputs" "$RESULTS_DIR" "$CHECKPOINTS_DIR"

if [[ -f "$PACK_ROOT/venv/bin/activate" ]]; then
  source "$PACK_ROOT/venv/bin/activate"
elif [[ -f "$PACK_ROOT/.venv/bin/activate" ]]; then
  source "$PACK_ROOT/.venv/bin/activate"
fi

echo "=========================================="
echo "BASELINE TRAINING (LOCAL)"
echo "=========================================="
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Checkpoints dir: $CHECKPOINTS_DIR"

python "$PACK_ROOT/src/train_baseline.py" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR"

echo "Baseline run completed successfully."
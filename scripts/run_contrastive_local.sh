#!/usr/bin/env bash
set -euo pipefail

FRACTION="${1:-1.0}"
PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/data/raw}"
CONTRASTIVE_DATA_DIR="${HAR_CONTRASTIVE_DATA_DIR:-$PACK_ROOT/contrastive_data}"
RESULTS_BASE_DIR="${HAR_RESULTS_BASE_DIR:-$PACK_ROOT/outputs}"
CHECKPOINTS_BASE_DIR="${HAR_CHECKPOINTS_BASE_DIR:-$PACK_ROOT/outputs}"

mkdir -p "$PACK_ROOT/logs" "$PACK_ROOT/outputs" "$CONTRASTIVE_DATA_DIR"

if [[ -f "$PACK_ROOT/venv/bin/activate" ]]; then
  source "$PACK_ROOT/venv/bin/activate"
elif [[ -f "$PACK_ROOT/.venv/bin/activate" ]]; then
  source "$PACK_ROOT/.venv/bin/activate"
fi

python - <<PY
f = float("$FRACTION")
if not (0.0 < f <= 1.0):
    raise SystemExit("Invalid fraction: must be in (0, 1]")
PY

echo "=========================================="
echo "CONTRASTIVE TRAINING (LOCAL)"
echo "=========================================="
echo "Fraction: $FRACTION"
echo "Raw data dir: $RAW_DATA_DIR"
echo "Contrastive data dir: $CONTRASTIVE_DATA_DIR"
echo "Results base dir: $RESULTS_BASE_DIR"
echo "Checkpoints base dir: $CHECKPOINTS_BASE_DIR"

if [[ ! -f "$CONTRASTIVE_DATA_DIR/preprocessed_data.pkl" ]]; then
  echo "Preprocessed data not found -> running preprocessing..."
  python "$PACK_ROOT/src/prepare_contrastive_dataset.py" \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$CONTRASTIVE_DATA_DIR"
fi

python "$PACK_ROOT/src/train_contrastive_model.py" \
  --loss-type triplet \
  --data-fraction "$FRACTION" \
  --strategies all \
  --data-dir "$CONTRASTIVE_DATA_DIR" \
  --results-base-dir "$RESULTS_BASE_DIR" \
  --checkpoints-base-dir "$CHECKPOINTS_BASE_DIR"

echo "Contrastive local run completed successfully (fraction=$FRACTION)."
_#!/usr/bin/env bash
#SBATCH --job-name=har_contrastive
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/contrastive_%j.out
#SBATCH --error=logs/contrastive_%j.err

set -euo pipefail

FRACTION="${1:-1.0}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PACK_ROOT="$SLURM_SUBMIT_DIR"
else
  PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$PACK_ROOT/src/train_contrastive_model.py" ]]; then
  PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

PROJECT_ROOT="$(cd "$PACK_ROOT/.." && pwd)"
cd "$PACK_ROOT"

if command -v module >/dev/null 2>&1; then
  module load python/3.10 || true
  module load gcc || true
fi

if [[ -f "$PACK_ROOT/venv/bin/activate" ]]; then
  source "$PACK_ROOT/venv/bin/activate"
elif [[ -f "$PACK_ROOT/.venv/bin/activate" ]]; then
  source "$PACK_ROOT/.venv/bin/activate"
elif [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/venv/bin/activate"
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "WARNING: No virtual environment found in pack or parent project."
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

RAW_DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/data/raw}"
CONTRASTIVE_DATA_DIR="${HAR_CONTRASTIVE_DATA_DIR:-$PACK_ROOT/contrastive_data}"
RESULTS_BASE_DIR="${HAR_RESULTS_BASE_DIR:-$PACK_ROOT/outputs}"
CHECKPOINTS_BASE_DIR="${HAR_CHECKPOINTS_BASE_DIR:-$PACK_ROOT/outputs}"
PRETRAIN_EPOCHS="${HAR_PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${HAR_FINETUNE_EPOCHS:-100}"
PRETRAIN_PATIENCE="${HAR_PRETRAIN_PATIENCE:-10}"
FINETUNE_PATIENCE="${HAR_FINETUNE_PATIENCE:-15}"

mkdir -p "$PACK_ROOT/logs" "$PACK_ROOT/outputs" "$CONTRASTIVE_DATA_DIR"

python - <<PY
f = float("$FRACTION")
if not (0.0 < f <= 1.0):
    raise SystemExit("Invalid fraction: must be in (0, 1]")
PY

echo "=========================================="
echo "CONTRASTIVE TRAINING (SLURM)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Fraction: $FRACTION"
echo "Raw data dir: $RAW_DATA_DIR"
echo "Contrastive data dir: $CONTRASTIVE_DATA_DIR"
echo "Pretrain epochs/patience: $PRETRAIN_EPOCHS/$PRETRAIN_PATIENCE"
echo "Finetune epochs/patience: $FINETUNE_EPOCHS/$FINETUNE_PATIENCE"

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
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --pretrain-patience "$PRETRAIN_PATIENCE" \
  --finetune-patience "$FINETUNE_PATIENCE" \
  --data-dir "$CONTRASTIVE_DATA_DIR" \
  --results-base-dir "$RESULTS_BASE_DIR" \
  --checkpoints-base-dir "$CHECKPOINTS_BASE_DIR"

echo "Contrastive SLURM run completed successfully (fraction=$FRACTION)."
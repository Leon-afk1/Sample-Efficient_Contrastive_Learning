#!/usr/bin/env bash
#SBATCH --job-name=har_semihard_allpct
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --account=def-s1gabour
#SBATCH --output=Log/slurm/semihard_%A_%a.out
#SBATCH --error=Log/slurm/semihard_%A_%a.err
#SBATCH --mail-user=leon.morales@utbm.fr
#SBATCH --mail-type=END,FAIL

set -euo pipefail

FRACTIONS=("1.0" "0.7" "0.6" "0.5" "0.4" "0.3")
PCT_LABELS=("100pct" "70pct" "60pct" "50pct" "40pct" "30pct")
IDX="${SLURM_ARRAY_TASK_ID:-0}"
FRACTION="${FRACTIONS[$IDX]}"
PCT_LABEL="${PCT_LABELS[$IDX]}"
METHOD="semihard"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PACK_ROOT="$SLURM_SUBMIT_DIR"
else
  PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$PACK_ROOT/src/train_contrastive_model.py" ]]; then
  PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "$PACK_ROOT"

if command -v module >/dev/null 2>&1; then
  module load python/3.10 || true
  module load gcc || true
fi

if [[ -f "$PACK_ROOT/venv/bin/activate" ]]; then
  source "$PACK_ROOT/venv/bin/activate"
elif [[ -f "$PACK_ROOT/.venv/bin/activate" ]]; then
  source "$PACK_ROOT/.venv/bin/activate"
elif [[ -f "$(cd "$PACK_ROOT/.." && pwd)/venv/bin/activate" ]]; then
  source "$(cd "$PACK_ROOT/.." && pwd)/venv/bin/activate"
elif [[ -f "$(cd "$PACK_ROOT/.." && pwd)/.venv/bin/activate" ]]; then
  source "$(cd "$PACK_ROOT/.." && pwd)/.venv/bin/activate"
else
  echo "WARNING: No virtual environment found."
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

CONTRASTIVE_DATA_DIR="${HAR_CONTRASTIVE_DATA_DIR:-$PACK_ROOT/contrastive_data}"
RESULTS_DIR="$PACK_ROOT/results/${PCT_LABEL}/${METHOD}"
CHECKPOINTS_DIR="$PACK_ROOT/checkpoints/${PCT_LABEL}/${METHOD}"
TSNE_DIR="$PACK_ROOT/images/tsne/${PCT_LABEL}/${METHOD}"
LOG_FILE="$PACK_ROOT/Log/${PCT_LABEL}/${METHOD}.log"
PRETRAIN_EPOCHS="${HAR_PRETRAIN_EPOCHS:-50}"
FINETUNE_EPOCHS="${HAR_FINETUNE_EPOCHS:-100}"
PRETRAIN_PATIENCE="${HAR_PRETRAIN_PATIENCE:-10}"
FINETUNE_PATIENCE="${HAR_FINETUNE_PATIENCE:-15}"

mkdir -p "$RESULTS_DIR/classification_report" "$CHECKPOINTS_DIR" "$TSNE_DIR" "$(dirname "$LOG_FILE")"

echo "==========================================" | tee -a "$LOG_FILE"
echo "SEMIHARD MINING — $PCT_LABEL" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}" | tee -a "$LOG_FILE"
echo "Array Task: $IDX  |  Fraction: $FRACTION" | tee -a "$LOG_FILE"
echo "Results:    $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "t-SNE:      $TSNE_DIR" | tee -a "$LOG_FILE"

if [[ ! -f "$CONTRASTIVE_DATA_DIR/preprocessed_data.pkl" ]]; then
  echo "Preprocessing data..." | tee -a "$LOG_FILE"
  python "$PACK_ROOT/src/prepare_contrastive_dataset.py" \
    --data-dir "${HAR_DATA_DIR:-$PACK_ROOT/Data Malwear/brut}" \
    --output-dir "$CONTRASTIVE_DATA_DIR" 2>&1 | tee -a "$LOG_FILE"
fi

python "$PACK_ROOT/src/train_contrastive_model.py" \
  --loss-type triplet \
  --mining-strategy semihard \
  --shift-prob 0.0 \
  --data-fraction "$FRACTION" \
  --strategies all \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --finetune-epochs "$FINETUNE_EPOCHS" \
  --pretrain-patience "$PRETRAIN_PATIENCE" \
  --finetune-patience "$FINETUNE_PATIENCE" \
  --data-dir "$CONTRASTIVE_DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --tsne-dir "$TSNE_DIR" \
  --method-name "$METHOD" 2>&1 | tee -a "$LOG_FILE"

echo "Done: $METHOD $PCT_LABEL" | tee -a "$LOG_FILE"

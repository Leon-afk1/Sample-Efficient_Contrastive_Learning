#!/usr/bin/env bash
#SBATCH --job-name=har_baseline_allpct
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=def-s1gabour
#SBATCH --output=Log/slurm/baseline_%A_%a.out
#SBATCH --error=Log/slurm/baseline_%A_%a.err
#SBATCH --mail-user=leon.morales@utbm.fr
#SBATCH --mail-type=END,FAIL

# ============================================================
# Baseline SDCNet — all fractions (100 → 30 %)
#
# Array index → fraction mapping:
#   0 → 100pct  (1.0)
#   1 →  70pct  (0.7)
#   2 →  60pct  (0.6)
#   3 →  50pct  (0.5)
#   4 →  40pct  (0.4)
#   5 →  30pct  (0.3)
#
# Runs only SDCNet (--sdcnet-only) with the corrected LOGO
# non-overlapping permutation partition (same as contrastive).
# ============================================================

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PACK_ROOT="$SLURM_SUBMIT_DIR"
else
  PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -f "$PACK_ROOT/src/train_baseline.py" ]]; then
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

# Map array index to fraction label and value
PCT_LABELS=(100pct  70pct 60pct 50pct 40pct 30pct)
PCT_VALUES=(1.0     0.7   0.6   0.5   0.4   0.3)

ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-0}"
PCT_LABEL="${PCT_LABELS[$ARRAY_IDX]}"
PCT_VALUE="${PCT_VALUES[$ARRAY_IDX]}"

DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/Data Malwear/brut}"
RESULTS_DIR="$PACK_ROOT/results/${PCT_LABEL}/baseline"
CHECKPOINTS_DIR="$PACK_ROOT/checkpoints/${PCT_LABEL}/baseline"
LOG_FILE="$PACK_ROOT/Log/${PCT_LABEL}/baseline.log"

mkdir -p "$RESULTS_DIR/classification_report" "$CHECKPOINTS_DIR" "$(dirname "$LOG_FILE")"

echo "==========================================" | tee -a "$LOG_FILE"
echo "BASELINE SDCNet — ${PCT_LABEL} (fraction=${PCT_VALUE})" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Job ID:    ${SLURM_JOB_ID:-N/A}"           | tee -a "$LOG_FILE"
echo "Array idx: ${ARRAY_IDX}"                    | tee -a "$LOG_FILE"
echo "Node:      ${SLURMD_NODENAME:-N/A}"         | tee -a "$LOG_FILE"
echo "Data dir:    $DATA_DIR"                     | tee -a "$LOG_FILE"
echo "Results dir: $RESULTS_DIR"                  | tee -a "$LOG_FILE"

python "$PACK_ROOT/src/train_baseline.py" \
  --data-dir       "$DATA_DIR" \
  --results-dir    "$RESULTS_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --data-fraction  "$PCT_VALUE" \
  --sdcnet-only \
  2>&1 | tee -a "$LOG_FILE"

echo "Baseline SDCNet ${PCT_LABEL} complete." | tee -a "$LOG_FILE"

#!/usr/bin/env bash
# NOTE: This script runs all 4 models (DNN, LSTM, Transformer, SDCNet) at 100%
# for model selection purposes. Results are saved to results/100pct/baseline_model_selection/.
# For the per-fraction SDCNet-only runs, use run_baseline_allpct_slurm.sh instead.
#SBATCH --job-name=har_baseline_model_sel
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=def-s1gabour
#SBATCH --output=Log/slurm/baseline_%j.out
#SBATCH --error=Log/slurm/baseline_%j.err
#SBATCH --mail-user=leon.morales@utbm.fr
#SBATCH --mail-type=END,FAIL

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

DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/Data Malwear/brut}"
RESULTS_DIR="$PACK_ROOT/results/100pct/baseline_model_selection"
CHECKPOINTS_DIR="$PACK_ROOT/checkpoints/100pct/baseline_model_selection"
LOG_FILE="$PACK_ROOT/Log/100pct/baseline_model_selection.log"

mkdir -p "$RESULTS_DIR/classification_report" "$CHECKPOINTS_DIR" "$(dirname "$LOG_FILE")"

echo "==========================================" | tee -a "$LOG_FILE"
echo "BASELINE TRAINING" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Job ID: ${SLURM_JOB_ID:-N/A}" | tee -a "$LOG_FILE"
echo "Node:   ${SLURMD_NODENAME:-N/A}" | tee -a "$LOG_FILE"
echo "Data dir:    $DATA_DIR" | tee -a "$LOG_FILE"
echo "Results dir: $RESULTS_DIR" | tee -a "$LOG_FILE"

python "$PACK_ROOT/src/train_baseline.py" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "Baseline training complete." | tee -a "$LOG_FILE"
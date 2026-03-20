#!/usr/bin/env bash
#SBATCH --job-name=har_baseline
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

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

DATA_DIR="${HAR_DATA_DIR:-$PACK_ROOT/data/raw}"
RESULTS_DIR="${HAR_BASELINE_RESULTS_DIR:-$PACK_ROOT/outputs/results_baseline}"
CHECKPOINTS_DIR="${HAR_BASELINE_CHECKPOINTS_DIR:-$PACK_ROOT/outputs/checkpoints_baseline}"

mkdir -p "$PACK_ROOT/logs" "$PACK_ROOT/outputs" "$RESULTS_DIR" "$CHECKPOINTS_DIR"

echo "=========================================="
echo "BASELINE TRAINING (SLURM)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data dir: $DATA_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Checkpoints dir: $CHECKPOINTS_DIR"

python "$PACK_ROOT/src/train_baseline.py" \
  --data-dir "$DATA_DIR" \
  --results-dir "$RESULTS_DIR" \
  --checkpoints-dir "$CHECKPOINTS_DIR"

echo "Baseline SLURM run completed successfully."
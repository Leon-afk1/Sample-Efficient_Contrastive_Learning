#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRACTIONS=("1.0" "0.7" "0.6" "0.5" "0.4" "0.3")

for FRACTION in "${FRACTIONS[@]}"; do
  echo ""
  echo "=========================================="
  echo "RUNNING CONTRASTIVE FRACTION=${FRACTION}"
  echo "=========================================="
  "$SCRIPT_DIR/run_contrastive_local.sh" "$FRACTION"
done

echo "All contrastive runs completed (100/70/60/50/40/30)."
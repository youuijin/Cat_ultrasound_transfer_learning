#!/usr/bin/env bash
# Frozen Human-adaptation downstream experiment commands.
# This script runs nothing unless one of the subcommands below is supplied.
# Usage (Git Bash / WSL):
#   PYTHON=/c/Users/skykkm/.conda/envs/brain/python.exe bash total_run.sh <command>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
METHODS=(imagenet full_human_mae human_mae_last2 human_mae_last4 human_mae_last6 human_mae_alpha_0p1 human_mae_pretrained_anchor_allblocks human_mae_cat_aware_preservation)

checkpoint_for() {
  case "$1" in
    imagenet) echo "" ;;
    full_human_mae) echo "runs/human_mae_recipe_ablation/baseline/last_encoder.pt" ;;
    human_mae_last2) echo "runs/human_mae_adaptation_depth/last2/seed0/mae/last_encoder.pt" ;;
    human_mae_last4) echo "runs/human_mae_recipe_ablation/partial_last4/last_encoder.pt" ;;
    human_mae_last6) echo "runs/human_mae_adaptation_depth/last6/seed0/mae/last_encoder.pt" ;;
    human_mae_alpha_0p1) echo "runs/human_mae_weight_interpolation/alpha_0p1/encoder.pt" ;;
    human_mae_pretrained_anchor_allblocks) echo "runs/human_mae_anchor_layer_ablation/anchor_all_blocks/last_encoder.pt" ;;
    human_mae_cat_aware_preservation) echo "runs/human_mae_cat_aware_anchor/lambda_cat_0p03/seed0/last_encoder.pt" ;;
  esac
}

run_binary() {
  "$PYTHON" scripts/run_human_adaptation_frozen_task.py --task classification_binary "$@"
}

run_detection() {
  "$PYTHON" scripts/run_human_adaptation_frozen_task.py --task detection "$@"
}

usage() {
  cat <<'EOF'
Commands:
  segmentation-dry-run [launcher options]
      Example: bash total_run.sh segmentation-dry-run --folds 0 1 2 3 4 --seeds 0 1 2
  segmentation-repeats [launcher options]
      Runs only missing exact frozen Cat runs; Human SSL checkpoint seed remains 0.
  aggregate-segmentation
      Rebuilds results/human_adaptation_frozen/segmentation/*.csv from frozen run directories.
  aggregate-classification
      Collects frozen binary-classification validation metrics into task-level CSV files.
  aggregate-detection
      Collects frozen detection validation metrics into task-level CSV files.
  classification-binary
      Runs only missing frozen binary-classification methods; accepts --folds, --seeds, --methods, --dry-run.
  detection
      Runs only missing frozen detection methods; accepts --folds, --seeds, --methods, --dry-run.

All segmentation repeated runs are stored as:
  runs/cat_frozen_screening/<method>/fold<FOLD>/seed<CAT_SEED>/
Each run contains best.pt, last.pt, config.json, metrics.csv, validation_metrics.json,
subject_dice.csv, parameter counts, split lists, previews, and TensorBoard files.
EOF
}

case "${1:-help}" in
  segmentation-dry-run) shift; "$PYTHON" scripts/run_cat_frozen_downstream_repeats.py --dry-run "$@" ;;
  segmentation-repeats) shift; "$PYTHON" scripts/run_cat_frozen_downstream_repeats.py "$@" ;;
  aggregate-segmentation) "$PYTHON" scripts/aggregate_human_adaptation_frozen.py ;;
  aggregate-classification) "$PYTHON" scripts/aggregate_human_adaptation_frozen_downstream.py --task classification_binary ;;
  aggregate-detection) "$PYTHON" scripts/aggregate_human_adaptation_frozen_downstream.py --task detection ;;
  classification-binary) shift; run_binary "$@" ;;
  detection) shift; run_detection "$@" ;;
  help|-h|--help) usage ;;
  *) echo "Unknown command: $1" >&2; usage; exit 2 ;;
esac

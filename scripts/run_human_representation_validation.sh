#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${HUMAN_MAE_CHECKPOINT:-checkpoints/human_mae_vit_b16_trajectory/last_encoder.pt}"
COMMON=(--dataset human2 --encoder vit_b16 --seed 0)

"$PYTHON_BIN" -m src.train_human_segmentation "${COMMON[@]}" \
  --encoder-init imagenet --transfer frozen
"$PYTHON_BIN" -m src.train_human_segmentation "${COMMON[@]}" \
  --encoder-init human_mae --encoder-checkpoint "$CHECKPOINT" --transfer frozen
"$PYTHON_BIN" -m src.train_human_segmentation "${COMMON[@]}" \
  --encoder-init imagenet --transfer full
"$PYTHON_BIN" -m src.train_human_segmentation "${COMMON[@]}" \
  --encoder-init human_mae --encoder-checkpoint "$CHECKPOINT" --transfer full

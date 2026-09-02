#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-C:/Users/skykkm/.conda/envs/brain/python.exe}"

"$PYTHON_BIN" train_human_ssl.py \
  --method dino \
  --encoder vit_b16 \
  --match-mae-config checkpoints/human_mae_vit_b16_trajectory/config.json \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  --num-workers 4 \
  --seed 0 \
  --amp \
  --save-encoder-epochs 10 25 50 75 99 \
  --output-dir checkpoints/human_dino_vit_b16_trajectory

"$PYTHON_BIN" -m src.run_human_ssl_trajectory_probe \
  --ssl-method dino \
  --checkpoint-dir checkpoints/human_dino_vit_b16_trajectory \
  --output-dir runs/human_dino_trajectory_probe

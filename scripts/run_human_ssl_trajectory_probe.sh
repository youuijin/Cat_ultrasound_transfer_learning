#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-C:/Users/skykkm/.conda/envs/brain/python.exe}"

"$PYTHON_BIN" -m src.run_human_ssl_trajectory_probe \
  --checkpoint-dir checkpoints/human_mae_vit_b16_trajectory \
  --output-dir runs/human_ssl_trajectory_probe

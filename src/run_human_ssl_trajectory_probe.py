"""Run frozen Human2 segmentation probes across the Human MAE trajectory."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


TRAJECTORY_EPOCHS = (0, 10, 25, 50, 75, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16_trajectory"))
    parser.add_argument("--ssl-method", choices=("mae", "dino"), default="mae")
    parser.add_argument("--data-root", type=Path,
                        default=Path("D:/_EUIJIN/Dataset/Human Ultrasound/OpenKidney"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/human_ssl_trajectory_probe"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def discover_checkpoints(directory: Path) -> dict[int, Path | None]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Trajectory checkpoint directory not found: {directory}")
    selected: dict[int, Path | None] = {0: None}
    for epoch in TRAJECTORY_EPOCHS[1:]:
        path = directory / f"epoch_{epoch:03d}_encoder.pt"
        fallback = False
        if not path.is_file() and epoch == 99:
            path = directory / "last_encoder.pt"
            fallback = path.is_file()
        if not path.is_file():
            raise FileNotFoundError(f"Required epoch {epoch} checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata_epoch = payload.get("epoch") if isinstance(payload, dict) else None
        if metadata_epoch is not None and int(metadata_epoch) != epoch:
            raise ValueError(f"{path} metadata epoch={metadata_epoch}, expected {epoch}")
        selected[epoch] = path.resolve()
        suffix = " (fallback: last_encoder.pt)" if fallback else ""
        print(f"[epoch {epoch:03d}] checkpoint={path.resolve()}{suffix}")
    return selected


def training_command(args, ssl_epoch: int, checkpoint: Path | None,
                     run_dir: Path) -> list[str]:
    command = [
        sys.executable, "-m", "src.train_human_segmentation",
        "--dataset", "human2", "--data-root", str(args.data_root),
        "--encoder", "vit_b16", "--transfer", "frozen",
        "--split-seed", str(args.split_seed), "--seed", str(args.seed),
        "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
        "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers), "--run-dir", str(run_dir),
        "--ssl-reference-config", str(args.checkpoint_dir / "config.json"),
        "--amp" if args.amp else "--no-amp",
    ]
    if ssl_epoch == 0:
        command.extend(("--encoder-init", "imagenet"))
    else:
        command.extend(("--encoder-init", f"human_{args.ssl_method}",
                        "--encoder-checkpoint", str(checkpoint)))
    return command


def summarize_run(ssl_epoch: int, checkpoint: Path | None, run_dir: Path,
                  seed: int) -> dict:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    if not metrics_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Completed outputs missing below {run_dir}")
    with metrics_path.open(encoding="utf-8") as handle:
        validation = [row for row in csv.DictReader(handle)
                      if row["phase"] == "validation"]
    if not validation:
        raise ValueError(f"No validation metrics in {metrics_path}")
    best = max(validation, key=lambda row: float(row["mean_dice"]))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "ssl_epoch": ssl_epoch,
        "encoder_checkpoint": "imagenet_pretrained" if checkpoint is None else str(checkpoint),
        "best_epoch": int(best["epoch"]),
        "val_dice": float(best["mean_dice"]),
        "val_iou": float(best["kidney_iou"]),
        "val_loss": float(best["loss"]),
        "train_subjects": len(config["train_subjects"]),
        "val_subjects": len(config["val_subjects"]),
        "seed": seed,
    }


def main() -> None:
    args = parse_args()
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ssl_epoch in TRAJECTORY_EPOCHS:
        run_dir = args.output_dir / f"epoch_{ssl_epoch:03d}"
        print(f"\n{'=' * 72}\nStarting frozen Human segmentation probe: epoch {ssl_epoch:03d}\n"
              f"{'=' * 72}", flush=True)
        command = training_command(args, ssl_epoch, checkpoints[ssl_epoch], run_dir)
        subprocess.run(command, check=True)
        rows.append(summarize_run(ssl_epoch, checkpoints[ssl_epoch], run_dir, args.seed))
        summary_path = args.output_dir / "trajectory_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"[epoch {ssl_epoch:03d}] summary updated: {summary_path.resolve()}")
    print(f"Trajectory probe complete: {(args.output_dir / 'trajectory_summary.csv').resolve()}")


if __name__ == "__main__":
    main()

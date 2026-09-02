"""Sequential one-factor-at-a-time Human MAE recipe screening launcher."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt

from src.human_ssl.data import DEFAULT_ROOTS, discover_ssl_samples


EXPERIMENTS = (
    ("baseline", "none", "baseline", []),
    ("mask050", "mask_ratio", "0.50", ["--mask-ratio", "0.50"]),
    ("mask060", "mask_ratio", "0.60", ["--mask-ratio", "0.60"]),
    ("normpix", "norm_pix_loss", "true", ["--norm-pixel-loss"]),
    ("encoder_lr_x0p1", "encoder_lr_scale", "0.1", ["--encoder-lr-scale", "0.1"]),
    ("partial_last4", "encoder_trainable_last_blocks", "4",
     ["--encoder-trainable-last-blocks", "4"]),
)
SUMMARY_COLUMNS = (
    "run_name", "changed_factor", "factor_value", "ssl_final_train_loss",
    "ssl_final_val_loss", "masked_mse", "masked_psnr", "masked_ssim",
    "gradient_mae", "human_frozen_dice", "human_frozen_iou",
    "human_frozen_loss", "final_checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("runs/human_mae_recipe_ablation"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/human_mae_recipe_ablation"))
    parser.add_argument("--baseline-config", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16_trajectory/config.json"))
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_audit(args) -> dict:
    source = json.loads(args.baseline_config.read_text(encoding="utf-8"))
    roots = {name: Path(source.get(f"{name}_root", DEFAULT_ROOTS[name]))
             for name in ("human1", "human2", "human3")}
    samples = discover_ssl_samples(roots)
    by_dataset = Counter(sample.dataset_id for sample in samples)
    subject_groups = {name: {sample.split_group for sample in samples
                             if sample.dataset_id == name}
                      for name in roots}
    subject_available = {name: any(sample.subject_level for sample in samples
                                   if sample.dataset_id == name) for name in roots}
    audit = {
        "source_config": str(args.baseline_config.resolve()),
        "encoder_architecture": "ViT-B/16",
        "pretrained_initialization": "ImageNet-1K supervised",
        "patch_size": int(source["patch_size"]), "image_size": int(source["image_size"]),
        "mask_ratio": float(source["mask_ratio"]),
        "reconstruction_target": "raw RGB pixel values per patch",
        "norm_pix_loss": bool(source["norm_pixel_loss"]),
        "encoder_learning_rate": float(source["lr"]),
        "decoder_learning_rate": float(source["lr"]),
        "optimizer": "AdamW", "weight_decay": float(source["weight_decay"]),
        "batch_size": int(source["batch_size"]),
        "effective_batch_size": int(source["batch_size"]),
        "gradient_accumulation_steps": 1,
        "scheduler": "linear warmup followed by cosine decay",
        "warmup_epochs": int(source["warmup_epochs"]), "epochs": int(source["epochs"]),
        "augmentation": "pad to square; bicubic resize; training horizontal flip p=0.5",
        "input_normalization": "[0,1] input; ImageNet mean/std inside VisionMAE encoder path",
        "decoder_embed_dim": int(source["decoder_dim"]),
        "decoder_depth": int(source["decoder_depth"]),
        "decoder_heads": int(source["decoder_heads"]), "seed": int(source["seed"]),
        "human_datasets": list(roots),
        "sampling_method": "concatenated image pool, shuffled uniformly by image",
        "dataset_mixing_ratio": {name: by_dataset[name] / len(samples) for name in roots},
        "split_limitations": source.get("split_limitations", {}),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "baseline_config.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.results_dir / "dataset_statistics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        columns = ("dataset", "root", "images", "subject_groups",
                   "subject_identifier_available", "sampling_fraction")
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader()
        for name, root in roots.items():
            writer.writerow({"dataset": name, "root": str(root), "images": by_dataset[name],
                             "subject_groups": len(subject_groups[name]),
                             "subject_identifier_available": subject_available[name],
                             "sampling_fraction": by_dataset[name] / len(samples)})
    print("Baseline MAE audit: " + json.dumps(audit, indent=2, ensure_ascii=False))
    return source


def run_command(command: list[str], label: str) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def mae_command(args, source: dict, name: str, run_dir: Path,
                overrides: list[str]) -> list[str]:
    final_epoch = args.epochs - 1
    evaluation_epochs = sorted({min(10, final_epoch), min(25, final_epoch), final_epoch})
    command = [args.python, "train_human_ssl.py", "--method", "mae", "--encoder", "vit_b16",
               "--human1-root", str(source["human1_root"]),
               "--human2-root", str(source["human2_root"]),
               "--human3-root", str(source["human3_root"]),
               "--val-fraction", str(source["val_fraction"]),
               "--mask-ratio", str(source["mask_ratio"]),
               "--decoder-dim", str(source["decoder_dim"]),
               "--decoder-depth", str(source["decoder_depth"]),
               "--decoder-heads", str(source["decoder_heads"]),
               "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
               "--lr", str(source["lr"]), "--weight-decay", str(source["weight_decay"]),
               "--warmup-epochs", str(min(int(source["warmup_epochs"]), args.epochs - 1)),
               "--num-workers", str(args.num_workers), "--seed", str(source["seed"]),
               "--run-name", name, "--output-dir", str(run_dir),
               "--save-encoder-epochs", *map(str, evaluation_epochs),
               "--reconstruction-eval-epochs", *map(str, evaluation_epochs),
               "--reconstruction-mask-seed", "12345",
               "--amp" if args.amp else "--no-amp"]
    command.append("--norm-pixel-loss" if source["norm_pixel_loss"] else "--no-norm-pixel-loss")
    return command + overrides


def probe_command(args, source: dict, run_dir: Path) -> list[str]:
    return [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
            "--data-root", str(source["human2_root"]), "--encoder", "vit_b16",
            "--encoder-init", "human_mae", "--encoder-checkpoint",
            str(run_dir / "last_encoder.pt"), "--transfer", "frozen",
            "--val-fraction", "0.2", "--split-seed", "42", "--seed", "0",
            "--batch-size", "8", "--epochs", "50", "--lr", "1e-4",
            "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
            "--run-dir", str(run_dir / "human_frozen_probe"),
            "--ssl-reference-config", str(run_dir / "config.json"),
            "--amp" if args.amp else "--no-amp"]


def best_validation(metrics_path: Path) -> dict:
    with metrics_path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["phase"] == "validation"]
    return max(rows, key=lambda row: float(row["mean_dice"]))


def collect_row(name: str, factor: str, value: str, run_dir: Path) -> dict:
    with (run_dir / "metrics.csv").open(encoding="utf-8") as handle:
        ssl_metrics = list(csv.DictReader(handle))[-1]
    with (run_dir / "reconstruction_metrics.csv").open(encoding="utf-8") as handle:
        reconstruction = list(csv.DictReader(handle))[-1]
    downstream = best_validation(run_dir / "human_frozen_probe" / "metrics.csv")
    return {"run_name": name, "changed_factor": factor, "factor_value": value,
            "ssl_final_train_loss": ssl_metrics["train_mae_loss"],
            "ssl_final_val_loss": ssl_metrics["validation_mae_loss"],
            "masked_mse": reconstruction["masked_mse"],
            "masked_psnr": reconstruction["masked_psnr"],
            "masked_ssim": reconstruction["masked_ssim"],
            "gradient_mae": reconstruction["gradient_mae"],
            "human_frozen_dice": downstream["mean_dice"],
            "human_frozen_iou": downstream["kidney_iou"],
            "human_frozen_loss": downstream["loss"],
            "final_checkpoint": str((run_dir / "last_encoder.pt").resolve())}


def image_net_reference(args) -> dict:
    expected = {"dataset": "human2", "encoder": "vit_b16", "encoder_init": "imagenet",
                "transfer": "frozen", "val_fraction": 0.2, "split_seed": 42, "seed": 0,
                "batch_size": 8, "epochs": 50, "lr": 1e-4, "weight_decay": 1e-4}
    run_dir = Path("runs/human_ssl_trajectory_probe/epoch_000")
    valid = False
    if (run_dir / "config.json").is_file() and (run_dir / "metrics.csv").is_file():
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        valid = all(config.get(key) == value for key, value in expected.items())
    if not valid:
        run_dir = args.runs_dir / "imagenet_frozen_reference"
        command = [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
                   "--encoder", "vit_b16", "--encoder-init", "imagenet",
                   "--transfer", "frozen", "--seed", "0", "--run-dir", str(run_dir),
                   "--num-workers", str(args.num_workers), "--amp" if args.amp else "--no-amp"]
        run_command(command, "ImageNet frozen reference (existing config did not match)")
    else:
        print(f"Reusing verified ImageNet frozen reference: {run_dir.resolve()}")
    best = best_validation(run_dir / "metrics.csv")
    return {"run_name": "imagenet_frozen_reference", "changed_factor": "initialization",
            "factor_value": "imagenet", "ssl_final_train_loss": "",
            "ssl_final_val_loss": "", "masked_mse": "", "masked_psnr": "",
            "masked_ssim": "", "gradient_mae": "",
            "human_frozen_dice": best["mean_dice"],
            "human_frozen_iou": best["kidney_iou"],
            "human_frozen_loss": best["loss"], "final_checkpoint": "imagenet_pretrained"}


def write_summary(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader(); writer.writerows(rows)


def make_plots(args, rows: list[dict]) -> None:
    experiment_rows = [row for row in rows if row["run_name"] != "imagenet_frozen_reference"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in experiment_rows:
        path = args.runs_dir / row["run_name"] / "metrics.csv"
        with path.open(encoding="utf-8") as handle: values = list(csv.DictReader(handle))
        epochs = [int(x["epoch"]) for x in values]
        line, = ax.plot(epochs, [float(x["validation_mae_loss"]) for x in values],
                        label=f"{row['run_name']} validation")
        ax.plot(epochs, [float(x["train_mae_loss"]) for x in values], linestyle="--",
                color=line.get_color(), alpha=0.65, label=f"{row['run_name']} train")
    ax.set(xlabel="SSL epoch", ylabel="MAE loss"); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(
        args.results_dir / "ssl_train_validation_loss_curves.png", dpi=200)
    plt.close(fig)
    names = [row["run_name"] for row in experiment_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, [float(row["masked_psnr"]) for row in experiment_rows])
    axes[0].set_ylabel("Masked PSNR")
    axes[1].bar(names, [float(row["masked_ssim"]) for row in experiment_rows])
    axes[1].set_ylabel("Masked SSIM")
    for ax in axes: ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(args.results_dir / "reconstruction_quality.png", dpi=200)
    plt.close(fig)
    reference = next(row for row in rows if row["run_name"] == "imagenet_frozen_reference")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, [float(row["human_frozen_dice"]) for row in experiment_rows])
    ax.axhline(float(reference["human_frozen_dice"]), color="black", linestyle="--",
               label="ImageNet frozen reference")
    ax.set_ylabel("Human frozen validation Dice"); ax.tick_params(axis="x", rotation=35)
    ax.legend(); fig.tight_layout()
    fig.savefig(args.results_dir / "human_frozen_dice.png", dpi=200); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2")
    source = write_audit(args)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [image_net_reference(args)], []
    for name, factor, value, overrides in EXPERIMENTS:
        run_dir = args.runs_dir / name
        try:
            run_command(mae_command(args, source, name, run_dir, overrides),
                        f"Human MAE OFAT: {name}")
            run_command(probe_command(args, source, run_dir),
                        f"Frozen Human2 downstream probe: {name}")
            rows.append(collect_row(name, factor, value, run_dir))
            write_summary(args.results_dir / "summary.csv", rows)
        except Exception as error:
            failures.append({"run_name": name, "error": repr(error)})
            print(f"[FAILED] {name}: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error: raise
    write_summary(args.results_dir / "summary.csv", rows)
    if failures:
        (args.results_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8")
    if len(rows) > 1: make_plots(args, rows)
    print(f"Summary: {(args.results_dir / 'summary.csv').resolve()}")
    if failures:
        raise SystemExit(f"{len(failures)} experiment(s) failed; see failures.json")


if __name__ == "__main__":
    main()

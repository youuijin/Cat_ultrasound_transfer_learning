"""Run the seed-0 update-budget constrained Human MAE feasibility experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_cat_cross_species_anchor_validation as catrun
from scripts import run_human_mae_cat_aware_anchor as diagnostics
from src.human_ssl.mae import VisionMAE
from src.human_ssl.update_budget import (
    checkpoint_encoder_state, copy_encoder_parameters, reference_update_norm,
    validate_encoder_state,
)


BETAS = (0.10, 0.25, 0.50)
BASELINE_CONFIG = Path("runs/human_mae_recipe_ablation/baseline/config.json")
FULL_CHECKPOINT = Path("runs/human_mae_recipe_ablation/baseline/last_encoder.pt")
LAST2_CHECKPOINT = Path("runs/human_mae_adaptation_depth/last2/seed0/mae/last_encoder.pt")
ALPHA_CHECKPOINT = Path("runs/human_mae_weight_interpolation/alpha_0p1/encoder.pt")
SUMMARY_COLUMNS = (
    "method", "beta", "final_relative_update_norm", "projection_fraction",
    "ssl_val_mae_loss", "human_frozen_dice", "mean_human_feature_drift",
    "mean_cat_feature_drift", "cat_val_dice", "cat_val_iou", "cat_val_loss",
    "encoder_checkpoint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("feasibility",), default="feasibility")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--beta", type=float, choices=BETAS,
                        help="Run one approved beta instead of all three.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/human_mae_update_budget"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/human_mae_update_budget"))
    parser.add_argument("--cat-data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG)
    parser.add_argument("--full-reference-checkpoint", type=Path, default=FULL_CHECKPOINT)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def execute(command: list[str], label: str) -> None:
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def beta_label(beta: float) -> str:
    return f"{beta:.2f}".replace(".", "p")


def method_name(beta: float) -> str:
    return f"update_budget_beta_{beta_label(beta)}"


def run_name(beta: float) -> str:
    return f"human_mae_update_budget_beta_{beta_label(beta)}"


def ssl_root(args, beta: float) -> Path:
    return args.runs_dir / run_name(beta)


def cat_root(args, beta: float) -> Path:
    return (ssl_root(args, beta) / "cat_segmentation" / "segmentation" / "vit_b16" /
            "full" / "fold_0" / "seed_0" / "init_human_mae")


def compute_reference(args) -> tuple[dict, float]:
    if not args.baseline_config.is_file():
        raise FileNotFoundError(f"Full MAE config not found: {args.baseline_config}")
    if not args.full_reference_checkpoint.is_file():
        raise FileNotFoundError(
            f"Full MAE reference checkpoint not found: {args.full_reference_checkpoint}")
    source = read_json(args.baseline_config)
    if int(source.get("seed", -1)) != 0 or source.get("encoder") != "vit_b16" or \
            source.get("encoder_trainable_last_blocks") is not None:
        raise RuntimeError("Reference config is not the completed full ViT-B/16 Human MAE seed0 run")
    model = VisionMAE("vit_b16", source["decoder_dim"], source["decoder_depth"],
                      source["decoder_heads"], source["norm_pixel_loss"])
    base = copy_encoder_parameters(model.encoder)
    full = checkpoint_encoder_state(args.full_reference_checkpoint)
    validate_encoder_state(model.encoder, model.encoder.state_dict(), "ImageNet encoder")
    validate_encoder_state(model.encoder, full, "Full Human MAE seed0 encoder")
    norm = reference_update_norm(model.encoder, base, full)
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Invalid full_update_norm: {norm}")
    return source, norm


def ssl_command(args, source: dict, beta: float) -> list[str]:
    command = [args.python, "train_human_ssl.py", "--method", "mae", "--encoder", "vit_b16"]
    for dataset in ("human1", "human2", "human3"):
        command.extend((f"--{dataset}-root", str(source[f"{dataset}_root"])))
    command.extend([
        "--val-fraction", str(source["val_fraction"]), "--mask-ratio", str(source["mask_ratio"]),
        "--norm-pixel-loss" if source["norm_pixel_loss"] else "--no-norm-pixel-loss",
        "--decoder-dim", str(source["decoder_dim"]), "--decoder-depth", str(source["decoder_depth"]),
        "--decoder-heads", str(source["decoder_heads"]), "--batch-size", str(source["batch_size"]),
        "--epochs", str(source["epochs"]), "--lr", str(source["lr"]),
        "--weight-decay", str(source["weight_decay"]), "--warmup-epochs", str(source["warmup_epochs"]),
        "--num-workers", str(args.num_workers), "--seed", "0", "--encoder-lr-scale", "1.0",
        "--run-name", run_name(beta), "--output-dir", str(ssl_root(args, beta)),
        "--update-budget-beta", str(beta), "--update-budget-reference-checkpoint",
        str(args.full_reference_checkpoint.resolve()), "--amp" if args.amp else "--no-amp",
    ])
    if source.get("max_images") is not None:
        command.extend(("--max-images", str(source["max_images"])))
    if source.get("save_encoder_epochs"):
        command.extend(("--save-encoder-epochs", *map(str, source["save_encoder_epochs"])))
    if source.get("reconstruction_eval_epochs"):
        command.extend(("--reconstruction-eval-epochs",
                        *map(str, source["reconstruction_eval_epochs"])))
    command.extend(("--reconstruction-mask-seed", str(source["reconstruction_mask_seed"]),
                    "--reconstruction-examples", str(source["reconstruction_examples"])))
    return command


def probe_command(args, source: dict, beta: float) -> list[str]:
    root = ssl_root(args, beta)
    return [args.python, "-m", "src.train_human_segmentation", "--dataset", "human2",
            "--data-root", str(source["human2_root"]), "--encoder", "vit_b16",
            "--encoder-init", "human_mae", "--encoder-checkpoint", str(root / "last_encoder.pt"),
            "--transfer", "frozen", "--val-fraction", "0.2", "--split-seed", "42",
            "--seed", "0", "--batch-size", "8", "--epochs", "50", "--lr", "1e-4",
            "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
            "--run-dir", str(root / "human_frozen_probe"), "--ssl-reference-config",
            str(root / "config.json"), "--amp" if args.amp else "--no-amp"]


def cat_command(args, beta: float) -> list[str]:
    root = ssl_root(args, beta)
    return [args.python, "-m", "src.segmentation.train", "--encoder", "vit_b16",
            "--encoder-init", "human_mae", "--encoder-checkpoint", str(root / "last_encoder.pt"),
            "--transfer", "full", "--data-root", str(args.cat_data_root), "--num-folds", "5",
            "--fold", "0", "--split-seed", "42", "--seed", "0", "--batch-size", "8",
            "--epochs", "50", "--lr", "1e-4", "--weight-decay", "1e-4",
            "--num-workers", str(args.num_workers), "--output-dir", str(root / "cat_segmentation"),
            "--amp" if args.amp else "--no-amp"]


def completed_ssl(root: Path, beta: float, epochs: int) -> bool:
    required = (root / "last_encoder.pt", root / "best_encoder.pt", root / "metrics.csv",
                root / "update_budget_final.json", root / "config.json")
    if not all(path.is_file() for path in required): return False
    config, final = read_json(root / "config.json"), read_json(root / "update_budget_final.json")
    rows = read_rows(root / "metrics.csv")
    return (float(config.get("update_budget_beta", -1)) == beta and
            int(config.get("seed", -1)) == 0 and int(config.get("epochs", -1)) == epochs and
            rows and int(rows[-1]["epoch"]) == epochs - 1 and
            float(final["final_relative_update_norm"]) <= beta + max(1e-8, beta * 1e-6))


def best_human(root: Path) -> tuple[float, float, float]:
    rows = [row for row in read_rows(root / "metrics.csv") if row["phase"] == "validation"]
    row = max(rows, key=lambda item: float(item["mean_dice"]))
    return float(row["mean_dice"]), float(row["kidney_iou"]), float(row["loss"])


def best_cat(root: Path) -> tuple[float, float, float]:
    row = max(read_rows(root / "metrics.csv"),
              key=lambda item: float(item["validation_mean_foreground_dice"]))
    return (float(row["validation_mean_foreground_dice"]),
            float(row["validation_mean_foreground_iou"]), float(row["validation_loss"]))


def mean_drift(path: Path) -> float:
    wanted = {3, 6, 9, 11}
    values = [float(row["drift_1_minus_cka"]) for row in read_rows(path)
              if int(row["layer"].split("_")[-1]) in wanted]
    if len(values) != 4: raise RuntimeError(f"Expected B3/B6/B9/B11 drift in {path}")
    return sum(values) / len(values)


def control_rows() -> list[dict]:
    cat = {row["method"]: row for row in read_rows(
        Path("results/constrained_human_mae_cat_transfer/all_seed_results.csv"))
           if int(row["cat_seed"]) == 0}
    human_depth = {row["adaptation_depth"]: row for row in read_rows(
        Path("results/human_mae_adaptation_depth_reproducibility/all_depth_seed_results.csv"))
                   if int(row["seed"]) == 0}
    full_human = read_rows(Path("results/human_mae_recipe_ablation/summary.csv"))[1]
    nan = float("nan")
    specs = (("imagenet", "imagenet", None, human_depth["imagenet"]["human_frozen_dice"], nan),
             ("full_human_mae", "full_human_mae", FULL_CHECKPOINT,
              full_human["human_frozen_dice"], 1.0),
             ("last2_human_mae", "last2", LAST2_CHECKPOINT,
              human_depth["last2"]["human_frozen_dice"], nan),
             ("alpha_0p1", "alpha0p1", ALPHA_CHECKPOINT, nan, 0.1))
    rows = []
    for name, cat_name, checkpoint, human_dice, relative in specs:
        source = cat[cat_name]
        rows.append({"method": name, "beta": nan,
                     "final_relative_update_norm": relative, "projection_fraction": nan,
                     "ssl_val_mae_loss": full_human["ssl_final_val_loss"] if name == "full_human_mae" else nan,
                     "human_frozen_dice": human_dice, "mean_human_feature_drift": nan,
                     "mean_cat_feature_drift": nan, "cat_val_dice": float(source["cat_val_dice"]),
                     "cat_val_iou": float(source["cat_val_iou"]),
                     "cat_val_loss": float(source["cat_val_loss"]),
                     "encoder_checkpoint": "imagenet_pretrained" if checkpoint is None else str(checkpoint.resolve())})
    return rows


def collect_budget(args, beta: float) -> dict:
    root = ssl_root(args, beta); final = read_json(root / "update_budget_final.json")
    ssl = read_rows(root / "metrics.csv")[-1]
    human_dice, _human_iou, _human_loss = best_human(root / "human_frozen_probe")
    cat_dice, cat_iou, cat_loss = best_cat(cat_root(args, beta))
    return {"method": method_name(beta), "beta": beta,
            "final_relative_update_norm": final["final_relative_update_norm"],
            "projection_fraction": final["projection_fraction"],
            "ssl_val_mae_loss": ssl["validation_mae_loss"], "human_frozen_dice": human_dice,
            "mean_human_feature_drift": mean_drift(root / "human_feature_drift.csv"),
            "mean_cat_feature_drift": mean_drift(root / "cat_train_feature_drift.csv"),
            "cat_val_dice": cat_dice, "cat_val_iou": cat_iou, "cat_val_loss": cat_loss,
            "encoder_checkpoint": str((root / "last_encoder.pt").resolve())}


def make_outputs(args, budget_rows: list[dict]) -> None:
    rows = control_rows() + budget_rows
    write_csv(args.results_dir / "feasibility_summary.csv", rows, SUMMARY_COLUMNS)
    index = {row["method"]: row for row in rows}
    deltas = []
    for row in budget_rows:
        deltas.append({"method": row["method"], "beta": row["beta"],
                       "dice_minus_imagenet": row["cat_val_dice"] - index["imagenet"]["cat_val_dice"],
                       "dice_minus_full_human_mae": row["cat_val_dice"] - index["full_human_mae"]["cat_val_dice"],
                       "dice_minus_last2": row["cat_val_dice"] - index["last2_human_mae"]["cat_val_dice"],
                       "dice_minus_alpha_0p1": row["cat_val_dice"] - index["alpha_0p1"]["cat_val_dice"]})
    write_csv(args.results_dir / "update_budget_deltas.csv", deltas)
    ordered = sorted(budget_rows, key=lambda row: row["beta"])
    plots = (
        ("beta", "cat_val_dice", "beta_vs_cat_dice.png", "Beta", "Cat validation Dice"),
        ("beta", "human_frozen_dice", "beta_vs_human_dice.png", "Beta", "Human frozen validation Dice"),
        ("final_relative_update_norm", "cat_val_dice", "relative_update_norm_vs_cat_dice.png",
         "Final relative update norm", "Cat validation Dice"),
        ("beta", "mean_cat_feature_drift", "update_budget_vs_feature_drift.png",
         "Beta", "Mean Cat feature drift"),
    )
    for xkey, ykey, filename, xlabel, ylabel in plots:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([float(row[xkey]) for row in ordered], [float(row[ykey]) for row in ordered], marker="o")
        if filename == "beta_vs_cat_dice.png":
            for method, label in (("imagenet", "ImageNet"), ("full_human_mae", "Full Human MAE"),
                                  ("last2_human_mae", "Last2"), ("alpha_0p1", "alpha=0.1")):
                ax.axhline(float(index[method]["cat_val_dice"]), linestyle="--", label=label)
            ax.legend()
        elif filename == "relative_update_norm_vs_cat_dice.png":
            for method, label in (("alpha_0p1", "alpha=0.1"), ("full_human_mae", "Full Human MAE")):
                ax.scatter(float(index[method]["final_relative_update_norm"]),
                           float(index[method]["cat_val_dice"]), label=label)
            ax.legend()
        ax.set(xlabel=xlabel, ylabel=ylabel); fig.tight_layout()
        fig.savefig(args.results_dir / filename, dpi=200); plt.close(fig)


def dry_run(args, source: dict, full_norm: float, betas: tuple[float, ...]) -> None:
    print(f"ImageNet checkpoint: torchvision ViT-B/16 ImageNet-1K supervised initialization")
    print(f"Full MAE reference checkpoint: {args.full_reference_checkpoint.resolve()}")
    print(f"computed full_update_norm: {full_norm:.12g}")
    print("beta values: " + ", ".join(f"{beta:.2f}" for beta in betas))
    for beta in betas: print(f"  beta={beta:.2f} max_update_norm={beta * full_norm:.12g}")
    print("controls to reuse: ImageNet, Full Human MAE, Last2 Human MAE, alpha=0.1")
    print("new runs that would execute:")
    for beta in betas:
        print(f"  {run_name(beta)}: Human MAE -> Human frozen probe -> Human/Cat-train drift -> Cat fold0 seed0 full transfer")
        print("    " + subprocess.list2cmdline(ssl_command(args, source, beta)))


def main() -> None:
    args = parse_args(); betas = (args.beta,) if args.beta is not None else BETAS
    source, full_norm = compute_reference(args)
    if args.dry_run:
        dry_run(args, source, full_norm, betas); return
    args.runs_dir.mkdir(parents=True, exist_ok=True); args.results_dir.mkdir(parents=True, exist_ok=True)
    for beta in betas:
        root = ssl_root(args, beta)
        if args.force or not completed_ssl(root, beta, int(source["epochs"])):
            execute(ssl_command(args, source, beta), f"Update-budget Human MAE beta={beta:.2f}")
        if not completed_ssl(root, beta, int(source["epochs"])):
            raise RuntimeError(f"Incomplete update-budget SSL run: {root}")
        probe = root / "human_frozen_probe"
        if args.force or not (probe / "metrics.csv").is_file():
            execute(probe_command(args, source, beta), f"Human frozen probe beta={beta:.2f}")
        diag_args = argparse.Namespace(cat_data_root=args.cat_data_root,
                                      num_workers=args.num_workers, force=args.force)
        diagnostics.diagnostic(diag_args, source, root / "last_encoder.pt",
                               root / "human_feature_drift.csv",
                               root / "cat_train_feature_drift.csv", 0)
        expected = catrun.expected_config(
            argparse.Namespace(data_root=args.cat_data_root, fold=0, amp=args.amp),
            method_name(beta), 0, (root / "last_encoder.pt").resolve())
        if args.force or not catrun.cat_complete(cat_root(args, beta), expected):
            execute(cat_command(args, beta), f"Cat segmentation fold0 seed0 beta={beta:.2f}")
        if not catrun.cat_complete(cat_root(args, beta), expected):
            raise RuntimeError(f"Incomplete Cat run: {cat_root(args, beta)}")
    available = [beta for beta in BETAS if completed_ssl(ssl_root(args, beta), beta,
                 int(source["epochs"])) and (ssl_root(args, beta) / "human_frozen_probe" / "metrics.csv").is_file()
                 and (ssl_root(args, beta) / "human_feature_drift.csv").is_file()
                 and (ssl_root(args, beta) / "cat_train_feature_drift.csv").is_file()
                 and (cat_root(args, beta) / "metrics.csv").is_file()]
    make_outputs(args, [collect_budget(args, beta) for beta in available])
    print(f"Outputs written to {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()

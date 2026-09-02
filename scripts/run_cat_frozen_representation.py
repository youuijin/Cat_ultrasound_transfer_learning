"""Seed-paired frozen-encoder Cat segmentation evaluation and aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = (
    "imagenet", "full_human_mae", "constrained_human_mae_last2",
    "human_mae_alpha_0p1",
)
DISPLAY = {
    "imagenet": "ImageNet", "full_human_mae": "Full Human MAE",
    "constrained_human_mae_last2": "Last2", "human_mae_alpha_0p1": "alpha=0.1",
}
RUN_FIELDS = ("method", "seed", "fold", "encoder_checkpoint", "transfer",
              "best_epoch", "val_dice", "val_iou", "val_loss",
              "reused_existing_run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("runs/cat_frozen_encoder_probe"))
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results/cat_frozen_encoder_probe"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 0 <= args.fold < 5:
        parser.error("--fold must be between 0 and 4")
    return args


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_checkpoint(seed: int) -> Path:
    root = (Path("runs/human_mae_recipe_ablation/baseline") if seed == 0 else
            Path(f"runs/human_mae_adaptation_depth/full/seed{seed}/mae"))
    return root / "last_encoder.pt"


def last2_checkpoint(seed: int) -> Path:
    return Path(f"runs/human_mae_adaptation_depth/last2/seed{seed}/mae/last_encoder.pt")


def alpha_checkpoint(seed: int) -> Path:
    if seed == 0:
        return Path("runs/human_mae_weight_interpolation/alpha_0p1/encoder.pt")
    return Path(f"runs/human_mae_alpha_0p1_reproducibility/seed{seed}/encoder.pt")


def checkpoint_for(method: str, seed: int) -> Path | None:
    path = {"full_human_mae": full_checkpoint,
            "constrained_human_mae_last2": last2_checkpoint,
            "human_mae_alpha_0p1": alpha_checkpoint}.get(method)
    return None if path is None else path(seed).resolve()


def validate_checkpoint(method: str, seed: int, path: Path | None,
                        reference_keys: dict | None) -> dict | None:
    if path is None:
        return reference_keys
    if not path.is_file():
        raise FileNotFoundError(f"Missing encoder checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no encoder-only state_dict: {path}")
    if payload.get("adaptation") != "human_kidney_ultrasound_mae":
        raise RuntimeError(f"Unexpected adaptation metadata: {path}")
    config = payload.get("config", {})
    checkpoint_seed = payload.get("human_mae_seed", config.get("seed"))
    if checkpoint_seed is not None and int(checkpoint_seed) != seed:
        raise RuntimeError(f"Checkpoint seed mismatch for {method}/seed{seed}: {checkpoint_seed}")
    if method == "constrained_human_mae_last2":
        source_config = read_json(path.parent / "config.json")
        if (source_config.get("encoder_trainable_last_blocks") != 2 or
                source_config.get("trainable_blocks") != [10, 11]):
            raise RuntimeError(f"Last2 definition mismatch: {path}")
    if method == "human_mae_alpha_0p1" and (
            not payload.get("interpolation") or
            float(payload.get("interpolation_alpha", -1)) != 0.1):
        raise RuntimeError(f"alpha=0.1 metadata mismatch: {path}")
    if reference_keys is None:
        reference_keys = state
    missing = sorted(set(reference_keys) - set(state))
    unexpected = sorted(set(state) - set(reference_keys))
    shape = sorted(k for k in set(reference_keys) & set(state)
                   if reference_keys[k].shape != state[k].shape)
    matched = len(set(reference_keys) & set(state)) - len(shape)
    print(f"checkpoint validation: matched={matched} missing={len(missing)} "
          f"unexpected={len(unexpected)} shape_mismatch={len(shape)}")
    if missing or shape:
        raise RuntimeError(f"Encoder checkpoint mismatch: missing={missing[:5]}, shape={shape[:5]}")
    if unexpected:
        raise RuntimeError(f"Encoder checkpoint has unexpected parameters: {unexpected[:5]}")
    return reference_keys


def run_dir(args: argparse.Namespace, method: str, seed: int) -> Path:
    return args.runs_dir / method / f"fold{args.fold}" / f"seed{seed}"


def expected_config(args: argparse.Namespace, method: str, seed: int,
                    checkpoint: Path | None) -> dict:
    return {"task": "segmentation", "encoder": "vit_b16",
            "encoder_init": "imagenet" if method == "imagenet" else "human_mae",
            "encoder_checkpoint": None if checkpoint is None else str(checkpoint),
            "transfer": "frozen", "data_root": str(args.data_root), "num_folds": 5,
            "fold": args.fold, "split_seed": 42, "seed": seed, "batch_size": 8,
            "epochs": 50, "lr": 1e-4, "weight_decay": 1e-4, "amp": args.amp}


def same_path(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return Path(left).resolve() == Path(right).resolve()


def complete(root: Path, expected: dict) -> bool:
    required = ("config.json", "metrics.csv", "last.pt", "best.pt",
                "validation_metrics.json", "validation_segmentation_preview.png",
                "train_subjects.txt", "val_subjects.txt", "parameter_counts.json", "subject_dice.csv",
                "trainable_parameters.json")
    if not all((root / name).is_file() for name in required):
        return False
    try:
        config, metrics = read_json(root / "config.json"), read_rows(root / "metrics.csv")
        last = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        counts = read_json(root / "parameter_counts.json")
        trainable = read_json(root / "trainable_parameters.json")
    except Exception:
        return False
    for key, value in expected.items():
        if key == "encoder_checkpoint":
            if not same_path(config.get(key), value):
                return False
        elif config.get(key) != value:
            return False
    if (len(metrics) != expected["epochs"] or int(metrics[-1]["epoch"]) != 49 or
            int(last.get("epoch", -1)) != 49):
        return False
    if (counts.get("trainable_encoder_parameters") != 0 or
            any(name.startswith("encoder.") for name in trainable)):
        return False
    summary = config.get("encoder_load_summary", {})
    if expected["encoder_checkpoint"] is not None and (
            summary.get("missing_keys") or summary.get("unexpected_keys") or
            summary.get("shape_mismatch_keys") or
            not same_path(summary.get("checkpoint"), expected["encoder_checkpoint"])):
        return False
    return True


def command(args, method, seed, checkpoint, target) -> list[str]:
    result = [args.python, "-m", "src.segmentation.train", "--encoder", "vit_b16",
              "--encoder-init", "imagenet" if method == "imagenet" else "human_mae",
              "--transfer", "frozen", "--data-root", str(args.data_root),
              "--num-folds", "5", "--fold", str(args.fold), "--split-seed", "42",
              "--seed", str(seed), "--batch-size", "8", "--epochs", "50",
              "--lr", "1e-4", "--weight-decay", "1e-4", "--num-workers",
              str(args.num_workers), "--exact-run-dir", str(target),
              "--amp" if args.amp else "--no-amp"]
    if checkpoint is not None:
        result.extend(("--encoder-checkpoint", str(checkpoint)))
    return result


def best_result(root: Path) -> dict:
    return max(read_rows(root / "metrics.csv"),
               key=lambda row: float(row["validation_mean_foreground_dice"]))


def print_subject_summary(method: str, fold: int, seed: int, root: Path) -> None:
    subject_rows = read_rows(root / "subject_dice.csv")
    counts = read_json(root / "parameter_counts.json")
    subject_dice = [float(row["mean_foreground_dice"]) for row in subject_rows]
    print("=" * 50)
    print(f"METHOD: {DISPLAY[method]}")
    print(f"FOLD: {fold}")
    print(f"SEED: {seed}")
    print("ENCODER_FROZEN: True")
    print(f"TRAINABLE_ENCODER_PARAMS: {counts['trainable_encoder_parameters']}")
    print("TEST_SUBJECTS: " + ", ".join(row["subject_id"] for row in subject_rows))
    print("SUBJECT_DICE:")
    for row, dice in zip(subject_rows, subject_dice):
        print(f"{row['subject_id']}: {dice:.6f}")
    print(f"MEAN_SUBJECT_DICE: {statistics.mean(subject_dice):.6f}")
    print("=" * 50)


def mean_std(values) -> tuple[float, float]:
    values = list(values)
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def full_ft_rows() -> list[dict]:
    sources = (Path("results/constrained_human_mae_cat_transfer/all_seed_results.csv"),
               Path("results/human_mae_alpha_0p1_reproducibility/all_seed_results.csv"))
    collected = {}
    aliases = {"last2": "constrained_human_mae_last2", "alpha0p1": "human_mae_alpha_0p1"}
    for source in sources:
        if not source.is_file():
            continue
        for row in read_rows(source):
            method = aliases.get(row["method"], row["method"])
            seed = int(row.get("cat_seed", row.get("seed", -1)))
            if method in METHODS and (method, seed) not in collected:
                collected[(method, seed)] = {"method": method, "seed": seed,
                    "dice": float(row.get("cat_val_dice", row.get("val_dice")))}
    return list(collected.values())


def aggregate(records: list[dict], results_dir: Path) -> None:
    import matplotlib.pyplot as plt
    write_rows(results_dir / "all_seed_results.csv", records, RUN_FIELDS)
    index = {(r["method"], int(r["seed"])): r for r in records}
    methods = [m for m in METHODS if any(r["method"] == m for r in records)]
    summary = []
    for method in methods:
        selected = [r for r in records if r["method"] == method]
        dm, ds = mean_std(float(r["val_dice"]) for r in selected)
        im, ios = mean_std(float(r["val_iou"]) for r in selected)
        lm, ls = mean_std(float(r["val_loss"]) for r in selected)
        summary.append({"method": method, "n_seeds": len(selected), "mean_dice": dm,
                        "std_dice": ds, "mean_iou": im, "std_iou": ios,
                        "mean_loss": lm, "std_loss": ls})
    write_rows(results_dir / "method_summary.csv", summary)
    paired, paired_summary = [], []
    if "imagenet" in methods:
        for method in methods:
            if method == "imagenet":
                continue
            values = []
            for seed in sorted({int(r["seed"]) for r in records}):
                if (method, seed) not in index or ("imagenet", seed) not in index:
                    continue
                image = float(index[("imagenet", seed)]["val_dice"])
                value = float(index[(method, seed)]["val_dice"]); delta = value - image
                values.append(delta); paired.append({"method": method, "seed": seed,
                    "imagenet_dice": image, "method_dice": value, "delta_dice": delta})
            if values:
                dm, ds = mean_std(values); paired_summary.append({"method": method,
                    "mean_delta_dice": dm, "std_delta_dice": ds,
                    "positive_seed_count": sum(x > 0 for x in values),
                    "negative_seed_count": sum(x < 0 for x in values),
                    "total_seeds": len(values)})
        if paired:
            write_rows(results_dir / "vs_imagenet.csv", paired)
            write_rows(results_dir / "vs_imagenet_summary.csv", paired_summary)
    full = {(r["method"], r["seed"]): r["dice"] for r in full_ft_rows()}
    comparison = []
    for (method, seed), frozen in index.items():
        keys = ((method, seed), ("imagenet", seed))
        if all(k in full for k in keys) and ("imagenet", seed) in index:
            comparison.append({"method": method, "seed": seed,
                "frozen_dice": frozen["val_dice"], "full_ft_dice": full[(method, seed)],
                "frozen_minus_imagenet": float(frozen["val_dice"]) - float(index[("imagenet", seed)]["val_dice"]),
                "full_ft_minus_imagenet": full[(method, seed)] - full[("imagenet", seed)]})
    if comparison:
        write_rows(results_dir / "frozen_vs_full_comparison.csv", comparison)
    labels = [DISPLAY[m] for m in methods]
    fig, ax = plt.subplots()
    for seed in sorted({int(r["seed"]) for r in records}):
        if all((m, seed) in index for m in methods):
            ax.plot(labels, [float(index[(m, seed)]["val_dice"]) for m in methods],
                    marker="o", alpha=.65)
    means = [next(x["mean_dice"] for x in summary if x["method"] == m) for m in methods]
    ax.scatter(labels, means, marker="D", s=80, color="black", label="mean", zorder=3)
    ax.set_ylabel("Cat frozen Dice"); ax.legend(); fig.tight_layout()
    fig.savefig(results_dir / "frozen_method_comparison.png", dpi=220); plt.close(fig)
    if comparison:
        fig, ax = plt.subplots()
        for row in comparison:
            label = f"{DISPLAY[row['method']]} s{row['seed']}"
            ax.plot(("Frozen", "Full fine-tuning"),
                    (float(row["frozen_dice"]), float(row["full_ft_dice"])),
                    marker="o", alpha=.55, label=label)
        ax.set_ylabel("Cat validation Dice"); fig.tight_layout()
        fig.savefig(results_dir / "frozen_vs_full_ft.png", dpi=220); plt.close(fig)
        fig, ax = plt.subplots()
        for row in comparison:
            if row["method"] == "imagenet": continue
            ax.scatter(float(row["frozen_minus_imagenet"]),
                       float(row["full_ft_minus_imagenet"]),
                       label=f"{DISPLAY[row['method']]} s{row['seed']}")
        ax.axhline(0, color="0.5", linewidth=1); ax.axvline(0, color="0.5", linewidth=1)
        ax.set(xlabel="Frozen delta vs ImageNet", ylabel="Full-FT delta vs ImageNet")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(results_dir / "frozen_delta_vs_full_delta.png", dpi=220); plt.close(fig)


def main() -> None:
    args = parse_args()
    plans, reference = [], None
    for seed in args.seeds:
        for method in args.methods:
            checkpoint = checkpoint_for(method, seed)
            if checkpoint is not None:
                print(f"\nmethod={method} fold={args.fold} seed={seed}\nencoder checkpoint={checkpoint}")
                reference = validate_checkpoint(method, seed, checkpoint, reference)
            else:
                print(f"\nmethod={method} fold={args.fold} seed={seed}\nencoder checkpoint=ImageNet pretrained ViT-B/16")
            target = run_dir(args, method, seed); expected = expected_config(args, method, seed, checkpoint)
            reuse = complete(target, expected)
            print("encoder trainable params expected = 0")
            print(f"[reuse] {target}" if reuse else f"[missing -> would run] {target}")
            plans.append((method, seed, checkpoint, target, expected, reuse))
    if args.dry_run:
        return
    records = []
    for method, seed, checkpoint, target, expected, reuse in plans:
        if not reuse:
            subprocess.run(command(args, method, seed, checkpoint, target), cwd=ROOT, check=True)
            if not complete(target, expected):
                raise RuntimeError(f"Completed process failed run-integrity check: {target}")
        best = best_result(target)
        print_subject_summary(method, args.fold, seed, target)
        records.append({"method": method, "seed": seed, "fold": args.fold,
            "encoder_checkpoint": "imagenet_pretrained" if checkpoint is None else str(checkpoint),
            "transfer": "frozen", "best_epoch": int(best["epoch"]),
            "val_dice": float(best["validation_mean_foreground_dice"]),
            "val_iou": float(best["validation_mean_foreground_iou"]),
            "val_loss": float(best["validation_loss"]), "reused_existing_run": reuse})
    subprocess.run([
        args.python, "scripts/aggregate_cat_frozen_encoder_probe.py",
        "--runs-dir", str(args.runs_dir), "--results-dir", str(args.results_dir),
        "--fold", str(args.fold), "--seeds", *[str(seed) for seed in args.seeds],
        "--methods", *args.methods,
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

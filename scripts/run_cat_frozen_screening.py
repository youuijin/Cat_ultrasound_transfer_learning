"""Run or aggregate the seed-0 Cat frozen-representation screening study.

This launcher never trains Human SSL.  It validates and reuses the four named
Human checkpoints, reuses exact completed frozen Cat runs where possible, and
runs only missing frozen Cat evaluations in ``runs/cat_frozen_screening``.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = (
    "imagenet", "full_human_mae", "human_mae_last2", "human_mae_last4",
    "human_mae_last6", "human_mae_alpha_0p1",
    "human_mae_pretrained_anchor_allblocks", "human_mae_cat_aware_preservation",
)
DISPLAY = {
    "imagenet": "ImageNet", "full_human_mae": "Full Human MAE",
    "human_mae_last2": "Human MAE Last2", "human_mae_last4": "Human MAE Last4",
    "human_mae_last6": "Human MAE Last6", "human_mae_alpha_0p1": "Human MAE alpha=0.1",
    "human_mae_pretrained_anchor_allblocks": "Human MAE + Pretrained Feature Anchor",
    "human_mae_cat_aware_preservation": "Human MAE + Cat-aware Preservation",
}
GROUP = {
    "imagenet": "baseline", "full_human_mae": "full_adaptation",
    "human_mae_last2": "constrained_depth", "human_mae_last4": "constrained_depth",
    "human_mae_last6": "constrained_depth", "human_mae_alpha_0p1": "task_vector",
    "human_mae_pretrained_anchor_allblocks": "feature_preservation",
    "human_mae_cat_aware_preservation": "cat_aware_preservation",
}
CHECKPOINTS = {
    "full_human_mae": Path("runs/human_mae_recipe_ablation/baseline/last_encoder.pt"),
    "human_mae_last2": Path("runs/human_mae_adaptation_depth/last2/seed0/mae/last_encoder.pt"),
    # The seed-0 adaptation-depth launcher records this exact source in
    # ``mae_reuse.json``; the checkpoint itself remains in its original run.
    "human_mae_last4": Path("runs/human_mae_recipe_ablation/partial_last4/last_encoder.pt"),
    "human_mae_last6": Path("runs/human_mae_adaptation_depth/last6/seed0/mae/last_encoder.pt"),
    "human_mae_alpha_0p1": Path("runs/human_mae_weight_interpolation/alpha_0p1/encoder.pt"),
    "human_mae_pretrained_anchor_allblocks": Path("runs/human_mae_anchor_layer_ablation/anchor_all_blocks/last_encoder.pt"),
    "human_mae_cat_aware_preservation": Path("runs/human_mae_cat_aware_anchor/lambda_cat_0p03/seed0/last_encoder.pt"),
}
OLD_FROZEN = {
    "imagenet": Path("runs/cat_frozen_encoder_probe/imagenet/fold0/seed0"),
    "full_human_mae": Path("runs/cat_frozen_encoder_probe/full_human_mae/fold0/seed0"),
    "human_mae_last2": Path("runs/cat_frozen_encoder_probe/constrained_human_mae_last2/fold0/seed0"),
    "human_mae_alpha_0p1": Path("runs/cat_frozen_encoder_probe/human_mae_alpha_0p1/fold0/seed0"),
}
FIELDS = ("method", "method_group", "fold", "seed", "transfer", "encoder_checkpoint",
          "source_run_path", "reused_existing_run", "best_epoch", "val_dice", "val_iou", "val_loss")


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cat_frozen_screening"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cat_frozen_screening"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.fold != 0 or args.seed != 0:
        parser.error("This screening is fixed to --fold 0 --seed 0.")
    return args


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, data: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(data)


def checkpoint(method: str) -> Path | None:
    return None if method == "imagenet" else (ROOT / CHECKPOINTS[method]).resolve()


def config_for_checkpoint(path: Path) -> dict:
    config = path.parent / "config.json"
    if not config.is_file():
        raise RuntimeError(f"Checkpoint metadata missing: {config}")
    return read_json(config)


def validate_checkpoint(method: str, path: Path | None, reference: dict | None) -> dict | None:
    if path is None:
        print("  checkpoint validation: ImageNet pretrained ViT-B/16")
        return reference
    if not path.is_file():
        raise FileNotFoundError(f"Missing Human checkpoint for {method}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is not an encoder state_dict: {path}")
    if payload.get("adaptation") != "human_kidney_ultrasound_mae":
        raise RuntimeError(f"Unexpected adaptation metadata for {method}: {path}")
    # Interpolated task-vector checkpoints are standalone artifacts: their
    # identifying configuration is embedded in the payload rather than a peer
    # config.json file.
    config = payload.get("config", {}) if method == "human_mae_alpha_0p1" else config_for_checkpoint(path)
    if method != "human_mae_alpha_0p1" and (int(config.get("seed", -1)) != 0 or int(config.get("epochs", 0)) < 50):
        raise RuntimeError(f"Incomplete or non-seed0 Human checkpoint for {method}: {path}")
    if method.startswith("human_mae_last"):
        expected = int(method.removeprefix("human_mae_last"))
        if config.get("encoder_trainable_last_blocks") != expected:
            raise RuntimeError(f"{method} configuration mismatch: {path}")
    if method == "human_mae_pretrained_anchor_allblocks":
        if config.get("feature_anchor_layers") != list(range(12)):
            raise RuntimeError(f"All-block pretrained anchor metadata mismatch: {path}")
    if method == "human_mae_alpha_0p1" and (not payload.get("interpolation") or float(payload.get("interpolation_alpha", -1)) != 0.1):
        raise RuntimeError(f"alpha=0.1 checkpoint metadata mismatch: {path}")
    if method == "human_mae_cat_aware_preservation":
        required = {"cat_anchor_lambda": 0.03, "cat_fold": 0, "cat_split_seed": 42,
                    "cat_anchor_uses_labels": False}
        if any(config.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Cat-aware checkpoint metadata mismatch: {path}")
        if config.get("feature_anchor_layers") != list(range(12)):
            raise RuntimeError(f"Cat-aware checkpoint lacks all-block pretrained feature anchor: {path}")
    if reference is None:
        reference = state
    missing = sorted(set(reference) - set(state)); unexpected = sorted(set(state) - set(reference))
    shape = sorted(key for key in set(reference) & set(state) if reference[key].shape != state[key].shape)
    print(f"  checkpoint validation: missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape)}")
    if missing or unexpected or shape:
        raise RuntimeError(f"Encoder tensor mismatch for {method}: missing={missing[:3]} unexpected={unexpected[:3]} shape={shape[:3]}")
    return reference


def expected(args: argparse.Namespace, method: str, path: Path | None) -> dict:
    return {"task": "segmentation", "encoder": "vit_b16", "encoder_init": "imagenet" if path is None else "human_mae",
            "encoder_checkpoint": None if path is None else str(path), "transfer": "frozen", "data_root": str(args.data_root),
            "num_folds": 5, "fold": 0, "split_seed": 42, "seed": 0, "batch_size": 8, "epochs": 50,
            "lr": 1e-4, "weight_decay": 1e-4, "amp": args.amp}


def same_path(left, right) -> bool:
    return left is None and right is None or (left is not None and right is not None and Path(left).resolve() == Path(right).resolve())


def complete(root: Path, required: dict) -> bool:
    names = ("config.json", "metrics.csv", "last.pt", "best.pt", "validation_metrics.json", "subject_dice.csv",
             "train_subjects.txt", "val_subjects.txt", "parameter_counts.json", "trainable_parameters.json")
    if not all((root / name).is_file() for name in names): return False
    try:
        config, metric_rows, counts = read_json(root / "config.json"), rows(root / "metrics.csv"), read_json(root / "parameter_counts.json")
        last = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    except Exception: return False
    for key, value in required.items():
        if key == "encoder_checkpoint":
            if not same_path(config.get(key), value): return False
        elif config.get(key) != value: return False
    if len(metric_rows) != 50 or int(metric_rows[-1]["epoch"]) != 49 or int(last.get("epoch", -1)) != 49: return False
    total = counts.get("total_encoder_parameters"); trainable = counts.get("trainable_encoder_parameters")
    if not total or trainable != 0 or total - trainable != total: return False
    if any(name.startswith("encoder.") for name in read_json(root / "trainable_parameters.json")): return False
    summary = config.get("encoder_load_summary", {})
    if required["encoder_checkpoint"] is not None and (summary.get("missing_keys") or summary.get("unexpected_keys") or summary.get("shape_mismatch_keys") or not same_path(summary.get("checkpoint"), required["encoder_checkpoint"])): return False
    return True


def candidate_runs(args: argparse.Namespace, method: str) -> list[Path]:
    candidates = [args.runs_dir / method / "fold0" / "seed0"]
    if method in OLD_FROZEN: candidates.append(ROOT / OLD_FROZEN[method])
    return candidates


def train_command(args: argparse.Namespace, method: str, path: Path | None, target: Path) -> list[str]:
    command = [args.python, "-m", "src.segmentation.train", "--encoder", "vit_b16", "--encoder-init",
               "imagenet" if path is None else "human_mae", "--transfer", "frozen", "--data-root", str(args.data_root),
               "--num-folds", "5", "--fold", "0", "--split-seed", "42", "--seed", "0", "--batch-size", "8",
               "--epochs", "50", "--lr", "1e-4", "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
               "--exact-run-dir", str(target), "--amp" if args.amp else "--no-amp"]
    if path is not None: command.extend(("--encoder-checkpoint", str(path)))
    return command


def metric_record(method: str, root: Path, reused: bool) -> dict:
    best = max(rows(root / "metrics.csv"), key=lambda row: float(row["validation_mean_foreground_dice"]))
    config = read_json(root / "config.json")
    return {"method": method, "method_group": GROUP[method], "fold": 0, "seed": 0, "transfer": "frozen",
            "encoder_checkpoint": "imagenet_pretrained" if method == "imagenet" else config["encoder_checkpoint"],
            "source_run_path": str(root.resolve()), "reused_existing_run": reused, "best_epoch": int(best["epoch"]),
            "val_dice": float(best["validation_mean_foreground_dice"]), "val_iou": float(best["validation_mean_foreground_iou"]),
            "val_loss": float(best["validation_loss"])}


def full_ft(method: str, checkpoint_path: Path | None) -> tuple[float, str] | tuple[str, str]:
    candidates = ROOT.glob("runs/**/config.json")
    matches = []
    for config_path in candidates:
        try: config = read_json(config_path)
        except Exception: continue
        if config.get("task") != "segmentation" or config.get("transfer") != "full" or config.get("fold") != 0 or config.get("seed") != 0: continue
        if checkpoint_path is None:
            correct = config.get("encoder_init") == "imagenet" and config.get("encoder_checkpoint") is None
        else:
            correct = same_path(config.get("encoder_checkpoint"), checkpoint_path)
        if not correct: continue
        metrics = config_path.parent / "metrics.csv"
        if metrics.is_file():
            best = max(rows(metrics), key=lambda row: float(row["validation_mean_foreground_dice"]))
            matches.append((float(best["validation_mean_foreground_dice"]), str(config_path.parent.resolve())))
    return matches[0] if matches else (float("nan"), "")


def aggregate(args: argparse.Namespace, records: list[dict]) -> None:
    if not records: return
    by_method = {row["method"]: row for row in records}; baseline = by_method.get("imagenet")
    write_rows(args.results_dir / "frozen_seed0_results.csv", records, FIELDS)
    delta = []
    for row in sorted(records, key=lambda item: item["val_dice"], reverse=True):
        image_dice = baseline["val_dice"] if baseline else float("nan")
        delta.append({"method": row["method"], "dice": row["val_dice"], "imagenet_dice": image_dice,
                      "delta_vs_imagenet": row["val_dice"] - image_dice if baseline else float("nan"),
                      "iou": row["val_iou"], "loss": row["val_loss"]})
    write_rows(args.results_dir / "frozen_seed0_vs_imagenet.csv", delta, tuple(delta[0]))
    full_rows = []
    for row in records:
        full_dice, full_path = full_ft(row["method"], checkpoint(row["method"]))
        image_delta = row["val_dice"] - baseline["val_dice"] if baseline else float("nan")
        full_image, _ = full_ft("imagenet", None)
        full_rows.append({"method": row["method"], "frozen_dice": row["val_dice"], "frozen_delta_vs_imagenet": image_delta,
                          "full_ft_dice": full_dice, "full_ft_delta_vs_imagenet": full_dice - full_image,
                          "frozen_source_path": row["source_run_path"], "full_ft_source_path": full_path})
    write_rows(args.results_dir / "frozen_vs_full_ft_seed0.csv", full_rows, tuple(full_rows[0]))
    table = [{"method": DISPLAY[row["method"]], "adaptation_type": GROUP[row["method"]],
              "uses_human_ssl": row["method"] != "imagenet", "uses_unlabeled_cat_during_ssl": row["method"] == "human_mae_cat_aware_preservation",
              "encoder_frozen_during_cat": True, "frozen_dice": row["val_dice"],
              "delta_vs_imagenet": next(item["delta_vs_imagenet"] for item in delta if item["method"] == row["method"]),
              "full_ft_dice": next(item["full_ft_dice"] for item in full_rows if item["method"] == row["method"])} for row in records]
    write_rows(args.results_dir / "frozen_screening_table.csv", table, tuple(table[0]))
    import matplotlib.pyplot as plt
    ordered = [method for method in METHODS if method in by_method]
    labels = [DISPLAY[method].replace("Human MAE ", "") for method in ordered]
    fig, ax = plt.subplots(figsize=(11, 4)); ax.bar(labels, [by_method[m]["val_dice"] for m in ordered])
    if baseline: ax.axhline(baseline["val_dice"], color="black", linewidth=1)
    ax.set_ylabel("subject-level Cat Dice"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(args.results_dir / "frozen_seed0_all_methods.png", dpi=220); plt.close(fig)
    other = [item for item in delta if item["method"] != "imagenet"]
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar([DISPLAY[x["method"]].replace("Human MAE ", "") for x in other], [x["delta_vs_imagenet"] for x in other])
    ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("delta Dice vs ImageNet"); ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(args.results_dir / "frozen_seed0_delta_vs_imagenet.png", dpi=220); plt.close(fig)
    paired = [item for item in full_rows if item["full_ft_dice"] == item["full_ft_dice"]]
    fig, ax = plt.subplots(figsize=(10, 4)); x = list(range(len(paired))); width = .38
    ax.bar([v - width / 2 for v in x], [r["frozen_dice"] for r in paired], width, label="Frozen")
    ax.bar([v + width / 2 for v in x], [r["full_ft_dice"] for r in paired], width, label="Full fine-tuning")
    ax.set_xticks(x, [DISPLAY[r["method"]].replace("Human MAE ", "") for r in paired], rotation=25); ax.set_ylabel("Cat Dice"); ax.legend(); fig.tight_layout(); fig.savefig(args.results_dir / "frozen_vs_full_ft_seed0.png", dpi=220); plt.close(fig)
    print("Method\tFrozen Dice\tDelta vs ImageNet\tFull-FT Dice\tSource checkpoint")
    for row in table:
        record = by_method[next(key for key, value in DISPLAY.items() if value == row["method"])]
        print(f"{row['method']}\t{row['frozen_dice']:.6f}\t{row['delta_vs_imagenet']:.6f}\t{row['full_ft_dice']}\t{record['encoder_checkpoint']}")


def main() -> None:
    args = args_parse(); planned = []; reference = None
    for method in METHODS:
        path = checkpoint(method); print(f"method={method}\n  encoder checkpoint={path or 'ImageNet pretrained ViT-B/16'}")
        reference = validate_checkpoint(method, path, reference)
        required = expected(args, method, path); found = next((root for root in candidate_runs(args, method) if complete(root, required)), None)
        action = "reuse" if found else "would run"
        print(f"  frozen run: {'existing' if found else 'missing'}\n  action: {action}")
        planned.append((method, path, required, found))
    existing, missing = sum(found is not None for *_, found in planned), sum(found is None for *_, found in planned)
    print(f"number existing: {existing}\nnumber missing: {missing}\nexpected new Cat frozen runs: {missing}")
    if args.dry_run: return
    records = []
    selected = set(args.methods)
    for method, path, required, found in planned:
        if found is None and method in selected:
            target = args.runs_dir / method / "fold0" / "seed0"; print(f"[run] {target}")
            subprocess.run(train_command(args, method, path, target), cwd=ROOT, check=True)
            if not complete(target, required): raise RuntimeError(f"Completed run failed integrity validation: {target}")
            found, reused = target, False
        elif found is not None:
            print(f"[reuse] {found}"); reused = True
        else:
            continue
        records.append(metric_record(method, found, reused))
    aggregate(args, records)


if __name__ == "__main__":
    main()

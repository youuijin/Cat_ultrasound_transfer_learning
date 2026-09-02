"""Collect completed frozen binary-classification or detection runs into CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("imagenet", "full_human_mae", "human_mae_last2", "human_mae_last4", "human_mae_last6",
           "human_mae_alpha_0p1", "human_mae_pretrained_anchor_allblocks", "human_mae_cat_aware_preservation")
DISPLAY = {"imagenet": "ImageNet", "full_human_mae": "Full Human MAE", "human_mae_last2": "Human MAE Last2",
           "human_mae_last4": "Human MAE Last4", "human_mae_last6": "Human MAE Last6",
           "human_mae_alpha_0p1": "Human MAE alpha=0.1",
           "human_mae_pretrained_anchor_allblocks": "Human MAE + Pretrained Feature Anchor",
           "human_mae_cat_aware_preservation": "Human MAE + Cat-aware Preservation"}
CHECKPOINTS = {
    "full_human_mae": ROOT / "runs/human_mae_recipe_ablation/baseline/last_encoder.pt",
    "human_mae_last2": ROOT / "runs/human_mae_adaptation_depth/last2/seed0/mae/last_encoder.pt",
    "human_mae_last4": ROOT / "runs/human_mae_recipe_ablation/partial_last4/last_encoder.pt",
    "human_mae_last6": ROOT / "runs/human_mae_adaptation_depth/last6/seed0/mae/last_encoder.pt",
    "human_mae_alpha_0p1": ROOT / "runs/human_mae_weight_interpolation/alpha_0p1/encoder.pt",
    "human_mae_pretrained_anchor_allblocks": ROOT / "runs/human_mae_anchor_layer_ablation/anchor_all_blocks/last_encoder.pt",
    "human_mae_cat_aware_preservation": ROOT / "runs/human_mae_cat_aware_anchor/lambda_cat_0p03/seed0/last_encoder.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification_binary", "detection"), required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def same_path(left: str | None, right: Path) -> bool:
    return left is not None and Path(left).resolve() == right.resolve()


def method_for(config: dict) -> str | None:
    if config.get("encoder") != "vit_b16" or config.get("transfer") != "frozen": return None
    if config.get("encoder_init") == "imagenet" and config.get("encoder_checkpoint") is None: return "imagenet"
    return next((method for method, checkpoint in CHECKPOINTS.items()
                 if same_path(config.get("encoder_checkpoint"), checkpoint)), None)


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else float("nan"))


def main() -> None:
    args = parse_args(); task = args.task
    run_root = ROOT / "runs/human_adaptation_frozen" / task
    output = ROOT / "results/human_adaptation_frozen" / task
    expected_task = "classification" if task == "classification_binary" else "detection"
    records, seen = [], set()
    for config_path in run_root.rglob("config.json") if run_root.is_dir() else ():
        root = config_path.parent
        try:
            config, metrics = load(config_path), load(root / "validation_metrics.json")
            trainable = load(root / "trainable_parameters.json")
        except (OSError, ValueError) as error:
            if args.verbose: print(f"[excluded] {root}: incomplete/unreadable ({error})")
            continue
        method = method_for(config)
        if method is None or config.get("task") != expected_task:
            if args.verbose: print(f"[excluded] {root}: not a canonical frozen {expected_task} run")
            continue
        if task == "classification_binary" and (config.get("classification_mode") != "binary" or config.get("condition") != "balanced_softmax"):
            if args.verbose: print(f"[excluded] {root}: binary balanced_softmax protocol mismatch")
            continue
        if any(name.startswith("encoder.") for name in trainable):
            print(f"[warning] excluded {root}: trainable encoder parameter found")
            continue
        try: key = (method, int(config["fold"]), int(config["seed"]))
        except (KeyError, ValueError):
            print(f"[warning] excluded {root}: missing fold/seed metadata"); continue
        if key in seen:
            print(f"[warning] excluded {root}: duplicate {method}/fold{key[1]}/seed{key[2]}"); continue
        seen.add(key)
        record = {"task": task, "method": method, "method_display_name": DISPLAY[method], "fold": key[1], "cat_seed": key[2],
                        "human_ssl_seed": "" if method == "imagenet" else config.get("human_ssl_seed", 0),
                        "transfer": "frozen", "encoder_checkpoint": "imagenet_pretrained" if method == "imagenet" else config["encoder_checkpoint"],
                        "run_path": str(root.resolve()), "result_file": str((root / "validation_metrics.json").resolve()),
                        "best_epoch": int(metrics["epoch"]), "loss": float(metrics["loss"])}
        if task == "classification_binary":
            record.update(balanced_accuracy=float(metrics["balanced_accuracy"]), macro_f1=float(metrics["macro_f1"]),
                          accuracy=float(metrics["accuracy"]))
        else:
            record.update(mean_iou=float(metrics["mean_iou"]), median_iou=float(metrics["median_iou"]),
                          center_error=float(metrics["center_error"]), width_error=float(metrics["width_error"]),
                          height_error=float(metrics["height_error"]))
        records.append(record)
        if args.verbose: print(f"[included] {method} fold{key[1]} seed{key[2]}: {root}")
    records.sort(key=lambda row: (METHODS.index(row["method"]), row["fold"], row["cat_seed"]))
    # Each task owns a task-specific schema: do not hide named metrics behind
    # generic "primary" or "secondary" columns.
    if task == "classification_binary":
        fields = ("task", "method", "method_display_name", "fold", "cat_seed", "human_ssl_seed", "transfer", "best_epoch",
                  "balanced_accuracy", "macro_f1", "accuracy", "loss", "encoder_checkpoint", "run_path", "result_file")
        metric_fields = ("balanced_accuracy", "macro_f1")
        summary_names = {"balanced_accuracy": ("mean_balanced_accuracy", "std_balanced_accuracy"),
                         "macro_f1": ("mean_macro_f1", "std_macro_f1"),
                         "loss": ("mean_loss", "std_loss")}
    else:
        fields = ("task", "method", "method_display_name", "fold", "cat_seed", "human_ssl_seed", "transfer", "best_epoch",
                  "mean_iou", "median_iou", "loss", "center_error", "width_error", "height_error", "encoder_checkpoint", "run_path", "result_file")
        metric_fields = ("mean_iou", "median_iou")
        summary_names = {"mean_iou": ("mean_iou", "std_iou"),
                         "median_iou": ("mean_median_iou", "std_median_iou"),
                         "loss": ("mean_loss", "std_loss")}
    write(output / "all_runs.csv", records, fields)
    summary = []
    for method in METHODS:
        selected = [row for row in records if row["method"] == method]
        values = {field: [float(row[field]) for row in selected] for field in (*metric_fields, "loss")}
        item = {"method": method, "method_display_name": DISPLAY[method], "n_runs": len(selected),
                        "folds_available": ",".join(str(x) for x in sorted({row["fold"] for row in selected})),
                        "cat_seeds_available": ",".join(str(x) for x in sorted({row["cat_seed"] for row in selected}))}
        for field, field_values in values.items():
            mean_name, std_name = summary_names[field]
            item[mean_name] = mean(field_values); item[std_name] = std(field_values)
        summary.append(item)
    summary_fields = ("method", "method_display_name", "n_runs", "folds_available", "cat_seeds_available",
                      *[name for field in (*metric_fields, "loss") for name in summary_names[field]])
    write(output / "summary_by_method.csv", summary, summary_fields)
    print(f"Collected {len(records)} completed frozen {task} runs.")
    for row in summary: print(f"{row['method']}: n={row['n_runs']} folds={row['folds_available'] or '-'} seeds={row['cat_seeds_available'] or '-'}")


if __name__ == "__main__": main()

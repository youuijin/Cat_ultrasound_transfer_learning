"""Aggregate completed Cat frozen-representation Human-adaptation runs only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "runs/cat_frozen_encoder_probe", ROOT / "runs/cat_frozen_screening")
# Task-specific output root.  Older top-level CSVs are retained as legacy
# artifacts and are never moved or deleted by this aggregator.
OUTPUT = ROOT / "results/human_adaptation_frozen/segmentation"
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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def same_path(value: str | None, expected: Path) -> bool:
    try:
        return value is not None and Path(value).resolve() == expected.resolve()
    except OSError:
        return False


def method_for(config: dict) -> str | None:
    if config.get("encoder") != "vit_b16" or config.get("transfer") != "frozen":
        return None
    if config.get("encoder_init") == "imagenet" and config.get("encoder_checkpoint") is None:
        return "imagenet"
    # The original probe has seed-specific reconstruction and interpolation
    # checkpoints.  Match their saved initialization metadata and canonical
    # path layout, rather than requiring the seed-0 checkpoint path.
    checkpoint = config.get("encoder_checkpoint")
    summary = config.get("encoder_load_summary", {})
    checkpoint_parts = Path(checkpoint).parts if checkpoint else ()
    if "human_mae_adaptation_depth" in checkpoint_parts:
        if "full" in checkpoint_parts and summary.get("adaptation") == "human_kidney_ultrasound_mae":
            return "full_human_mae"
        if "last2" in checkpoint_parts and summary.get("adaptation") == "human_kidney_ultrasound_mae":
            return "human_mae_last2"
    if ("human_mae_alpha_0p1_reproducibility" in checkpoint_parts and
            summary.get("interpolation") and float(summary.get("interpolation_alpha", -1)) == 0.1):
        return "human_mae_alpha_0p1"
    return next((method for method, path in CHECKPOINTS.items()
                 if same_path(checkpoint, path)), None)


def validate(root: Path, config: dict) -> tuple[bool, str]:
    required = ("metrics.csv", "validation_metrics.json", "subject_dice.csv", "parameter_counts.json",
                "trainable_parameters.json", "train_subjects.txt", "val_subjects.txt")
    if not all((root / name).is_file() for name in required):
        return False, "incomplete frozen run"
    try:
        metrics = read_rows(root / "metrics.csv")
        counts = read_json(root / "parameter_counts.json")
        trainable = read_json(root / "trainable_parameters.json")
    except (OSError, ValueError, KeyError) as error:
        return False, f"unreadable run metadata: {error}"
    if not metrics:
        return False, "empty metrics.csv"
    if counts.get("trainable_encoder_parameters") != 0:
        return False, "encoder was trainable"
    if any(name.startswith("encoder.") for name in trainable):
        return False, "encoder parameter listed as trainable"
    if config.get("task") != "segmentation" or config.get("transfer") != "frozen":
        return False, "not frozen segmentation"
    return True, "included"


def numeric(value: str | float | int) -> float:
    return float(value)


def metadata(method: str, config: dict) -> dict:
    defaults = {"human_adaptation_type": "none", "human_adaptation_depth": "none", "alpha": "",
                "feature_anchor": False, "feature_anchor_layers": "", "lambda_feature": "",
                "cat_aware": False, "lambda_cat": ""}
    if method == "full_human_mae": defaults.update(human_adaptation_type="mae", human_adaptation_depth="full")
    if method.startswith("human_mae_last"):
        defaults.update(human_adaptation_type="mae", human_adaptation_depth=method.removeprefix("human_mae_"))
    if method == "human_mae_alpha_0p1": defaults.update(human_adaptation_type="mae_task_vector", alpha=0.1)
    if method in ("human_mae_pretrained_anchor_allblocks", "human_mae_cat_aware_preservation"):
        defaults.update(human_adaptation_type="mae", human_adaptation_depth="full", feature_anchor=True,
                        feature_anchor_layers="0,1,2,3,4,5,6,7,8,9,10,11", lambda_feature=0.01)
    if method == "human_mae_cat_aware_preservation": defaults.update(cat_aware=True, lambda_cat=0.03)
    return defaults


def discover(verbose: bool) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for source in SOURCE_ROOTS:
        if not source.is_dir():
            if verbose: print(f"[excluded] {source}: source root missing")
            continue
        for config_path in source.rglob("config.json"):
            root = config_path.parent
            try: config = read_json(config_path)
            except (OSError, ValueError) as error:
                if verbose: print(f"[excluded] {root}: invalid config ({error})")
                continue
            method = method_for(config)
            if method is None:
                if verbose: print(f"[excluded] {root}: not a canonical frozen Human-adaptation method")
                continue
            ok, why = validate(root, config)
            if not ok:
                print(f"[warning] excluded {root}: {why}")
                continue
            try:
                fold, seed = int(config["fold"]), int(config["seed"])
                best = max(read_rows(root / "metrics.csv"), key=lambda row: numeric(row["validation_mean_foreground_dice"]))
            except (KeyError, ValueError) as error:
                print(f"[warning] excluded {root}: invalid stored metrics ({error})")
                continue
            key = method, fold, seed
            if key in seen:
                print(f"[warning] excluded {root}: duplicate canonical {method}/fold{fold}/seed{seed}")
                continue
            seen.add(key)
            info = metadata(method, config)
            found.append({"method": method, "method_display_name": DISPLAY[method], "method_group": GROUP[method],
                          "fold": fold, "seed": seed,
                          "human_ssl_seed": "" if method == "imagenet" else config.get("human_ssl_seed", seed),
                          "encoder": config["encoder"], "encoder_init": config["encoder_init"], "transfer": config["transfer"],
                          **info, "encoder_checkpoint": "imagenet_pretrained" if method == "imagenet" else config["encoder_checkpoint"],
                          "run_path": str(root.resolve()), "result_file": str((root / "validation_metrics.json").resolve()),
                          "best_epoch": int(best["epoch"]), "subject_dice": numeric(best["validation_mean_foreground_dice"]),
                          "iou": numeric(best["validation_mean_foreground_iou"]), "loss": numeric(best["validation_loss"]),
                          "delta_vs_imagenet": float("nan")})
            if verbose: print(f"[included] {method} fold{fold} seed{seed}: {root}")
    paired = {(row["fold"], row["seed"]): row["subject_dice"] for row in found if row["method"] == "imagenet"}
    for row in found:
        if row["method"] != "imagenet" and (row["fold"], row["seed"]) in paired:
            row["delta_vs_imagenet"] = row["subject_dice"] - paired[row["fold"], row["seed"]]
    return sorted(found, key=lambda row: (METHODS.index(row["method"]), row["fold"], row["seed"]))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else float("nan"))


def aggregate(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summary, ablation = [], []
    for method in METHODS:
        selected = [row for row in records if row["method"] == method]
        dice, iou, loss = ([row[field] for row in selected] for field in ("subject_dice", "iou", "loss"))
        paired = [row["delta_vs_imagenet"] for row in selected if not math.isnan(row["delta_vs_imagenet"])]
        summary.append({"method": method, "method_display_name": DISPLAY[method], "method_group": GROUP[method],
                        "n_runs": len(selected), "folds_available": ",".join(map(str, sorted({row["fold"] for row in selected}))),
                        "seeds_available": ",".join(map(str, sorted({row["seed"] for row in selected}))),
                        "mean_dice": mean(dice), "std_dice": std(dice), "median_dice": statistics.median(dice) if dice else float("nan"),
                        "min_dice": min(dice) if dice else float("nan"), "max_dice": max(dice) if dice else float("nan"),
                        "mean_iou": mean(iou), "std_iou": std(iou), "mean_loss": mean(loss), "std_loss": std(loss),
                        "paired_n": len(paired), "mean_delta_vs_imagenet": mean(paired), "std_delta_vs_imagenet": std(paired),
                        "positive_pair_count": sum(value > 0 for value in paired), "negative_pair_count": sum(value < 0 for value in paired),
                        "zero_pair_count": sum(value == 0 for value in paired)})
        ablation.append({"method": DISPLAY[method], "adaptation_category": GROUP[method], "n_runs": len(selected),
                         "dice_mean": mean(dice), "dice_std": std(dice), "delta_vs_imagenet_mean": mean(paired),
                         "positive_pairs": sum(value > 0 for value in paired), "total_pairs": len(paired)})
    coverage = sorted({(row["fold"], row["seed"]) for row in records})
    horizontal = []
    aliases = {"full_human_mae": "full_human_mae", "human_mae_last2": "last2", "human_mae_last4": "last4", "human_mae_last6": "last6", "human_mae_alpha_0p1": "alpha_0p1", "human_mae_pretrained_anchor_allblocks": "pretrained_anchor", "human_mae_cat_aware_preservation": "cat_aware"}
    index = {(row["method"], row["fold"], row["seed"]): row["subject_dice"] for row in records}
    for fold, seed in coverage:
        row = {"fold": fold, "seed": seed, "imagenet_dice": index.get(("imagenet", fold, seed), float("nan"))}
        for method, alias in aliases.items():
            value = index.get((method, fold, seed), float("nan")); row[f"{alias}_dice"] = value
            row[f"{alias}_minus_imagenet"] = value - row["imagenet_dice"] if not math.isnan(value) and not math.isnan(row["imagenet_dice"]) else float("nan")
        horizontal.append(row)
    return summary, horizontal, ablation


def main() -> None:
    args = parse_args(); records = discover(args.verbose)
    summary, by_seed, ablation = aggregate(records)
    # Put the comparison values before provenance paths, so this raw table is
    # readable without horizontal scrolling.  Exact run provenance remains
    # available at the end of every row.
    all_fields = ("method", "method_display_name", "method_group", "fold", "seed", "human_ssl_seed", "transfer", "best_epoch", "subject_dice", "iou", "loss", "delta_vs_imagenet", "encoder", "encoder_init", "human_adaptation_type", "human_adaptation_depth", "alpha", "feature_anchor", "feature_anchor_layers", "lambda_feature", "cat_aware", "lambda_cat", "encoder_checkpoint", "run_path", "result_file")
    summary_fields = ("method", "method_display_name", "method_group", "n_runs", "folds_available", "seeds_available", "mean_dice", "std_dice", "median_dice", "min_dice", "max_dice", "mean_iou", "std_iou", "mean_loss", "std_loss", "paired_n", "mean_delta_vs_imagenet", "std_delta_vs_imagenet", "positive_pair_count", "negative_pair_count", "zero_pair_count")
    seed_fields = ("fold", "seed", "imagenet_dice", "full_human_mae_dice", "last2_dice", "last4_dice", "last6_dice", "alpha_0p1_dice", "pretrained_anchor_dice", "cat_aware_dice", "full_human_mae_minus_imagenet", "last2_minus_imagenet", "last4_minus_imagenet", "last6_minus_imagenet", "alpha_0p1_minus_imagenet", "pretrained_anchor_minus_imagenet", "cat_aware_minus_imagenet")
    write_rows(OUTPUT / "all_runs.csv", records, all_fields)
    write_rows(OUTPUT / "summary_by_method.csv", summary, summary_fields)
    write_rows(OUTPUT / "by_seed.csv", by_seed, seed_fields)
    write_rows(OUTPUT / "ablation_table.csv", ablation, tuple(ablation[0]))
    inventory = [{key: row[key] for key in ("method", "fold", "seed", "human_ssl_seed", "encoder_checkpoint")} | {"frozen_run_path": row["run_path"], "result_file": row["result_file"]} for row in records]
    write_rows(OUTPUT / "checkpoint_inventory.csv", inventory, ("method", "fold", "seed", "human_ssl_seed", "encoder_checkpoint", "frozen_run_path", "result_file"))
    print("All discovered frozen runs:")
    for row in records: print(f"  {row['method']} fold{row['fold']} seed{row['seed']}: {row['run_path']}")
    print("Methods found: " + ", ".join(row["method"] for row in summary if row["n_runs"]))
    print("Fold/seed coverage:")
    for row in summary: print(f"  {row['method']}: fold(s) {row['folds_available'] or '-'}; seed(s) {row['seeds_available'] or '-'}")
    expected = {(method, 0, seed) for method in METHODS for seed in (0, 1, 2)}
    observed = {(row["method"], row["fold"], row["seed"]) for row in records}
    print("Missing expected combinations: " + ", ".join(f"{m}/fold{f}/seed{s}" for m, f, s in sorted(expected - observed)) if expected - observed else "Missing expected combinations: none")
    print("Method\tn\tDice mean ± std\tDelta vs ImageNet\tPositive paired runs")
    for row in summary:
        print(f"{row['method']}\t{row['n_runs']}\t{row['mean_dice']} ± {row['std_dice']}\t{row['mean_delta_vs_imagenet']}\t{row['positive_pair_count']}")


if __name__ == "__main__":
    main()

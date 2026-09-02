"""Aggregate the fixed-subset 25% Cat-label full-fine-tuning pilot."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logging_utils import enable_timestamped_prints

METHODS = ("imagenet", "full_human_mae", "constrained_human_mae_last2", "human_mae_alpha_0p1")
DISPLAY = {"imagenet": "ImageNet", "full_human_mae": "Full Human MAE",
           "constrained_human_mae_last2": "Last2", "human_mae_alpha_0p1": "alpha0.1"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/lowlabel25_fold0"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/lowlabel25_fold0"))
    return parser.parse_args()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, data, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(data)


def main():
    enable_timestamped_prints()
    args = parse_args(); subsets = (0, 1, 2); train_seed = 0
    detail, run_means, train_lists = [], {}, {}
    for subset_seed in subsets:
        for method in METHODS:
            root = args.runs_dir / method / f"subset_seed{subset_seed}"
            config_path, dice_path = root / "config.json", root / "subject_dice.csv"
            if not config_path.is_file() or not dice_path.is_file():
                raise FileNotFoundError(f"Incomplete run: {root}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            train_ids = (root / "train_subjects.txt").read_text(encoding="utf-8").splitlines()
            if config.get("transfer") != "full" or config.get("seed") != train_seed:
                raise RuntimeError(f"Not a valid low-label full-fine-tuning run: {root}")
            if subset_seed in train_lists and train_lists[subset_seed] != train_ids:
                raise RuntimeError(f"Methods use different train subjects for subset {subset_seed}")
            train_lists[subset_seed] = train_ids
            dice_rows = rows(dice_path)
            values = [float(row["mean_foreground_dice"]) for row in dice_rows]
            run_means[(method, subset_seed)] = statistics.mean(values)
            for row, value in zip(dice_rows, values):
                detail.append({"method": method, "fold": 0, "subset_seed": subset_seed,
                               "train_seed": train_seed, "train_subject_ids": "|".join(train_ids),
                               "test_subject_id": row["subject_id"], "subject_dice": value})
    aggregate = []
    for method in METHODS:
        for subset_seed in subsets:
            value = run_means[(method, subset_seed)]
            aggregate.append({"method": method, "subset_seed": subset_seed,
                              "mean_subject_dice": value,
                              "delta_vs_imagenet": value - run_means[("imagenet", subset_seed)]})
    write(args.results_dir / "subject_level_results.csv", detail,
          ("method", "fold", "subset_seed", "train_seed", "train_subject_ids", "test_subject_id", "subject_dice"))
    write(args.results_dir / "aggregate_by_subset.csv", aggregate,
          ("method", "subset_seed", "mean_subject_dice", "delta_vs_imagenet"))
    print("=" * 60); print("CAT 25% LOW-LABEL FULL FINETUNING - FOLD0"); print("TRAIN_SEED = 0")
    print("=" * 60); print("\nOriginal fold0 train subjects: 180\nSubjects per low-label subset: 45\nActual fraction: 0.250000")
    for subset_seed in subsets:
        print(f"\n---\n\nSUBSET SEED {subset_seed}\nTrain subjects:\n{train_lists[subset_seed]}")
        for method in METHODS: print(f"{DISPLAY[method]}: {run_means[(method, subset_seed)]:.6f}")
        print("\npaired deltas vs ImageNet:")
        for method in METHODS[1:]: print(f"{DISPLAY[method]}: {run_means[(method, subset_seed)] - run_means[('imagenet', subset_seed)]:+.6f}")
    print("\n" + "=" * 60 + "\nAGGREGATE ACROSS SUBJECT SUBSETS\n" + "=" * 32)
    for method in METHODS:
        values = [run_means[(method, subset_seed)] for subset_seed in subsets]
        print(f"\n{DISPLAY[method]}")
        for subset_seed, value in zip(subsets, values): print(f"subset{subset_seed}: {value:.6f}")
        print(f"mean +/- std: {statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}")
        if method != "imagenet":
            deltas = [value - run_means[("imagenet", subset_seed)] for subset_seed, value in zip(subsets, values)]
            print(f"mean delta vs ImageNet: {statistics.mean(deltas):+.6f}\npaired subset deltas:")
            for subset_seed, delta in zip(subsets, deltas): print(f"subset{subset_seed}: {delta:+.6f}")
            print(f"positive subsets vs ImageNet: {sum(delta > 0 for delta in deltas)}/3")
    print("\n" + "=" * 60 + "\nSUBJECT-LEVEL DETAIL\n" + "=" * 20)
    for subject_id in sorted({row["test_subject_id"] for row in detail}):
        print(f"\n{subject_id}")
        for method in METHODS:
            values = [row["subject_dice"] for row in detail if row["method"] == method and row["test_subject_id"] == subject_id]
            print(f"{DISPLAY[method]}: " + ", ".join(f"subset{seed}={value:.6f}" for seed, value in zip(subsets, values)))


if __name__ == "__main__": main()

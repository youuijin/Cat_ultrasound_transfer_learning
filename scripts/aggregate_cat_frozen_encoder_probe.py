"""Aggregate subject-level results from the Cat frozen-encoder probe."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHODS = ("imagenet", "full_human_mae", "constrained_human_mae_last2", "human_mae_alpha_0p1")
DISPLAY = {
    "imagenet": "ImageNet", "full_human_mae": "Full Human MAE",
    "constrained_human_mae_last2": "Last2", "human_mae_alpha_0p1": "alpha0.1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cat_frozen_encoder_probe"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cat_frozen_encoder_probe"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_root(args: argparse.Namespace, method: str, seed: int) -> Path:
    return args.runs_dir / method / f"fold{args.fold}" / f"seed{seed}"


def main() -> None:
    args = parse_args()
    records = []
    run_summary = {}
    for method in args.methods:
        for seed in args.seeds:
            root = run_root(args, method, seed)
            subject_path, config_path = root / "subject_dice.csv", root / "config.json"
            if not subject_path.is_file() or not config_path.is_file():
                raise FileNotFoundError(f"Incomplete probe run for {method}/seed{seed}: {root}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            counts = json.loads((root / "parameter_counts.json").read_text(encoding="utf-8"))
            if config.get("transfer") != "frozen" or counts.get("trainable_encoder_parameters") != 0:
                raise RuntimeError(f"Run is not a valid frozen-encoder probe: {root}")
            subject_rows = read_rows(subject_path)
            if not subject_rows:
                raise RuntimeError(f"No subject-level Dice rows: {subject_path}")
            values = []
            for row in subject_rows:
                dice = float(row["mean_foreground_dice"])
                values.append(dice)
                records.append({"method": method, "fold": args.fold, "seed": seed,
                                "subject_id": row["subject_id"], "subject_dice": dice})
            run_summary[(method, seed)] = statistics.mean(values)

    write_rows(args.results_dir / "subject_level_results.csv", records,
               ("method", "fold", "seed", "subject_id", "subject_dice"))
    baseline = {seed: run_summary[("imagenet", seed)] for seed in args.seeds
                if ("imagenet", seed) in run_summary}
    summaries = []
    print("=" * 60)
    print(f"CAT FROZEN ENCODER PROBE - FOLD{args.fold}")
    print("=" * 60)
    for method in args.methods:
        values = [run_summary[(method, seed)] for seed in args.seeds]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        deltas = ([run_summary[(method, seed)] - baseline[seed] for seed in args.seeds]
                  if method != "imagenet" else [])
        summary = {"method": method, "mean_dice": mean, "std_dice": std,
                   "delta_vs_imagenet": statistics.mean(deltas) if deltas else 0.0,
                   "positive_seed_count": sum(delta > 0 for delta in deltas) if deltas else 0}
        summaries.append(summary)
        print(f"\n{DISPLAY[method]}")
        for seed, value in zip(args.seeds, values):
            print(f"seed{seed}: {value:.6f}")
        print(f"mean +/- std: {mean:.6f} +/- {std:.6f}")
        if deltas:
            print(f"delta vs ImageNet: {summary['delta_vs_imagenet']:+.6f}")
            print("paired seed deltas:")
            for seed, delta in zip(args.seeds, deltas):
                print(f"seed{seed}: {delta:+.6f}")
            print(f"positive seeds vs ImageNet: {summary['positive_seed_count']}/{len(deltas)}")
    print("\n" + "=" * 60)
    write_rows(args.results_dir / "aggregate_summary.csv", summaries,
               ("method", "mean_dice", "std_dice", "delta_vs_imagenet", "positive_seed_count"))


if __name__ == "__main__":
    main()

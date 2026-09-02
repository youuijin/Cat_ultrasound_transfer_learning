"""Run the fixed-subset 25% Cat-label full-fine-tuning pilot (fold 0 only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cat_frozen_representation import (DISPLAY, METHODS, checkpoint_for,
                                                    validate_checkpoint)
from src.classification.data import split_subjects
from src.logging_utils import enable_timestamped_prints

TRAIN_SEED = 0
FOLD = 0
FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/lowlabel25_fold0"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/lowlabel25_fold0"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Write/verify subsets and checkpoints without launching training.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def subset_path(args: argparse.Namespace, subset_seed: int) -> Path:
    return args.results_dir / "subsets" / f"lowlabel_fold0_subset_seed{subset_seed}.txt"


def prepare_subsets(args: argparse.Namespace) -> tuple[list[str], list[str], dict[int, list[str]]]:
    train, heldout, _, _ = split_subjects(str(args.data_root), "four_class", 5, FOLD, 42)
    train_ids, heldout_ids = [subject.subject_id for subject in train], [subject.subject_id for subject in heldout]
    overlap = set(train_ids) & set(heldout_ids)
    if overlap:
        raise RuntimeError(f"Fold0 train/held-out overlap: {sorted(overlap)}")
    subset_root = args.results_dir / "subsets"
    subset_root.mkdir(parents=True, exist_ok=True)
    (subset_root / "original_fold0_train_subjects.txt").write_text(
        "\n".join(train_ids) + "\n", encoding="utf-8")
    (subset_root / "fold0_heldout_validation_subjects.txt").write_text(
        "\n".join(heldout_ids) + "\n", encoding="utf-8")
    n_low = max(3, math.ceil(FRACTION * len(train_ids)))
    subsets = {}
    for subset_seed in args.subset_seeds:
        selected = [train_ids[index] for index in np.random.default_rng(subset_seed).choice(
            len(train_ids), size=n_low, replace=False)]
        path = subset_path(args, subset_seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_text(encoding="utf-8").splitlines() != selected:
            raise RuntimeError(f"Existing subset file disagrees with deterministic selection: {path}")
        path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        subsets[subset_seed] = selected
    return train_ids, heldout_ids, subsets


def run_dir(args: argparse.Namespace, method: str, subset_seed: int) -> Path:
    return args.runs_dir / method / f"subset_seed{subset_seed}"


def complete(root: Path, expected_train: list[str], heldout: list[str], checkpoint: Path | None) -> bool:
    needed = ("config.json", "metrics.csv", "last.pt", "best.pt", "subject_dice.csv",
              "parameter_counts.json", "train_subjects.txt", "val_subjects.txt")
    if not all((root / name).is_file() for name in needed):
        return False
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        counts = json.loads((root / "parameter_counts.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    loaded = config.get("encoder_load_summary", {})
    return (
        config.get("transfer") == "full" and config.get("seed") == TRAIN_SEED and
        config.get("fold") == FOLD and config.get("num_folds") == 5 and
        (root / "train_subjects.txt").read_text(encoding="utf-8").splitlines() == expected_train and
        (root / "val_subjects.txt").read_text(encoding="utf-8").splitlines() == heldout and
        counts.get("trainable_encoder_parameters", 0) > 0 and
        counts.get("trainable_decoder_parameters", 0) > 0 and
        (checkpoint is None or Path(loaded.get("checkpoint", "")).resolve() == checkpoint.resolve())
    )


def command(args: argparse.Namespace, method: str, subset_seed: int, checkpoint: Path | None) -> list[str]:
    result = [args.python, "-m", "src.segmentation.train", "--encoder", "vit_b16",
              "--encoder-init", "imagenet" if method == "imagenet" else "human_mae",
              "--transfer", "full", "--data-root", str(args.data_root), "--num-folds", "5",
              "--fold", "0", "--split-seed", "42", "--seed", str(TRAIN_SEED),
              "--train-subject-list", str(subset_path(args, subset_seed)), "--batch-size", "8",
              "--epochs", "50", "--lr", "1e-4", "--weight-decay", "1e-4",
              "--num-workers", str(args.num_workers), "--exact-run-dir",
              str(run_dir(args, method, subset_seed)), "--amp" if args.amp else "--no-amp"]
    if checkpoint is not None:
        result.extend(("--encoder-checkpoint", str(checkpoint)))
    return result


def print_sanity(method: str, subset_seed: int, selected: list[str], heldout: list[str],
                 checkpoint: Path | None, target: Path) -> None:
    print("=" * 50)
    print(f"METHOD: {DISPLAY[method]}")
    print("FOLD: 0")
    print(f"SUBSET_SEED: {subset_seed}")
    print("TRAIN_SEED: 0")
    print("LOW_LABEL_FRACTION_REQUESTED: 0.25")
    print("TOTAL_ORIGINAL_TRAIN_SUBJECTS: 180")
    print(f"SELECTED_TRAIN_SUBJECTS: {len(selected)}")
    print(f"ACTUAL_LOW_LABEL_FRACTION: {len(selected) / 180:.6f}")
    print("TRAIN_SUBJECT_IDS: " + ", ".join(selected))
    print("VAL_SUBJECT_IDS: " + ", ".join(heldout))
    print("TEST_SUBJECT_IDS: no distinct test split in the existing pipeline")
    print("ENCODER_FROZEN: False")
    print("TRAINABLE_ENCODER_PARAMS: checked after model construction (> 0)")
    print("TRAINABLE_HEAD_PARAMS: checked after model construction (> 0)")
    print("INITIALIZATION_CHECKPOINT: " + ("ImageNet pretrained ViT-B/16" if checkpoint is None else str(checkpoint)))
    print(f"RESULT_DIR: {target}")
    print("=" * 50)


def main() -> None:
    enable_timestamped_prints()
    args = parse_args()
    original, heldout, subsets = prepare_subsets(args)
    if len(original) != 180:
        raise RuntimeError(f"Unexpected fold0 training-subject count: {len(original)}")
    checkpoints, reference = {}, None
    for method in args.methods:
        for subset_seed in args.subset_seeds:
            checkpoint = checkpoint_for(method, TRAIN_SEED)
            if checkpoint is not None:
                reference = validate_checkpoint(method, TRAIN_SEED, checkpoint, reference)
            checkpoints[method] = checkpoint
            selected = subsets[subset_seed]
            if set(selected) & set(heldout):
                raise RuntimeError(f"Low-label subset overlaps held-out fold subjects: {method}/{subset_seed}")
            target = run_dir(args, method, subset_seed)
            print_sanity(method, subset_seed, selected, heldout, checkpoint, target)
            if args.prepare_only:
                continue
            if not complete(target, selected, heldout, checkpoint):
                subprocess.run(command(args, method, subset_seed, checkpoint), cwd=ROOT, check=True)
            if not complete(target, selected, heldout, checkpoint):
                raise RuntimeError(f"Run-integrity check failed: {target}")
            rows = read_rows(target / "subject_dice.csv")
            counts = json.loads((target / "parameter_counts.json").read_text(encoding="utf-8"))
            print(f"TRAINABLE_ENCODER_PARAMS: {counts['trainable_encoder_parameters']}")
            print(f"TRAINABLE_HEAD_PARAMS: {counts['trainable_decoder_parameters']}")
            print("LOW-LABEL CAT RESULT")
            print(f"METHOD: {DISPLAY[method]}\nFOLD: 0\nSUBSET_SEED: {subset_seed}\nTRAIN_SEED: 0")
            print("TRAIN_SUBJECT_IDS: " + ", ".join(selected))
            print(f"NUMBER_OF_TRAIN_SUBJECTS: {len(selected)}")
            print(f"NUMBER_OF_TRAIN_FRAMES: {json.loads((target / 'config.json').read_text(encoding='utf-8'))['train_samples']}")
            print("SUBJECT_DICE:")
            for row in rows:
                print(f"{row['subject_id']}: {float(row['mean_foreground_dice']):.6f}")
            print("MEAN_SUBJECT_DICE: " + format(
                float(np.mean([float(row['mean_foreground_dice']) for row in rows])), ".6f"))


if __name__ == "__main__":
    main()

"""Run frozen Cat segmentation repeats while holding every Human checkpoint fixed.

Human SSL is always seed 0 in this launcher.  ``--folds`` and ``--seeds``
refer only to Cat downstream splits and decoder-training randomness.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_cat_frozen_screening import METHODS, checkpoint, complete, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]
# Only these legacy names belong to the original frozen probe.  Human-method
# seed1/2 runs are intentionally *not* reused when their checkpoint differs
# from this launcher's fixed Human SSL seed-0 checkpoint.
LEGACY_METHODS = {
    "imagenet": "imagenet",
    "full_human_mae": "full_human_mae",
    "human_mae_last2": "constrained_human_mae_last2",
    "human_mae_alpha_0p1": "human_mae_alpha_0p1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/cat_frozen_screening"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(not 0 <= fold < 5 for fold in args.folds):
        parser.error("--folds values must be between 0 and 4.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds values must be non-negative.")
    return args


def expected(args: argparse.Namespace, fold: int, seed: int, path: Path | None) -> dict:
    return {"task": "segmentation", "encoder": "vit_b16",
            "encoder_init": "imagenet" if path is None else "human_mae",
            "encoder_checkpoint": None if path is None else str(path), "transfer": "frozen",
            "data_root": str(args.data_root), "num_folds": 5, "fold": fold,
            "split_seed": 42, "seed": seed, "batch_size": 8, "epochs": 50,
            "lr": 1e-4, "weight_decay": 1e-4, "amp": args.amp}


def command(args: argparse.Namespace, fold: int, seed: int, path: Path | None, target: Path) -> list[str]:
    result = [args.python, "-m", "src.segmentation.train", "--encoder", "vit_b16", "--encoder-init",
              "imagenet" if path is None else "human_mae", "--transfer", "frozen",
              "--data-root", str(args.data_root), "--num-folds", "5", "--fold", str(fold),
              "--split-seed", "42", "--seed", str(seed), "--batch-size", "8", "--epochs", "50",
              "--lr", "1e-4", "--weight-decay", "1e-4", "--num-workers", str(args.num_workers),
              "--human-ssl-seed", "0",
              "--exact-run-dir", str(target), "--amp" if args.amp else "--no-amp"]
    if path is not None:
        result.extend(("--encoder-checkpoint", str(path)))
    return result


def existing_run(args: argparse.Namespace, method: str, fold: int, seed: int,
                 required: dict, target: Path) -> Path | None:
    """Return an exact completed run, preferring the new screening location."""
    candidates = [target]
    legacy = LEGACY_METHODS.get(method)
    if legacy is not None:
        candidates.append(ROOT / "runs/cat_frozen_encoder_probe" / legacy / f"fold{fold}" / f"seed{seed}")
    return next((candidate for candidate in candidates if complete(candidate, required)), None)


def main() -> None:
    args = parse_args()
    reference = None
    checkpoints: dict[str, Path | None] = {}
    for method in args.methods:
        path = checkpoint(method)
        # All selected Human encoders are validated once as seed-0 artifacts.
        reference = validate_checkpoint(method, path, reference)
        checkpoints[method] = path
    plans = []
    for method in args.methods:
        for fold in args.folds:
            for seed in args.seeds:
                path = checkpoints[method]
                target = args.runs_dir / method / f"fold{fold}" / f"seed{seed}"
                required = expected(args, fold, seed, path)
                found = existing_run(args, method, fold, seed, required, target)
                reuse = found is not None
                print(f"[reuse] {method} fold={fold} cat_seed={seed}: {found}" if reuse else
                      f"[run] {method} fold={fold} cat_seed={seed}: {target}")
                plans.append((method, fold, seed, path, target, required, reuse))
    print(f"Human SSL seed: 0 (fixed)\nexisting matching Cat runs: {sum(item[-1] for item in plans)}"
          f"\nmissing Cat runs: {sum(not item[-1] for item in plans)}")
    if args.dry_run:
        return
    for method, fold, seed, path, target, required, reuse in plans:
        if reuse:
            continue
        subprocess.run(command(args, fold, seed, path, target), cwd=ROOT, check=True)
        if not complete(target, required):
            raise RuntimeError(f"Completed run failed frozen-run validation: {target}")


if __name__ == "__main__":
    main()

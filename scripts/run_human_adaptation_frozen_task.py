"""Run only missing exact frozen binary-classification or detection experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from run_cat_frozen_screening import METHODS, checkpoint, same_path, validate_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification_binary", "detection"), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Cat downstream seeds only.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/human_adaptation_frozen"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if any(not 0 <= fold < 5 for fold in args.folds): parser.error("--folds values must be between 0 and 4.")
    if any(seed < 0 for seed in args.seeds): parser.error("--seeds values must be non-negative.")
    return args


def run_root(args: argparse.Namespace, method: str, fold: int, seed: int, path: Path | None) -> Path:
    if args.task == "classification_binary":
        root = args.runs_dir / "classification_binary" / method / "classification" / "vit_b16" / "frozen" / f"fold_{fold}" / f"seed_{seed}" / "balanced_softmax"
    else:
        root = args.runs_dir / "detection" / method / "detection" / "vit_b16" / "frozen" / f"fold_{fold}" / f"seed_{seed}"
    return root / "init_human_mae" if path is not None else root


def expected(args: argparse.Namespace, fold: int, seed: int, path: Path | None) -> dict:
    expected = {"task": "classification" if args.task == "classification_binary" else "detection", "encoder": "vit_b16",
                "encoder_init": "imagenet" if path is None else "human_mae", "encoder_checkpoint": None if path is None else str(path),
                "transfer": "frozen", "data_root": str(args.data_root), "num_folds": 5, "fold": fold,
                "split_seed": 42, "seed": seed, "batch_size": 32, "epochs": 50, "lr": 1e-4,
                "weight_decay": 1e-4, "amp": args.amp}
    if args.task == "classification_binary": expected.update(classification_mode="binary", condition="balanced_softmax")
    return expected


def complete(root: Path, required: dict) -> bool:
    names = ("config.json", "last.pt", "best.pt", "validation_metrics.json", "parameter_counts.json", "trainable_parameters.json")
    if not all((root / name).is_file() for name in names): return False
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        trainable = json.loads((root / "trainable_parameters.json").read_text(encoding="utf-8"))
        last = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    except Exception: return False
    for key, value in required.items():
        if key == "encoder_checkpoint":
            if not same_path(config.get(key), value): return False
        elif key == "data_root":
            if Path(config.get(key, "")).resolve() != Path(value).resolve(): return False
        elif config.get(key) != value: return False
    return int(last.get("epoch", -1)) == 49 and not any(name.startswith("encoder.") for name in trainable)


def command(args: argparse.Namespace, method: str, fold: int, seed: int, path: Path | None) -> list[str]:
    output = args.runs_dir / args.task / method
    module = "src.classification.train" if args.task == "classification_binary" else "src.detection.train"
    result = [args.python, "-m", module, "--encoder", "vit_b16", "--encoder-init", "imagenet" if path is None else "human_mae",
              "--transfer", "frozen", "--data-root", str(args.data_root), "--num-folds", "5", "--fold", str(fold),
              "--split-seed", "42", "--seed", str(seed), "--batch-size", "32", "--epochs", "50", "--lr", "1e-4",
              "--weight-decay", "1e-4", "--num-workers", str(args.num_workers), "--output-dir", str(output),
              "--amp" if args.amp else "--no-amp"]
    if args.task == "classification_binary": result.extend(("--task", "classification", "--classification-mode", "binary", "--condition", "balanced_softmax"))
    else: result.extend(("--task", "detection"))
    if path is not None: result.extend(("--encoder-checkpoint", str(path)))
    return result


def main() -> None:
    args = parse_args(); reference = None; paths = {}
    for method in args.methods:
        path = checkpoint(method); reference = validate_checkpoint(method, path, reference); paths[method] = path
    plans = []
    for method in args.methods:
        for fold in args.folds:
            for seed in args.seeds:
                path = paths[method]; root = run_root(args, method, fold, seed, path); ok = complete(root, expected(args, fold, seed, path))
                print(f"[reuse] {args.task} {method} fold={fold} cat_seed={seed}: {root}" if ok else
                      f"[run] {args.task} {method} fold={fold} cat_seed={seed}: {root}")
                plans.append((method, fold, seed, path, root, ok))
    print(f"Human SSL seed: 0 (fixed)\nexisting matching runs: {sum(plan[-1] for plan in plans)}\nmissing runs: {sum(not plan[-1] for plan in plans)}")
    if args.dry_run: return
    for method, fold, seed, path, root, ok in plans:
        if ok: continue
        subprocess.run(command(args, method, fold, seed, path), cwd=ROOT, check=True)
        if not complete(root, expected(args, fold, seed, path)): raise RuntimeError(f"Completed run failed integrity validation: {root}")


if __name__ == "__main__": main()

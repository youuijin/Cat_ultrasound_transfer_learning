"""Measure Cat representation drift along Human MAE adaptation checkpoints.

This is deliberately a read-only analysis: it discovers encoder checkpoints,
uses only fold-training Cat images as an unlabeled probe, and compares every
representation with the torchvision ImageNet ViT-B/16 using linear CKA.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.classification.data import split_subjects
from src.classification.dataset import CatImageTransform, load_nifti_image
from src.encoders import get_encoder


LAYER_NAMES = ["patch_embed", *[f"block_{i}" for i in range(12)], "final_norm"]
HIGHLIGHT_LAYERS = ("patch_embed", "block_3", "block_6", "block_9", "block_11", "final_norm")
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


@dataclass
class CheckpointInfo:
    filename: str
    path: str
    epoch: int | None
    epoch_source: str
    checkpoint_type: str
    encoder_parameter_count: int
    format: str | None
    adaptation: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--human-checkpoint-dir", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/human_ssl_cat_drift"))
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Analyze every discovered checkpoint instead of base/middle/final.")
    parser.add_argument("--max-images", type=int,
                        help="Deterministic image subset (after sorting by subject and path).")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--token", choices=("mean_patch", "cls"), default="mean_patch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inventory-only", action="store_true",
                        help="Write checkpoint inventory/selection without loading images or ImageNet weights.")
    args = parser.parse_args()
    if args.num_folds < 2 or not 0 <= args.fold < args.num_folds:
        parser.error("--num-folds must be >= 2 and --fold must be in range")
    if args.batch_size < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error("--batch-size and --max-images must be positive")
    return args


def _checkpoint_type(path: Path) -> str:
    name = path.stem.lower()
    if "best" in name:
        return "best"
    if "last" in name or "final" in name:
        return "last/final"
    if re.search(r"(?:epoch|ep)[_-]?\d+", name):
        return "epoch"
    if "encoder" in name:
        return "encoder-only"
    return "unknown"


def _state_from_payload(payload: Any, path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"], payload
        if isinstance(payload.get("model_state_dict"), dict):
            return payload["model_state_dict"], payload
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload, {}
    raise ValueError(f"Unsupported checkpoint structure: {path}")


def _epoch_from_name(path: Path) -> int | None:
    matches = re.findall(r"(?:epoch|ep)[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def inspect_checkpoint(path: Path) -> tuple[CheckpointInfo, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state, metadata = _state_from_payload(payload, path)
    raw_epoch = metadata.get("epoch")
    if raw_epoch is not None:
        epoch, source = int(raw_epoch), "metadata"
    else:
        epoch = _epoch_from_name(path)
        source = "filename" if epoch is not None else "unknown"
    info = CheckpointInfo(
        filename=path.name,
        path=str(path.resolve()),
        epoch=epoch,
        epoch_source=source,
        checkpoint_type=_checkpoint_type(path),
        encoder_parameter_count=sum(value.numel() for value in state.values() if torch.is_tensor(value)),
        format=metadata.get("format"),
        adaptation=metadata.get("adaptation"),
    )
    return info, state


def discover_checkpoints(directory: Path) -> list[CheckpointInfo]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Human MAE checkpoint directory not found: {directory}")
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found below {directory}")
    inventory = []
    print("Found Human MAE checkpoints\n" + "-" * 60)
    for path in paths:
        info, _ = inspect_checkpoint(path)
        inventory.append(info)
        shown_epoch = "?" if info.epoch is None else str(info.epoch)
        print(f"{info.filename:<28} epoch={shown_epoch:<5} type={info.checkpoint_type:<12} "
              f"params={info.encoder_parameter_count:,}")
    return inventory


def select_checkpoints(inventory: list[CheckpointInfo], all_checkpoints: bool) -> list[dict[str, Any]]:
    known = [item for item in inventory if item.epoch is not None]
    epoch_checkpoints = [item for item in known if item.checkpoint_type == "epoch"]
    ordered = sorted(inventory, key=lambda item: (
        math.inf if item.epoch is None else item.epoch, item.filename))
    selected: list[dict[str, Any]] = [{
        "selection": "initial", "checkpoint_name": "imagenet_base", "path": "",
        "epoch": 0, "epoch_source": "synthetic_base", "checkpoint_type": "imagenet_base",
        "encoder_parameter_count": "",
    }]
    if all_checkpoints:
        chosen = ordered
        labels = ["available"] * len(chosen)
    elif epoch_checkpoints:
        final = max(known, key=lambda item: (item.epoch, item.checkpoint_type == "last/final"))
        total_epoch = final.epoch
        candidates = [item for item in epoch_checkpoints if item.path != final.path]
        if candidates:
            middle = min(candidates, key=lambda item: (abs(item.epoch - total_epoch / 2), item.epoch))
            chosen, labels = [middle, final], ["middle", "final"]
        else:
            chosen, labels = [final], ["final"]
            warnings.warn("No intermediate epoch checkpoint was found; computing base-to-final drift only.")
    elif inventory:
        # best/last are outcome aliases, not intermediate trajectory points.  Do not
        # mislabel a late best checkpoint as the middle merely because it has metadata.
        final = max(inventory, key=lambda item: (
            item.checkpoint_type == "last/final",
            item.checkpoint_type == "best",
            item.epoch if item.epoch is not None else -1))
        chosen, labels = [final], ["final"]
        warnings.warn("No epoch checkpoints were found; computing base-to-final drift only "
                      "from the available best/last checkpoint.")
    else:
        chosen, labels = [], []
    for label, item in zip(labels, chosen):
        row = asdict(item)
        row.update({"selection": label, "checkpoint_name": item.filename})
        selected.append(row)
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class CatProbeDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: CatImageTransform) -> None:
        self.rows, self.transform = rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(load_nifti_image(self.rows[index]["image_path"]))


def cat_probe_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    train, _validation, _classes, issues = split_subjects(
        str(args.data_root), "four_class", args.num_folds, args.fold, args.split_seed)
    rows = [{"subject_id": subject.subject_id, "class_name": subject.class_name,
             "side": side, "image_path": str(path.resolve())}
            for subject in train for side, path in sorted(subject.images.items())]
    rows.sort(key=lambda row: (row["subject_id"], row["side"], row["image_path"]))
    if args.max_images is not None:
        # A sorted prefix is stable across machines and never draws from validation subjects.
        rows = rows[:args.max_images]
    if not rows:
        raise ValueError("The fold-0 training split contains no Cat probe images.")
    print(f"Cat probe: {len(set(row['subject_id'] for row in rows))} subjects, "
          f"{len(rows)} images (fold={args.fold} training split only)")
    return rows, issues


def _pool(tokens: torch.Tensor, token: str) -> torch.Tensor:
    return tokens[:, 0] if token == "cls" else tokens[:, 1:].mean(dim=1)


@torch.no_grad()
def extract_layer_features(encoder, loader: DataLoader, device: str,
                           token: str) -> dict[str, np.ndarray]:
    model = encoder.model
    collected = {name: [] for name in LAYER_NAMES}
    for images in loader:
        images = images.to(device, non_blocking=True)
        images = encoder.adapt_input_channels(images, encoder.repeat_grayscale)
        patches = model._process_input(images)
        batch = patches.shape[0]
        # patch_embed is the pooled projected-patch output before positional embedding.
        patch_vector = patches.mean(dim=1) if token == "mean_patch" else model.class_token.expand(batch, -1, -1)[:, 0]
        collected["patch_embed"].append(patch_vector.float().cpu().numpy())
        tokens = torch.cat((model.class_token.expand(batch, -1, -1), patches), dim=1)
        tokens = model.encoder.dropout(tokens + model.encoder.pos_embedding)
        for index, block in enumerate(model.encoder.layers):
            tokens = block(tokens)
            collected[f"block_{index}"].append(_pool(tokens, token).float().cpu().numpy())
        tokens = model.encoder.ln(tokens)
        collected["final_norm"].append(_pool(tokens, token).float().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA in float64 without constructing n-by-n Gram matrices."""
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("CKA requires matching feature rows from at least two images")
    x = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    y = y.astype(np.float64, copy=False) - y.mean(axis=0, keepdims=True)
    numerator = np.square(np.linalg.norm(x.T @ y, ord="fro"))
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("CKA is undefined because a centered feature matrix has zero variance")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def load_human_checkpoint(encoder, path: Path) -> None:
    _info, state = inspect_checkpoint(path)
    expected = encoder.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(key for key in set(expected) & set(state)
                            if expected[key].shape != state[key].shape)
    matched = sorted(key for key in set(expected) & set(state)
                     if expected[key].shape == state[key].shape)
    matched_params = sum(expected[key].numel() for key in matched)
    print(f"[{path.name}] matched encoder params={matched_params:,}; "
          f"missing={len(missing)} unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}")
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(f"Incompatible encoder checkpoint {path}: missing={missing}, "
                           f"unexpected={unexpected}, shape_mismatch={shape_mismatch}")
    encoder.load_state_dict(state, strict=True)


def summarize(layerwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (epoch, checkpoint), values in layerwise.groupby(["epoch", "checkpoint"], sort=False):
        drift = values.set_index("layer")["drift_1_minus_cka"]
        rows.append({
            "epoch": epoch, "checkpoint": checkpoint,
            "mean_layer_drift": drift.mean(),
            "early_mean": drift[["patch_embed", "block_0", "block_1", "block_2", "block_3"]].mean(),
            "middle_mean": drift[[f"block_{i}" for i in range(4, 9)]].mean(),
            "late_mean": drift[["block_9", "block_10", "block_11", "final_norm"]].mean(),
            "n_layers": len(drift),
            "aggregation": "equal weight over patch_embed, block_0..11, final_norm",
        })
    return pd.DataFrame(rows)


def make_plots(layerwise: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    ordered = layerwise.sort_values(["epoch", "layer_order"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer in HIGHLIGHT_LAYERS:
        values = ordered[ordered.layer == layer]
        ax.plot(values.epoch, values.drift_1_minus_cka, marker="o", label=layer)
    ax.plot(summary.epoch, summary.mean_layer_drift, color="black", linewidth=3,
            marker="o", label="mean_layer_drift")
    ax.set(xlabel="Human MAE epoch", ylabel="Cat feature drift (1 - linear CKA)")
    ax.set_ylim(bottom=0); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(output / "cat_drift_trajectory.png", dpi=200); plt.close(fig)

    matrix = ordered.pivot(index="layer", columns="checkpoint", values="drift_1_minus_cka")
    matrix = matrix.reindex(LAYER_NAMES)
    column_order = (ordered[["checkpoint", "epoch"]].drop_duplicates()
                    .sort_values("epoch").checkpoint.tolist())
    matrix = matrix[column_order]
    fig, ax = plt.subplots(figsize=(max(6, len(column_order) * 1.5), 7))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="Checkpoint", ylabel="Encoder layer")
    fig.colorbar(image, ax=ax, label="1 - linear CKA")
    fig.tight_layout(); fig.savefig(output / "cat_drift_heatmap.png", dpi=200); plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = discover_checkpoints(args.human_checkpoint_dir)
    inventory_rows = [asdict(item) for item in inventory]
    selected = select_checkpoints(inventory, args.all_checkpoints)
    _write_csv(args.output_dir / "checkpoint_inventory.csv", inventory_rows)
    _write_csv(args.output_dir / "selected_checkpoints.csv", selected)
    if args.inventory_only:
        print(f"Inventory written to {args.output_dir.resolve()}")
        return

    probe, issues = cat_probe_rows(args)
    _write_csv(args.output_dir / "cat_probe_images.csv", probe)
    if issues:
        (args.output_dir / "cat_discovery_issues.json").write_text(
            json.dumps(issues, indent=2, ensure_ascii=False), encoding="utf-8")
    encoder = get_encoder("vit_b16_imagenet", pretrained=True).eval().to(args.device)
    transform = CatImageTransform(encoder.preprocess.image_size, encoder.preprocess.input_channels,
                                  encoder.preprocess.mean, encoder.preprocess.std, training=False)
    loader = DataLoader(CatProbeDataset(probe, transform), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=str(args.device).startswith("cuda"))
    base = extract_layer_features(encoder, loader, args.device, args.token)
    rows = []
    for selection in selected:
        if selection["checkpoint_type"] == "imagenet_base":
            current = base
        else:
            load_human_checkpoint(encoder, Path(selection["path"]))
            encoder.eval()
            current = extract_layer_features(encoder, loader, args.device, args.token)
        for order, layer in enumerate(LAYER_NAMES):
            cka = linear_cka(base[layer], current[layer])
            rows.append({"epoch": selection["epoch"], "checkpoint": selection["checkpoint_name"],
                         "selection": selection["selection"], "layer": layer,
                         "layer_order": order, "token": args.token,
                         "cka_to_imagenet": cka, "drift_1_minus_cka": 1.0 - cka,
                         "n_images": len(probe), "feature_dim": current[layer].shape[1]})
    layerwise = pd.DataFrame(rows)
    layerwise.to_csv(args.output_dir / "layerwise_cat_cka_drift.csv", index=False)
    summary = summarize(layerwise)
    summary.to_csv(args.output_dir / "cat_cka_drift_summary.csv", index=False)
    make_plots(layerwise, summary, args.output_dir)
    print(f"Analysis outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

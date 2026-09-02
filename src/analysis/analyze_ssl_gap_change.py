"""Layer-wise Human-Cat MMD change after Human SSL adaptation."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.config.datasets import DATASETS
from src.classification.model import build_encoder


DATASET_ORDER = ("human1", "human2", "human3", "cat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b16",), default="vit_b16")
    parser.add_argument("--ssl-checkpoint", type=Path, required=True)
    parser.add_argument("--ssl-name", choices=("human_mae", "human_dino"), required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/ssl_gap_change"))
    parser.add_argument("--max-images-per-dataset", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def load_encoders(args: argparse.Namespace):
    base = build_encoder(
        name="vit_b16",
        transfer="frozen",
        partial_blocks=0,
        encoder_init="imagenet",
    ).cpu().eval()
    adapted = build_encoder(
        name="vit_b16",
        transfer="frozen",
        partial_blocks=0,
        encoder_init=args.ssl_name,
        encoder_checkpoint=str(args.ssl_checkpoint),
    ).cpu().eval()
    return base, adapted


def extract_features(encoder, core, device: str) -> tuple[dict, dict]:
    """Use the established deterministic dataset extraction unchanged."""
    by_feature: dict[str, dict[str, np.ndarray]] = {}
    encoder.to(device).eval()
    for dataset_name in DATASET_ORDER:
        extracted = core.extract_dataset_layer_features(
            encoder=encoder,
            dataset_name=dataset_name,
            root=DATASETS[dataset_name],
            device=device,
        )
        for feature_key, values in extracted.items():
            by_feature.setdefault(feature_key, {})[dataset_name] = values
    metadata = {row["feature_key"]: row for row in encoder.layer_feature_metadata}
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return by_feature, metadata


def selected_representations(metadata: dict) -> list[str]:
    keys = [
        key for key, row in metadata.items()
        if row["representation_type"] == "patch_mean"
        and row["layer"].startswith("block_")
    ]
    post_norm = [
        key for key, row in metadata.items()
        if row["representation_type"] == "post_norm_patch_mean"
    ]
    return sorted(keys, key=lambda key: metadata[key]["layer_index"]) + post_norm


def calculate_layer_mmd(dataset_features: dict, core) -> tuple[float, float]:
    """Return existing human-cat mean MMD² and its deterministic bandwidth."""
    matrix, sigma_squared = core.build_mmd_matrix(dataset_features)
    summary = core.calculate_domain_gap_summary(matrix).iloc[0]
    return float(summary["human_cat_mean_mmd2"]), float(sigma_squared)


def compare_models(base_features: dict, base_metadata: dict,
                   ssl_features: dict, ssl_metadata: dict,
                   ssl_name: str, epsilon: float, core) -> pd.DataFrame:
    base_keys = selected_representations(base_metadata)
    ssl_keys = selected_representations(ssl_metadata)
    if base_keys != ssl_keys:
        raise RuntimeError(
            f"Base/SSL representation mismatch: base={base_keys}, ssl={ssl_keys}"
        )

    rows = []
    for feature_key in base_keys:
        base_row = base_metadata[feature_key]
        ssl_row = ssl_metadata[feature_key]
        if base_row["feature_dim"] != ssl_row["feature_dim"]:
            raise RuntimeError(f"Feature dimension mismatch at {feature_key}.")
        imagenet_mmd2, imagenet_sigma = calculate_layer_mmd(
            base_features[feature_key], core
        )
        ssl_mmd2, ssl_sigma = calculate_layer_mmd(ssl_features[feature_key], core)
        delta = ssl_mmd2 - imagenet_mmd2
        layer_name = (
            "final_norm" if base_row["representation_type"] == "post_norm_patch_mean"
            else base_row["layer"]
        )
        rows.append({
            "ssl_name": ssl_name,
            "encoder": "vit_b16_imagenet",
            "layer_name": layer_name,
            "layer_index": base_row["layer_index"],
            "representation_type": base_row["representation_type"],
            "pooling": base_row["pooling"],
            "feature_dim": base_row["feature_dim"],
            "imagenet_rbf_sigma_squared": imagenet_sigma,
            "human_ssl_rbf_sigma_squared": ssl_sigma,
            "imagenet_human_cat_mmd2": imagenet_mmd2,
            "human_ssl_human_cat_mmd2": ssl_mmd2,
            "delta_mmd2": delta,
            "relative_mmd_change": delta / (abs(imagenet_mmd2) + epsilon),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.max_images_per_dataset < 1 or args.batch_size < 1:
        raise ValueError("Sample and batch sizes must be positive.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")

    core = importlib.import_module("analysis.3_encoder_feature")
    core.MAX_IMAGES_PER_DATASET = args.max_images_per_dataset
    core.BATCH_SIZE = args.batch_size
    core.RANDOM_SEED = args.seed
    core.set_seed(args.seed)
    device = core.DEVICE

    base_encoder, ssl_encoder = load_encoders(args)
    base_features, base_metadata = extract_features(base_encoder, core, device)
    ssl_features, ssl_metadata = extract_features(ssl_encoder, core, device)
    result = compare_models(
        base_features, base_metadata, ssl_features, ssl_metadata,
        args.ssl_name, args.epsilon, core,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.ssl_name}_vit_b16_ssl_gap_change.csv"
    result.to_csv(output_path, index=False)

    print("\nLayer        ImageNet MMD²    Human-SSL MMD²    Delta")
    print("-------------------------------------------------------")
    for row in result.itertuples(index=False):
        print(f"{row.layer_name:<12} {row.imagenet_human_cat_mmd2:<16.8f} "
              f"{row.human_ssl_human_cat_mmd2:<17.8f} {row.delta_mmd2:.8f}")
    print(f"\nCSV saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()

"""Compare ImageNet and human-SSL-adapted torchvision ViT-B/16 encoders."""
from __future__ import annotations

import argparse
import importlib
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from analysis.config.datasets import DATASETS
from analysis.utils.dataset_io import load_analysis_samples, read_analysis_image
from src.encoders import get_encoder


DATASET_ORDER = ("human1", "human2", "human3", "cat")
ENCODER_ORDER = ("imagenet", "human_mae", "human_dino")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", choices=("parameter_change", "cka", "domain_gap",
                                                "visualization", "all"), required=True)
    parser.add_argument("--imagenet-checkpoint", type=Path)
    parser.add_argument("--human-mae-checkpoint", type=Path)
    parser.add_argument("--human-dino-checkpoint", type=Path)
    for name in DATASET_ORDER:
        parser.add_argument(f"--{name}-root", type=Path, default=DATASETS[name])
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER,
                        default=list(DATASET_ORDER))
    parser.add_argument("--cka-dataset", choices=DATASET_ORDER, default="cat")
    parser.add_argument("--samples-per-dataset", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Reserved for CLI compatibility; image IO is intentionally sequential.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token", choices=("cls", "mean_patch"), default="cls")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/representation_analysis"))
    parser.add_argument("--umap", action="store_true",
                        help="Create UMAP plots only if umap-learn is already installed.")
    args = parser.parse_args()
    if args.samples_per_dataset < 1 or args.batch_size < 1:
        parser.error("sample and batch sizes must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"], payload
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, {}
    raise ValueError(f"{path} must be an encoder state_dict or contain 'state_dict'")


def _load_checkpoint(encoder, path: Path, label: str, expected_adaptation: str | None) -> None:
    state, metadata = _checkpoint_state(path)
    if expected_adaptation and metadata.get("adaptation") not in (None, expected_adaptation):
        raise ValueError(f"{label}: expected adaptation={expected_adaptation!r}, "
                         f"got {metadata.get('adaptation')!r}")
    incompatible = encoder.load_state_dict(state, strict=False)
    print(f"[{label}] checkpoint={path.expanduser().resolve()}")
    print(f"[{label}] missing keys ({len(incompatible.missing_keys)}): "
          f"{list(incompatible.missing_keys)}")
    print(f"[{label}] unexpected keys ({len(incompatible.unexpected_keys)}): "
          f"{list(incompatible.unexpected_keys)}")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"{label}: checkpoint does not exactly match the repository ViT-B/16")


def load_encoders(args: argparse.Namespace) -> dict[str, torch.nn.Module]:
    # The repository wrapper and official ImageNet initialization are used once; adapted
    # encoders are exact copies before their encoder-only checkpoints are loaded.
    imagenet = get_encoder("vit_b16_imagenet", pretrained=True)
    if args.imagenet_checkpoint:
        _load_checkpoint(imagenet, args.imagenet_checkpoint, "imagenet", None)
    else:
        print("[imagenet] torchvision ViT_B_16_Weights.IMAGENET1K_V1 (no checkpoint path)")
        print("[imagenet] missing keys (0): []\n[imagenet] unexpected keys (0): []")
    encoders = {"imagenet": imagenet, "human_mae": deepcopy(imagenet),
                "human_dino": deepcopy(imagenet)}
    required = (("human_mae", args.human_mae_checkpoint, "human_kidney_ultrasound_mae"),
                ("human_dino", args.human_dino_checkpoint, "human_kidney_ultrasound_dino"))
    for label, path, adaptation in required:
        if path is None:
            raise ValueError(f"--{label.replace('_', '-')}-checkpoint is required")
        _load_checkpoint(encoders[label], path, label, adaptation)
    for encoder in encoders.values():
        encoder.eval().to(args.device)
    return encoders


def select_samples(args: argparse.Namespace, names: list[str]) -> dict[str, list]:
    selected = {}
    for offset, name in enumerate(names):
        root = getattr(args, f"{name}_root")
        samples = load_analysis_samples(name, root)
        rng = np.random.default_rng(args.seed + offset)
        if len(samples) > args.samples_per_dataset:
            indices = np.sort(rng.choice(len(samples), args.samples_per_dataset, replace=False))
            samples = [samples[int(i)] for i in indices]
        if not samples:
            raise ValueError(f"No samples found for {name}: {root}")
        selected[name] = samples
        print(f"[{name}] selected {len(samples)} / root={root}")
    return selected


def prepare_tensor(sample, encoder) -> torch.Tensor:
    # Reuse the established feature-analysis preprocessing without copying it.
    feature_analysis = importlib.import_module("analysis.3_encoder_feature")
    return feature_analysis.prepare_image_tensor(read_analysis_image(sample, mode="raw"), encoder)


@torch.no_grad()
def extract_features(encoder, samples: list, batch_size: int, device: str) -> np.ndarray:
    chunks = []
    for start in range(0, len(samples), batch_size):
        batch = torch.stack([prepare_tensor(s, encoder)
                             for s in samples[start:start + batch_size]]).to(device)
        chunks.append(F.normalize(encoder.forward_features(batch), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _component_parameters(encoder) -> dict[str, list[torch.Tensor]]:
    model = encoder.model
    # Treat all input-token parameters as the patch/input embedding component so
    # class/position-token adaptation is not silently omitted from the report.
    groups = {"patch_embedding": [*model.conv_proj.parameters(), model.class_token,
                                  model.encoder.pos_embedding]}
    for index, block in enumerate(model.encoder.layers):
        groups[f"block_{index}"] = list(block.parameters())
    groups["final_norm"] = list(model.encoder.ln.parameters())
    return groups


def run_parameter_change(encoders: dict, output: Path) -> None:
    base = _component_parameters(encoders["imagenet"])
    rows = []
    for label in ("human_mae", "human_dino"):
        adapted = _component_parameters(encoders[label])
        for order, name in enumerate(base):
            numerator = sum(torch.sum((a.detach().cpu() - b.detach().cpu()) ** 2).item()
                            for b, a in zip(base[name], adapted[name]))
            denominator = sum(torch.sum(b.detach().cpu() ** 2).item() for b in base[name])
            rows.append({"encoder": label, "layer": name, "layer_order": order,
                         "normalized_l2_change": np.sqrt(numerator) /
                         max(np.sqrt(denominator), 1e-12)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "parameter_change.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, values in frame.groupby("encoder", sort=False):
        ax.plot(values.layer_order, values.normalized_l2_change, marker="o", label=label)
    ax.set_xticks(range(len(base)), list(base), rotation=45, ha="right")
    ax.set_ylabel("Normalized L2 change")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "parameter_change.png", dpi=200)
    plt.close(fig)


@torch.no_grad()
def block_features(encoder, samples: list, args: argparse.Namespace) -> list[np.ndarray]:
    collected = [[] for _ in range(12)]
    hooks = []
    for index, block in enumerate(encoder.model.encoder.layers):
        def capture(_module, _inputs, result, index=index):
            vector = result[:, 0] if args.token == "cls" else result[:, 1:].mean(dim=1)
            collected[index].append(vector.detach().cpu().float().numpy())
        hooks.append(block.register_forward_hook(capture))
    try:
        for start in range(0, len(samples), args.batch_size):
            batch = torch.stack([prepare_tensor(s, encoder)
                                 for s in samples[start:start + args.batch_size]]).to(args.device)
            encoder.forward_features(batch)
    finally:
        for hook in hooks:
            hook.remove()
    return [np.concatenate(parts) for parts in collected]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / max(denom, 1e-12))


def run_cka(encoders: dict, args: argparse.Namespace, output: Path) -> None:
    samples = select_samples(args, [args.cka_dataset])[args.cka_dataset]
    base = block_features(encoders["imagenet"], samples, args)
    rows = []
    for label in ("human_mae", "human_dino"):
        current = block_features(encoders[label], samples, args)
        rows.extend({"comparison": f"imagenet_vs_{label}", "block": index,
                     "token": args.token, "dataset": args.cka_dataset,
                     "n_samples": len(samples), "linear_cka": linear_cka(x, y)}
                    for index, (x, y) in enumerate(zip(base, current)))
    frame = pd.DataFrame(rows); frame.to_csv(output / "cka.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in frame.groupby("comparison", sort=False):
        ax.plot(values.block, values.linear_cka, marker="o", label=label)
    ax.set(xticks=range(12), xlabel="Transformer block", ylabel="Linear CKA", ylim=(0, 1.02))
    ax.legend(); fig.tight_layout(); fig.savefig(output / "cka.png", dpi=200); plt.close(fig)


def extract_all(encoders: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    samples = select_samples(args, list(args.datasets))
    features = {label: {name: extract_features(encoder, values, args.batch_size, args.device)
                        for name, values in samples.items()}
                for label, encoder in encoders.items()}
    return features, samples


def run_domain_gap(features: dict, args: argparse.Namespace, output: Path) -> None:
    legacy = importlib.import_module("analysis.4_feature_domain_gap")
    rows, summaries = [], []
    domain_output = output / "domain_gap"
    domain_output.mkdir(parents=True, exist_ok=True)
    for encoder_name, dataset_features in features.items():
        # These are the repository's existing normalization and metric implementations.
        normalized = {k: v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
                      for k, v in dataset_features.items()}
        centroids = legacy.calculate_centroids(normalized)
        euclidean = legacy.build_centroid_euclidean_matrix(centroids)
        cosine = legacy.build_centroid_cosine_matrix(centroids)
        old_seed, old_limit = legacy.RANDOM_SEED, legacy.MAX_MMD_SAMPLES
        legacy.RANDOM_SEED = args.seed
        legacy.MAX_MMD_SAMPLES = args.samples_per_dataset * len(normalized)
        try:
            sigma = legacy.estimate_sigma_squared(normalized)
        finally:
            legacy.RANDOM_SEED, legacy.MAX_MMD_SAMPLES = old_seed, old_limit
        names = [name for name in DATASET_ORDER if name in normalized]
        mmd = pd.DataFrame(0.0, index=names, columns=names)
        for a, b in combinations([n for n in DATASET_ORDER if n in normalized], 2):
            mmd_value = legacy.mmd_squared(normalized[a], normalized[b], sigma)
            mmd.loc[a, b] = mmd.loc[b, a] = mmd_value
            rows.append({"encoder": encoder_name, "dataset_a": a, "dataset_b": b,
                         "n_a": len(normalized[a]), "n_b": len(normalized[b]),
                         "mmd_squared": mmd_value,
                         "rbf_sigma_squared": sigma,
                         "centroid_euclidean": euclidean.loc[a, b],
                         "centroid_cosine": cosine.loc[a, b]})
        encoder_output = domain_output / encoder_name
        encoder_output.mkdir(parents=True, exist_ok=True)
        matrices = ((euclidean, "centroid_euclidean", "Centroid Euclidean Distance",
                     "Euclidean distance"),
                    (cosine, "centroid_cosine", "Centroid Cosine Distance",
                     "Cosine distance"),
                    (mmd, "mmd", "Feature Distribution Distance", "RBF-MMD²"))
        for matrix, filename, title, colorbar in matrices:
            matrix.to_csv(encoder_output / f"{filename}_matrix.csv")
            legacy.plot_heatmap(matrix, f"{title} — {encoder_name}", colorbar,
                                encoder_output / f"{filename}_heatmap.png")
        summaries.append({
            "encoder": encoder_name,
            "rbf_sigma_squared": sigma,
            **legacy.summarize_pairwise_matrix(euclidean, "centroid_euclidean"),
            **legacy.summarize_pairwise_matrix(cosine, "centroid_cosine"),
            **legacy.summarize_pairwise_matrix(mmd, "mmd2"),
        })
    pd.DataFrame(rows).to_csv(output / "domain_gap.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(domain_output / "encoder_comparison.csv", index=False)
    _plot_domain_gap_comparison(summary, domain_output / "encoder_comparison.png")


def _plot_domain_gap_comparison(summary: pd.DataFrame, path: Path) -> None:
    """Plot the legacy human–cat summary for all three initializations."""
    metrics = (("centroid_euclidean_human_cat", "Centroid Euclidean"),
               ("centroid_cosine_human_cat", "Centroid cosine"),
               ("mmd2_human_cat", "RBF-MMD²"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for ax, (column, title) in zip(axes, metrics):
        values = summary[column].to_numpy(dtype=float)
        bars = ax.bar(summary.encoder, values, color=colors[:len(values)])
        ax.set_title(title); ax.set_ylabel("Mean Human–Cat distance")
        ax.tick_params(axis="x", rotation=25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def _plot_embedding(values: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in DATASET_ORDER:
        mask = labels == name
        if mask.any(): ax.scatter(values[mask, 0], values[mask, 1], s=12, alpha=.7, label=name)
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_visualization(features: dict, args: argparse.Namespace, output: Path) -> None:
    for encoder_name, by_dataset in features.items():
        names = [name for name in DATASET_ORDER if name in by_dataset]
        x = np.concatenate([by_dataset[name] for name in names])
        labels = np.concatenate([np.repeat(name, len(by_dataset[name])) for name in names])
        pca = PCA(n_components=2, random_state=args.seed).fit_transform(x)
        _plot_embedding(pca, labels, f"PCA — {encoder_name}", output / f"pca_{encoder_name}.png")
        if args.umap:
            try:
                umap_module = importlib.import_module("umap")
            except ImportError:
                print("[visualization] umap-learn is not installed; skipping UMAP")
            else:
                values = umap_module.UMAP(n_components=2, random_state=args.seed).fit_transform(x)
                _plot_embedding(values, labels, f"UMAP — {encoder_name}",
                                output / f"umap_{encoder_name}.png")


def main() -> None:
    args = parse_args(); set_seed(args.seed); args.output_dir.mkdir(parents=True, exist_ok=True)
    encoders = load_encoders(args)
    requested = {args.analysis} if args.analysis != "all" else {
        "parameter_change", "cka", "domain_gap", "visualization"}
    if "parameter_change" in requested: run_parameter_change(encoders, args.output_dir)
    if "cka" in requested: run_cka(encoders, args, args.output_dir)
    if requested & {"domain_gap", "visualization"}:
        features, _ = extract_all(encoders, args)
        if "domain_gap" in requested: run_domain_gap(features, args, args.output_dir)
        if "visualization" in requested: run_visualization(features, args, args.output_dir)


if __name__ == "__main__":
    main()

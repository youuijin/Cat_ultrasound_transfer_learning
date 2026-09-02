"""Minimal Human kidney segmentation validation for ImageNet versus Human MAE."""
from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from analysis.config.datasets import DATASETS
from analysis.utils.dataset_io import load_analysis_samples
from analysis.utils.utils import load_roi_mask
from src.classification.training_utils import save_checkpoint, seed_worker, set_seed
from src.encoders import get_encoder
from src.human_ssl.data import discover_ssl_samples, split_ssl_samples
from src.segmentation.data import PairedTransform
from src.segmentation.model import LightweightDecoder


@dataclass(frozen=True)
class HumanSegmentationSample:
    image_path: Path
    subject_id: str

    @property
    def path(self) -> Path:
        """Compatibility with the repository ROI-mask utility's sample interface."""
        return self.image_path


class HumanSegmentationDataset(Dataset):
    def __init__(self, samples: list[HumanSegmentationSample], root: Path,
                 transform: PairedTransform) -> None:
        self.samples, self.root, self.transform = samples, root, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        with Image.open(item.image_path) as handle:
            image = handle.convert("L")
            shape = (image.height, image.width)
        mask = load_roi_mask("human2", item, self.root, shape)
        if mask is None:
            raise FileNotFoundError(f"No reviewer capsule mask for {item.image_path}")
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        image_tensor, mask_tensor = self.transform(image, mask_image)
        return image_tensor, mask_tensor, item.subject_id


class HumanBinarySegmenter(nn.Module):
    def __init__(self, encoder, frozen_encoder: bool) -> None:
        super().__init__()
        self.encoder, self.frozen_encoder = encoder, frozen_encoder
        self.decoder = LightweightDecoder(encoder.spatial_feature_dim, num_classes=2)

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder.forward_spatial_features(images)
        return self.decoder(features, images.shape[-2:])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("human2",), default="human2")
    parser.add_argument("--data-root", type=Path, default=DATASETS["human2"])
    parser.add_argument("--encoder", choices=("vit_b16",), default="vit_b16")
    parser.add_argument("--encoder-init", choices=("imagenet", "human_mae", "human_dino", "human_barlow"),
                        required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--transfer", choices=("frozen", "full"), required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("runs/human_downstream_validation"))
    parser.add_argument("--run-dir", type=Path,
                        help="Exact run directory override (used by trajectory probes).")
    parser.add_argument("--ssl-reference-config", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16_trajectory/config.json"),
                        help="Existing MAE config used to reconstruct the Human2 SSL subject split.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data, split, checkpoint, and outputs without training.")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between zero and one")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.encoder_init in ("human_mae", "human_dino", "human_barlow") and args.encoder_checkpoint is None:
        parser.error("--encoder-checkpoint is required with a Human SSL initialization")
    if args.encoder_init == "imagenet" and args.encoder_checkpoint is not None:
        parser.error("--encoder-checkpoint is only valid with --encoder-init human_mae")
    return args


def _human2_subject_id(path: Path) -> str:
    token = path.stem.split("_", 1)[0]
    if not token:
        raise ValueError(f"Cannot derive Human2 subject ID from {path.name}")
    return token


def discover_samples(root: Path) -> list[HumanSegmentationSample]:
    samples = []
    for sample in load_analysis_samples("human2", root):
        with Image.open(sample.path) as image:
            shape = (image.height, image.width)
        if load_roi_mask("human2", sample, root, shape) is not None:
            samples.append(HumanSegmentationSample(sample.path, _human2_subject_id(sample.path)))
    if not samples:
        raise ValueError(f"No Human2 images with capsule masks found under {root}")
    return samples


def split_by_subject(samples: list[HumanSegmentationSample], val_fraction: float,
                     seed: int) -> tuple[list[HumanSegmentationSample], list[HumanSegmentationSample]]:
    subjects = sorted({sample.subject_id for sample in samples})
    if len(subjects) < 2:
        raise ValueError("Subject-level split requires at least two subjects")
    random.Random(seed).shuffle(subjects)
    val_count = min(len(subjects) - 1, max(1, round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:val_count])
    train = [sample for sample in samples if sample.subject_id not in val_subjects]
    validation = [sample for sample in samples if sample.subject_id in val_subjects]
    if {x.subject_id for x in train} & {x.subject_id for x in validation}:
        raise RuntimeError("Subject leakage detected in downstream split")
    return train, validation


def audit_ssl_overlap(args, downstream_train: list[HumanSegmentationSample],
                      downstream_val: list[HumanSegmentationSample]) -> dict:
    config_path = args.ssl_reference_config.expanduser().resolve()
    if not config_path.is_file():
        return {"status": "unknown", "reason": f"SSL config not found: {config_path}"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ssl_root = Path(config.get("human2_root", args.data_root))
    ssl_samples = discover_ssl_samples({"human2": ssl_root})
    ssl_train, ssl_val = split_ssl_samples(
        ssl_samples, float(config.get("val_fraction", 0.1)), int(config.get("seed", 0)),
        config.get("max_images"))
    ssl_train_subjects = {sample.split_group for sample in ssl_train}
    ssl_val_subjects = {sample.split_group for sample in ssl_val}
    train_subjects = {sample.subject_id for sample in downstream_train}
    val_subjects = {sample.subject_id for sample in downstream_val}
    return {
        "status": "overlap_detected" if val_subjects & ssl_train_subjects else "no_train_overlap",
        "ssl_reference_config": str(config_path),
        "ssl_train_subjects": len(ssl_train_subjects),
        "ssl_validation_subjects": len(ssl_val_subjects),
        "downstream_train_subjects": len(train_subjects),
        "downstream_validation_subjects": len(val_subjects),
        "downstream_train_overlap_ssl_train": len(train_subjects & ssl_train_subjects),
        "downstream_validation_overlap_ssl_train": len(val_subjects & ssl_train_subjects),
        "downstream_validation_overlap_ssl_validation": len(val_subjects & ssl_val_subjects),
    }


def _checkpoint_state(path: Path, encoder_init: str) -> tuple[dict[str, Tensor], dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Human MAE checkpoint must contain encoder-only 'state_dict'")
    expected_adaptation = {"human_mae": "human_kidney_ultrasound_mae",
                           "human_dino": "human_kidney_ultrasound_dino",
                           "human_barlow": "human_kidney_ultrasound_barlow"}[encoder_init]
    if payload.get("adaptation") not in (None, expected_adaptation):
        raise ValueError(f"Unexpected checkpoint adaptation: {payload.get('adaptation')!r}")
    return payload["state_dict"], payload


def load_encoder(args: argparse.Namespace):
    encoder = get_encoder("vit_b16_imagenet", pretrained=True)
    summary = {"encoder_init": "imagenet", "checkpoint": None, "matched_parameters": 0,
               "missing_keys": [], "unexpected_keys": [], "shape_mismatch": []}
    if args.encoder_init in ("human_mae", "human_dino", "human_barlow"):
        state, payload = _checkpoint_state(args.encoder_checkpoint, args.encoder_init)
        expected = encoder.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        shape_mismatch = sorted(key for key in set(expected) & set(state)
                                if expected[key].shape != state[key].shape)
        matched = sorted(key for key in set(expected) & set(state)
                         if expected[key].shape == state[key].shape)
        matched_parameters = sum(expected[key].numel() for key in matched)
        print(f"Human {args.encoder_init.removeprefix('human_').upper()} encoder compatibility")
        print(f"  matched parameters: {matched_parameters:,}")
        print(f"  missing: {len(missing)}")
        print(f"  unexpected: {len(unexpected)}")
        print(f"  shape mismatch: {len(shape_mismatch)}")
        if missing or shape_mismatch or unexpected:
            raise RuntimeError(f"Incompatible encoder checkpoint: missing={missing}, "
                               f"unexpected={unexpected}, shape_mismatch={shape_mismatch}")
        encoder.load_state_dict(state, strict=True)
        summary = {"encoder_init": args.encoder_init,
                   "checkpoint": str(args.encoder_checkpoint.resolve()),
                   "checkpoint_epoch": payload.get("epoch"),
                   "matched_parameters": matched_parameters,
                   "missing_keys": missing, "unexpected_keys": unexpected,
                   "shape_mismatch": shape_mismatch,
                   "mae_decoder_weights_loaded": False,
                   "dino_teacher_or_head_weights_loaded": False}
    if args.transfer == "frozen":
        encoder.freeze()
    else:
        encoder.unfreeze()
    return encoder, summary


def make_loaders(args, encoder, train_samples, val_samples):
    config = encoder.preprocess
    train = HumanSegmentationDataset(
        train_samples, args.data_root,
        PairedTransform(config.image_size, config.mean, config.std, training=True))
    validation = HumanSegmentationDataset(
        val_samples, args.data_root,
        PairedTransform(config.image_size, config.mean, config.std, training=False))
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0,
                  worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(args.seed)
    return (DataLoader(train, shuffle=True, generator=generator, **common),
            DataLoader(validation, shuffle=False, **common))


def dice_loss(logits: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = logits.softmax(1)
    foreground = (targets == 1).float()
    intersection = (probabilities[:, 1] * foreground).sum()
    denominator = probabilities[:, 1].sum() + foreground.sum()
    return 1 - (2 * intersection + epsilon) / (denominator + epsilon)


def _metrics(confusion: Tensor) -> dict[str, float]:
    matrix = confusion.double()
    true_positive = matrix.diag()
    false_positive = matrix.sum(0) - true_positive
    false_negative = matrix.sum(1) - true_positive
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1e-12)
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1e-12)
    return {"kidney_dice": float(dice[1]), "kidney_iou": float(iou[1]),
            "mean_dice": float(dice[1]), "background_dice": float(dice[0]),
            "background_iou": float(iou[0])}


def run_epoch(model, loader, device, scaler, amp_enabled, optimizer=None):
    training = optimizer is not None
    model.train(training)
    confusion = torch.zeros(2, 2, dtype=torch.long)
    loss_sum, count = 0.0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for images, targets, _subjects in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets) + dice_loss(logits, targets)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            predictions = logits.argmax(1)
            indices = targets.flatten().cpu() * 2 + predictions.flatten().cpu()
            confusion += torch.bincount(indices, minlength=4).reshape(2, 2)
            loss_sum += float(loss) * images.shape[0]
            count += images.shape[0]
    metrics = _metrics(confusion)
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics


def _write_subjects(path: Path, samples: list[HumanSegmentationSample]) -> None:
    path.write_text("\n".join(sorted({sample.subject_id for sample in samples})) + "\n",
                    encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    samples = discover_samples(args.data_root)
    train_samples, val_samples = split_by_subject(samples, args.val_fraction, args.split_seed)
    train_subjects = sorted({sample.subject_id for sample in train_samples})
    val_subjects = sorted({sample.subject_id for sample in val_samples})
    overlap_audit = audit_ssl_overlap(args, train_samples, val_samples)
    warning = ("LEAKAGE LIMITATION: the existing Human SSL run did not reserve an SSL-unseen "
               "downstream test set. This is a representation utility feasibility check, not a "
               "strictly held-out evaluation. SSL overlap audit: " + json.dumps(overlap_audit))
    warnings.warn(warning)
    print(warning)
    print(f"Human2 split: train={len(train_subjects)} subjects/{len(train_samples)} images, "
          f"validation={len(val_subjects)} subjects/{len(val_samples)} images")
    encoder, load_summary = load_encoder(args)
    model = HumanBinarySegmenter(encoder, args.transfer == "frozen")
    run_dir = (args.run_dir if args.run_dir is not None else
               args.output_dir / "human2_segmentation" / args.encoder_init / args.transfer /
               f"seed_{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_subjects(run_dir / "train_subjects.txt", train_samples)
    _write_subjects(run_dir / "val_subjects.txt", val_samples)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key != "dry_run"}
    config.update({"task": "binary_kidney_segmentation",
                   "mask_definition": "union of available Human2 reviewer capsule masks",
                   "train_subjects": train_subjects, "val_subjects": val_subjects,
                   "train_images": len(train_samples), "val_images": len(val_samples),
                   "encoder_load_summary": load_summary,
                   "ssl_subject_overlap_audit": overlap_audit,
                   "ssl_downstream_leakage_warning": warning})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    encoder_trainable = sum(parameter.numel() for parameter in model.encoder.parameters()
                            if parameter.requires_grad)
    head_trainable = sum(parameter.numel() for parameter in model.decoder.parameters()
                         if parameter.requires_grad)
    print(f"Human dataset: Human2 OpenKidney binary capsule segmentation")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Transfer mode: {args.transfer}")
    print(f"Head trainable params: {head_trainable:,}")
    print(f"Encoder trainable params: {encoder_trainable:,}")
    if args.dry_run:
        print(f"Dry run complete: {run_dir.resolve()}")
        return
    train_loader, val_loader = make_loaders(args, encoder, train_samples, val_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_score, metric_rows = -float("inf"), []
    writer = SummaryWriter(run_dir / "tensorboard")
    try:
        for epoch in range(args.epochs):
            train_metrics = run_epoch(model, train_loader, device, scaler, amp_enabled, optimizer)
            validation_metrics = run_epoch(model, val_loader, device, scaler, amp_enabled)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            for phase, metrics in (("train", train_metrics),
                                   ("validation", validation_metrics)):
                metric_rows.append({"epoch": epoch, "phase": phase,
                                    "learning_rate": current_lr, **metrics})
                for name, value in metrics.items():
                    writer.add_scalar(f"{phase}/{name}", value, epoch)
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            is_best = validation_metrics["mean_dice"] > best_score
            if is_best:
                best_score = validation_metrics["mean_dice"]
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler,
                            epoch, best_score, args)
            if is_best:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler,
                                epoch, best_score, args)
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
                csv_writer.writeheader(); csv_writer.writerows(metric_rows)
            writer.flush()
            print(f"epoch {epoch + 1}/{args.epochs} lr={current_lr:.6g} "
                  f"val_dice={validation_metrics['mean_dice']:.4f} "
                  f"val_iou={validation_metrics['kidney_iou']:.4f}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.classification.model import ENCODER_NAMES, parameter_report
from src.classification.training_utils import save_checkpoint, set_seed
from src.detection.data import build_detection_loaders, detection_split
from src.detection.model import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat kidney bounding-box regression benchmark")
    parser.add_argument("--task", choices=("detection",), default="detection")
    parser.add_argument("--encoder", choices=tuple(ENCODER_NAMES), required=True)
    parser.add_argument("--encoder-init", choices=("imagenet", "human_mae", "human_dino"))
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--transfer", choices=(
        "scratch", "frozen", "partial", "full", "lora", "adapter"), required=True)
    parser.add_argument("--data-root", default="data/cat_dataset")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partial-blocks", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def _xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(1)
    return torch.stack((cx - width / 2, cy - height / 2,
                        cx + width / 2, cy + height / 2), dim=1).clamp(0, 1)


def _iou(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred, target = _xyxy(predictions), _xyxy(targets)
    left_top = torch.maximum(pred[:, :2], target[:, :2])
    right_bottom = torch.minimum(pred[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp_min(0).prod(1)
    pred_area = (pred[:, 2:] - pred[:, :2]).prod(1)
    target_area = (target[:, 2:] - target[:, :2]).prod(1)
    return intersection / (pred_area + target_area - intersection).clamp_min(1e-8)


def detection_epoch(model, loader, device, criterion, scaler, amp_enabled, optimizer=None,
                    collect_previews: int = 0):
    training = optimizer is not None
    model.train(training)
    losses, predictions_all, targets_all, previews = [], [], [], []
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, targets, keys in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(images)
                loss = criterion(predictions, targets)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.extend([float(loss)] * images.shape[0])
            predictions_all.append(predictions.detach().cpu())
            targets_all.append(targets.detach().cpu())
            needed = collect_previews - len(previews)
            if needed > 0:
                previews.extend(zip(images[:needed].detach().cpu(), targets[:needed].detach().cpu(),
                                    predictions[:needed].detach().cpu(), keys[:needed]))
    predictions = torch.cat(predictions_all)
    targets = torch.cat(targets_all)
    ious = _iou(predictions, targets)
    center_error = torch.linalg.vector_norm(predictions[:, :2] - targets[:, :2], dim=1)
    metrics = {"loss": float(np.mean(losses)), "mean_iou": float(ious.mean()),
               "median_iou": float(ious.quantile(0.5)),
               "center_error": float(center_error.mean()),
               "width_error": float((predictions[:, 2] - targets[:, 2]).abs().mean()),
               "height_error": float((predictions[:, 3] - targets[:, 3]).abs().mean())}
    return metrics, previews


def _save_overlay(previews, mean, std, path: Path) -> None:
    size = previews[0][0].shape[-1]
    canvas = Image.new("RGB", (size * len(previews), size + 28), "white")
    mean_tensor, std_tensor = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    for index, (tensor, target, prediction, key) in enumerate(previews):
        array = ((tensor * std_tensor + mean_tensor).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        image = Image.fromarray(array, mode="RGB")
        draw = ImageDraw.Draw(image)
        for box, color in ((_xyxy(target[None])[0], (255, 50, 50)),
                           (_xyxy(prediction[None])[0], (50, 255, 50))):
            draw.rectangle(tuple(float(value) * size for value in box), outline=color, width=3)
        canvas.paste(image, (index * size, 0))
        ImageDraw.Draw(canvas).text((index * size + 4, size + 7), str(key), fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.encoder == "vit_b16" and args.transfer != "scratch":
        args.encoder_init = args.encoder_init or "imagenet"
    elif args.encoder_init is not None or args.encoder_checkpoint is not None:
        raise ValueError("--encoder-init/--encoder-checkpoint currently apply only to pretrained vit_b16.")
    else:
        args.encoder_init = "random" if args.transfer == "scratch" else "native"
    human_ssl_init = args.encoder_init in ("human_mae", "human_dino")
    if human_ssl_init and not args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is required with Human SSL initialization.")
    if not human_ssl_init and args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is only valid with Human SSL initialization.")
    set_seed(args.seed)
    train_subjects, val_subjects, issues = detection_split(
        args.data_root, args.num_folds, args.fold, args.split_seed)
    model = build_detector(args.encoder, args.transfer, args.partial_blocks,
                           args.dropout, args.checkpoint_path, args.lora_r,
                           args.lora_alpha, args.lora_dropout, args.adapter_dim,
                           args.adapter_dropout, args.encoder_init,
                           args.encoder_checkpoint)
    counts = parameter_report(model, args.transfer, args.partial_blocks)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if args.transfer == "lora":
        application = model.encoder.lora_application
        counts.update({
            "lora_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and "lora_" in n),
            "head_trainable_parameters": sum(p.numel() for p in model.head.parameters()
                                               if p.requires_grad),
            "lora_rank": application.rank, "lora_alpha": application.alpha,
            "lora_dropout": application.dropout, "lora_targets": list(application.targets),
        })
    elif args.transfer == "adapter":
        application = model.encoder.adapter_application
        counts.update({
            "adapter_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                  if p.requires_grad and "adapter_modules" in n),
            "head_trainable_parameters": sum(p.numel() for p in model.head.parameters()
                                               if p.requires_grad),
            "trainable_ratio_percent": counts["trainable_ratio"] * 100,
            "adapter_dim": application.adapter_dim,
            "adapter_dropout": application.dropout,
            "adapter_targets": list(application.targets),
            "trainable_encoder_blocks": len(application.targets),
        })
    load_summary = getattr(model.encoder, "initialization_summary", {
        "encoder_init": args.encoder_init, "checkpoint": None, "human_ssl_method": None})
    run_dir = (args.output_dir / "detection" / args.encoder / args.transfer /
               f"fold_{args.fold}" / f"seed_{args.seed}")
    if human_ssl_init:
        init_directory = f"init_{args.encoder_init}"
        hybrid_blocks = load_summary.get("hybrid_ssl_blocks")
        if load_summary.get("hybrid") and hybrid_blocks:
            init_directory += "_blocks_" + "_".join(str(index) for index in hybrid_blocks)
        run_dir = run_dir / init_directory
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"human_ssl_method": ("mae" if args.encoder_init == "human_mae" else
                                        "dino" if args.encoder_init == "human_dino" else None),
                   "encoder_load_summary": load_summary,
                   "train_subjects": [str(item.directory) for item in train_subjects],
                   "val_subjects": [str(item.directory) for item in val_subjects], "data_issues": issues})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    (run_dir / "trainable_parameters.json").write_text(
        json.dumps(trainable_names, indent=2), encoding="utf-8")
    if human_ssl_init:
        (run_dir / "encoder_load_summary.json").write_text(
            json.dumps(load_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("encoder initialization summary: " + json.dumps(load_summary, ensure_ascii=False))
    for key, value in counts.items(): print(f"{key}: {value}")
    if args.transfer in ("lora", "adapter"):
        print("trainable parameter names:\n  " + "\n  ".join(trainable_names))
    train_loader, val_loader = build_detection_loaders(
        args.data_root, train_subjects, val_subjects, model.encoder.preprocess,
        args.batch_size, args.num_workers, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    writer = SummaryWriter(run_dir / "tensorboard")
    best_score = -float("inf")
    try:
        progress = tqdm(range(args.epochs), desc=f"detection fold{args.fold} seed{args.seed}", unit="epoch")
        for epoch in progress:
            train_metrics, _ = detection_epoch(
                model, train_loader, device, criterion, scaler, amp_enabled, optimizer)
            metrics, previews = detection_epoch(
                model, val_loader, device, criterion, scaler, amp_enabled, None, 5)
            for phase, values in (("train", train_metrics), ("validation", metrics)):
                for name, value in values.items(): writer.add_scalar(f"{phase}/{name}", value, epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            scheduler.step()
            is_best = metrics["mean_iou"] > best_score
            if is_best:
                best_score = metrics["mean_iou"]
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_score, args)
            if is_best:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_score, args)
                (run_dir / "validation_metrics.json").write_text(
                    json.dumps({"epoch": epoch, **metrics}, indent=2), encoding="utf-8")
                _save_overlay(previews, model.encoder.preprocess.mean, model.encoder.preprocess.std,
                              run_dir / "validation_bbox_overlay.png")
            writer.flush()
            progress.set_postfix(lr=f"{current_lr:.2e}", val_iou=f"{metrics['mean_iou']:.4f}",
                                 val_loss=f"{metrics['loss']:.4f}", best=is_best)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

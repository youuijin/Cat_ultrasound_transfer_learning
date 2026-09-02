from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint_utils import portable_config
from src.classification.training_utils import set_seed
from src.human_ssl.data import (
    DEFAULT_ROOTS, build_dino_loaders, dataset_counts, discover_ssl_samples, split_ssl_samples,
)
from src.human_ssl.dino import ENCODER_REGISTRY, DINOLoss, HumanDINO, cosine_teacher_momentum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human kidney ultrasound DINO adaptation")
    parser.add_argument("--method", choices=("dino",), default="dino")
    parser.add_argument("--encoder", choices=tuple(ENCODER_REGISTRY), default="vit_b16")
    parser.add_argument("--checkpoint-path", type=str,
                        help="Optional official checkpoint override (primarily USFM).")
    parser.add_argument("--human1-root", type=Path, default=DEFAULT_ROOTS["human1"])
    parser.add_argument("--human2-root", type=Path, default=DEFAULT_ROOTS["human2"])
    parser.add_argument("--human3-root", type=Path, default=DEFAULT_ROOTS["human3"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--head-hidden-dim", type=int, default=1024)
    parser.add_argument("--bottleneck-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--student-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-temperature", type=float, default=0.04)
    parser.add_argument("--center-momentum", type=float, default=0.9)
    parser.add_argument("--teacher-momentum", type=float, default=0.996)
    parser.add_argument("--gradient-clip", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-images", type=int, help="Optional smoke/debug subset size.")
    parser.add_argument("--match-mae-config", type=Path,
                        default=Path("checkpoints/human_mae_vit_b16/config.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--save-encoder-epochs", nargs="+", type=int,
        help="Optional zero-based epochs at which to save periodic student encoder checkpoints.",
    )
    args = parser.parse_args()
    if args.save_encoder_epochs is not None:
        if len(args.save_encoder_epochs) != len(set(args.save_encoder_epochs)):
            parser.error("--save-encoder-epochs must not contain duplicate epochs")
        invalid = [epoch for epoch in args.save_encoder_epochs
                   if epoch < 0 or epoch >= args.epochs]
        if invalid:
            parser.error("--save-encoder-epochs values must satisfy "
                         f"0 <= epoch < {args.epochs}; invalid: {invalid}")
    return args


def _epoch(model, criterion, loader, device, scaler, amp_enabled, optimizer=None,
           global_step: int = 0, total_steps: int = 1, base_momentum: float = 0.996,
           gradient_clip: float = 3.0):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "embedding_std": 0.0, "teacher_output_entropy": 0.0,
              "prototype_usage_entropy": 0.0, "max_prototype_probability": 0.0,
              "teacher_momentum": 0.0}
    count = steps = 0
    with torch.enable_grad() if training else torch.no_grad():
        for view1, view2, _dataset_ids in loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                student1, embedding1 = model.student(view1)
                student2, embedding2 = model.student(view2)
                with torch.no_grad():
                    teacher1, _ = model.teacher(view1)
                    teacher2, _ = model.teacher(view2)
                loss, diagnostics = criterion(
                    (student1, student2), (teacher1, teacher2), update_center=training)
            momentum = cosine_teacher_momentum(base_momentum, global_step + steps, total_steps)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.student.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                model.update_teacher(momentum)
                steps += 1
            embedding = torch.cat((embedding1.detach(), embedding2.detach()))
            batch = view1.shape[0]
            totals["loss"] += float(loss) * batch
            totals["embedding_std"] += float(embedding.std(0).mean()) * batch
            for key, value in diagnostics.items():
                totals[key] += value * batch
            totals["teacher_momentum"] += momentum * batch
            count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}, steps, count


def _args_config(args) -> dict:
    values = vars(args).copy()
    if values.get("save_encoder_epochs") is None:
        values.pop("save_encoder_epochs", None)
    return portable_config(values)


def _save_encoder(path: Path, model: HumanDINO, args, epoch: int,
                  validation_loss: float, optimizer_steps: int,
                  images_seen: int, checkpoint_type: str | None = None) -> None:
    encoder = model.student.encoder
    payload = {
        "format": "feline_transfer_learning.vision_encoder.v1",
        "encoder_name": model.registry_name,
        "initialization": encoder.pretraining,
        "adaptation": "human_kidney_ultrasound_dino",
        "human_ssl_method": "dino",
        "epoch": epoch,
        "validation_dino_loss": validation_loss,
        "optimizer_steps": optimizer_steps,
        "total_train_images_seen": images_seen,
        "state_dict": encoder.state_dict(),
        "model_state_dict": encoder.model.state_dict(),
        "config": _args_config(args),
    }
    if checkpoint_type is not None:
        payload["checkpoint_type"] = checkpoint_type
    torch.save(payload, path)


def _save_views(loader, path: Path) -> None:
    view1, view2, _ = next(iter(loader))
    view1, view2 = view1[:5], view2[:5]
    size, caption = view1.shape[-1], 24
    canvas = Image.new("RGB", (size * 2, (size + caption) * len(view1)), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(len(view1)):
        for column, tensor in enumerate((view1[row], view2[row])):
            array = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array, mode="RGB"),
                         (column * size, row * (size + caption)))
        y = row * (size + caption) + size + 4
        draw.text((4, y), "Global view 1", fill="black")
        draw.text((size + 4, y), "Global view 2", fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than epochs.")
    set_seed(args.seed)
    if args.output_dir is None:
        args.output_dir = Path(f"checkpoints/human_dino_{args.encoder}")
    roots = {"human1": args.human1_root, "human2": args.human2_root,
             "human3": args.human3_root}
    samples = discover_ssl_samples(roots)
    train_samples, val_samples = split_ssl_samples(
        samples, args.val_fraction, args.seed, args.max_images)
    reference = json.loads(args.match_mae_config.read_text(encoding="utf-8"))
    matched_fields = ("encoder", "val_fraction", "batch_size", "epochs", "lr", "weight_decay",
                      "warmup_epochs", "seed")
    mismatches = {}
    if args.max_images is None:
        mismatches.update({key: (reference.get(key), getattr(args, key)) for key in matched_fields
                           if reference.get(key) != getattr(args, key)})
        if len(train_samples) != reference.get("train_images") or len(val_samples) != reference.get(
                "validation_images"):
            mismatches["split_counts"] = (
                (reference.get("train_images"), reference.get("validation_images")),
                (len(train_samples), len(val_samples)))
        for key, current in (("dataset_counts", dataset_counts(samples)),
                             ("train_dataset_counts", dataset_counts(train_samples)),
                             ("validation_dataset_counts", dataset_counts(val_samples))):
            if reference.get(key) != current:
                mismatches[key] = (reference.get(key), current)
    if mismatches:
        raise ValueError(f"DINO budget does not match Human-MAE config: {mismatches}")

    model = HumanDINO(args.encoder, args.head_hidden_dim, args.bottleneck_dim,
                      args.output_dim, args.checkpoint_path)
    train_loader, val_loader = build_dino_loaders(
        train_samples, val_samples, model.student.encoder.preprocess.image_size,
        args.batch_size, args.num_workers, args.seed)
    total_steps = len(train_loader) * args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = DINOLoss(args.output_dim, args.student_temperature,
                         args.teacher_temperature, args.center_momentum).to(device)
    optimizer = torch.optim.AdamW(model.student.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_factor(epoch: int) -> float:
        if args.warmup_epochs and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs - 1)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in _args_config(args).items()}
    config.update({
        "initialization_source": model.student.encoder.pretraining,
        "dataset_counts": dataset_counts(samples),
        "train_dataset_counts": dataset_counts(train_samples),
        "validation_dataset_counts": dataset_counts(val_samples),
        "train_images": len(train_samples), "validation_images": len(val_samples),
        "planned_optimizer_steps": total_steps,
        "planned_source_images_seen": len(train_samples) * args.epochs,
        "planned_augmented_views_seen": 2 * len(train_samples) * args.epochs,
        "augmentation": "two global crops: scale .80-1.0, hflip .5, mild intensity/contrast/gamma",
        "checkpoint_selection": "minimum validation DINO loss",
        "mae_reference_config": str(args.match_mae_config),
        "budget_matched_to_mae": args.max_images is None,
    })
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_views(val_loader, args.output_dir / "two_global_views.png")
    print(f"images per dataset: {dataset_counts(samples)}")
    print(f"train / validation: {len(train_samples):,} / {len(val_samples):,}")
    print(f"optimizer steps / source images seen: {total_steps:,} / "
          f"{len(train_samples) * args.epochs:,}")
    print(f"student/teacher initialization: {model.registry_name} exact copies")
    if args.save_encoder_epochs is not None:
        print(f"Periodic encoder checkpoints: {sorted(args.save_encoder_epochs)}")

    writer = SummaryWriter(args.output_dir / "tensorboard")
    best_loss, global_step, images_seen = float("inf"), 0, 0
    periodic_epochs = set(args.save_encoder_epochs or ())
    try:
        for epoch in range(args.epochs):
            train_metrics, steps, epoch_images = _epoch(
                model, criterion, train_loader, device, scaler, amp_enabled, optimizer,
                global_step, total_steps, args.teacher_momentum, args.gradient_clip)
            global_step += steps
            images_seen += epoch_images
            val_metrics, _, _ = _epoch(
                model, criterion, val_loader, device, scaler, amp_enabled, None,
                global_step, total_steps, args.teacher_momentum, args.gradient_clip)
            current_lr = optimizer.param_groups[0]["lr"]
            for phase, metrics in (("train", train_metrics), ("validation", val_metrics)):
                for key, value in metrics.items():
                    writer.add_scalar(f"dino/{phase}_{key}", value, epoch)
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            writer.add_scalar("train/optimizer_steps", global_step, epoch)
            writer.add_scalar("train/source_images_seen", images_seen, epoch)
            scheduler.step()
            _save_encoder(args.output_dir / "last_encoder.pt", model, args, epoch,
                          val_metrics["loss"], global_step, images_seen)
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                _save_encoder(args.output_dir / "best_encoder.pt", model, args, epoch,
                              best_loss, global_step, images_seen)
                (args.output_dir / "validation_metrics.json").write_text(json.dumps({
                    "epoch": epoch, "validation_dino_loss": best_loss,
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"validation_{key}": value for key, value in val_metrics.items()},
                }, indent=2), encoding="utf-8")
            if epoch in periodic_epochs:
                periodic_path = args.output_dir / f"epoch_{epoch:03d}_encoder.pt"
                _save_encoder(periodic_path, model, args, epoch, val_metrics["loss"],
                              global_step, images_seen, checkpoint_type="periodic")
                print(f"[checkpoint] saved periodic DINO student encoder: {periodic_path.name}")
            config.update({"completed_epochs": epoch + 1, "optimizer_steps": global_step,
                           "total_source_images_seen": images_seen,
                           "best_validation_dino_loss": best_loss})
            (args.output_dir / "config.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            writer.flush()
            print(f"epoch {epoch + 1}/{args.epochs} lr={current_lr:.6g} "
                  f"loss={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
                  f"std={train_metrics['embedding_std']:.4f} "
                  f"usage_H={train_metrics['prototype_usage_entropy']:.4f} "
                  f"m={train_metrics['teacher_momentum']:.6f}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.classification.model import ENCODER_NAMES, parameter_report
from src.classification.training_utils import save_checkpoint, set_seed
from src.segmentation.data import build_segmentation_loaders
from src.segmentation.model import build_segmenter

CLASS_NAMES = ("background", "cortex", "medulla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat cortex/medulla segmentation benchmark")
    parser.add_argument("--task", choices=("segmentation",), default="segmentation")
    parser.add_argument("--encoder", choices=tuple(ENCODER_NAMES), required=True)
    parser.add_argument("--encoder-init", choices=("imagenet", "human_mae", "human_dino", "human_barlow"))
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--transfer", choices=(
        "scratch", "frozen", "partial", "full", "lora", "adapter"), required=True)
    parser.add_argument("--data-root", default="data/cat_dataset")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of fold-training subjects (default: 1.0).")
    parser.add_argument("--train-fraction-seed", type=int, default=None,
                        help="Nested-subset seed (default: --split-seed).")
    parser.add_argument("--train-subject-list", type=Path, default=None,
                        help="Text file of subject IDs selected from the original fold-training split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-ssl-seed", type=int, default=None,
                        help="Metadata only: seed used to create a fixed Human SSL encoder checkpoint.")
    parser.add_argument("--partial-blocks", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--exact-run-dir", type=Path, default=None,
                        help="Write directly to this run directory instead of the standard nested layout.")
    return parser.parse_args()


def dice_loss(logits: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    probabilities = logits.softmax(1)
    one_hot = nn.functional.one_hot(targets, 3).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    return 1 - ((2 * intersection + epsilon) / (denominator + epsilon))[1:].mean()


def _metrics(confusion: Tensor) -> dict[str, float]:
    matrix = confusion.double()
    true_positive = matrix.diag()
    false_positive = matrix.sum(0) - true_positive
    false_negative = matrix.sum(1) - true_positive
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1e-12)
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1e-12)
    return {"cortex_dice": float(dice[1]), "medulla_dice": float(dice[2]),
            "mean_foreground_dice": float(dice[1:].mean()),
            "cortex_iou": float(iou[1]), "medulla_iou": float(iou[2]),
            "mean_foreground_iou": float(iou[1:].mean()),
            "background_dice": float(dice[0]), "background_iou": float(iou[0])}


def segmentation_epoch(model, loader, device, scaler, amp_enabled, optimizer=None, preview_count=0,
                       collect_subject_metrics=False):
    training = optimizer is not None
    model.train(training)
    # ``model.train()`` also toggles the encoder.  A frozen representation is
    # intentionally kept in inference mode even while its decoder is trained.
    if training and all(not parameter.requires_grad for parameter in model.encoder.parameters()):
        model.encoder.eval()
        assert not model.encoder.training, "Frozen encoder must remain in eval mode"
    confusion = torch.zeros(3, 3, dtype=torch.long)
    loss_sum, count, previews = 0.0, 0, []
    subject_confusions = {} if collect_subject_metrics else None
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, targets, keys in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, targets) + dice_loss(logits, targets)
            if training:
                scaler.scale(loss).backward()
                if all(not parameter.requires_grad for parameter in model.encoder.parameters()):
                    assert all(parameter.grad is None for parameter in model.encoder.parameters()), (
                        "Frozen encoder received a gradient")
                scaler.step(optimizer); scaler.update()
            predictions = logits.argmax(1)
            indices = targets.flatten().cpu() * 3 + predictions.flatten().cpu()
            confusion += torch.bincount(indices, minlength=9).reshape(3, 3)
            if subject_confusions is not None:
                for index, key in enumerate(keys):
                    subject_id, _ = str(key).rsplit("_", 1)
                    subject_indices = (targets[index].flatten().cpu() * 3 +
                                       predictions[index].flatten().cpu())
                    subject_confusions.setdefault(subject_id, torch.zeros(3, 3, dtype=torch.long))
                    subject_confusions[subject_id] += torch.bincount(
                        subject_indices, minlength=9).reshape(3, 3)
            loss_sum += float(loss) * images.shape[0]; count += images.shape[0]
            needed = preview_count - len(previews)
            if needed > 0:
                previews.extend(zip(images[:needed].detach().cpu(), targets[:needed].detach().cpu(),
                                    predictions[:needed].detach().cpu(), keys[:needed]))
    metrics = _metrics(confusion)
    metrics["loss"] = loss_sum / max(count, 1)
    subject_metrics = ([] if subject_confusions is None else [
        {"subject_id": subject_id, **_metrics(subject_confusion)}
        for subject_id, subject_confusion in sorted(subject_confusions.items())
    ])
    return metrics, previews, subject_metrics


def _colorize(mask: Tensor) -> Image.Image:
    colors = torch.tensor(((0, 0, 0), (255, 70, 70), (50, 210, 255)), dtype=torch.uint8)
    return Image.fromarray(colors[mask.long()].numpy(), mode="RGB")


def save_preview(previews, mean, std, path: Path) -> None:
    size, caption = previews[0][0].shape[-1], 24
    canvas = Image.new("RGB", (size * 3, (size + caption) * len(previews)), "white")
    mean_tensor, std_tensor = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    draw = ImageDraw.Draw(canvas)
    for row, (tensor, target, prediction, key) in enumerate(previews):
        image = Image.fromarray(((tensor * std_tensor + mean_tensor).clamp(0, 1) * 255)
                                .byte().permute(1, 2, 0).numpy(), mode="RGB")
        y = row * (size + caption)
        canvas.paste(image, (0, y)); canvas.paste(_colorize(target), (size, y))
        canvas.paste(_colorize(prediction), (size * 2, y))
        draw.text((4, y + size + 5), f"{key} | input", fill="black")
        draw.text((size + 4, y + size + 5), "GT", fill="black")
        draw.text((size * 2 + 4, y + size + 5), "prediction", fill="black")
    canvas.save(path)


def main() -> None:
    args = parse_args()
    using_train_fraction = any(
        token == "--train-fraction" or token.startswith("--train-fraction=")
        for token in sys.argv[1:]
    )
    label_efficiency = using_train_fraction or args.train_subject_list is not None
    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction must be in (0, 1].")
    if args.train_fraction != 1.0 and args.train_subject_list is not None:
        raise ValueError("--train-fraction cannot be combined with --train-subject-list.")
    fraction_seed = (args.split_seed if args.train_fraction_seed is None
                     else args.train_fraction_seed)
    if args.encoder == "vit_b16" and args.transfer != "scratch":
        args.encoder_init = args.encoder_init or "imagenet"
    elif args.encoder_init is not None or args.encoder_checkpoint is not None:
        raise ValueError("--encoder-init/--encoder-checkpoint currently apply only to pretrained vit_b16.")
    else:
        args.encoder_init = "random" if args.transfer == "scratch" else "native"
    human_ssl_init = args.encoder_init in ("human_mae", "human_dino", "human_barlow")
    if human_ssl_init and not args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is required with Human SSL initialization.")
    if not human_ssl_init and args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is only valid with Human SSL initialization.")
    set_seed(args.seed)
    model = build_segmenter(args.encoder, args.transfer, args.partial_blocks, args.checkpoint_path,
                            args.lora_r, args.lora_alpha, args.lora_dropout,
                            args.adapter_dim, args.adapter_dropout, args.encoder_init,
                            args.encoder_checkpoint)
    counts = parameter_report(model, args.transfer, args.partial_blocks)
    counts.update({
        "total_encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "trainable_encoder_parameters": sum(
            p.numel() for p in model.encoder.parameters() if p.requires_grad),
        "total_decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
        "trainable_decoder_parameters": sum(
            p.numel() for p in model.decoder.parameters() if p.requires_grad),
    })
    if args.transfer == "frozen":
        assert counts["trainable_encoder_parameters"] == 0, (
            "Frozen transfer requires trainable encoder parameters = 0")
        assert counts["trainable_decoder_parameters"] > 0, (
            "Frozen transfer requires trainable segmentation decoder parameters > 0")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if args.transfer == "lora":
        application = model.encoder.lora_application
        counts.update({
            "lora_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                               if p.requires_grad and "lora_" in n),
            "decoder_trainable_parameters": sum(p.numel() for p in model.decoder.parameters()
                                                  if p.requires_grad),
            "lora_rank": application.rank, "lora_alpha": application.alpha,
            "lora_dropout": application.dropout, "lora_targets": list(application.targets),
        })
    elif args.transfer == "adapter":
        application = model.encoder.adapter_application
        counts.update({
            "adapter_trainable_parameters": sum(p.numel() for n, p in model.named_parameters()
                                                  if p.requires_grad and "adapter_modules" in n),
            "decoder_trainable_parameters": sum(p.numel() for p in model.decoder.parameters()
                                                  if p.requires_grad),
            "trainable_ratio_percent": counts["trainable_ratio"] * 100,
            "adapter_dim": application.adapter_dim,
            "adapter_dropout": application.dropout,
            "adapter_targets": list(application.targets),
            "trainable_encoder_blocks": len(application.targets),
        })
    load_summary = getattr(model.encoder, "initialization_summary", {
        "encoder_init": args.encoder_init, "checkpoint": None, "human_ssl_method": None})
    if args.exact_run_dir is not None:
        run_dir = args.exact_run_dir
    else:
        run_root = args.output_dir / "label_efficiency" if label_efficiency else args.output_dir
        run_dir = (run_root / "segmentation" / args.encoder / args.transfer /
                   f"fold_{args.fold}" / f"seed_{args.seed}")
        if label_efficiency:
            fraction_label = f"{round(args.train_fraction * 100):03d}"
            run_dir = run_dir / f"train_fraction_{fraction_label}"
        if human_ssl_init or label_efficiency:
            init_directory = f"init_{args.encoder_init}"
            hybrid_blocks = load_summary.get("hybrid_ssl_blocks")
            interpolation_alpha = load_summary.get("interpolation_alpha")
            if load_summary.get("interpolation") and interpolation_alpha is not None:
                alpha_suffix = f"{round(float(interpolation_alpha) * 100):03d}"
                init_directory += f"_interp_a{alpha_suffix}"
            elif load_summary.get("hybrid") and hybrid_blocks:
                init_directory += "_blocks_" + "_".join(str(index) for index in hybrid_blocks)
            run_dir = run_dir / init_directory
    run_dir.mkdir(parents=True, exist_ok=True)
    (train_loader, val_loader, train_subjects, val_subjects, issues,
     subset_metadata) = build_segmentation_loaders(
        args.data_root, args.num_folds, args.fold, args.split_seed, model.encoder.preprocess,
        args.batch_size, args.num_workers, args.seed,
        train_fraction=args.train_fraction if using_train_fraction else None,
        train_fraction_seed=fraction_seed,
        train_subject_list=args.train_subject_list)
    subset_metadata.update({
        "train_fraction": args.train_fraction,
        "train_fraction_seed": fraction_seed,
        "train_subject_list": (None if args.train_subject_list is None else
                               str(args.train_subject_list.resolve())),
    })
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if not label_efficiency:
        config.pop("train_fraction", None)
        config.pop("train_fraction_seed", None)
    experiment_metadata = subset_metadata if label_efficiency else {}
    config.update({"human_ssl_method": ({"human_mae": "mae", "human_dino": "dino",
                                         "human_barlow": "barlow"}.get(args.encoder_init)),
                   "encoder_load_summary": load_summary,
                   "classes": CLASS_NAMES, "train_subjects": [str(x.directory) for x in train_subjects],
                   "val_subjects": [str(x.directory) for x in val_subjects], "data_issues": issues,
                   "train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset),
                   **experiment_metadata})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    (run_dir / "trainable_parameters.json").write_text(
        json.dumps(trainable_names, indent=2), encoding="utf-8")
    if human_ssl_init:
        (run_dir / "encoder_load_summary.json").write_text(
            json.dumps(load_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "train_subjects.txt").write_text(
        "\n".join(subject.subject_id for subject in train_subjects) + "\n", encoding="utf-8")
    (run_dir / "val_subjects.txt").write_text(
        "\n".join(subject.subject_id for subject in val_subjects) + "\n", encoding="utf-8")
    if label_efficiency:
        print("Label-efficiency experiment")
        print(f"Fold: {args.fold}")
        print(f"Train fraction: {args.train_fraction}")
        print(f"Full train subjects: {subset_metadata['num_train_subjects_full']}")
        print(f"Used train subjects: {subset_metadata['num_train_subjects_used']}")
        print(f"Full train images: {subset_metadata['num_train_images_full']}")
        print(f"Used train images: {subset_metadata['num_train_images_used']}")
        print(f"Validation subjects: {len(val_subjects)}")
        print(f"Subset seed: {fraction_seed}")
    print("encoder initialization summary: " + json.dumps(load_summary, ensure_ascii=False))
    for key, value in counts.items(): print(f"{key}: {value}")
    if args.transfer in ("lora", "adapter"):
        print("trainable parameter names:\n  " + "\n  ".join(trainable_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
    if args.transfer == "frozen":
        encoder_parameter_ids = {id(p) for p in model.encoder.parameters()}
        assert not any(id(p) in encoder_parameter_ids for p in optimizer_parameters), (
            "Frozen encoder parameter was included in the optimizer")
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    writer, best_score = SummaryWriter(run_dir / "tensorboard"), -float("inf")
    metric_rows = []
    try:
        progress = tqdm(range(args.epochs), desc=f"segmentation fold{args.fold} seed{args.seed}", unit="epoch")
        for epoch in progress:
            train_metrics, _, _ = segmentation_epoch(
                model, train_loader, device, scaler, amp_enabled, optimizer)
            metrics, previews, subject_metrics = segmentation_epoch(
                model, val_loader, device, scaler, amp_enabled, None, 5,
                collect_subject_metrics=True)
            for phase, values in (("train", train_metrics), ("validation", metrics)):
                for name, value in values.items(): writer.add_scalar(f"{phase}/{name}", value, epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", current_lr, epoch); scheduler.step()
            is_best = metrics["mean_foreground_dice"] > best_score
            if is_best: best_score = metrics["mean_foreground_dice"]
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_score, args)
            if is_best:
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_score, args)
                (run_dir / "validation_metrics.json").write_text(
                    json.dumps({"epoch": epoch, **metrics, **experiment_metadata}, indent=2),
                    encoding="utf-8")
                with (run_dir / "subject_dice.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer_subject = csv.DictWriter(
                        handle, fieldnames=("subject_id", "cortex_dice", "medulla_dice",
                                            "mean_foreground_dice", "cortex_iou", "medulla_iou",
                                            "mean_foreground_iou", "background_dice", "background_iou"))
                    writer_subject.writeheader()
                    writer_subject.writerows(subject_metrics)
                save_preview(previews, model.encoder.preprocess.mean, model.encoder.preprocess.std,
                             run_dir / "validation_segmentation_preview.png")
            metric_rows.append({"epoch": epoch,
                                **{f"train_{key}": value for key, value in train_metrics.items()},
                                **{f"validation_{key}": value for key, value in metrics.items()},
                                "learning_rate": current_lr, "is_best": is_best})
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
                csv_writer.writeheader(); csv_writer.writerows(metric_rows)
            writer.flush()
            progress.set_postfix(lr=f"{current_lr:.2e}",
                                 val_dice=f"{metrics['mean_foreground_dice']:.4f}",
                                 val_iou=f"{metrics['mean_foreground_iou']:.4f}",
                                 val_loss=f"{metrics['loss']:.4f}", best=is_best)
    finally: writer.close()
    subject_rows = list(csv.DictReader((run_dir / "subject_dice.csv").open(encoding="utf-8")))
    print("=" * 50)
    print(f"FOLD: {args.fold}")
    print(f"SEED: {args.seed}")
    print(f"ENCODER_FROZEN: {args.transfer == 'frozen'}")
    print(f"TRAINABLE_ENCODER_PARAMS: {counts['trainable_encoder_parameters']}")
    print("TEST_SUBJECTS: " + ", ".join(row["subject_id"] for row in subject_rows))
    print("SUBJECT_DICE:")
    for row in subject_rows:
        print(f"{row['subject_id']}: {float(row['mean_foreground_dice']):.6f}")
    print("MEAN_SUBJECT_DICE: " + format(
        sum(float(row["mean_foreground_dice"]) for row in subject_rows) / len(subject_rows), ".6f"))
    print("=" * 50)


if __name__ == "__main__": main()

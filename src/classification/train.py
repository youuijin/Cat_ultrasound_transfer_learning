from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.classification.data import build_loaders, split_subjects
from src.classification.model import ENCODER_NAMES, build_classifier, parameter_report
from src.classification.training_utils import (
    BalancedSoftmaxLoss, LDAMDRWLoss, classification_epoch, save_checkpoint, set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat classification transfer-learning benchmark")
    parser.add_argument("--task", choices=("classification",), default="classification")
    parser.add_argument("--encoder", choices=tuple(ENCODER_NAMES), required=True)
    parser.add_argument(
        "--encoder-init",
        choices=("imagenet", "human_mae", "human_dino"),
        default=None,
        help="Pretrained initialization source; Human SSL must match --encoder.",
    )
    parser.add_argument("--encoder-checkpoint", type=str)
    parser.add_argument("--transfer", choices=(
        "scratch", "frozen", "partial", "full", "lora", "adapter"), required=True)
    parser.add_argument("--data-root", type=str, default="data/cat_dataset")
    parser.add_argument("--classification-mode", choices=("binary", "four_class", "abnormal_subtype"),
                        default="four_class")
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
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--condition", choices=(
        "plain", "baseline", "weighted_ce", "strong_aug",
        "weighted_ce_strong_aug", "balanced_softmax", "ldam_drw",
        "classifier_retrain"),
        default="baseline")
    parser.add_argument("--drw-start-epoch", type=int, default=20)
    parser.add_argument("--classifier-retrain-epochs", type=int, default=20)
    parser.add_argument("--classifier-retrain-reset-head",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier-retrain-balance",
                        choices=("sampler", "weighted_ce"), default="sampler")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def _jsonable(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _log_metrics(writer: SummaryWriter, phase: str, metrics: dict[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{phase}/{name}", value, epoch)


def _confusion_table(confusion: torch.Tensor, class_names: list[str]) -> str:
    header = "|true / pred|" + "|".join(class_names) + "|"
    divider = "|---|" + "|".join("---:" for _ in class_names) + "|"
    rows = [f"|{name}|" + "|".join(str(int(value)) for value in row) + "|"
            for name, row in zip(class_names, confusion)]
    return "\n".join([header, divider, *rows])


def _run_stage(model, train_loader, val_loader, criterion, class_names, device, scaler,
               amp_enabled, optimizer, scheduler, epochs, args, run_dir, writer,
               stage: str = ""):
    best_score, best_metrics, best_confusion = -float("inf"), {}, None
    tag = f"{stage}/" if stage else ""
    file_prefix = f"{stage}_" if stage else ""
    progress = tqdm(range(epochs), desc=f"classification {stage or 'main'}", unit="epoch")
    for epoch in progress:
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        train_metrics, _ = classification_epoch(
            model, train_loader, device, criterion, class_names,
            scaler, amp_enabled, optimizer)
        metrics, confusion = classification_epoch(
            model, val_loader, device, criterion, class_names,
            scaler, amp_enabled, None)
        _log_metrics(writer, f"{tag}train", train_metrics, epoch)
        _log_metrics(writer, f"{tag}validation", metrics, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar(f"{tag}train/learning_rate", current_lr, epoch)
        writer.add_text(f"{tag}validation/confusion_matrix",
                        _confusion_table(confusion, class_names), epoch)
        scheduler.step()
        score = metrics["balanced_accuracy"]
        is_best = score > best_score
        if is_best:
            best_score, best_metrics, best_confusion = score, metrics, confusion.tolist()
        save_checkpoint(run_dir / f"{file_prefix}last.pt", model, optimizer, scheduler,
                        epoch, best_score, args)
        if is_best:
            save_checkpoint(run_dir / f"{file_prefix}best.pt", model, optimizer, scheduler,
                            epoch, best_score, args)
            result = {"epoch": epoch, **best_metrics, "confusion_matrix": best_confusion}
            (run_dir / f"{file_prefix}validation_metrics.json").write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        writer.flush()
        progress.set_postfix(lr=f"{current_lr:.2e}", balanced_acc=f"{score:.4f}",
                             macro_f1=f"{metrics['macro_f1']:.4f}", loss=f"{metrics['loss']:.4f}",
                             best=is_best)
    return best_score


def main() -> None:
    args = parse_args()
    if args.transfer == "scratch":
        if args.encoder_init is not None or args.encoder_checkpoint is not None:
            raise ValueError("--encoder-init/--encoder-checkpoint cannot be used with scratch.")
        args.encoder_init = "random"
    elif args.encoder_init is None:
        args.encoder_init = "imagenet" if args.encoder == "vit_b16" else "native"
    elif args.encoder_init == "imagenet" and args.encoder != "vit_b16":
        raise ValueError("--encoder-init imagenet is only valid with --encoder vit_b16.")
    if args.encoder_init in ("human_mae", "human_dino") and not args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is required with Human SSL initialization.")
    if args.encoder_init not in ("human_mae", "human_dino") and args.encoder_checkpoint:
        raise ValueError("--encoder-checkpoint is only valid with Human SSL initialization.")
    set_seed(args.seed)
    train_subjects, val_subjects, class_names, issues = split_subjects(
        args.data_root, args.classification_mode, args.num_folds, args.fold, args.split_seed
    )
    model = build_classifier(args.encoder, args.transfer, len(class_names), args.partial_blocks,
                             args.dropout, args.checkpoint_path, args.lora_r,
                             args.lora_alpha, args.lora_dropout, args.adapter_dim,
                             args.adapter_dropout, args.encoder_init,
                             args.encoder_checkpoint)
    counts = parameter_report(model, args.transfer, args.partial_blocks)
    trainable_names = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad]
    if args.transfer == "lora":
        application = model.encoder.lora_application
        counts.update({
            "lora_trainable_parameters": sum(
                parameter.numel() for name, parameter in model.named_parameters()
                if parameter.requires_grad and "lora_" in name),
            "head_trainable_parameters": sum(
                parameter.numel() for parameter in model.head.parameters()
                if parameter.requires_grad),
            "trainable_ratio_percent": counts["trainable_ratio"] * 100,
            "lora_rank": application.rank,
            "lora_alpha": application.alpha,
            "lora_dropout": application.dropout,
            "lora_targets": list(application.targets),
        })
    elif args.transfer == "adapter":
        application = model.encoder.adapter_application
        counts.update({
            "adapter_trainable_parameters": sum(
                parameter.numel() for name, parameter in model.named_parameters()
                if parameter.requires_grad and "adapter_modules" in name),
            "head_trainable_parameters": sum(
                parameter.numel() for parameter in model.head.parameters()
                if parameter.requires_grad),
            "trainable_ratio_percent": counts["trainable_ratio"] * 100,
            "adapter_dim": application.adapter_dim,
            "adapter_dropout": application.dropout,
            "adapter_targets": list(application.targets),
            "trainable_encoder_blocks": len(application.targets),
        })
    load_summary = getattr(model.encoder, "initialization_summary", {
        "encoder_init": args.encoder_init,
        "checkpoint": None,
        "human_ssl_method": None,
    })
    run_dir = (args.output_dir / "classification" / args.encoder / args.transfer /
               f"fold_{args.fold}" / f"seed_{args.seed}" / args.condition)
    if args.encoder_init in ("human_mae", "human_dino"):
        init_directory = f"init_{args.encoder_init}"
        hybrid_blocks = load_summary.get("hybrid_ssl_blocks")
        if load_summary.get("hybrid") and hybrid_blocks:
            block_suffix = "_".join(str(index) for index in hybrid_blocks)
            init_directory += f"_blocks_{block_suffix}"
        run_dir = run_dir / init_directory
    run_dir.mkdir(parents=True, exist_ok=True)
    print("encoder initialization summary: " + json.dumps(load_summary, ensure_ascii=False))
    if args.encoder_init in ("human_mae", "human_dino"):
        (run_dir / "encoder_load_summary.json").write_text(
            json.dumps(load_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"total parameters: {counts['total_parameters']:,}")
    print(f"trainable parameters: {counts['trainable_parameters']:,}")
    print(f"trainable ratio: {counts['trainable_ratio']:.6f}")
    print(f"trainable encoder blocks: {counts['trainable_encoder_blocks']}")
    if args.transfer == "lora":
        print(f"LoRA trainable parameters: {counts['lora_trainable_parameters']:,}")
        print(f"head trainable parameters: {counts['head_trainable_parameters']:,}")
        print(f"trainable ratio: {counts['trainable_ratio_percent']:.4f}%")
        print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
        print("LoRA targets: " + ", ".join(counts["lora_targets"]))
        print("Trainable parameters:\n  " + "\n  ".join(trainable_names))
    elif args.transfer == "adapter":
        print(f"adapter trainable parameters: {counts['adapter_trainable_parameters']:,}")
        print(f"head trainable parameters: {counts['head_trainable_parameters']:,}")
        print(f"trainable ratio: {counts['trainable_ratio_percent']:.4f}%")
        print(f"adapter dim/dropout: {args.adapter_dim}/{args.adapter_dropout}")
        print("Adapter targets: " + ", ".join(counts["adapter_targets"]))
        print("Trainable parameters:\n  " + "\n  ".join(trainable_names))

    weighted_condition = args.condition in ("weighted_ce", "weighted_ce_strong_aug")
    use_sampler = args.condition == "baseline"
    augmentation = "strong" if args.condition in ("strong_aug", "weighted_ce_strong_aug") else "baseline"
    train_loader, val_loader, loss_weights = build_loaders(
        train_subjects, val_subjects, model.encoder.preprocess, args.batch_size,
        args.num_workers, args.seed, len(class_names), augmentation, use_sampler,
        weighted_condition)
    subject_counts = torch.bincount(torch.tensor(
        [subject.class_index for subject in train_subjects]), minlength=len(class_names))
    sampler_name = "weighted_random" if use_sampler else "shuffle"
    loss_name = ("balanced_softmax" if args.condition == "balanced_softmax" else
                 "ldam_drw" if args.condition == "ldam_drw" else
                 "weighted_ce" if weighted_condition else "cross_entropy")
    if args.condition == "classifier_retrain":
        loss_name = "stage1_ce_then_balanced_head"
    config = _jsonable(args)
    config.update({"human_ssl_method": (
                       "mae" if args.encoder_init == "human_mae" else
                       "dino" if args.encoder_init == "human_dino" else None),
                   "encoder_load_summary": load_summary})
    config.update({"class_names": class_names, "train_subjects": [str(x.directory) for x in train_subjects],
                   "val_subjects": [str(x.directory) for x in val_subjects], "data_issues": issues,
                   "augmentation": augmentation,
                   "subject_class_counts": subject_counts.tolist(),
                   "sampler": sampler_name, "loss": loss_name,
                   "loss_class_weights": loss_weights.tolist() if loss_weights is not None else None})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    (run_dir / "trainable_parameters.json").write_text(
        json.dumps(trainable_names, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_weights = loss_weights.to(device) if loss_weights is not None else None
    if args.condition == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(subject_counts).to(device)
    elif args.condition == "ldam_drw":
        criterion = LDAMDRWLoss(subject_counts, args.drw_start_epoch).to(device)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    writer = SummaryWriter(run_dir / "tensorboard")
    writer.add_text("run/config", json.dumps(config, indent=2, ensure_ascii=False), 0)
    for name, value in counts.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"parameters/{name}", value, 0)
        else:
            writer.add_text(f"parameters/{name}", str(value), 0)
    try:
        if args.condition != "classifier_retrain":
            _run_stage(model, train_loader, val_loader, criterion, class_names, device,
                       scaler, amp_enabled, optimizer, scheduler, args.epochs, args,
                       run_dir, writer)
        else:
            _run_stage(model, train_loader, val_loader, criterion, class_names, device,
                       scaler, amp_enabled, optimizer, scheduler, args.epochs, args,
                       run_dir, writer, "stage1")
            checkpoint = torch.load(run_dir / "stage1_best.pt", map_location="cpu",
                                    weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            model.encoder.freeze(); model.frozen_encoder = True
            if args.classifier_retrain_reset_head:
                for module in model.head.modules():
                    if isinstance(module, nn.Linear): module.reset_parameters()
            stage2_sampler = args.classifier_retrain_balance == "sampler"
            stage2_weighted = args.classifier_retrain_balance == "weighted_ce"
            retrain_loader, _, retrain_weights = build_loaders(
                train_subjects, val_subjects, model.encoder.preprocess, args.batch_size,
                args.num_workers, args.seed, len(class_names), "baseline",
                stage2_sampler, stage2_weighted)
            retrain_weights = retrain_weights.to(device) if retrain_weights is not None else None
            retrain_criterion = nn.CrossEntropyLoss(weight=retrain_weights)
            retrain_optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.lr, weight_decay=args.weight_decay)
            retrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                retrain_optimizer, T_max=args.classifier_retrain_epochs)
            stage2_counts = parameter_report(model, "frozen", 0)
            (run_dir / "stage2_parameter_counts.json").write_text(
                json.dumps(stage2_counts, indent=2), encoding="utf-8")
            _run_stage(model, retrain_loader, val_loader, retrain_criterion, class_names,
                       device, scaler, amp_enabled, retrain_optimizer, retrain_scheduler,
                       args.classifier_retrain_epochs, args, run_dir, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

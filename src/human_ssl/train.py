from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

from src.checkpoint_utils import portable_config
from src.classification.training_utils import set_seed
from src.encoders import get_encoder
from src.human_ssl.data import (
    DEFAULT_ROOTS, build_ssl_loaders, dataset_counts, discover_ssl_samples, split_ssl_samples,
)
from src.human_ssl.cat_anchor import build_cat_anchor_loader, write_subject_csv
from src.human_ssl.mae import ENCODER_REGISTRY, VisionMAE
from src.human_ssl.feature_anchor import (
    ANCHOR_BLOCKS, encoder_checksum, feature_preservation_loss,
    final_feature_drift, normalized_images,
)
from src.human_ssl.reconstruction import (
    append_reconstruction_metrics, evaluate_fixed_reconstruction,
)
from src.human_ssl.update_budget import (
    checkpoint_encoder_state, copy_encoder_parameters, encoder_update_norm,
    project_encoder_to_update_budget, reference_update_norm, validate_encoder_state,
)
from src.logging_utils import enable_timestamped_prints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human kidney ultrasound MAE adaptation")
    parser.add_argument("--method", choices=("mae",), default="mae")
    parser.add_argument("--encoder", choices=tuple(ENCODER_REGISTRY), default="vit_b16")
    parser.add_argument("--checkpoint-path", type=str,
                        help="Optional official checkpoint override (primarily USFM).")
    parser.add_argument("--human1-root", type=Path, default=DEFAULT_ROOTS["human1"])
    parser.add_argument("--human2-root", type=Path, default=DEFAULT_ROOTS["human2"])
    parser.add_argument("--human3-root", type=Path, default=DEFAULT_ROOTS["human3"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--norm-pixel-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-images", type=int, help="Optional smoke/debug subset size.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--save-encoder-epochs", nargs="+", type=int,
        help="Optional zero-based epoch indices at which to save periodic encoder checkpoints.",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=1.0)
    parser.add_argument("--encoder-trainable-last-blocks", type=int)
    parser.add_argument("--run-name", default="human_mae")
    parser.add_argument("--reconstruction-eval-epochs", nargs="+", type=int)
    parser.add_argument("--reconstruction-mask-seed", type=int, default=12345)
    parser.add_argument("--reconstruction-examples", type=int, default=3)
    parser.add_argument("--feature-anchor-lambda", type=float, default=0.0)
    parser.add_argument("--feature-anchor-layers", nargs="+", type=int,
                        default=list(ANCHOR_BLOCKS))
    parser.add_argument("--cat-anchor-lambda", type=float, default=0.0)
    parser.add_argument("--cat-data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--cat-num-folds", type=int, default=5)
    parser.add_argument("--cat-fold", type=int, default=0)
    parser.add_argument("--cat-split-seed", type=int, default=42)
    parser.add_argument("--update-budget-beta", type=float)
    parser.add_argument("--update-budget-reference-checkpoint", type=Path)
    args = parser.parse_args()
    if args.save_encoder_epochs is not None:
        if len(args.save_encoder_epochs) != len(set(args.save_encoder_epochs)):
            parser.error("--save-encoder-epochs must not contain duplicate epochs")
        invalid = [epoch for epoch in args.save_encoder_epochs
                   if epoch < 0 or epoch >= args.epochs]
        if invalid:
            parser.error(
                "--save-encoder-epochs values must satisfy "
                f"0 <= epoch < {args.epochs}; invalid: {invalid}"
            )
    if args.encoder_lr_scale <= 0:
        parser.error("--encoder-lr-scale must be positive")
    if (args.encoder_trainable_last_blocks is not None and
            not 1 <= args.encoder_trainable_last_blocks <= 12):
        parser.error("--encoder-trainable-last-blocks must be between 1 and 12")
    if args.reconstruction_examples < 0:
        parser.error("--reconstruction-examples must be non-negative")
    if args.feature_anchor_lambda < 0:
        parser.error("--feature-anchor-lambda must be non-negative")
    if args.feature_anchor_lambda > 0 and args.encoder_trainable_last_blocks is not None:
        parser.error("Feature anchoring cannot be combined with partial encoder adaptation")
    if args.cat_anchor_lambda < 0:
        parser.error("--cat-anchor-lambda must be non-negative")
    if args.cat_anchor_lambda > 0 and args.feature_anchor_lambda <= 0:
        parser.error("Cat anchoring requires pretrained feature anchoring")
    if (args.update_budget_beta is None) != (args.update_budget_reference_checkpoint is None):
        parser.error("Update budgeting requires both --update-budget-beta and "
                     "--update-budget-reference-checkpoint")
    if args.update_budget_beta is not None and not 0 < args.update_budget_beta < 1:
        parser.error("--update-budget-beta must be between 0 and 1")
    if args.update_budget_beta is not None and (args.encoder != "vit_b16" or
            args.encoder_trainable_last_blocks is not None or
            args.feature_anchor_lambda > 0 or args.cat_anchor_lambda > 0):
        parser.error("Update budgeting requires fully trainable unanchored vit_b16 MAE")
    if (not args.feature_anchor_layers or len(args.feature_anchor_layers) !=
            len(set(args.feature_anchor_layers)) or
            any(block < 0 or block > 11 for block in args.feature_anchor_layers)):
        parser.error("--feature-anchor-layers must be unique ViT block indices from 0 to 11")
    if args.reconstruction_eval_epochs is not None:
        invalid = [epoch for epoch in args.reconstruction_eval_epochs
                   if epoch < 0 or epoch >= args.epochs]
        if invalid:
            parser.error("--reconstruction-eval-epochs values must satisfy "
                         f"0 <= epoch < {args.epochs}; invalid: {invalid}")
    return args


def _epoch(model, loader, device, scaler, amp_enabled, mask_ratio, optimizer=None,
           teacher=None, feature_anchor_lambda: float = 0.0,
           feature_anchor_layers: tuple[int, ...] = ANCHOR_BLOCKS,
           cat_loader=None, cat_anchor_lambda: float = 0.0,
           update_budget=None):
    training = optimizer is not None
    model.train(training)
    totals = {"mae_loss": 0.0, "feature_preserve_loss": 0.0,
              "cat_anchor_loss": 0.0, "total_loss": 0.0,
              **{f"feature_loss_block{block}": 0.0 for block in feature_anchor_layers}}
    count = 0
    cat_iterator = iter(cat_loader) if cat_loader is not None else None
    with torch.enable_grad() if training else torch.no_grad():
        for images, _dataset_ids in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                mae_loss, _, _ = model(images, mask_ratio)
                if teacher is not None:
                    feature_loss, layer_losses = feature_preservation_loss(
                        model.encoder, teacher, normalized_images(model, images),
                        feature_anchor_layers)
                else:
                    feature_loss = mae_loss.new_zeros(())
                    layer_losses = {block: feature_loss for block in feature_anchor_layers}
                if cat_iterator is not None:
                    try: cat_images, _cat_keys = next(cat_iterator)
                    except StopIteration:
                        cat_iterator = iter(cat_loader); cat_images, _cat_keys = next(cat_iterator)
                    cat_images = cat_images.to(device, non_blocking=True)
                    cat_loss, _cat_layers = feature_preservation_loss(
                        model.encoder, teacher, normalized_images(model, cat_images),
                        feature_anchor_layers)
                else: cat_loss = mae_loss.new_zeros(())
                loss = (mae_loss + feature_anchor_lambda * feature_loss +
                        cat_anchor_lambda * cat_loss)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if update_budget is not None:
                    _norm, projected = project_encoder_to_update_budget(
                        model.encoder, update_budget["pretrained_state"],
                        update_budget["max_update_norm"])
                    update_budget["total_optimizer_steps"] += 1
                    update_budget["projection_step_count"] += int(projected)
            batch = images.shape[0]
            totals["mae_loss"] += float(mae_loss) * batch
            totals["feature_preserve_loss"] += float(feature_loss) * batch
            totals["cat_anchor_loss"] += float(cat_loss) * batch
            totals["total_loss"] += float(loss) * batch
            for block, value in layer_losses.items():
                totals[f"feature_loss_block{block}"] += float(value) * batch
            count += images.shape[0]
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def _epoch_cat_anchor(model, teacher, loader, device, amp_enabled, blocks):
    model.eval(); teacher.eval(); total = 0.0; count = 0
    for images, _keys in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss, _by_layer = feature_preservation_loss(
                model.encoder, teacher, normalized_images(model, images), blocks)
        total += float(loss) * images.shape[0]; count += images.shape[0]
    return total / max(count, 1)


def _args_config(args) -> dict:
    """Keep legacy config payloads unchanged when periodic saving is not requested."""
    values = vars(args).copy()
    if values.get("save_encoder_epochs") is None:
        values.pop("save_encoder_epochs", None)
    return portable_config(values)


def _save_encoder(path: Path, model: VisionMAE, args, epoch: int, val_loss: float,
                  checkpoint_type: str | None = None) -> None:
    payload = {
        "format": "feline_transfer_learning.vision_encoder.v1",
        "encoder_name": model.registry_name,
        "initialization": model.encoder.pretraining,
        "adaptation": "human_kidney_ultrasound_mae",
        "epoch": epoch,
        "validation_reconstruction_loss": val_loss,
        "state_dict": model.encoder.state_dict(),
        "model_state_dict": model.encoder.model.state_dict(),
        "config": _args_config(args),
    }
    if checkpoint_type is not None:
        payload["checkpoint_type"] = checkpoint_type
    torch.save(payload, path)


def _save_preview(model: VisionMAE, loader, device, mask_ratio: float, path: Path) -> None:
    images, _ = next(iter(loader))
    images = images[:5].to(device)
    model.eval()
    with torch.no_grad():
        _, prediction, mask = model(images, mask_ratio)
    target_patches = model.patchify(images)
    if model.norm_pixel_loss:
        target_mean = target_patches.mean(-1, keepdim=True)
        target_variance = target_patches.var(-1, keepdim=True, unbiased=False)
        prediction = prediction * (target_variance + 1e-6).sqrt() + target_mean
    masked = model.unpatchify(target_patches * (1 - mask.unsqueeze(-1)))
    combined = model.unpatchify(
        target_patches * (1 - mask.unsqueeze(-1)) + prediction * mask.unsqueeze(-1))
    images = images.cpu(); masked = masked.clamp(0, 1).cpu(); combined = combined.clamp(0, 1).cpu()
    size, caption = images.shape[-1], 24
    canvas = Image.new("RGB", (size * 3, (size + caption) * len(images)), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(len(images)):
        for column, tensor in enumerate((images[row], masked[row], combined[row])):
            array = (tensor * 255).byte().permute(1, 2, 0).numpy()
            canvas.paste(Image.fromarray(array, mode="RGB"), (column * size, row * (size + caption)))
        y = row * (size + caption) + size + 4
        for column, label in enumerate(("Original", "Masked", "Reconstruction")):
            draw.text((column * size + 4, y), label, fill="black")
    canvas.save(path)


def main() -> None:
    enable_timestamped_prints()
    args = parse_args()
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be non-negative and smaller than epochs.")
    set_seed(args.seed)
    if args.output_dir is None:
        args.output_dir = Path(f"checkpoints/human_mae_{args.encoder}")
    roots = {"human1": args.human1_root, "human2": args.human2_root,
             "human3": args.human3_root}
    all_samples = discover_ssl_samples(roots)
    train_samples, val_samples = split_ssl_samples(
        all_samples, args.val_fraction, args.seed, args.max_images)
    model = VisionMAE(args.encoder, args.decoder_dim, args.decoder_depth,
                      args.decoder_heads, args.norm_pixel_loss, args.checkpoint_path)
    update_budget = None
    if args.update_budget_beta is not None:
        pretrained_state = copy_encoder_parameters(model.encoder)
        full_state = checkpoint_encoder_state(args.update_budget_reference_checkpoint)
        validate_encoder_state(model.encoder, model.encoder.state_dict(), "ImageNet encoder")
        validate_encoder_state(model.encoder, full_state, "Full Human MAE reference")
        full_update_norm = reference_update_norm(model.encoder, pretrained_state, full_state)
        if not math.isfinite(full_update_norm) or full_update_norm <= 0:
            raise RuntimeError(f"Invalid Full Human MAE update norm: {full_update_norm}")
        max_update_norm = args.update_budget_beta * full_update_norm
        update_budget = {"pretrained_state": pretrained_state,
                         "full_update_norm": full_update_norm,
                         "max_update_norm": max_update_norm,
                         "projection_step_count": 0,
                         "total_optimizer_steps": 0}
        print(f"full_update_norm: {full_update_norm:.12g}")
        print(f"beta: {args.update_budget_beta:.2f}")
        print(f"max_update_norm: {max_update_norm:.12g}")
    teacher = None
    teacher_initial_checksum = None
    if args.feature_anchor_lambda > 0:
        if args.encoder != "vit_b16":
            raise ValueError("Feature-anchor feasibility screening currently requires vit_b16")
        teacher = deepcopy(model.encoder).requires_grad_(False).eval()
        teacher_initial_checksum = encoder_checksum(teacher)
        max_initial_difference = max(
            float((student.detach() - reference.detach()).abs().max())
            for student, reference in zip(model.encoder.parameters(), teacher.parameters()))
        if max_initial_difference != 0:
            raise RuntimeError(f"Teacher/student initialization differs: {max_initial_difference}")
    if args.encoder_trainable_last_blocks is not None:
        if args.encoder != "vit_b16":
            raise ValueError("Partial MAE adaptation is currently defined only for vit_b16.")
        model.encoder.freeze()
        blocks = list(model.encoder.model.encoder.layers)
        for block in blocks[-args.encoder_trainable_last_blocks:]:
            block.requires_grad_(True)
        model.encoder.model.encoder.ln.requires_grad_(True)
    if args.encoder == "vit_b16":
        block_count = len(model.encoder.model.encoder.layers)
        trainable_blocks = (list(range(block_count)) if args.encoder_trainable_last_blocks is None
                            else list(range(block_count - args.encoder_trainable_last_blocks,
                                            block_count)))
        frozen_blocks = [index for index in range(block_count) if index not in trainable_blocks]
    else:
        trainable_blocks, frozen_blocks = [], []
    train_loader, val_loader = build_ssl_loaders(
        train_samples, val_samples, model.image_size, args.batch_size,
        args.num_workers, args.seed)
    cat_anchor_loader = None; cat_diagnostic_loader = None
    cat_train_subjects = []; cat_val_subjects = []
    if args.cat_anchor_lambda > 0:
        cat_anchor_loader, cat_train_subjects, cat_val_subjects = build_cat_anchor_loader(
            args.cat_data_root, model.image_size, args.batch_size, args.num_workers, args.seed,
            args.cat_num_folds, args.cat_fold, args.cat_split_seed, training=True)
        cat_diagnostic_loader, _train_again, _val_again = build_cat_anchor_loader(
            args.cat_data_root, model.image_size, args.batch_size, args.num_workers, args.seed,
            args.cat_num_folds, args.cat_fold, args.cat_split_seed, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if teacher is not None: teacher.to(device).eval()
    encoder_parameters = [parameter for parameter in model.encoder.parameters()
                          if parameter.requires_grad]
    decoder_parameters = [parameter for name, parameter in model.named_parameters()
                          if not name.startswith("encoder.") and parameter.requires_grad]
    if args.encoder_lr_scale == 1.0:
        optimizer = torch.optim.AdamW(
            [*encoder_parameters, *decoder_parameters], lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": args.lr * args.encoder_lr_scale,
             "group_name": "encoder"},
            {"params": decoder_parameters, "lr": args.lr, "group_name": "decoder"},
        ], weight_decay=args.weight_decay)

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
    config.update({"dataset_counts": dataset_counts(all_samples),
                   "train_dataset_counts": dataset_counts(train_samples),
                   "validation_dataset_counts": dataset_counts(val_samples),
                   "train_images": len(train_samples), "validation_images": len(val_samples),
                   "image_size": model.image_size, "patch_size": model.patch_size,
                   "mae_masking_mode": (
                       "full_grid_mask_tokens" if args.encoder == "usfm" else
                       "visible_tokens_only"),
                   "split_limitations": {
                       "human1": "image-level: no subject identifier in current layout",
                       "human2": "subject-level: leading anonymous filename token",
                       "human3": "image-level: no subject identifier in current layout",
                   },
                   "adaptation_depth": ("full" if args.encoder_trainable_last_blocks is None
                                          else f"last{args.encoder_trainable_last_blocks}"),
                   "trainable_blocks": trainable_blocks,
                   "frozen_blocks": frozen_blocks,
                   "patch_embed_trainable": args.encoder_trainable_last_blocks is None,
                   "final_norm_trainable": True,
                   "trainable_encoder_params": sum(p.numel() for p in encoder_parameters),
                   "total_encoder_params": sum(p.numel() for p in model.encoder.parameters()),
                   "trainable_encoder_percent": 100 * sum(
                       p.numel() for p in encoder_parameters) / sum(
                       p.numel() for p in model.encoder.parameters()),
                   "mae_decoder_trainable_params": sum(p.numel() for p in decoder_parameters)})
    config.update({"feature_anchor_lambda": args.feature_anchor_lambda,
                   "feature_anchor_layers": list(args.feature_anchor_layers),
                   "feature_anchor_pooling": "mean patch tokens; CLS excluded",
                   "feature_anchor_input": "same unmasked normalized Human image",
                   "feature_anchor_teacher": "frozen ImageNet ViT-B/16" if teacher else None})
    if update_budget is not None:
        config.update({"update_budget_method": "optimizer_step_projection",
                       "update_budget_beta": args.update_budget_beta,
                       "update_budget_reference_checkpoint": str(
                           args.update_budget_reference_checkpoint.resolve()),
                       "full_update_norm": update_budget["full_update_norm"],
                       "max_update_norm": update_budget["max_update_norm"],
                       "update_budget_encoder_scope": "patch_embed, blocks_0_11, final_norm",
                       "update_budget_decoder_constrained": False})
    if args.cat_anchor_lambda > 0:
        config.update({"cat_anchor_train_subject_count": len({x.subject_id for x in cat_train_subjects}),
                       "cat_heldout_val_subject_count": len({x.subject_id for x in cat_val_subjects}),
                       "cat_subject_overlap_count": len({x.subject_id for x in cat_train_subjects} &
                                                        {x.subject_id for x in cat_val_subjects}),
                       "cat_anchor_uses_labels": False})
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.cat_anchor_lambda > 0:
        write_subject_csv(args.output_dir / "cat_anchor_train_subjects.csv", cat_train_subjects)
        write_subject_csv(args.output_dir / "cat_heldout_val_subjects.csv", cat_val_subjects)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"images per dataset: {dataset_counts(all_samples)}")
    print(f"total SSL images: {len(all_samples):,}")
    print(f"train / validation images: {len(train_samples):,} / {len(val_samples):,}")
    print(f"encoder initialization: {model.encoder.model_name} / "
          f"{model.encoder.pretraining}")
    print(f"total / trainable parameters: {total:,} / {trainable:,}")
    print(f"mask ratio: {args.mask_ratio}; image/patch size: "
          f"{model.image_size}/{model.patch_size}")
    print(f"optimizer: AdamW; lr: {args.lr}; weight decay: {args.weight_decay}")
    print(f"encoder / decoder LR: {args.lr * args.encoder_lr_scale:g} / {args.lr:g}")
    print(f"trainable encoder / decoder parameters: "
          f"{sum(p.numel() for p in encoder_parameters):,} / "
          f"{sum(p.numel() for p in decoder_parameters):,}")
    print(f"Adaptation: {config['adaptation_depth']}")
    print(f"Trainable blocks: {trainable_blocks}")
    print(f"Frozen blocks: {frozen_blocks}")
    print(f"Patch embed: {'trainable' if config['patch_embed_trainable'] else 'frozen'}")
    print("Final norm: trainable")
    print(f"Encoder trainable: {config['trainable_encoder_params']:,} / "
          f"{config['total_encoder_params']:,} "
          f"({config['trainable_encoder_percent']:.4f}%)")
    print(f"MAE decoder params: {config['mae_decoder_trainable_params']:,}")
    print(f"Seed: {args.seed}")
    if teacher is not None:
        print("Teacher encoder parameters: frozen")
        print("Student encoder parameters: trainable")
        print(f"Selected anchor layers: {list(args.feature_anchor_layers)}")
        print(f"lambda_feature: {args.feature_anchor_lambda}")
        print("Initial teacher/student max absolute parameter difference: 0.0")
    if args.cat_anchor_lambda > 0:
        train_ids={x.subject_id for x in cat_train_subjects}; val_ids={x.subject_id for x in cat_val_subjects}
        human_groups={(x.dataset_id,x.split_group) for x in train_samples}
        print(f"Human subjects/groups: {len(human_groups)}; training images: {len(train_samples)}")
        print(f"Cat anchor train subjects: {len(train_ids)}")
        print(f"Cat downstream validation subjects: {len(val_ids)}")
        print(f"intersection(train Cat anchor, Cat val) = {len(train_ids & val_ids)}")
        print(f"lambda_cat: {args.cat_anchor_lambda}")
    if args.save_encoder_epochs is not None:
        print(f"Periodic encoder checkpoints: {sorted(args.save_encoder_epochs)}")

    writer = SummaryWriter(args.output_dir / "tensorboard")
    best_loss = float("inf")
    periodic_epochs = set(args.save_encoder_epochs or ())
    reconstruction_epochs = set(args.reconstruction_eval_epochs or ())
    metric_rows = []
    try:
        for epoch in range(args.epochs):
            train_metrics = _epoch(model, train_loader, device, scaler, amp_enabled,
                                   args.mask_ratio, optimizer, teacher,
                                   args.feature_anchor_lambda,
                                   tuple(args.feature_anchor_layers), cat_anchor_loader,
                                   args.cat_anchor_lambda, update_budget)
            val_metrics = _epoch(model, val_loader, device, scaler, amp_enabled,
                                 args.mask_ratio, None, teacher,
                                 args.feature_anchor_lambda,
                                 tuple(args.feature_anchor_layers), None, 0.0)
            if cat_anchor_loader is not None:
                cat_validation = _epoch_cat_anchor(
                    model, teacher, cat_diagnostic_loader, device, amp_enabled,
                    tuple(args.feature_anchor_layers))
                val_metrics["cat_anchor_loss"] = cat_validation
                val_metrics["total_loss"] += args.cat_anchor_lambda * cat_validation
            train_loss, val_loss = train_metrics["mae_loss"], val_metrics["mae_loss"]
            if update_budget is not None:
                current_update_norm = encoder_update_norm(
                    model.encoder, update_budget["pretrained_state"])
                relative_update_norm = current_update_norm / update_budget["full_update_norm"]
                projection_fraction = (update_budget["projection_step_count"] /
                                       max(update_budget["total_optimizer_steps"], 1))
                tolerance = max(1e-8, args.update_budget_beta * 1e-6)
                if relative_update_norm > args.update_budget_beta + tolerance:
                    raise RuntimeError("Encoder update budget violated at logged epoch: "
                                       f"{relative_update_norm} > {args.update_budget_beta}")
            else:
                current_update_norm = relative_update_norm = projection_fraction = float("nan")
            current_lr = optimizer.param_groups[0]["lr"]
            current_decoder_lr = next((group["lr"] for group in optimizer.param_groups
                                       if group.get("group_name") == "decoder"), current_lr)
            writer.add_scalar("reconstruction/train_loss", train_loss, epoch)
            writer.add_scalar("reconstruction/validation_loss", val_loss, epoch)
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            scheduler.step()
            _save_encoder(args.output_dir / "last_encoder.pt", model, args, epoch, val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                _save_encoder(args.output_dir / "best_encoder.pt", model, args, epoch, val_loss)
                _save_preview(model, val_loader, device, args.mask_ratio,
                              args.output_dir / "validation_reconstruction.png")
                (args.output_dir / "validation_metrics.json").write_text(
                    json.dumps({"epoch": epoch, "validation_reconstruction_loss": val_loss,
                                "train_reconstruction_loss": train_loss}, indent=2),
                    encoding="utf-8")
            if epoch in periodic_epochs:
                periodic_path = args.output_dir / f"epoch_{epoch:03d}_encoder.pt"
                _save_encoder(periodic_path, model, args, epoch, val_loss,
                              checkpoint_type="periodic")
                print(f"[checkpoint] saved periodic encoder: {periodic_path.name}")
            metric_rows.append({"epoch": epoch, "train_mae_loss": train_loss,
                                "validation_mae_loss": val_loss,
                                "train_feature_preserve_loss": train_metrics["feature_preserve_loss"],
                                "validation_feature_preserve_loss": val_metrics["feature_preserve_loss"],
                                "train_cat_anchor_loss": train_metrics["cat_anchor_loss"],
                                "validation_cat_anchor_loss": val_metrics["cat_anchor_loss"],
                                "train_total_loss": train_metrics["total_loss"],
                                "validation_total_loss": val_metrics["total_loss"],
                                **{f"train_feature_loss_block{block}":
                                   train_metrics[f"feature_loss_block{block}"]
                                   for block in args.feature_anchor_layers},
                                **{f"validation_feature_loss_block{block}":
                                   val_metrics[f"feature_loss_block{block}"]
                                   for block in args.feature_anchor_layers},
                                "encoder_learning_rate": current_lr,
                                "decoder_learning_rate": current_decoder_lr})
            if update_budget is not None:
                metric_rows[-1].update({
                    "encoder_update_norm": current_update_norm,
                    "relative_update_norm": relative_update_norm,
                    "projection_fraction": projection_fraction,
                    "projection_step_count": update_budget["projection_step_count"],
                    "total_optimizer_steps": update_budget["total_optimizer_steps"],
                })
            with (args.output_dir / "metrics.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
                csv_writer.writeheader(); csv_writer.writerows(metric_rows)
            with (args.output_dir / "ssl_metrics.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
                csv_writer.writeheader(); csv_writer.writerows(metric_rows)
            if epoch in reconstruction_epochs:
                reconstruction = evaluate_fixed_reconstruction(
                    model, val_loader, device, args.mask_ratio,
                    args.reconstruction_mask_seed, args.run_name, epoch,
                    args.output_dir, args.reconstruction_examples)
                reconstruction["val_mae_loss"] = val_loss
                append_reconstruction_metrics(
                    args.output_dir / "reconstruction_metrics.csv", reconstruction)
            writer.flush()
            print(f"epoch {epoch + 1}/{args.epochs} lr={current_lr:.6g} "
                  f"train_mae={train_loss:.6f} val_mae={val_loss:.6f} "
                  f"train_feature={train_metrics['feature_preserve_loss']:.6f} "
                  f"val_feature={val_metrics['feature_preserve_loss']:.6f} "
                  f"train_cat_anchor={train_metrics['cat_anchor_loss']:.6f} "
                  f"val_cat_anchor={val_metrics['cat_anchor_loss']:.6f}")
            if update_budget is not None:
                print(f"  encoder_update_norm={current_update_norm:.12g} "
                      f"relative_update_norm={relative_update_norm:.8f} "
                      f"projection_fraction={projection_fraction:.8f} "
                      f"projection_steps={update_budget['projection_step_count']}/"
                      f"{update_budget['total_optimizer_steps']}")
        if update_budget is not None:
            final_norm = encoder_update_norm(model.encoder, update_budget["pretrained_state"])
            final_relative = final_norm / update_budget["full_update_norm"]
            tolerance = max(1e-8, args.update_budget_beta * 1e-6)
            assert final_relative <= args.update_budget_beta + tolerance, (
                f"Final relative update norm {final_relative} exceeds beta "
                f"{args.update_budget_beta}")
            (args.output_dir / "update_budget_final.json").write_text(json.dumps({
                "beta": args.update_budget_beta,
                "full_update_norm": update_budget["full_update_norm"],
                "max_update_norm": update_budget["max_update_norm"],
                "encoder_update_norm": final_norm,
                "final_relative_update_norm": final_relative,
                "projection_step_count": update_budget["projection_step_count"],
                "total_optimizer_steps": update_budget["total_optimizer_steps"],
                "projection_fraction": update_budget["projection_step_count"] /
                    max(update_budget["total_optimizer_steps"], 1),
            }, indent=2), encoding="utf-8")
        if teacher is not None:
            final_feature_drift(model, teacher, val_loader, device,
                                args.output_dir / "final_feature_drift.csv", ANCHOR_BLOCKS)
            final_feature_drift(model, teacher, val_loader, device,
                                args.output_dir / "all_layer_feature_drift.csv", tuple(range(12)))
            if cat_diagnostic_loader is not None:
                final_feature_drift(model, teacher, cat_diagnostic_loader, device,
                                    args.output_dir / "cat_train_feature_drift.csv", ANCHOR_BLOCKS)
            teacher_final_checksum = encoder_checksum(teacher)
            if teacher_final_checksum != teacher_initial_checksum:
                raise RuntimeError("Frozen teacher checksum changed during training")
            print("Teacher parameter checksum unchanged: verified")
    finally:
        writer.close()


if __name__ == "__main__":
    main()

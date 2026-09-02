from __future__ import annotations

import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.checkpoint_utils import portable_config


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)


def class_weights(targets: list[int], num_classes: int) -> Tensor:
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    return counts.sum() / counts.clamp_min(1) / num_classes


def _metrics(confusion: Tensor, class_names: list[str]) -> dict[str, float]:
    matrix = confusion.double()
    true_counts, predicted_counts, correct = matrix.sum(1), matrix.sum(0), matrix.diag()
    recall = correct / true_counts.clamp_min(1)
    precision = correct / predicted_counts.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    metrics = {"accuracy": float(correct.sum() / matrix.sum().clamp_min(1)),
               "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean())}
    for index, name in enumerate(class_names):
        key = name.replace(" ", "_").replace("/", "_")
        metrics[f"class_precision/{key}"] = float(precision[index])
        metrics[f"class_recall/{key}"] = float(recall[index])
        metrics[f"class_f1/{key}"] = float(f1[index])
        metrics[f"class_prediction_count/{key}"] = float(predicted_counts[index])
    return metrics


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: Tensor) -> None:
        super().__init__()
        self.register_buffer("log_class_counts", class_counts.float().clamp_min(1).log())

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return nn.functional.cross_entropy(logits + self.log_class_counts, targets)


class LDAMDRWLoss(nn.Module):
    def __init__(self, class_counts: Tensor, drw_start_epoch: int,
                 max_margin: float = 0.5, scale: float = 30.0) -> None:
        super().__init__()
        counts = class_counts.float().clamp_min(1)
        margins = counts.pow(-0.25)
        self.register_buffer("margins", margins * (max_margin / margins.max()))
        weights = counts.rsqrt()
        self.register_buffer("drw_weights", weights / weights.mean())
        self.drw_start_epoch, self.scale, self.epoch = drw_start_epoch, scale, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        adjusted = logits.clone()
        rows = torch.arange(targets.shape[0], device=targets.device)
        adjusted[rows, targets] -= self.margins[targets]
        weights = self.drw_weights if self.epoch >= self.drw_start_epoch else None
        return nn.functional.cross_entropy(self.scale * adjusted, targets, weight=weights)


def _update(confusion: Tensor, predictions: Tensor, targets: Tensor) -> None:
    size = confusion.shape[0]
    indices = targets.long() * size + predictions.long()
    confusion += torch.bincount(indices.cpu(), minlength=size**2).reshape(size, size)


def classification_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
                         criterion: nn.Module, class_names: list[str], scaler,
                         amp_enabled: bool, optimizer=None):
    training = optimizer is not None
    model.train(training)
    kidney_confusion = torch.zeros(len(class_names), len(class_names), dtype=torch.long)
    subject_logits: dict[str, list[Tensor]] = {}
    subject_targets: dict[str, int] = {}
    loss_sum = count = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, targets, subject_keys in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            _update(kidney_confusion, logits.argmax(1), targets)
            for key, logit, target in zip(subject_keys, logits.detach().cpu(), targets.cpu()):
                subject_logits.setdefault(key, []).append(logit)
                subject_targets[key] = int(target)
            loss_sum += float(loss) * targets.shape[0]; count += targets.shape[0]
    confusion = torch.zeros_like(kidney_confusion)
    for key, logits in subject_logits.items():
        _update(confusion, torch.stack(logits).mean(0).argmax().reshape(1),
                torch.tensor([subject_targets[key]]))
    metrics = _metrics(confusion, class_names)
    metrics.update({f"kidney/{key}": value for key, value in _metrics(kidney_confusion, class_names).items()})
    metrics["loss"] = loss_sum / max(count, 1)
    return metrics, confusion


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, epoch: int,
                    best_score: float, args: Namespace) -> None:
    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_score": best_score,
                "args": portable_config(vars(args))}, path)

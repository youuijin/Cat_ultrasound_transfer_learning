"""Feature-preservation helpers for Human MAE feasibility experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ANCHOR_BLOCKS = (3, 6, 9, 11)


def normalized_images(model, images: torch.Tensor) -> torch.Tensor:
    return (images - model.image_mean) / model.image_std


def patch_mean_features(encoder, images: torch.Tensor,
                        blocks: tuple[int, ...] = ANCHOR_BLOCKS) -> dict[int, torch.Tensor]:
    features = encoder.forward_layer_features(images)
    return {block: features[f"block_{block}::patch_mean"] for block in blocks}


def feature_preservation_loss(student, teacher, images: torch.Tensor,
                              blocks: tuple[int, ...] = ANCHOR_BLOCKS):
    student_features = patch_mean_features(student, images, blocks)
    with torch.no_grad():
        teacher_features = patch_mean_features(teacher, images, blocks)
    by_layer = {}
    for block in blocks:
        student_value = F.normalize(student_features[block], dim=-1)
        teacher_value = F.normalize(teacher_features[block], dim=-1)
        by_layer[block] = (1 - (student_value * teacher_value).sum(-1)).mean()
    return torch.stack(list(by_layer.values())).mean(), by_layer


def encoder_checksum(encoder) -> tuple[float, float, int]:
    total = squared = 0.0
    count = 0
    with torch.no_grad():
        for parameter in encoder.parameters():
            value = parameter.detach().double()
            total += float(value.sum()); squared += float(value.square().sum())
            count += value.numel()
    return total, squared, count


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False) - x.mean(0, keepdims=True)
    y = y.astype(np.float64, copy=False) - y.mean(0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("Linear CKA is undefined for zero-variance features")
    return float(np.clip(numerator / denominator, 0, 1))


@torch.no_grad()
def final_feature_drift(model, teacher, loader, device: torch.device,
                        output: Path, blocks: tuple[int, ...] = ANCHOR_BLOCKS) -> list[dict]:
    model.eval(); teacher.eval()
    student_parts = {block: [] for block in blocks}
    teacher_parts = {block: [] for block in blocks}
    for images, _dataset_ids in loader:
        images = normalized_images(model, images.to(device, non_blocking=True))
        student = patch_mean_features(model.encoder, images, blocks)
        reference = patch_mean_features(teacher, images, blocks)
        for block in blocks:
            student_parts[block].append(student[block].float().cpu().numpy())
            teacher_parts[block].append(reference[block].float().cpu().numpy())
    rows = []
    for block in blocks:
        teacher_value = np.concatenate(teacher_parts[block])
        student_value = np.concatenate(student_parts[block])
        cka = linear_cka(teacher_value, student_value)
        teacher_normalized = teacher_value / np.clip(
            np.linalg.norm(teacher_value, axis=1, keepdims=True), 1e-12, None)
        student_normalized = student_value / np.clip(
            np.linalg.norm(student_value, axis=1, keepdims=True), 1e-12, None)
        cosine = float((teacher_normalized * student_normalized).sum(1).mean())
        rows.append({"layer": f"block_{block}", "cka_to_imagenet": cka,
                     "drift_1_minus_cka": 1 - cka,
                     "cosine_similarity_to_imagenet": cosine,
                     "feature_preserve_loss": 1 - cosine,
                     "n_validation_images": sum(len(part) for part in student_parts[block])})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows

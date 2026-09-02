from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.encoders import get_encoder


ENCODER_REGISTRY = {
    "vit_b16": "vit_b16_imagenet",
    "dinov2": "dinov2_vitb14",
    "biomedclip": "biomedclip_vitb16",
    "usfm": "usfm",
}


class PrototypeLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class DINOHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024,
                 bottleneck_dim: int = 256, output_dim: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototypes = PrototypeLayer(bottleneck_dim, output_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        embedding = F.normalize(self.mlp(features), dim=-1)
        return self.prototypes(embedding), embedding


class DINOBranch(nn.Module):
    def __init__(self, encoder, head: DINOHead) -> None:
        super().__init__()
        self.encoder, self.head = encoder, head
        self.register_buffer(
            "image_mean", torch.tensor(encoder.preprocess.mean)[None, :, None, None])
        self.register_buffer(
            "image_std", torch.tensor(encoder.preprocess.std)[None, :, None, None])

    def forward(self, images: Tensor):
        normalized = (images - self.image_mean) / self.image_std
        return self.head(self.encoder.forward_features(normalized))


class HumanDINO(nn.Module):
    def __init__(self, encoder_name: str = "vit_b16", head_hidden_dim: int = 1024,
                 bottleneck_dim: int = 256, output_dim: int = 1024,
                 checkpoint_path: str | None = None) -> None:
        super().__init__()
        if encoder_name not in ENCODER_REGISTRY:
            raise ValueError(f"Unsupported DINO encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.registry_name = ENCODER_REGISTRY[encoder_name]
        encoder = get_encoder(
            self.registry_name, pretrained=True, checkpoint_path=checkpoint_path)
        head = DINOHead(encoder.feature_dim, head_hidden_dim, bottleneck_dim, output_dim)
        self.student = DINOBranch(encoder, head)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student, teacher in zip(self.student.parameters(), self.teacher.parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1 - momentum)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self


class DINOLoss(nn.Module):
    def __init__(self, output_dim: int, student_temperature: float = 0.1,
                 teacher_temperature: float = 0.04,
                 center_momentum: float = 0.9) -> None:
        super().__init__()
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(self, student_logits: tuple[Tensor, Tensor],
                teacher_logits: tuple[Tensor, Tensor], update_center: bool = True):
        student = [F.log_softmax(logits / self.student_temperature, dim=-1)
                   for logits in student_logits]
        teacher = [F.softmax((logits - self.center) / self.teacher_temperature, dim=-1).detach()
                   for logits in teacher_logits]
        loss = -0.5 * (
            (teacher[0] * student[1]).sum(-1).mean()
            + (teacher[1] * student[0]).sum(-1).mean())
        if update_center:
            with torch.no_grad():
                batch_center = torch.cat(teacher_logits).mean(0, keepdim=True)
                self.center.mul_(self.center_momentum).add_(
                    batch_center, alpha=1 - self.center_momentum)
        probabilities = torch.cat(teacher)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
        usage = probabilities.mean(0)
        usage_entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
        normalized_usage_entropy = usage_entropy / math.log(usage.numel())
        return loss, {
            "teacher_output_entropy": float(entropy),
            "prototype_usage_entropy": float(normalized_usage_entropy),
            "max_prototype_probability": float(usage.max()),
        }


def cosine_teacher_momentum(base: float, step: int, total_steps: int) -> float:
    progress = min(1.0, step / max(1, total_steps - 1))
    return 1 - (1 - base) * (math.cos(math.pi * progress) + 1) / 2

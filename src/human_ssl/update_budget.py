"""Encoder update-norm accounting and trust-region projection utilities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch


TensorState = Mapping[str, torch.Tensor]


def checkpoint_encoder_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint does not contain an encoder state_dict: {path}")
    return payload["state_dict"]


def validate_encoder_state(encoder, state: TensorState, label: str) -> None:
    expected = encoder.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        name for name in set(expected) & set(state)
        if expected[name].shape != state[name].shape
    )
    print(f"{label}: missing={len(missing)} unexpected={len(unexpected)} "
          f"shape_mismatch={len(shape_mismatch)}")
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"Incompatible encoder state for {label}: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatch={shape_mismatch}")


def copy_encoder_parameters(encoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in encoder.named_parameters()}


def encoder_update_norm(encoder, pretrained_state: TensorState) -> float:
    total = 0.0
    for name, parameter in encoder.named_parameters():
        reference = pretrained_state[name].to(device=parameter.device, dtype=parameter.dtype)
        total += float((parameter.detach() - reference).double().square().sum())
    return math.sqrt(total)


def reference_update_norm(encoder, pretrained_state: TensorState,
                          adapted_state: TensorState) -> float:
    names = {name for name, _ in encoder.named_parameters()}
    total = sum(float((adapted_state[name].detach().cpu().double() -
                       pretrained_state[name].detach().cpu().double()).square().sum())
                for name in names)
    return math.sqrt(total)


@torch.no_grad()
def project_encoder_to_update_budget(encoder, pretrained_state: TensorState,
                                     max_update_norm: float) -> tuple[float, bool]:
    """Project all encoder parameter differences with one global scale factor."""
    if max_update_norm < 0:
        raise ValueError("max_update_norm must be non-negative")
    current_norm = encoder_update_norm(encoder, pretrained_state)
    if current_norm <= max_update_norm or current_norm == 0.0:
        return current_norm, False
    scale = max_update_norm / current_norm
    for name, parameter in encoder.named_parameters():
        reference = pretrained_state[name].to(device=parameter.device, dtype=parameter.dtype)
        parameter.copy_(reference + scale * (parameter - reference))
    return encoder_update_norm(encoder, pretrained_state), True


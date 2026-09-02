from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from src.adapter import apply_adapters
from src.checkpoint_utils import load_torch_checkpoint
from src.encoders import VisionEncoder, get_encoder
from src.lora import apply_qv_lora

ENCODER_NAMES = {
    "vit_b16": "vit_b16_imagenet",
    "dinov2": "dinov2_vitb14",
    "biomedclip": "biomedclip_vitb16",
    "usfm": "usfm",
}


class EncoderClassifier(nn.Module):
    def __init__(self, encoder: VisionEncoder, num_classes: int, dropout: float,
                 frozen_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.frozen_encoder = frozen_encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.feature_dim, num_classes))

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder.forward_features(images))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self


def build_encoder(name: str, transfer: str, partial_blocks: int,
                  checkpoint_path: str | None = None, lora_r: int = 8,
                  lora_alpha: float = 16.0, lora_dropout: float = 0.0,
                  adapter_dim: int = 64, adapter_dropout: float = 0.0,
                  encoder_init: str | None = None,
                  encoder_checkpoint: str | None = None) -> VisionEncoder:
    if transfer == "scratch" and name != "vit_b16":
        raise ValueError(
            "scratch is defined only for the common ViT-B/16 baseline; use "
            "--encoder vit_b16 --transfer scratch."
        )
    encoder_name = "vit_b16_scratch" if transfer == "scratch" else ENCODER_NAMES[name]
    encoder = get_encoder(encoder_name, pretrained=transfer != "scratch",
                          checkpoint_path=checkpoint_path)
    if encoder_init in ("human_mae", "human_dino", "human_barlow"):
        if transfer == "scratch":
            raise ValueError(f"{encoder_init} initialization requires a pretrained encoder.")
        if not encoder_checkpoint:
            raise ValueError(f"--encoder-checkpoint is required for {encoder_init} initialization.")
        path = Path(encoder_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Human SSL encoder checkpoint not found: {path}")
        payload = load_torch_checkpoint(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("Human SSL checkpoint must contain an encoder-only 'state_dict'.")
        compatible_encoder_names = {encoder_name}
        if encoder_name == "vit_b16_imagenet":
            compatible_encoder_names.add("vit_b16")  # Internal MAE registry alias.
        if payload.get("encoder_name") not in ({None} | compatible_encoder_names):
            raise ValueError(
                f"Expected {encoder_name!r} checkpoint for --encoder {name!r}, "
                f"got {payload.get('encoder_name')!r}.")
        expected_adaptation = {
            "human_mae": "human_kidney_ultrasound_mae",
            "human_dino": "human_kidney_ultrasound_dino",
            "human_barlow": "human_kidney_ultrasound_barlow",
        }[encoder_init]
        if payload.get("adaptation") != expected_adaptation:
            raise ValueError(
                f"Expected adaptation={expected_adaptation!r}, got {payload.get('adaptation')!r}.")
        state_dict = payload["state_dict"]
        base_state = encoder.state_dict()
        missing = sorted(set(base_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(base_state))
        shape_mismatch = sorted(
            key for key in set(base_state) & set(state_dict)
            if base_state[key].shape != state_dict[key].shape)
        matched = len(set(base_state) & set(state_dict)) - len(shape_mismatch)
        print(f"matched encoder parameters: {matched}")
        print(f"missing: {len(missing)}")
        print(f"unexpected: {len(unexpected)}")
        print(f"shape mismatch: {len(shape_mismatch)}")
        if missing or unexpected or shape_mismatch:
            raise RuntimeError("Human SSL encoder checkpoint does not exactly match the encoder")
        tensors_different_from_base = sum(
            key in base_state and base_state[key].shape == value.shape
            and not torch.equal(base_state[key].cpu(), value.cpu())
            for key, value in state_dict.items())
        result = encoder.load_state_dict(state_dict, strict=True)
        verification_key = next(iter(state_dict))
        loaded_state = encoder.state_dict()
        if not torch.equal(loaded_state[verification_key].cpu(), state_dict[verification_key].cpu()):
            raise RuntimeError("Human SSL encoder load verification failed.")
        encoder.initialization_summary = {
            "encoder_init": encoder_init,
            "human_ssl_method": {"human_mae": "mae", "human_dino": "dino",
                                 "human_barlow": "barlow"}[encoder_init],
            "adaptation": payload.get("adaptation"),
            "checkpoint": str(path),
            "checkpoint_format": payload.get("format"),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_validation_reconstruction_loss": payload.get(
                "validation_reconstruction_loss"),
            "checkpoint_validation_dino_loss": payload.get("validation_dino_loss"),
            "loaded_state_tensors": len(state_dict),
            "matched_encoder_parameters": matched,
            "shape_mismatch_keys": shape_mismatch,
            "base_encoder": encoder_name,
            "tensors_different_from_base": tensors_different_from_base,
            # Retained for compatibility with existing ViT run summaries.
            "tensors_different_from_imagenet": (
                tensors_different_from_base if encoder_name == "vit_b16_imagenet" else None
            ),
            "verification_key": verification_key,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "mae_decoder_weights_loaded": False,
            "dino_teacher_or_head_weights_loaded": False,
            "hybrid": bool(payload.get("hybrid", False)),
            "hybrid_base": payload.get("hybrid_base"),
            "hybrid_ssl_source": payload.get("hybrid_ssl_source"),
            "hybrid_ssl_blocks": payload.get("hybrid_ssl_blocks"),
            "interpolation": bool(payload.get("interpolation", False)),
            "interpolation_base": payload.get("interpolation_base"),
            "interpolation_ssl_source": payload.get("interpolation_ssl_source"),
            "interpolation_alpha": payload.get("interpolation_alpha"),
        }
    if transfer == "frozen":
        encoder.freeze()
    elif transfer == "partial":
        encoder.freeze_except_last_n_blocks(partial_blocks)
    elif transfer == "lora":
        encoder.freeze()
        encoder.lora_application = apply_qv_lora(
            encoder, lora_r, lora_alpha, lora_dropout)
    elif transfer == "adapter":
        encoder.freeze()
        encoder.adapter_application = apply_adapters(
            encoder, adapter_dim, adapter_dropout)
    else:
        encoder.unfreeze()
    return encoder


def build_classifier(name: str, transfer: str, num_classes: int, partial_blocks: int,
                     dropout: float, checkpoint_path: str | None = None,
                     lora_r: int = 8, lora_alpha: float = 16.0,
                     lora_dropout: float = 0.0, adapter_dim: int = 64,
                     adapter_dropout: float = 0.0,
                     encoder_init: str | None = None,
                     encoder_checkpoint: str | None = None) -> EncoderClassifier:
    encoder = build_encoder(name, transfer, partial_blocks, checkpoint_path,
                            lora_r, lora_alpha, lora_dropout,
                            adapter_dim, adapter_dropout,
                            encoder_init, encoder_checkpoint)
    return EncoderClassifier(encoder, num_classes, dropout, transfer == "frozen")


def parameter_report(model: nn.Module, transfer: str, partial_blocks: int) -> dict[str, object]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total,
        "trainable_encoder_blocks": partial_blocks if transfer == "partial" else (
            "all" if transfer in ("full", "scratch") else 0
        ),
    }

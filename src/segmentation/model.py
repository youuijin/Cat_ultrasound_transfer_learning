from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from src.classification.model import build_encoder
from src.encoders import VisionEncoder


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class LightweightDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 3) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False), nn.GroupNorm(8, 256), nn.GELU())
        self.blocks = nn.Sequential(
            UpsampleBlock(256, 128), UpsampleBlock(128, 64),
            UpsampleBlock(64, 32), UpsampleBlock(32, 32))
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def forward(self, features: Tensor, output_size: tuple[int, int]) -> Tensor:
        logits = self.classifier(self.blocks(self.projection(features)))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class EncoderSegmenter(nn.Module):
    """Segmenter whose frozen encoder stays in eval mode while its decoder trains.

    ``segmentation_epoch`` calls ``model.train(True)`` for each training epoch;
    this override immediately restores the frozen encoder to eval mode so any
    normalization running state and dropout behavior remain fixed.
    """
    def __init__(self, encoder: VisionEncoder, frozen_encoder: bool) -> None:
        super().__init__()
        self.encoder, self.frozen_encoder = encoder, frozen_encoder
        self.decoder = LightweightDecoder(encoder.spatial_feature_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.decoder(self.encoder.forward_spatial_features(images), images.shape[-2:])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder: self.encoder.eval()
        return self


def build_segmenter(name: str, transfer: str, partial_blocks: int,
                    checkpoint_path: str | None = None, lora_r: int = 8,
                    lora_alpha: float = 16.0,
                    lora_dropout: float = 0.0, adapter_dim: int = 64,
                    adapter_dropout: float = 0.0,
                    encoder_init: str | None = None,
                    encoder_checkpoint: str | None = None) -> EncoderSegmenter:
    encoder = build_encoder(name, transfer, partial_blocks, checkpoint_path,
                            lora_r, lora_alpha, lora_dropout,
                            adapter_dim, adapter_dropout,
                            encoder_init, encoder_checkpoint)
    return EncoderSegmenter(encoder, transfer == "frozen")

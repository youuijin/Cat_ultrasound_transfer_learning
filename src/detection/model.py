from __future__ import annotations

from torch import Tensor, nn

from src.classification.model import build_encoder
from src.encoders import VisionEncoder


class EncoderBBoxRegressor(nn.Module):
    def __init__(self, encoder: VisionEncoder, dropout: float, frozen_encoder: bool) -> None:
        super().__init__()
        self.encoder = encoder
        self.frozen_encoder = frozen_encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.feature_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder.forward_features(images))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self


def build_detector(name: str, transfer: str, partial_blocks: int, dropout: float,
                   checkpoint_path: str | None = None, lora_r: int = 8,
                   lora_alpha: float = 16.0,
                   lora_dropout: float = 0.0, adapter_dim: int = 64,
                   adapter_dropout: float = 0.0,
                   encoder_init: str | None = None,
                   encoder_checkpoint: str | None = None) -> EncoderBBoxRegressor:
    encoder = build_encoder(name, transfer, partial_blocks, checkpoint_path,
                            lora_r, lora_alpha, lora_dropout,
                            adapter_dim, adapter_dropout,
                            encoder_init, encoder_checkpoint)
    return EncoderBBoxRegressor(encoder, dropout, transfer == "frozen")

"""Shared encoder interface and preprocessing metadata."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PreprocessConfig:
    input_channels: int
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    patch_size: Optional[int] = None
    interpolation: str = "bicubic"
    crop_pct: Optional[float] = None
    grayscale_mode: str = "repeat"


class VisionEncoder(nn.Module, ABC):
    """Base class for headless vision encoders.

    ``forward_features(..., return_spatial=False)`` returns a global feature.
    With ``return_spatial=True``, implementations return native patch tokens or a
    native feature map and raise ``NotImplementedError`` when unavailable.
    """

    model_name: str
    architecture: str
    pretraining: str
    feature_dim: int
    preprocess: PreprocessConfig

    @abstractmethod
    def forward_features(self, x: Tensor, return_spatial: bool = False) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)

    def forward_layer_features(self, x: Tensor) -> dict[str, Tensor]:
        """Return pooled intermediate representations in shallow-to-deep order.

        Implementations must include ``final`` and must leave
        :meth:`forward_features` unchanged so legacy analyses remain reproducible.
        ``layer_feature_metadata`` is populated by the same call with the raw and
        pooled tensor shapes actually observed at runtime.
        """
        raise NotImplementedError(
            f"Layer-wise extraction is not defined for {self.model_name}."
        )

    @property
    def layer_feature_metadata(self) -> list[dict[str, Any]]:
        metadata = getattr(self, "_layer_feature_metadata", None)
        if metadata is None:
            raise RuntimeError("Call forward_layer_features() before reading layer metadata.")
        return metadata

    def _set_layer_metadata(
        self,
        entries: list[dict[str, Any]],
    ) -> None:
        self._layer_feature_metadata = []
        for entry in entries:
            row = dict(entry)
            raw, pooled = row.pop("raw"), row.pop("pooled")
            row.update({
                "encoder": self.model_name,
                "architecture": self.architecture,
                "raw_shape": str(list(raw.shape)),
                "pooled_shape": str(list(pooled.shape)),
                "feature_dim": int(pooled.shape[1]),
            })
            self._layer_feature_metadata.append(row)

    @property
    def spatial_feature_dim(self) -> int:
        return self.feature_dim

    def forward_spatial_features(self, x: Tensor) -> Tensor:
        """Return the final patch representation as a BCHW feature map."""
        features = self.forward_features(x, return_spatial=True)
        if features.ndim == 4:
            return features
        if features.ndim != 3:
            raise ValueError(f"Expected BCN/NLC spatial features, got {features.shape}")
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError(f"Patch-token count is not square: {features.shape[1]}")
        return features.reshape(features.shape[0], side, side, features.shape[2]).permute(0, 3, 1, 2)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "architecture": self.architecture,
            "pretraining": self.pretraining,
            "total_params": self.total_params,
            "trainable_params": self.trainable_params,
            "feature_dim": self.feature_dim,
            "preprocess": asdict(self.preprocess),
        }

    def freeze(self) -> "VisionEncoder":
        self.requires_grad_(False)
        return self

    def unfreeze(self) -> "VisionEncoder":
        self.requires_grad_(True)
        return self

    def _freeze_except_blocks(
        self, blocks: Iterable[nn.Module], n: int, always_trainable: Iterable[nn.Module] = ()
    ) -> "VisionEncoder":
        blocks = list(blocks)
        if not isinstance(n, int) or n < 0 or n > len(blocks):
            raise ValueError(f"n must be between 0 and {len(blocks)}, got {n!r}")
        self.freeze()
        for block in blocks[len(blocks) - n :] if n else ():
            block.requires_grad_(True)
        for module in always_trainable:
            module.requires_grad_(True)
        return self

    def freeze_except_last_n_blocks(self, n: int) -> "VisionEncoder":
        raise NotImplementedError(
            f"Partial unfreezing is not defined for {self.model_name}; use freeze/unfreeze."
        )

    @staticmethod
    def adapt_input_channels(x: Tensor, enabled: bool = True) -> Tensor:
        """Optionally repeat NCHW one-channel input to RGB."""
        if enabled and x.ndim == 4 and x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

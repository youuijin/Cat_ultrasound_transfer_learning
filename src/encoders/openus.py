"""OpenUS-S encoder wrapper using the official cloned implementation."""
from __future__ import annotations

import sys
from pathlib import Path

from torch import Tensor, nn

from .base import PreprocessConfig, VisionEncoder


class OpenUSVMambaSEncoder(VisionEncoder):
    def __init__(self, checkpoint_path: str | None = None,
                 source_path: str | None = None,
                 repeat_grayscale: bool = True) -> None:
        super().__init__()
        project_root = Path(__file__).resolve().parents[2]
        source_path = Path(source_path or project_root / "external" / "OpenUS").resolve()
        checkpoint_path = Path(
            checkpoint_path or project_root / "weights" / "openus" / "OpenUS-S.pth"
        ).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"OpenUS source repository not found: {source_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"OpenUS-S checkpoint not found: {checkpoint_path}")

        source_str = str(source_path)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)
        try:
            from vmamba_models.dino_vmamba import (
                dinov2_vmamba_small,
                Backbone_DINOv2_VSSM_2,
            )
        except Exception as e:
            raise ImportError(
                "Could not import the official OpenUS VMamba-S implementation.\n"
                f"Original error: {type(e).__name__}: {e}"
            ) from e

        # Exact vmamba_small backbone used by official OpenUS training. The
        # ultrasound-pretrained teacher weights are loaded on top immediately.
        self.model = Backbone_DINOv2_VSSM_2(
            depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3,
            ssm_ratio=2.0, patch_size=4, masked_im_modeling=False,
            pretrained=None, training=False,
        )
        load_openus_backbone(self.model, str(checkpoint_path), key="teacher")

        self.feature_dim = int(self.model.dims[-1])
        self.model_name, self.architecture, self.pretraining = (
            "openus_vmamba_s", "OpenUS-S / VMamba-Small", "OpenUS ultrasound pretraining"
        )
        self.preprocess = PreprocessConfig(
            input_channels=3, image_size=224,
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
            patch_size=4, interpolation="bicubic", grayscale_mode="repeat",
        )
        self.repeat_grayscale = repeat_grayscale

    def forward_features(self, x: Tensor, return_spatial: bool = False) -> Tensor:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        tokens = self.model(x)
        return tokens[:, 1:] if return_spatial else tokens[:, 0]

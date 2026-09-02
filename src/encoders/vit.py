"""Torchvision ViT-B/16 baselines."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import PreprocessConfig, VisionEncoder

IMAGENET_PREPROCESS = PreprocessConfig(
    input_channels=3, image_size=224,
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), patch_size=16,
)


class TorchvisionViTB16Encoder(VisionEncoder):
    def __init__(self, pretrained: bool, repeat_grayscale: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except ImportError as exc:
            raise ImportError("vit_b16 requires torchvision.") from exc
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.model_name = "vit_b16_imagenet" if pretrained else "vit_b16_scratch"
        self.architecture = "ViT-B/16"
        self.pretraining = "ImageNet-1K supervised" if pretrained else "None (random initialization)"
        self.feature_dim = int(self.model.hidden_dim)
        self.preprocess = IMAGENET_PREPROCESS
        self.repeat_grayscale = repeat_grayscale

    def _tokens(self, x: Tensor) -> Tensor:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        n = x.shape[0]
        x = self.model._process_input(x)
        cls = self.model.class_token.expand(n, -1, -1)
        return self.model.encoder(torch.cat([cls, x], dim=1))

    def forward_features(self, x: Tensor, return_spatial: bool = False) -> Tensor:
        tokens = self._tokens(x)
        return tokens[:, 1:] if return_spatial else tokens[:, 0]

    def forward_layer_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        n = x.shape[0]
        x = self.model._process_input(x)
        x = torch.cat([self.model.class_token.expand(n, -1, -1), x], dim=1)
        encoder = self.model.encoder
        x = encoder.dropout(x + encoder.pos_embedding)
        features, metadata = {}, []
        special_tokens = 1
        for index, block in enumerate(encoder.layers):
            x = block(x)
            name = f"block_{index}"
            variants = (("patch_mean", x[:, special_tokens:].mean(1), "mean_patch_tokens"),
                        ("cls_token", x[:, 0], "cls_token"))
            for representation_type, pooled, pooling in variants:
                key = f"{name}::{representation_type}"
                features[key] = pooled
                metadata.append({"feature_key": key, "layer": name, "layer_index": index,
                    "representation_type": representation_type, "module_name": f"model.encoder.layers.{index}",
                    "raw": x, "pooled": pooled, "pooling": pooling, "is_final_representation": False,
                    "total_tokens": x.shape[1], "special_tokens_excluded": special_tokens,
                    "patch_tokens_used": x.shape[1] - special_tokens})
        raw_final = encoder.ln(x)
        post_norm = raw_final[:, special_tokens:].mean(1)
        features["post_norm::patch_mean"] = post_norm
        metadata.append({"feature_key": "post_norm::patch_mean", "layer": "post_norm", "layer_index": len(encoder.layers),
            "representation_type": "post_norm_patch_mean", "module_name": "model.encoder.ln", "raw": raw_final,
            "pooled": post_norm, "pooling": "mean_post_norm_patch_tokens", "is_final_representation": False,
            "total_tokens": raw_final.shape[1], "special_tokens_excluded": special_tokens,
            "patch_tokens_used": raw_final.shape[1] - special_tokens})
        final = raw_final[:, 0]
        features["final::native_final"] = final
        metadata.append({"feature_key": "final::native_final", "layer": "final", "layer_index": len(encoder.layers),
            "representation_type": "native_final", "module_name": "model.encoder.ln", "raw": raw_final,
            "pooled": final, "pooling": "normalized_cls_token", "is_final_representation": True,
            "total_tokens": raw_final.shape[1], "special_tokens_excluded": special_tokens,
            "patch_tokens_used": raw_final.shape[1] - special_tokens})
        self._set_layer_metadata(metadata)
        return features

    def freeze_except_last_n_blocks(self, n: int) -> "TorchvisionViTB16Encoder":
        self._freeze_except_blocks(
            self.model.encoder.layers, n, (self.model.encoder.ln,)
        )
        return self

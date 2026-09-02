"""Official Meta DINOv2 wrapper."""
from __future__ import annotations

import torch
from torch import Tensor

from .base import PreprocessConfig, VisionEncoder


class DINOv2ViTB14Encoder(VisionEncoder):
    def __init__(self, pretrained: bool = True, repeat_grayscale: bool = True) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError("dinov2_vitb14 is an official-pretrained setting; pretrained=False is unsupported.")
        try:
            import torch
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load official DINOv2. Ensure network/cache access and a compatible PyTorch; "
                "torch.hub uses facebookresearch/dinov2."
            ) from exc
        self.model_name, self.architecture, self.pretraining = (
            "dinov2_vitb14", "ViT-B/14", "DINOv2"
        )
        self.feature_dim = int(self.model.embed_dim)
        self.preprocess = PreprocessConfig(3, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 14)
        self.repeat_grayscale = repeat_grayscale

    def forward_features(self, x: Tensor, return_spatial: bool = False) -> Tensor:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"] if return_spatial else out["x_norm_clstoken"]

    def forward_layer_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        captured: list[Tensor] = []
        handles = [block.register_forward_hook(lambda _m, _i, out: captured.append(out))
                   for block in self.model.blocks]
        try:
            output = self.model.forward_features(x)
        finally:
            for handle in handles:
                handle.remove()
        features, metadata = {}, []
        register_tokens = int(getattr(self.model, "num_register_tokens", 0))
        special_tokens = 1 + register_tokens
        for index, raw in enumerate(captured):
            name = f"block_{index}"
            variants = (("patch_mean", raw[:, special_tokens:].mean(1), "mean_patch_tokens"),
                        ("cls_token", raw[:, 0], "cls_token"))
            for representation_type, pooled, pooling in variants:
                key = f"{name}::{representation_type}"
                features[key] = pooled
                metadata.append({"feature_key": key, "layer": name, "layer_index": index,
                    "representation_type": representation_type, "module_name": f"model.blocks.{index}",
                    "raw": raw, "pooled": pooled, "pooling": pooling, "is_final_representation": False,
                    "total_tokens": raw.shape[1], "special_tokens_excluded": special_tokens,
                    "patch_tokens_used": raw.shape[1] - special_tokens, "register_tokens_excluded": register_tokens})
        final = output["x_norm_clstoken"]
        normalized_parts = [final.unsqueeze(1)]
        normalized_registers = output.get("x_norm_regtokens")
        if normalized_registers is not None and normalized_registers.shape[1] > 0:
            normalized_parts.append(normalized_registers)
        normalized_parts.append(output["x_norm_patchtokens"])
        raw_final = torch.cat(normalized_parts, dim=1)
        post_norm = output["x_norm_patchtokens"].mean(1)
        features["post_norm::patch_mean"] = post_norm
        metadata.append({"feature_key": "post_norm::patch_mean", "layer": "post_norm", "layer_index": len(captured),
            "representation_type": "post_norm_patch_mean", "module_name": "model.norm / x_norm_patchtokens",
            "raw": output["x_norm_patchtokens"], "pooled": post_norm, "pooling": "mean_post_norm_patch_tokens",
            "is_final_representation": False, "total_tokens": raw_final.shape[1],
            "special_tokens_excluded": special_tokens, "patch_tokens_used": output["x_norm_patchtokens"].shape[1],
            "register_tokens_excluded": register_tokens})
        features["final::native_final"] = final
        metadata.append({"feature_key": "final::native_final", "layer": "final", "layer_index": len(captured),
            "representation_type": "native_final", "module_name": "model.norm / x_norm_clstoken",
            "raw": raw_final, "pooled": final, "pooling": "normalized_cls_token", "is_final_representation": True,
            "total_tokens": raw_final.shape[1], "special_tokens_excluded": special_tokens,
            "patch_tokens_used": output["x_norm_patchtokens"].shape[1], "register_tokens_excluded": register_tokens})
        self._set_layer_metadata(metadata)
        return features

    def freeze_except_last_n_blocks(self, n: int) -> "DINOv2ViTB14Encoder":
        self._freeze_except_blocks(self.model.blocks, n, (self.model.norm,))
        return self

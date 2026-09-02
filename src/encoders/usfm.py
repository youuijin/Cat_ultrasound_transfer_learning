"""USFM encoder wrapper using the official cloned USFM implementation."""
from __future__ import annotations

import sys
import logging
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from .base import PreprocessConfig, VisionEncoder


class USFMEncoder(VisionEncoder):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        source_path: str | None = None,
        repeat_grayscale: bool = True,
    ) -> None:
        super().__init__()

        project_root = Path(__file__).resolve().parents[2]
        source_path = Path(source_path or project_root / "external" / "USFM").resolve()
        checkpoint_path = Path(
            checkpoint_path or project_root / "weights" / "usfm" / "USFM_latest.pth"
        ).resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"USFM source repository not found: {source_path}"
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"USFM checkpoint not found: {checkpoint_path}"
            )

        # Add official USFM repository to Python import path.
        source_str = str(source_path)
        if source_str not in sys.path:
            sys.path.insert(0, source_str)

        try:
            from usdsgen.modules.backbone.vision_transformer import (
                VisionTransformer,
            )
            from usdsgen.utils.modelutils import load_pretrained
        except ImportError as e:
            raise ImportError(
                "Could not import the official USFM VisionTransformer. "
                f"Check source_path={source_path} and install USFM dependencies."
            ) from e

        # ------------------------------------------------------------
        # Official USFM ViT-B configuration
        #
        # Taken from:
        # configs/model/Cls/vit.yaml
        # ------------------------------------------------------------
        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_values=0.1,
            use_abs_pos_emb=False,
            use_rel_pos_bias=True,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # Use the repository's official ViT loader. In particular, it expands
        # USFM_latest.pth's shared relative-position bias into the per-block
        # tensors required by the official downstream classification config.
        load_pretrained(
            SimpleNamespace(type="vit", pretrained=str(checkpoint_path)),
            self.model,
            logging.getLogger(__name__),
        )

        self.feature_dim = 768
        self.model_name = "usfm"
        self.architecture = "USFM ViT-B/16"
        self.pretraining = "USFM ultrasound pretraining"

        self.preprocess = PreprocessConfig(
            input_channels=3,
            image_size=224,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            patch_size=16,
            interpolation="bicubic",
            crop_pct=None,
            grayscale_mode="repeat",
        )

        self.repeat_grayscale = repeat_grayscale

    def freeze_except_last_n_blocks(self, n: int) -> "USFMEncoder":
        output_norms = tuple(
            module for module in (self.model.norm, self.model.fc_norm)
            if module is not None
        )
        self._freeze_except_blocks(self.model.blocks, n, output_norms)
        return self

    def forward_features(
        self,
        x: Tensor,
        return_spatial: bool = False,
    ) -> Tensor:

        x = self.adapt_input_channels(
            x,
            self.repeat_grayscale,
        )

        if not return_spatial:
            return self.model.forward_features(x)

        model = self.model
        x = model.patch_embed(x)
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if model.pos_embed is not None:
            x = x + model.pos_embed
        x = model.pos_drop(x)
        rel_pos_bias = model.rel_pos_bias() if model.rel_pos_bias is not None else None
        for block in model.blocks:
            x = block(x, rel_pos_bias=rel_pos_bias)
        return model.norm(x)[:, 1:]

    def forward_layer_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        captured: list[Tensor] = []
        handles = [block.register_forward_hook(lambda _m, _i, out: captured.append(out))
                   for block in self.model.blocks]
        try:
            final = self.model.forward_features(x)
        finally:
            for handle in handles:
                handle.remove()
        features, metadata = {}, []
        special_tokens = 1
        # USFM blocks contain a CLS token, but the official final representation
        # uses mean pooling over patch tokens. Use that rule consistently here.
        for index, raw in enumerate(captured):
            pooled = raw[:, special_tokens:].mean(dim=1)
            name = f"block_{index}"
            key = f"{name}::patch_mean"
            features[key] = pooled
            metadata.append({"feature_key": key, "layer": name, "layer_index": index,
                "representation_type": "patch_mean", "module_name": f"model.blocks.{index}",
                "raw": raw, "pooled": pooled, "pooling": "mean_patch_tokens",
                "is_final_representation": False, "total_tokens": raw.shape[1],
                "special_tokens_excluded": special_tokens, "patch_tokens_used": raw.shape[1] - special_tokens})
        # USFM's configured token norm is Identity; its learned fc_norm is
        # applied after patch averaging, so this harmonized post-norm feature is
        # numerically the native final but remains explicitly labeled.
        features["post_norm::patch_mean"] = final
        metadata.append({"feature_key": "post_norm::patch_mean", "layer": "post_norm",
            "layer_index": len(captured), "representation_type": "post_norm_patch_mean",
            "module_name": "model.fc_norm (after patch mean)", "raw": final, "pooled": final,
            "pooling": "fc_norm_after_mean_patch_tokens", "is_final_representation": False,
            "total_tokens": captured[-1].shape[1], "special_tokens_excluded": special_tokens,
            "patch_tokens_used": captured[-1].shape[1] - special_tokens})
        features["final::native_final"] = final
        metadata.append({"feature_key": "final::native_final", "layer": "final", "layer_index": len(captured),
            "representation_type": "native_final", "module_name": "model.forward_features (fc_norm after patch mean)",
            "raw": final, "pooled": final, "pooling": "official_mean_pool",
            "is_final_representation": True, "total_tokens": captured[-1].shape[1],
            "special_tokens_excluded": special_tokens, "patch_tokens_used": captured[-1].shape[1] - special_tokens})
        self._set_layer_metadata(metadata)
        return features

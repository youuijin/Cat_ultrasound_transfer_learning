"""BiomedCLIP vision-encoder wrapper."""
from __future__ import annotations

import torch
from torch import Tensor

from .base import PreprocessConfig, VisionEncoder

HF_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedCLIPViTB16Encoder(VisionEncoder):
    def __init__(self, pretrained: bool = True, repeat_grayscale: bool = True) -> None:
        super().__init__()
        if not pretrained:
            raise ValueError("biomedclip_vitb16 is an official-pretrained setting; pretrained=False is unsupported.")
        try:
            import open_clip
            clip_model, self.official_preprocess = open_clip.create_model_from_pretrained(HF_ID)
        except ImportError as exc:
            raise ImportError("biomedclip_vitb16 requires open_clip_torch and huggingface_hub.") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to load official BiomedCLIP model {HF_ID!r}.") from exc
        self.model = clip_model.visual
        self.model_name, self.architecture, self.pretraining = (
            "biomedclip_vitb16", "ViT-B/16", "BiomedCLIP"
        )
        self.feature_dim = int(getattr(self.model, "output_dim", 512))
        # Published OpenCLIP preprocessing; official_preprocess is retained too.
        self.preprocess = PreprocessConfig(
            3, 224, (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711), 16,
        )
        self.repeat_grayscale = repeat_grayscale

    @property
    def spatial_feature_dim(self) -> int:
        trunk = getattr(self.model, "trunk", None)
        if trunk is not None:
            return int(trunk.num_features)
        return int(self.model.conv1.out_channels)

    def forward_spatial_features(self, x: Tensor) -> Tensor:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        if not hasattr(self.model, "forward_intermediates"):
            raise NotImplementedError("Installed OpenCLIP does not expose forward_intermediates().")
        output = self.model.forward_intermediates(
            x, indices=1, normalize_intermediates=True, intermediates_only=True,
            output_fmt="NCHW",
        )
        return output["image_intermediates"][-1]

    def forward_features(self, x: Tensor, return_spatial: bool = False) -> Tensor:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        if return_spatial:
            raise NotImplementedError(
                "BiomedCLIP's stable OpenCLIP vision API exposes the projected global image embedding; "
                "native patch tokens are version-dependent and are not exposed by this wrapper."
            )
        return self.model(x)

    def forward_layer_features(self, x: Tensor) -> dict[str, Tensor]:
        x = self.adapt_input_channels(x, self.repeat_grayscale)
        trunk = getattr(self.model, "trunk", None)
        blocks = getattr(trunk, "blocks", None)
        prefix = "model.trunk.blocks"
        if blocks is None:
            blocks = getattr(getattr(self.model, "transformer", None), "resblocks", None)
            prefix = "model.transformer.resblocks"
        if blocks is None:
            raise NotImplementedError("This OpenCLIP visual implementation does not expose transformer blocks.")
        captured: list[Tensor] = []
        normalized: list[Tensor] = []
        handles = [block.register_forward_hook(lambda _m, _i, out: captured.append(out))
                   for block in blocks]
        norm_module = getattr(trunk, "norm", None) if trunk is not None else getattr(self.model, "ln_post", None)
        if norm_module is not None:
            handles.append(norm_module.register_forward_hook(lambda _m, _i, out: normalized.append(out)))
        try:
            final = self.model(x)
        finally:
            for handle in handles:
                handle.remove()
        features, metadata = {}, []
        special_tokens = int(getattr(trunk, "num_prefix_tokens", 1)) if trunk is not None else 1
        for index, raw in enumerate(captured):
            # timm/OpenCLIP trunks are batch-first; legacy OpenCLIP resblocks are
            # sequence-first. Detect the latter from the known batch dimension.
            batch_first = raw if raw.shape[0] == x.shape[0] else raw.transpose(0, 1)
            name = f"block_{index}"
            variants = (("patch_mean", batch_first[:, special_tokens:].mean(1), "mean_patch_tokens"),
                        ("cls_token", batch_first[:, 0], "cls_token"))
            for representation_type, pooled, pooling in variants:
                key = f"{name}::{representation_type}"
                features[key] = pooled
                metadata.append({"feature_key": key, "layer": name, "layer_index": index,
                    "representation_type": representation_type, "module_name": f"{prefix}.{index}",
                    "raw": raw, "pooled": pooled, "pooling": pooling, "is_final_representation": False,
                    "total_tokens": batch_first.shape[1], "special_tokens_excluded": special_tokens,
                    "patch_tokens_used": batch_first.shape[1] - special_tokens})
        if normalized:
            norm_raw = normalized[-1]
            norm_batch_first = norm_raw if norm_raw.shape[0] == x.shape[0] else norm_raw.transpose(0, 1)
            post_norm = norm_batch_first[:, special_tokens:].mean(1)
            features["post_norm::patch_mean"] = post_norm
            metadata.append({"feature_key": "post_norm::patch_mean", "layer": "post_norm",
                "layer_index": len(captured), "representation_type": "post_norm_patch_mean",
                "module_name": "model.trunk.norm" if trunk is not None else "model.ln_post",
                "raw": norm_raw, "pooled": post_norm, "pooling": "mean_post_norm_patch_tokens",
                "is_final_representation": False, "total_tokens": norm_batch_first.shape[1],
                "special_tokens_excluded": special_tokens,
                "patch_tokens_used": norm_batch_first.shape[1] - special_tokens})
        features["final::native_final"] = final
        metadata.append({"feature_key": "final::native_final", "layer": "final", "layer_index": len(captured),
            "representation_type": "native_final", "module_name": "model (OpenCLIP visual pooled projection)",
            "raw": final, "pooled": final, "pooling": "official_projected_global_embedding",
            "is_final_representation": True, "total_tokens": None, "special_tokens_excluded": None,
            "patch_tokens_used": None})
        self._set_layer_metadata(metadata)
        return features

    def freeze_except_last_n_blocks(self, n: int) -> "BiomedCLIPViTB16Encoder":
        trunk = getattr(self.model, "trunk", None)
        blocks = getattr(trunk, "blocks", None)
        if blocks is not None:
            trainable = tuple(module for module in
                              (getattr(trunk, "norm", None), getattr(self.model, "head", None))
                              if module is not None)
            self._freeze_except_blocks(blocks, n, trainable)
            return self
        blocks = getattr(getattr(self.model, "transformer", None), "resblocks", None)
        if blocks is None:
            raise NotImplementedError("This OpenCLIP visual implementation does not expose transformer blocks.")
        trainable = tuple(m for m in (getattr(self.model, "ln_post", None),) if m is not None)
        self._freeze_except_blocks(blocks, n, trainable)
        return self

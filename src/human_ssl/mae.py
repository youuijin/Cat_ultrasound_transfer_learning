from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.encoders import VisionEncoder, get_encoder


ENCODER_REGISTRY = {
    "vit_b16": "vit_b16_imagenet",
    "dinov2": "dinov2_vitb14",
    "biomedclip": "biomedclip_vitb16",
    "usfm": "usfm",
}


class VisionMAE(nn.Module):
    def __init__(self, encoder_name: str = "vit_b16", decoder_dim: int = 256,
                 decoder_depth: int = 4, decoder_heads: int = 8,
                 norm_pixel_loss: bool = False,
                 checkpoint_path: str | None = None) -> None:
        super().__init__()
        if encoder_name not in ENCODER_REGISTRY:
            raise ValueError(f"Unsupported MAE encoder: {encoder_name}")
        self.encoder_name = encoder_name
        self.registry_name = ENCODER_REGISTRY[encoder_name]
        self.encoder: VisionEncoder = get_encoder(
            self.registry_name, pretrained=True, checkpoint_path=checkpoint_path)
        self.image_size = self.encoder.preprocess.image_size
        self.patch_size = int(self.encoder.preprocess.patch_size or 16)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.norm_pixel_loss = norm_pixel_loss
        self.encoder_dim = self._encoder_width()
        self.decoder_embed = nn.Linear(self.encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim))
        layer = nn.TransformerEncoderLayer(
            decoder_dim, decoder_heads, decoder_dim * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, decoder_depth, nn.LayerNorm(decoder_dim))
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size**2 * 3)
        self.register_buffer(
            "image_mean", torch.tensor(self.encoder.preprocess.mean)[None, :, None, None])
        self.register_buffer(
            "image_std", torch.tensor(self.encoder.preprocess.std)[None, :, None, None])
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.encoder_mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def _encoder_width(self) -> int:
        model = self.encoder.model
        if self.encoder_name == "vit_b16":
            return int(model.hidden_dim)
        if self.encoder_name in ("dinov2", "usfm"):
            return int(model.embed_dim)
        return int(model.trunk.num_features)

    def patchify(self, images: Tensor) -> Tensor:
        patch = self.patch_size
        batch, channels, height, width = images.shape
        if height != width or height % patch:
            raise ValueError(f"Expected square images divisible by {patch}, got {images.shape}")
        side = height // patch
        return images.reshape(batch, channels, side, patch, side, patch).permute(
            0, 2, 4, 3, 5, 1).reshape(batch, side * side, patch * patch * channels)

    def unpatchify(self, patches: Tensor) -> Tensor:
        patch, side = self.patch_size, math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("Patch count is not square.")
        return patches.reshape(patches.shape[0], side, side, patch, patch, 3).permute(
            0, 5, 1, 3, 2, 4).reshape(patches.shape[0], 3, side * patch, side * patch)

    @staticmethod
    def _random_mask(batch: int, patches: int, visible: int, device):
        noise = torch.rand(batch, patches, device=device)
        ids_shuffle = noise.argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        ids_keep = ids_shuffle[:, :visible]
        mask = torch.ones(batch, patches, device=device)
        mask[:, :visible] = 0
        return ids_keep, ids_restore, torch.gather(mask, 1, ids_restore)

    @staticmethod
    def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def _encode_visible(self, normalized: Tensor, ids_keep: Tensor, mask: Tensor):
        model = self.encoder.model
        batch = normalized.shape[0]
        if self.encoder_name == "vit_b16":
            patches = model._process_input(normalized)
            positions = model.encoder.pos_embedding.expand(batch, -1, -1)
            visible = self._gather(patches + positions[:, 1:], ids_keep)
            cls = model.class_token.expand(batch, -1, -1) + positions[:, :1]
            tokens = model.encoder.dropout(torch.cat((cls, visible), dim=1))
            return model.encoder.ln(model.encoder.layers(tokens)), False

        if self.encoder_name == "dinov2":
            if getattr(model, "num_register_tokens", 0):
                raise RuntimeError("MAE currently expects DINOv2 without register tokens.")
            full = model.prepare_tokens_with_masks(normalized)
            tokens = torch.cat((full[:, :1], self._gather(full[:, 1:], ids_keep)), dim=1)
            for block in model.blocks:
                tokens = block(tokens)
            return model.norm(tokens), False

        if self.encoder_name == "biomedclip":
            trunk = model.trunk
            patches = trunk.patch_embed(normalized)
            full = trunk._pos_embed(patches)
            prefix = int(getattr(trunk, "num_prefix_tokens", 1))
            if prefix != 1:
                raise RuntimeError("MAE currently expects one BiomedCLIP prefix token.")
            tokens = torch.cat((full[:, :1], self._gather(full[:, 1:], ids_keep)), dim=1)
            tokens = trunk.norm_pre(trunk.patch_drop(trunk.pos_drop(tokens)))
            tokens = trunk.blocks(tokens)
            return trunk.norm(tokens), False

        # USFM's relative-position tables require the complete 14x14 grid.
        patches = model.patch_embed(normalized)
        patches = torch.where(mask.bool().unsqueeze(-1),
                              self.encoder_mask_token.to(patches.dtype), patches)
        cls = model.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        if model.pos_embed is not None:
            tokens = tokens + model.pos_embed
        tokens = model.pos_drop(tokens)
        shared_bias = model.rel_pos_bias() if model.rel_pos_bias is not None else None
        for block in model.blocks:
            tokens = block(tokens, rel_pos_bias=shared_bias)
        return model.norm(tokens), True

    def forward(self, images: Tensor, mask_ratio: float = 0.75):
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be between zero and one.")
        normalized = (images - self.image_mean) / self.image_std
        batch = images.shape[0]
        visible = max(1, int(self.num_patches * (1 - mask_ratio)))
        ids_keep, ids_restore, mask = self._random_mask(
            batch, self.num_patches, visible, images.device)
        encoded, keeps_full_grid = self._encode_visible(normalized, ids_keep, mask)
        decoded_encoded = self.decoder_embed(encoded)
        if keeps_full_grid:
            decoded = decoded_encoded
        else:
            missing = self.num_patches - visible
            shuffled = torch.cat((decoded_encoded[:, 1:],
                                  self.mask_token.expand(batch, missing, -1)), dim=1)
            decoded_patches = self._gather(shuffled, ids_restore)
            decoded = torch.cat((decoded_encoded[:, :1], decoded_patches), dim=1)
        decoded = self.decoder(decoded + self.decoder_pos_embed)
        prediction = self.decoder_pred(decoded[:, 1:])

        target = self.patchify(images)
        if self.norm_pixel_loss:
            mean = target.mean(-1, keepdim=True)
            variance = target.var(-1, keepdim=True, unbiased=False)
            target = (target - mean) / (variance + 1e-6).sqrt()
        patch_loss = (prediction - target).square().mean(-1)
        loss = (patch_loss * mask).sum() / mask.sum().clamp_min(1)
        return loss, prediction, mask


class ViTB16MAE(VisionMAE):
    """Backward-compatible ImageNet ViT-B/16 MAE constructor."""

    def __init__(self, decoder_dim: int = 256, decoder_depth: int = 4,
                 decoder_heads: int = 8, norm_pixel_loss: bool = False) -> None:
        super().__init__("vit_b16", decoder_dim, decoder_depth,
                         decoder_heads, norm_pixel_loss)

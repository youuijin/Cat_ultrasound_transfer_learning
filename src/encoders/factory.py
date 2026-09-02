"""Public encoder registry and construction helpers."""
from __future__ import annotations

from typing import Any

from .base import PreprocessConfig, VisionEncoder

SUPPORTED_ENCODERS = (
    "vit_b16_scratch",
    "vit_b16_imagenet",
    "dinov2_vitb14",
    "biomedclip_vitb16",
    "usfm",
    "openus_vmamba_s",
)


def get_encoder(name: str, pretrained: bool | None = None,
                checkpoint_path: str | None = None, **kwargs: Any) -> VisionEncoder:
    if name not in SUPPORTED_ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}. Supported encoders: {', '.join(SUPPORTED_ENCODERS)}")
    if name == "vit_b16_scratch":
        if pretrained is True:
            raise ValueError("vit_b16_scratch cannot be constructed with pretrained=True.")
        from .vit import TorchvisionViTB16Encoder
        return TorchvisionViTB16Encoder(pretrained=False, **kwargs)
    if name == "vit_b16_imagenet":
        if pretrained is False:
            raise ValueError("vit_b16_imagenet requires pretrained weights; use vit_b16_scratch instead.")
        from .vit import TorchvisionViTB16Encoder
        return TorchvisionViTB16Encoder(pretrained=True, **kwargs)
    if name == "dinov2_vitb14":
        from .dinov2 import DINOv2ViTB14Encoder
        return DINOv2ViTB14Encoder(pretrained=True if pretrained is None else pretrained, **kwargs)
    if name == "biomedclip_vitb16":
        from .biomedclip import BiomedCLIPViTB16Encoder
        return BiomedCLIPViTB16Encoder(pretrained=True if pretrained is None else pretrained, **kwargs)
    if pretrained is False:
        raise ValueError(f"{name} represents an official-pretrained setting; pretrained=False is unsupported.")
    if name == "usfm":
        from .usfm import USFMEncoder
        return USFMEncoder(checkpoint_path=checkpoint_path, **kwargs)
    from .openus import OpenUSVMambaSEncoder
    return OpenUSVMambaSEncoder(checkpoint_path=checkpoint_path, **kwargs)


def get_encoder_transform(name: str, repeat_grayscale: bool = True):
    """Build a torchvision transform from static native preprocessing metadata.

    For BiomedCLIP, ``encoder.official_preprocess`` remains the authoritative
    OpenCLIP transform. USFM/OpenUS require metadata from their official release.
    """
    from .vit import IMAGENET_PREPROCESS
    if name in ("vit_b16_scratch", "vit_b16_imagenet", "dinov2_vitb14"):
        config = IMAGENET_PREPROCESS
    elif name == "biomedclip_vitb16":
        config = PreprocessConfig(3, 224, (0.48145466, 0.4578275, 0.40821073),
                                  (0.26862954, 0.26130258, 0.27577711), 16)
    elif name in ("usfm", "openus_vmamba_s"):
        raise ValueError(f"{name} preprocessing must come from its official release; use encoder.preprocess.")
    else:
        raise ValueError(f"Unknown encoder {name!r}. Supported encoders: {', '.join(SUPPORTED_ENCODERS)}")
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:
        raise ImportError("get_encoder_transform requires torchvision.") from exc
    operations = [transforms.Resize(config.image_size, interpolation=InterpolationMode.BICUBIC),
                  transforms.CenterCrop(config.image_size)]
    if repeat_grayscale:
        operations.append(transforms.Lambda(
            lambda image: image.convert("RGB") if getattr(image, "mode", None) in ("1", "L", "I", "F") else image
        ))
    operations.extend([transforms.ToTensor(), transforms.Normalize(config.mean, config.std)])
    return transforms.Compose(operations)

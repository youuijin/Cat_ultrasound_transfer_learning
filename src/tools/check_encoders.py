from src.encoders import get_encoder

from src.config.encoder_paths import (
    USFM_SOURCE,
    USFM_CHECKPOINT,
    OPENUS_SOURCE,
    OPENUS_CHECKPOINT,
)


MODEL_CONFIGS = [
    {
        "name": "vit_b16_scratch",
    },
    {
        "name": "vit_b16_imagenet",
    },
    {
        "name": "dinov2_vitb14",
    },
    {
        "name": "biomedclip_vitb16",
    },
    {
        "name": "usfm",
        "source_path": str(USFM_SOURCE),
        "checkpoint_path": str(USFM_CHECKPOINT),
    },
    # {
    #     "name": "openus_vmamba_s",
    #     "source_path": str(OPENUS_SOURCE),
    #     "checkpoint_path": str(OPENUS_CHECKPOINT),
    # },
]


for cfg in MODEL_CONFIGS:
    name = cfg["name"]

    print("=" * 80)
    print(f"Model: {name}")

    try:
        encoder = get_encoder(**cfg)

        total_params = sum(
            p.numel()
            for p in encoder.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in encoder.parameters()
            if p.requires_grad
        )

        print(f"Total params     : {total_params:,}")
        print(f"Trainable params : {trainable_params:,}")
        print(
            f"Total params (M) : "
            f"{total_params / 1e6:.2f} M"
        )

        if hasattr(encoder, "feature_dim"):
            print(
                f"Feature dim      : "
                f"{encoder.feature_dim}"
            )

        if hasattr(encoder, "get_model_info"):
            print("Model info:")
            print(
                encoder.get_model_info()
            )

    except Exception as e:
        print(f"[FAILED] {name}")
        print(e)
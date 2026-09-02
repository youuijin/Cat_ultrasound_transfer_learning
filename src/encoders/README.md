# Unified vision encoders

```python
from src.encoders import PreprocessConfig, get_encoder

scratch = get_encoder("vit_b16_scratch")
imagenet = get_encoder("vit_b16_imagenet")
dinov2 = get_encoder("dinov2_vitb14")
biomedclip = get_encoder("biomedclip_vitb16")

features = dinov2.forward_features(images)              # [B, 768] CLS
patches = dinov2.forward_features(images, return_spatial=True)  # [B, N, 768]
```

The two torchvision ViTs likewise return `[B, 768]` CLS features or `[B, N,
768]` patch tokens. BiomedCLIP returns its native projected global image
embedding. Its patch-token API varies between OpenCLIP versions, so spatial
extraction deliberately raises `NotImplementedError`.

One-channel NCHW tensors are repeated to RGB by default inside wrappers. Set
`repeat_grayscale=False` to require already-RGB input. `get_encoder_transform`
provides deterministic PIL grayscale-to-RGB conversion and native normalization
for the first four settings. BiomedCLIP also retains the authoritative transform
as `encoder.official_preprocess`.

## Official USFM and OpenUS source integration

Architecture details are intentionally not copied or guessed. Check out each
official repository yourself, install its documented dependencies, identify its
official builder, and pass that builder plus the official preprocessing values:

```python
native_preprocess = PreprocessConfig(
    input_channels=3, image_size=OFFICIAL_SIZE,
    mean=OFFICIAL_MEAN, std=OFFICIAL_STD, patch_size=OFFICIAL_PATCH_SIZE,
)

usfm = get_encoder(
    "usfm", checkpoint_path="/weights/USFM_latest.pth",
    source_path="/repos/official-usfm",
    model_factory="official_python_module:model_builder",
    factory_kwargs={"official_argument": "value"},
    feature_dim=OFFICIAL_FEATURE_DIM,
    preprocess_config=native_preprocess,
)

openus = get_encoder(
    "openus_vmamba_s", checkpoint_path="/weights/openus_vmamba_s.pth",
    source_path="/repos/official-openus",
    model_factory="official_python_module:vmamba_s_builder",
    factory_kwargs={"official_argument": "value"},
    feature_dim=OFFICIAL_FEATURE_DIM,
    preprocess_config=native_preprocess,
)
```

`model_factory` may instead be a callable. Checkpoints must exist and default to
strict loading. Nested `model` and `state_dict` formats are recognized. A
non-strict load is still rejected if it reports any missing or unexpected keys.

USFM delegates spatial extraction only when its official `forward_features`
accepts `return_spatial`. OpenUS returns the native `forward_features` tensor for
spatial requests and mean-pools 3D/4D native features for global requests.
Because official repository APIs can change, adapt a small callable locally if
their builder or feature API differs; do not replace either native architecture.

All wrappers support `freeze()`, `unfreeze()`, and model information via
`get_model_info()`. Reliable last-N-block unfreezing is implemented only for the
three ViT wrappers whose block layout is known; external USFM/OpenUS wrappers
explicitly leave it unsupported.

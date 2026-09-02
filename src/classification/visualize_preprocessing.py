from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.classification.dataset import (
    CatImageTransform,
    discover_cat_subjects,
    four_class_classification_subjects,
    load_nifti_image,
)
from src.classification.model import ENCODER_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save original/preprocessed/augmented comparisons")
    parser.add_argument("--data-root", default="data/cat_dataset")
    parser.add_argument("--encoder", choices=tuple(ENCODER_NAMES), default="vit_b16")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/outputs/preprocessing_preview"))
    return parser.parse_args()


def _config_for(name: str):
    # Static metadata avoids downloading/loading the pretrained model for visualization.
    if name == "biomedclip":
        from src.encoders.base import PreprocessConfig
        return PreprocessConfig(3, 224, (0.48145466, 0.4578275, 0.40821073),
                                (0.26862954, 0.26130258, 0.27577711), 16)
    from src.encoders.vit import IMAGENET_PREPROCESS
    return IMAGENET_PREPROCESS


def _tensor_to_image(tensor: torch.Tensor, mean, std) -> Image.Image:
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    array = ((tensor.cpu() * std_tensor + mean_tensor).clamp(0, 1) * 255)
    array = array.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _fit_for_display(image: Image.Image, size: int) -> Image.Image:
    display = Image.new("RGB", (size, size), (64, 64, 64))
    fitted = image.convert("RGB").copy()
    fitted.thumbnail((size, size), Image.Resampling.BILINEAR)
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    display.paste(fitted, (left, top))
    return display


def _comparison(original: Image.Image, preprocessed: Image.Image, augmented: Image.Image,
                label: str, source: str) -> Image.Image:
    size, header, footer = 224, 28, 34
    canvas = Image.new("RGB", (size * 3, header + size + footer), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate((
        ("Original", _fit_for_display(original, size)),
        ("Preprocessed", preprocessed),
        ("Augmented", augmented),
    )):
        draw.text((index * size + 6, 7), title, fill="black")
        canvas.paste(image, (index * size, header))
    draw.text((6, header + size + 8), f"{label} | {source}", fill="black")
    return canvas


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    torch.manual_seed(args.seed)
    subjects, _, _ = discover_cat_subjects(args.data_root)
    subjects = four_class_classification_subjects(subjects)
    samples = [(subject, side, path) for subject in subjects
               for side, path in sorted(subject.images.items())]
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(samples), generator=generator)[:args.num_images].tolist()
    config = _config_for(args.encoder)
    preprocess = CatImageTransform(config.image_size, config.input_channels,
                                   config.mean, config.std, False)
    augment = CatImageTransform(config.image_size, config.input_channels,
                                config.mean, config.std, True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output_index, sample_index in enumerate(indices, start=1):
        subject, side, path = samples[sample_index]
        original = load_nifti_image(path)
        preprocessed = _tensor_to_image(preprocess(original), config.mean, config.std)
        augmented = _tensor_to_image(augment(original), config.mean, config.std)
        panel = _comparison(original, preprocessed, augmented,
                            subject.class_name, f"{subject.subject_id} {side}")
        panel.save(args.output_dir / f"sample_{output_index:02d}.png")
    print(f"Saved {len(indices)} comparisons to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

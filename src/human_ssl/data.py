from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

from analysis.utils.dataset_io import load_analysis_samples
from src.classification.training_utils import seed_worker


DEFAULT_ROOTS = {
    "human1": Path("D:/_EUIJIN/Dataset/Human Ultrasound/NormalSegmentation"),
    "human2": Path("D:/_EUIJIN/Dataset/Human Ultrasound/OpenKidney"),
    "human3": Path("D:/_EUIJIN/Dataset/Human Ultrasound/Stone"),
}


@dataclass(frozen=True)
class SSLSample:
    path: Path
    dataset_id: str
    split_group: str
    subject_level: bool


def _sample_group(dataset_id: str, path: Path) -> tuple[str, bool]:
    if dataset_id == "human2":
        # OpenKidney filenames use the leading anonymous patient identifier,
        # e.g. 100_IM-0246-0030_anon.png.
        return path.stem.split("_", 1)[0], True
    # No patient identifier is exposed by the current Human1/Human3 layout.
    return path.stem, False


def discover_ssl_samples(roots: dict[str, Path]) -> list[SSLSample]:
    samples: list[SSLSample] = []
    for dataset_id, root in roots.items():
        for item in load_analysis_samples(dataset_id, root):
            group, subject_level = _sample_group(dataset_id, item.path)
            samples.append(SSLSample(item.path, dataset_id, group, subject_level))
    return samples


def split_ssl_samples(samples: list[SSLSample], val_fraction: float, seed: int,
                      max_images: int | None = None) -> tuple[list[SSLSample], list[SSLSample]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    rng = random.Random(seed)
    by_dataset: dict[str, list[SSLSample]] = defaultdict(list)
    for sample in samples:
        by_dataset[sample.dataset_id].append(sample)

    train, validation = [], []
    for dataset_id, dataset_samples in sorted(by_dataset.items()):
        groups: dict[str, list[SSLSample]] = defaultdict(list)
        for sample in dataset_samples:
            groups[sample.split_group].append(sample)
        keys = sorted(groups)
        rng.shuffle(keys)
        val_groups = max(1, round(len(keys) * val_fraction))
        val_keys = set(keys[:val_groups])
        validation.extend(sample for key in keys if key in val_keys for sample in groups[key])
        train.extend(sample for key in keys if key not in val_keys for sample in groups[key])

    if max_images is not None:
        if max_images < 2:
            raise ValueError("max_images must be at least two.")
        train_limit = max(1, round(max_images * (1 - val_fraction)))
        val_limit = max(1, max_images - train_limit)
        rng.shuffle(train); rng.shuffle(validation)
        train, validation = train[:train_limit], validation[:val_limit]
    return train, validation


class HumanSSLDataset(Dataset):
    def __init__(self, samples: list[SSLSample], image_size: int, training: bool) -> None:
        self.samples, self.image_size, self.training = samples, image_size, training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        width, height = image.size
        size = max(width, height)
        left, top = (size - width) // 2, (size - height) // 2
        image = TF.pad(image, [left, top, size - width - left, size - height - top], fill=0)
        image = TF.resize(image, [self.image_size, self.image_size],
                          InterpolationMode.BICUBIC, antialias=True)
        if self.training and torch.rand(()) < 0.5:
            image = TF.hflip(image)
        return TF.to_tensor(image), sample.dataset_id


class DINOViewTransform:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size
        self.crop = transforms.RandomResizedCrop(
            image_size, scale=(0.80, 1.0), ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.jitter = transforms.ColorJitter(brightness=0.20, contrast=0.20)

    def __call__(self, source: Image.Image) -> torch.Tensor:
        image = self.crop(source)
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
        if torch.rand(()) < 0.8:
            image = self.jitter(image)
        if torch.rand(()) < 0.5:
            image = TF.adjust_gamma(image, float(torch.empty(1).uniform_(0.85, 1.15)))
        return TF.to_tensor(image)


class HumanDINODataset(Dataset):
    def __init__(self, samples: list[SSLSample], image_size: int) -> None:
        self.samples, self.transform = samples, DINOViewTransform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        width, height = image.size
        size = max(width, height)
        left, top = (size - width) // 2, (size - height) // 2
        image = TF.pad(image, [left, top, size - width - left, size - height - top], fill=0)
        return self.transform(image), self.transform(image), sample.dataset_id


class BarlowViewTransform:
    """Conservative stochastic views for kidney ultrasound anatomy."""
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.crop = transforms.RandomResizedCrop(
            image_size, scale=(0.90, 1.0), ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.jitter = transforms.ColorJitter(brightness=0.12, contrast=0.12)

    def __call__(self, image: Image.Image):
        image = self.crop(image)
        if torch.rand(()) < 0.5: image = TF.hflip(image)
        angle = float(torch.empty(1).uniform_(-5.0, 5.0))
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        if torch.rand(()) < 0.8: image = self.jitter(image)
        if torch.rand(()) < 0.5:
            image = TF.adjust_gamma(image, float(torch.empty(1).uniform_(0.92, 1.08)))
        tensor = TF.to_tensor(image)
        if torch.rand(()) < 0.3: tensor = (tensor + torch.randn_like(tensor) * 0.01).clamp(0, 1)
        return tensor


class HumanBarlowDataset(Dataset):
    def __init__(self, samples, image_size, training):
        self.samples, self.image_size, self.training = samples, image_size, training
        self.view = BarlowViewTransform(image_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as source: image = source.convert("RGB")
        width, height = image.size; size = max(width, height)
        left, top = (size-width)//2, (size-height)//2
        image = TF.pad(image, [left, top, size-width-left, size-height-top], fill=0)
        if self.training: return self.view(image), self.view(image), sample.dataset_id
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BICUBIC, antialias=True)
        tensor = TF.to_tensor(image)
        return tensor, tensor.clone(), sample.dataset_id


def build_ssl_loaders(train_samples: list[SSLSample], val_samples: list[SSLSample],
                      image_size: int, batch_size: int, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                  worker_init_fn=seed_worker)
    train = DataLoader(HumanSSLDataset(train_samples, image_size, True), shuffle=True,
                       generator=generator, **common)
    validation = DataLoader(HumanSSLDataset(val_samples, image_size, False), shuffle=False,
                            **common)
    return train, validation


def build_dino_loaders(train_samples: list[SSLSample], val_samples: list[SSLSample],
                       image_size: int, batch_size: int, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                  worker_init_fn=seed_worker)
    train = DataLoader(HumanDINODataset(train_samples, image_size), shuffle=True,
                       generator=generator, **common)
    validation = DataLoader(HumanDINODataset(val_samples, image_size), shuffle=False,
                            **common)
    return train, validation


def build_barlow_loaders(train_samples, val_samples, image_size, batch_size, workers, seed):
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                  worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(HumanBarlowDataset(train_samples, image_size, True), shuffle=True,
                       generator=generator, **common)
    validation = DataLoader(HumanBarlowDataset(val_samples, image_size, True), shuffle=False,
                            **common)
    return train, validation


def dataset_counts(samples: list[SSLSample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.dataset_id for sample in samples).items()))

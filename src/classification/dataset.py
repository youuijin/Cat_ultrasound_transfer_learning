from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

SIDES = ("LK", "RK")
BINARY_CLASS_NAMES = ("normal", "abnormal")
FOUR_CLASS_NAMES = ("Normal", "AKI", "CKD", "ACKD")
ABNORMAL_SUBTYPE_NAMES = ("AKI", "CKD", "ACKD")
FOUR_CLASS_MAP = {
    "1": "Normal", "8": "Normal", "2": "AKI", "3": "CKD",
    "4": "CKD", "5": "CKD", "6": "CKD", "7": "ACKD",
}


@dataclass(frozen=True)
class CatSubject:
    class_name: str
    class_index: int
    subject_id: str
    directory: Path
    images: dict[str, Path]


def discover_cat_subjects(root: str | Path) -> tuple[list[CatSubject], list[str], list[dict[str, Any]]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Cat dataset root not found: {root}")
    class_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    class_names = [path.name for path in class_dirs]
    subjects: list[CatSubject] = []
    issues: list[dict[str, Any]] = []
    for class_index, class_dir in enumerate(class_dirs):
        for subject_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            images: dict[str, Path] = {}
            for side in SIDES:
                candidates = sorted(subject_dir.glob(f"*_{side}*.nii"))
                if len(candidates) == 1:
                    images[side] = candidates[0]
                elif len(candidates) > 1:
                    issues.append({"type": "ambiguous_kidney_image", "class": class_dir.name,
                                   "subject": subject_dir.name, "side": side,
                                   "candidates": [str(path) for path in candidates]})
            if images:
                subjects.append(CatSubject(class_dir.name, class_index, subject_dir.name,
                                           subject_dir, images))
            else:
                issues.append({"type": "excluded_subject_without_kidney_image",
                               "class": class_dir.name, "subject": subject_dir.name})
    return subjects, class_names, issues


def binary_classification_subjects(subjects: list[CatSubject]) -> list[CatSubject]:
    result = []
    for subject in subjects:
        prefix = subject.class_name.split(".", 1)[0].strip()
        index = 0 if prefix in ("1", "8") else 1
        result.append(replace(subject, class_name=BINARY_CLASS_NAMES[index], class_index=index))
    return result


def four_class_classification_subjects(subjects: list[CatSubject]) -> list[CatSubject]:
    result = []
    for subject in subjects:
        prefix = subject.class_name.split(".", 1)[0].strip()
        if prefix not in FOUR_CLASS_MAP:
            raise ValueError(f"Unexpected disease folder: {subject.class_name}")
        name = FOUR_CLASS_MAP[prefix]
        result.append(replace(subject, class_name=name, class_index=FOUR_CLASS_NAMES.index(name)))
    return result


def abnormal_subtype_subjects(subjects: list[CatSubject]) -> list[CatSubject]:
    return [replace(subject, class_index=ABNORMAL_SUBTYPE_NAMES.index(subject.class_name))
            for subject in subjects if subject.class_name != "Normal"]


def stratified_fold_split(subjects: list[CatSubject], num_folds: int, fold: int, seed: int):
    if num_folds < 2 or not 0 <= fold < num_folds:
        raise ValueError("num_folds must be >=2 and fold must be in [0, num_folds).")
    grouped: dict[int, list[CatSubject]] = {}
    for subject in subjects:
        grouped.setdefault(subject.class_index, []).append(subject)
    if not grouped or min(map(len, grouped.values())) < num_folds:
        raise ValueError("num_folds exceeds the number of subjects in the smallest class.")
    train, val = [], []
    for class_index, class_subjects in sorted(grouped.items()):
        shuffled = list(class_subjects)
        random.Random(seed + class_index).shuffle(shuffled)
        for index, subject in enumerate(shuffled):
            (val if index % num_folds == fold else train).append(subject)
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(val)
    return train, val


def _uint8_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    low, high = np.percentile(array, (1, 99))
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    return (np.clip((array - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def load_nifti_image(path: str | Path) -> Image.Image:
    array = np.asanyarray(nib.load(path).dataobj)
    if array.dtype.names:
        array = np.stack([array[name] for name in array.dtype.names[:3]], axis=-1)
    if array.ndim >= 3 and array.shape[2] == 1:
        array = np.squeeze(array, axis=2)
    if array.ndim not in (2, 3):
        raise ValueError(f"Unsupported NIfTI image shape {array.shape} at {path}")
    return Image.fromarray(_uint8_image(np.swapaxes(array, 0, 1)))


def _pad_to_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    size = max(width, height)
    left, top = (size - width) // 2, (size - height) // 2
    return TF.pad(image, [left, top, size - width - left, size - height - top], fill=0)


class CatImageTransform:
    def __init__(self, image_size: int, channels: int, mean: tuple[float, ...],
                 std: tuple[float, ...], training: bool,
                 augmentation: str = "baseline") -> None:
        self.image_size, self.channels, self.training = image_size, channels, training
        self.augmentation = augmentation
        jitter_strength = 0.20 if augmentation == "strong" else 0.25
        self.jitter = transforms.ColorJitter(
            brightness=jitter_strength, contrast=jitter_strength)
        self.normalize = transforms.Normalize(mean, std)
        self.erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), value=0.0)

    def __call__(self, image: Image.Image) -> Tensor:
        image = _pad_to_square(image.convert("L" if self.channels == 1 else "RGB"))
        image = TF.resize(image, [self.image_size] * 2, InterpolationMode.BILINEAR, antialias=True)
        if self.training:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
            scale_range = (0.90, 1.10) if self.augmentation == "strong" else (0.85, 1.15)
            params = transforms.RandomAffine.get_params(
                (-15.0, 15.0), (0.08, 0.08), scale_range, (-5.0, 5.0),
                [self.image_size, self.image_size])
            image = TF.affine(image, *params, interpolation=InterpolationMode.BILINEAR, fill=0)
            image = self.jitter(image)
            if self.augmentation == "strong":
                image = TF.adjust_gamma(image, float(torch.empty(1).uniform_(0.85, 1.15)))
        raw_tensor = TF.to_tensor(image)
        if self.training and self.augmentation == "strong":
            gaussian = torch.randn_like(raw_tensor) * 0.02
            speckle = raw_tensor * torch.randn_like(raw_tensor) * 0.03
            raw_tensor = (raw_tensor + gaussian + speckle).clamp(0, 1)
        tensor = self.normalize(raw_tensor)
        if self.training:
            if self.augmentation == "baseline":
                tensor = tensor + torch.randn_like(tensor) * 0.03
            tensor = self.erasing(tensor)
        return tensor


class CatSingleKidneyClassificationDataset(Dataset):
    def __init__(self, subjects: list[CatSubject], transform: CatImageTransform) -> None:
        self.samples = [(subject, path) for subject in subjects
                        for _, path in sorted(subject.images.items())]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        subject, image_path = self.samples[index]
        return (self.transform(load_nifti_image(image_path)),
                torch.tensor(subject.class_index, dtype=torch.long), str(subject.directory))

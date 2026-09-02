from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import nibabel as nib
import nrrd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.classification.data import split_subjects
from src.classification.dataset import CatSubject, load_nifti_image
from src.classification.training_utils import seed_worker


@dataclass(frozen=True)
class SegmentationSample:
    subject_id: str
    side: str
    image_path: Path
    mask_path: Path


def build_samples(subjects: list[CatSubject]) -> list[SegmentationSample]:
    samples = []
    for subject in subjects:
        for side, image_path in sorted(subject.images.items()):
            mask_path = subject.directory / f"{side}.seg.nrrd"
            if not mask_path.is_file():
                continue
            image_shape = tuple(nib.load(image_path).shape[:3])
            mask_shape = tuple(int(value) for value in nrrd.read_header(str(mask_path))["sizes"][:3])
            if image_shape == mask_shape:
                samples.append(SegmentationSample(subject.subject_id, side, image_path, mask_path))
    return samples


def load_mask(path: Path) -> Image.Image:
    array, _ = nrrd.read(str(path))
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.squeeze(array, axis=2)
    if array.ndim != 2 or array.min() < 0 or array.max() > 2:
        raise ValueError(f"Expected a 2D mask with labels 0/1/2: {path}")
    return Image.fromarray(np.swapaxes(array, 0, 1).astype(np.uint8), mode="L")


class PairedTransform:
    def __init__(self, size: int, mean, std, training: bool) -> None:
        self.size, self.training = size, training
        self.normalize = transforms.Normalize(mean, std)
        self.jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        square = max(width, height)
        padding = [(square - width) // 2, (square - height) // 2,
                   square - width - (square - width) // 2,
                   square - height - (square - height) // 2]
        image, mask = TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=0)
        output_size = [self.size, self.size]
        image = TF.resize(image, output_size, InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, output_size, InterpolationMode.NEAREST)
        if self.training:
            if torch.rand(()) < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle, translate, scale, shear = transforms.RandomAffine.get_params(
                (-8.0, 8.0), (0.04, 0.04), (0.96, 1.04), None, output_size)
            image = TF.affine(image, angle, translate, scale, shear,
                              interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear,
                             interpolation=InterpolationMode.NEAREST, fill=0)
            image = self.jitter(image)
        return self.normalize(TF.to_tensor(image)), TF.pil_to_tensor(mask).squeeze(0).long()


class CatSegmentationDataset(Dataset):
    def __init__(self, samples: list[SegmentationSample], transform: PairedTransform) -> None:
        self.samples, self.transform = samples, transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image, mask = self.transform(load_nifti_image(sample.image_path), load_mask(sample.mask_path))
        return image, mask, f"{sample.subject_id}_{sample.side}"


def select_train_subjects(subjects: list[CatSubject], fraction: float,
                          fraction_seed: int) -> list[CatSubject]:
    """Select a deterministic nested prefix of fold-training subjects."""
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}.")
    if not subjects:
        raise ValueError("The fold contains no training subjects.")
    if fraction >= 1.0:
        return list(subjects)
    rng = np.random.default_rng(fraction_seed)
    order = rng.permutation(len(subjects))
    count = max(1, math.ceil(len(subjects) * fraction))
    return [subjects[int(index)] for index in order[:count]]


def build_segmentation_loaders(root, folds, fold, split_seed, config, batch_size, workers, seed,
                               train_fraction: float | None = None,
                               train_fraction_seed: int | None = None,
                               train_subject_list: str | Path | None = None):
    train_subjects, val_subjects, _, issues = split_subjects(root, "four_class", folds, fold, split_seed)
    full_train_subjects = list(train_subjects)
    full_train_samples = build_samples(full_train_subjects)
    if train_fraction is not None and train_subject_list is not None:
        raise ValueError("Specify either train_fraction or train_subject_list, not both.")
    if train_subject_list is not None:
        list_path = Path(train_subject_list)
        requested_ids = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError(f"Duplicate subject IDs in train_subject_list: {list_path}")
        by_id = {subject.subject_id: subject for subject in full_train_subjects}
        unknown = sorted(set(requested_ids) - set(by_id))
        if unknown:
            raise ValueError(f"Subject IDs are not in the original fold train split: {unknown}")
        train_subjects = [by_id[subject_id] for subject_id in requested_ids]
        if not train_subjects:
            raise ValueError("train_subject_list selected no subjects.")
    elif train_fraction is not None:
        train_subjects = select_train_subjects(
            full_train_subjects,
            train_fraction,
            split_seed if train_fraction_seed is None else train_fraction_seed,
        )
    train_samples = (full_train_samples if len(train_subjects) == len(full_train_subjects)
                     else build_samples(train_subjects))
    val_samples = build_samples(val_subjects)
    if not train_samples:
        raise ValueError("Training-subject fraction produced no valid segmentation images.")
    train = CatSegmentationDataset(
        train_samples, PairedTransform(config.image_size, config.mean, config.std, True))
    val = CatSegmentationDataset(
        val_samples, PairedTransform(config.image_size, config.mean, config.std, False))
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0, worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(seed)
    subset_metadata = {
        "num_train_subjects_full": len(full_train_subjects),
        "num_train_subjects_used": len(train_subjects),
        "num_train_images_full": len(full_train_samples),
        "num_train_images_used": len(train_samples),
    }
    return (DataLoader(train, shuffle=True, generator=generator, **common),
            DataLoader(val, shuffle=False, **common), train_subjects, val_subjects, issues,
            subset_metadata)

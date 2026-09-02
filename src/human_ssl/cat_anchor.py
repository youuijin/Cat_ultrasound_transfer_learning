"""Unlabeled Cat train-fold input pipeline for target-aware Human MAE anchoring."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, functional as TF

from src.classification.data import split_subjects
from src.classification.dataset import load_nifti_image
from src.classification.training_utils import seed_worker


class UnlabeledCatDataset(Dataset):
    def __init__(self, subjects, image_size: int, training: bool):
        self.samples = [(subject.subject_id, side, path) for subject in subjects
                        for side, path in sorted(subject.images.items())]
        self.image_size, self.training = image_size, training

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        subject, side, path = self.samples[index]
        image = load_nifti_image(path).convert("RGB")
        width, height = image.size; size = max(width, height)
        left, top = (size-width)//2, (size-height)//2
        image = TF.pad(image, [left, top, size-width-left, size-height-top], fill=0)
        image = TF.resize(image, [self.image_size, self.image_size],
                          InterpolationMode.BICUBIC, antialias=True)
        if self.training and torch.rand(()) < 0.5: image = TF.hflip(image)
        return TF.to_tensor(image), f"{subject}_{side}"


def cat_anchor_split(root, folds=5, fold=0, split_seed=42):
    train, validation, _classes, _issues = split_subjects(root, "four_class", folds, fold, split_seed)
    train_ids, val_ids = {x.subject_id for x in train}, {x.subject_id for x in validation}
    overlap = train_ids & val_ids
    if overlap: raise RuntimeError(f"Cat train/validation subject leakage: {sorted(overlap)}")
    return train, validation


def build_cat_anchor_loader(root, image_size, batch_size, workers, seed,
                            folds=5, fold=0, split_seed=42, training=True):
    train, validation = cat_anchor_split(root, folds, fold, split_seed)
    dataset = UnlabeledCatDataset(train, image_size, training)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training,
                        generator=generator if training else None, num_workers=workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                        worker_init_fn=seed_worker)
    return loader, train, validation


def write_subject_csv(path: Path, subjects):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id", "directory")); writer.writeheader()
        writer.writerows({"subject_id": x.subject_id, "directory": str(x.directory)} for x in subjects)

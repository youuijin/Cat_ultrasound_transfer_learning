from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.encoders import PreprocessConfig
from src.classification.dataset import (
    ABNORMAL_SUBTYPE_NAMES, BINARY_CLASS_NAMES, FOUR_CLASS_NAMES,
    CatImageTransform, CatSingleKidneyClassificationDataset, CatSubject,
    abnormal_subtype_subjects, binary_classification_subjects,
    discover_cat_subjects, four_class_classification_subjects, stratified_fold_split,
)
from src.classification.training_utils import class_weights, seed_worker


def split_subjects(root: str, mode: str, folds: int, fold: int, split_seed: int):
    subjects, class_names, issues = discover_cat_subjects(root)
    if mode == "binary":
        subjects, class_names = binary_classification_subjects(subjects), list(BINARY_CLASS_NAMES)
    else:
        subjects, class_names = four_class_classification_subjects(subjects), list(FOUR_CLASS_NAMES)
    train, val = stratified_fold_split(subjects, folds, fold, split_seed)
    if mode == "abnormal_subtype":
        train, val = abnormal_subtype_subjects(train), abnormal_subtype_subjects(val)
        class_names = list(ABNORMAL_SUBTYPE_NAMES)
    return train, val, class_names, issues


def _dataset(subjects: list[CatSubject], config: PreprocessConfig, training: bool,
             augmentation: str = "baseline"):
    transform = CatImageTransform(
        config.image_size, config.input_channels, config.mean, config.std, training,
        augmentation if training else "baseline",
    )
    return CatSingleKidneyClassificationDataset(subjects, transform)


def build_loaders(train_subjects, val_subjects, config, batch_size, workers, seed,
                  num_classes, augmentation="baseline", use_sampler=False,
                  subject_sqrt_weighting=False):
    train_dataset = _dataset(train_subjects, config, True, augmentation)
    val_dataset = _dataset(val_subjects, config, False)
    targets = [subject.class_index for subject in train_subjects for _ in subject.images]
    if subject_sqrt_weighting:
        subject_counts = torch.bincount(
            torch.tensor([subject.class_index for subject in train_subjects]),
            minlength=num_classes).float()
        weights = subject_counts.clamp_min(1).rsqrt()
        weights /= weights.mean()
    else:
        weights = class_weights(targets, num_classes)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if use_sampler:
        sample_weights = weights[torch.tensor(targets)]
        sampler = WeightedRandomSampler(sample_weights, len(targets), replacement=True,
                                        generator=generator)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler,
                              generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, weights if subject_sqrt_weighting else None

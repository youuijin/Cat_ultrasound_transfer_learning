from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from analysis.utils.utils import load_roi_mask
from src.analysis.analyze_kidney_bbox import bbox_from_mask
from src.classification.data import split_subjects
from src.classification.dataset import CatSubject, load_nifti_image
from src.classification.training_utils import seed_worker


class DetectionTransform:
    def __init__(self, image_size: int, mean, std, training: bool) -> None:
        self.size, self.training = image_size, training
        self.normalize = transforms.Normalize(mean, std)
        self.jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        width, height = image.size
        size = max(width, height)
        padding = [(size - width) // 2, (size - height) // 2,
                   size - width - (size - width) // 2, size - height - (size - height) // 2]
        image, mask = TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=0)
        output_size = [self.size, self.size]
        image = TF.resize(image, output_size, InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, output_size, InterpolationMode.NEAREST)
        if self.training:
            if torch.rand(()) < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle, translate, scale, shear = transforms.RandomAffine.get_params(
                (-15.0, 15.0), (0.08, 0.08), (0.85, 1.15), (-5.0, 5.0), output_size)
            image = TF.affine(image, angle, translate, scale, shear,
                              interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear,
                             interpolation=InterpolationMode.NEAREST, fill=0)
            image = self.jitter(image)
        mask_array = np.asarray(mask, dtype=np.uint8) > 0
        x0, y0, x1, y1 = bbox_from_mask(mask_array)
        target = torch.tensor([((x0 + x1) / 2) / self.size, ((y0 + y1) / 2) / self.size,
                               (x1 - x0) / self.size, (y1 - y0) / self.size], dtype=torch.float32)
        tensor = self.normalize(TF.to_tensor(image))
        if self.training:
            tensor = tensor + torch.randn_like(tensor) * 0.03
        return tensor, target


class KidneyDetectionDataset(Dataset):
    def __init__(self, subjects: list[CatSubject], root: Path, transform: DetectionTransform) -> None:
        self.root, self.transform = root, transform
        self.samples = []
        for subject in subjects:
            for side, path in sorted(subject.images.items()):
                if (path.parent / f"{side}.seg.nrrd").is_file():
                    self.samples.append((subject, side, path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        subject, side, path = self.samples[index]
        image = load_nifti_image(path)
        mask = load_roi_mask("cat", SimpleNamespace(path=path, side=side), self.root,
                             (image.height, image.width))
        mask_image = Image.fromarray(mask.astype("uint8") * 255, mode="L")
        image_tensor, target = self.transform(image, mask_image)
        return image_tensor, target, f"{subject.subject_id}_{side}"


def build_detection_loaders(root, train_subjects, val_subjects, config, batch_size, workers, seed):
    train = KidneyDetectionDataset(
        train_subjects, Path(root), DetectionTransform(config.image_size, config.mean, config.std, True))
    val = KidneyDetectionDataset(
        val_subjects, Path(root), DetectionTransform(config.image_size, config.mean, config.std, False))
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=workers > 0, worker_init_fn=seed_worker)
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(train, shuffle=True, generator=generator, **common),
            DataLoader(val, shuffle=False, **common))


def detection_split(root, folds, fold, split_seed):
    train, val, _, issues = split_subjects(root, "four_class", folds, fold, split_seed)
    return train, val, issues

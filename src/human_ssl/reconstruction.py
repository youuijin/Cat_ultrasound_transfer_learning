"""Fixed-mask reconstruction diagnostics for the repository VisionMAE."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def _pixel_mask(model, patch_mask: torch.Tensor) -> torch.Tensor:
    values = patch_mask.unsqueeze(-1).expand(-1, -1, model.patch_size ** 2 * 3)
    return model.unpatchify(values)[:, :1]


def _gradient_error(target: np.ndarray, reconstruction: np.ndarray,
                    mask: np.ndarray) -> tuple[float, int]:
    target_y, target_x = np.gradient(target.mean(2))
    recon_y, recon_x = np.gradient(reconstruction.mean(2))
    difference = np.abs(np.hypot(target_x, target_y) - np.hypot(recon_x, recon_y))
    return float(difference[mask].sum()), int(mask.sum())


def _save_example(path: Path, tensors: tuple[torch.Tensor, ...], label: str) -> None:
    names = ("Original", "Masked input", "Reconstruction", "Composite")
    size, caption = tensors[0].shape[-1], 26
    canvas = Image.new("RGB", (size * 4, size + caption), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, tensor) in enumerate(zip(names, tensors)):
        array = (tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        canvas.paste(Image.fromarray(array, mode="RGB"), (column * size, 0))
        draw.text((column * size + 4, size + 5), name, fill="black")
    draw.text((4, 4), label, fill="white")
    canvas.save(path)


@torch.no_grad()
def evaluate_fixed_reconstruction(model, loader, device: torch.device, mask_ratio: float,
                                  mask_seed: int, run_name: str, epoch: int,
                                  output_dir: Path, example_count: int = 3) -> dict:
    model.eval()
    example_dir = output_dir / "recon_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    squared_sum = ssim_sum = gradient_sum = 0.0
    masked_values = ssim_count = gradient_count = image_count = examples = 0
    for batch_index, (images, _dataset_ids) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(mask_seed + batch_index)
            if device.type == "cuda": torch.cuda.manual_seed_all(mask_seed + batch_index)
            _loss, prediction, patch_mask = model(images, mask_ratio)
        target_patches = model.patchify(images)
        if model.norm_pixel_loss:
            mean = target_patches.mean(-1, keepdim=True)
            variance = target_patches.var(-1, keepdim=True, unbiased=False)
            prediction = prediction * (variance + 1e-6).sqrt() + mean
        reconstruction = model.unpatchify(prediction).clamp(0, 1)
        pixel_mask = _pixel_mask(model, patch_mask)
        masked = images * (1 - pixel_mask)
        composite = masked + reconstruction * pixel_mask
        squared_sum += float(((reconstruction - images).square() * pixel_mask).sum())
        masked_values += int(pixel_mask.sum().item() * images.shape[1])
        for index in range(images.shape[0]):
            target = images[index].permute(1, 2, 0).float().cpu().numpy()
            reconstructed = reconstruction[index].permute(1, 2, 0).float().cpu().numpy()
            mask = pixel_mask[index, 0].bool().cpu().numpy()
            if structural_similarity is not None:
                _score, similarity = structural_similarity(
                    target, reconstructed, data_range=1.0, channel_axis=2, full=True)
                ssim_sum += float(similarity[mask].sum()); ssim_count += int(similarity[mask].size)
            value, count = _gradient_error(target, reconstructed, mask)
            gradient_sum += value; gradient_count += count
            if examples < example_count:
                _save_example(
                    example_dir / f"sample_{examples + 1:03d}_epoch{epoch:03d}.png",
                    (images[index], masked[index], reconstruction[index], composite[index]),
                    f"{run_name} | epoch {epoch}")
                examples += 1
        image_count += images.shape[0]
    mse = squared_sum / max(masked_values, 1)
    return {"run_name": run_name, "epoch": epoch, "n_validation_images": image_count,
            "mask_seed": mask_seed, "masked_mse": mse,
            "masked_psnr": -10 * math.log10(max(mse, 1e-12)),
            "masked_ssim": ssim_sum / ssim_count if ssim_count else float("nan"),
            "gradient_mae": gradient_sum / max(gradient_count, 1)}


def append_reconstruction_metrics(path: Path, row: dict) -> None:
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    rows = [value for value in rows if int(value["epoch"]) != int(row["epoch"])]
    rows.append(row); rows.sort(key=lambda value: int(value["epoch"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerows(rows)

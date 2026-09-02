from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from analysis.utils.utils import load_roi_mask
from src.classification.dataset import discover_cat_subjects, load_nifti_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cat kidney boxes from segmentation masks")
    parser.add_argument("--data-root", type=Path, default=Path("data/cat_dataset"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/outputs/kidney_bbox"))
    parser.add_argument("--num-previews", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot calculate a bounding box from an empty mask")
    # Exclusive maximum coordinates, matching common detection target conventions.
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalized_measurements(box: tuple[int, int, int, int], width: int, height: int):
    x0, y0, x1, y1 = box
    box_width, box_height = x1 - x0, y1 - y0
    return {
        "center_x": ((x0 + x1) / 2) / width,
        "center_y": ((y0 + y1) / 2) / height,
        "width": box_width / width,
        "height": box_height / height,
        "area_ratio": (box_width * box_height) / (width * height),
        "aspect_ratio": box_width / box_height,
    }


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for name in ("center_x", "center_y", "width", "height", "area_ratio", "aspect_ratio"):
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        result[name] = {"mean": float(values.mean()), "std": float(values.std()),
                        "min": float(values.min()), "max": float(values.max())}
    return result


def make_preview(items, output_path: Path) -> None:
    panel_size, caption_height = 320, 34
    canvas = Image.new("RGB", (panel_size * len(items), panel_size + caption_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image_path, box, label) in enumerate(items):
        image = load_nifti_image(image_path).convert("RGB")
        original_width, original_height = image.size
        display = image.resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        x0, y0, x1, y1 = box
        scaled = (x0 / original_width * panel_size, y0 / original_height * panel_size,
                  x1 / original_width * panel_size, y1 / original_height * panel_size)
        ImageDraw.Draw(display).rectangle(scaled, outline=(255, 40, 40), width=3)
        canvas.paste(display, (index * panel_size, 0))
        draw.text((index * panel_size + 5, panel_size + 8), label, fill="black")
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    subjects, _, discovery_issues = discover_cat_subjects(args.data_root)
    records: list[dict[str, float]] = []
    valid_items = []
    skipped = []
    for subject in subjects:
        for side, image_path in sorted(subject.images.items()):
            image = load_nifti_image(image_path)
            width, height = image.size
            sample = SimpleNamespace(path=image_path, side=side)
            try:
                mask = load_roi_mask("cat", sample, args.data_root, (height, width))
                box = bbox_from_mask(mask)
            except (FileNotFoundError, ValueError) as error:
                skipped.append({"path": str(image_path), "reason": str(error)})
                continue
            record = normalized_measurements(box, width, height)
            record.update({"subject": subject.subject_id, "side": side,
                           "image_path": str(image_path)})
            records.append(record)
            valid_items.append((image_path, box, f"{subject.subject_id} {side}"))
    if not records:
        raise RuntimeError("No valid image/mask pairs were found")
    stats = summarize(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"num_samples": len(records), "num_skipped": len(skipped), "statistics": stats,
              "skipped": skipped, "discovery_issues": discovery_issues}
    (args.output_dir / "bbox_statistics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    count = min(args.num_previews, len(valid_items))
    selected = [valid_items[index] for index in rng.choice(len(valid_items), count, replace=False)]
    make_preview(selected, args.output_dir / "bbox_sanity_check.png")
    print(json.dumps({"num_samples": len(records), "num_skipped": len(skipped), **stats}, indent=2))


if __name__ == "__main__":
    main()

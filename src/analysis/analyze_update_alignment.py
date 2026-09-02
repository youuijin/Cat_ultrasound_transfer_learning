"""Compare Human SSL and Cat fine-tuning parameter-update directions."""
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders import get_encoder


EPSILON = 1e-24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human SSL vs Cat segmentation encoder update alignment")
    parser.add_argument("--encoder", choices=("vit_b16",), required=True)
    parser.add_argument("--human-checkpoint", type=Path, required=True)
    parser.add_argument("--cat-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/update_alignment"))
    parser.add_argument("--output-name", default="human_mae_vs_cat_seg_fold0_update_alignment.csv")
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return payload


def tensor_state(payload: dict, label: str) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} checkpoint has no state_dict")
    return {str(key): value for key, value in state.items() if torch.is_tensor(value)}


def strip_common_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "student.", "teacher.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
                break
    return key


def extract_encoder_state(state: dict[str, torch.Tensor], base_keys: set[str],
                          label: str, segmentation: bool) -> tuple[dict[str, torch.Tensor], list[str]]:
    extracted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for original_key, value in state.items():
        key = original_key
        if segmentation:
            while key.startswith("module."):
                key = key[len("module."):]
            if not key.startswith("encoder."):
                # Segmentation decoder/head and other training state are outside
                # the comparison by definition.
                continue
            key = key[len("encoder."):]
        else:
            # Encoder-only repository checkpoints already use the exact
            # ``model.*`` namespace. Preserve an exact match before attempting
            # to remove wrappers used by other SSL checkpoint formats.
            if key not in base_keys:
                candidates = [key]
                while candidates[-1].startswith("module."):
                    candidates.append(candidates[-1][len("module."):])
                unwrapped = candidates[-1]
                for prefix in ("student.", "teacher.", "encoder."):
                    if unwrapped.startswith(prefix):
                        candidates.append(unwrapped[len(prefix):])
                candidates.extend(
                    f"model.{candidate}" for candidate in tuple(candidates)
                    if not candidate.startswith("model."))
                key = next((candidate for candidate in candidates if candidate in base_keys), key)
        if key in base_keys:
            if key in extracted:
                raise RuntimeError(f"{label}: duplicate encoder parameter {key!r}")
            extracted[key] = value
        else:
            # SSL decoder/projector/predictor tensors are outside the encoder.
            if segmentation and key.startswith("model."):
                ignored.append(original_key)
    return extracted, ignored


def validate_pair(label: str, base: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor],
                  ignored: list[str]) -> None:
    missing = sorted(set(base) - set(candidate))
    unexpected = sorted(ignored)
    mismatches = sorted(
        key for key in set(base) & set(candidate) if base[key].shape != candidate[key].shape)
    print(f"Base vs {label}:")
    print(f"Matched encoder parameters: {len(set(base) & set(candidate))}")
    print(f"Missing: {len(missing)}")
    print(f"Unexpected: {len(unexpected)}")
    print(f"Shape mismatches: {len(mismatches)}\n")
    if missing:
        raise RuntimeError(f"Base vs {label}: missing encoder parameters: {missing[:10]}")
    if mismatches:
        details = [(key, tuple(base[key].shape), tuple(candidate[key].shape))
                   for key in mismatches[:10]]
        raise RuntimeError(f"Base vs {label}: shape mismatches: {details}")


def validate_cat_metadata(payload: dict) -> None:
    raw_args = payload.get("args")
    if not isinstance(raw_args, dict):
        raise RuntimeError("Cat checkpoint has no saved args; cannot verify its run conditions.")
    required = {"task": "segmentation", "encoder": "vit_b16", "transfer": "full",
                "fold": 0, "seed": 0}
    mismatches = {key: (raw_args.get(key), expected) for key, expected in required.items()
                  if raw_args.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"Cat checkpoint run metadata does not match the required run: {mismatches}")
    # Older repository checkpoints predate --encoder-init. In those runs,
    # pretrained vit_b16 with transfer=full was unambiguously ImageNet initialized.
    encoder_init = raw_args.get("encoder_init")
    if encoder_init not in (None, "imagenet"):
        raise RuntimeError(
            f"Cat checkpoint must be ImageNet initialized, got encoder_init={encoder_init!r}")


def layer_group(name: str) -> tuple[str, int | str] | None:
    if (name == "model.class_token" or name.startswith("model.conv_proj.")
            or name == "model.encoder.pos_embedding"):
        return "patch_embed", "patch_embed"
    prefix = "model.encoder.layers.encoder_layer_"
    if name.startswith(prefix):
        suffix = name[len(prefix):]
        index_text = suffix.split(".", 1)[0]
        if index_text.isdigit():
            index = int(index_text)
            return f"block_{index}", index
    if name.startswith("model.encoder.ln."):
        return "final_norm", "final_norm"
    return None


def calculate(keys: list[str], base: dict[str, torch.Tensor], human: dict[str, torch.Tensor],
              cat: dict[str, torch.Tensor]) -> dict[str, float | int]:
    human_sq = cat_sq = dot = 0.0
    num_parameters = 0
    with torch.no_grad():
        for key in keys:
            base_value = base[key].detach().cpu().to(torch.float64)
            human_update = human[key].detach().cpu().to(torch.float64) - base_value
            cat_update = cat[key].detach().cpu().to(torch.float64) - base_value
            human_sq += float(torch.sum(human_update * human_update))
            cat_sq += float(torch.sum(cat_update * cat_update))
            dot += float(torch.sum(human_update * cat_update))
            num_parameters += base_value.numel()
    human_norm, cat_norm = math.sqrt(human_sq), math.sqrt(cat_sq)
    if human_norm <= EPSILON or cat_norm <= EPSILON:
        cosine = math.nan
        warnings.warn("Zero update norm encountered; cosine_similarity is NaN.", stacklevel=2)
    else:
        cosine = dot / (human_norm * cat_norm + EPSILON)
    return {"num_parameters": num_parameters, "human_update_norm": human_norm,
            "cat_update_norm": cat_norm, "norm_ratio": human_norm / (cat_norm + EPSILON),
            "dot_product": dot, "cosine_similarity": cosine,
            "projection_coefficient": dot / (cat_sq + EPSILON)}


def main() -> None:
    args = parse_args()
    base_encoder = get_encoder("vit_b16_imagenet", pretrained=True)
    base = {key: value.detach().cpu() for key, value in base_encoder.named_parameters()}
    if args.base_checkpoint:
        payload = load_payload(args.base_checkpoint)
        supplied, ignored = extract_encoder_state(
            tensor_state(payload, "Base"), set(base), "Base", segmentation=False)
        validate_pair("provided Base", base, supplied, ignored)
        base = supplied

    human_payload = load_payload(args.human_checkpoint)
    if human_payload.get("encoder_name") not in (None, "vit_b16_imagenet"):
        raise RuntimeError(f"Human checkpoint encoder_name is {human_payload.get('encoder_name')!r}")
    adaptation = human_payload.get("adaptation")
    if adaptation not in (None, "human_kidney_ultrasound_mae"):
        raise RuntimeError(f"Expected Human MAE checkpoint, got adaptation={adaptation!r}")
    human, human_ignored = extract_encoder_state(
        tensor_state(human_payload, "Human"), set(base), "Human", segmentation=False)

    cat_payload = load_payload(args.cat_checkpoint)
    validate_cat_metadata(cat_payload)
    cat, cat_ignored = extract_encoder_state(
        tensor_state(cat_payload, "Cat"), set(base), "Cat", segmentation=True)
    validate_pair("Human", base, human, human_ignored)
    validate_pair("Cat", base, cat, cat_ignored)

    grouped: dict[tuple[str, int | str], list[str]] = {}
    for key in base:
        group = layer_group(key)
        if group is None:
            raise RuntimeError(f"Unassigned ViT encoder parameter: {key}")
        grouped.setdefault(group, []).append(key)

    rows = [{"group_type": "global", "layer_index": "global", "layer_name": "encoder",
             **calculate(list(base), base, human, cat)}]
    ordered_groups = [("patch_embed", "patch_embed")] + [
        (f"block_{index}", index) for index in range(12)] + [("final_norm", "final_norm")]
    for layer_name, layer_index in ordered_groups:
        keys = grouped.get((layer_name, layer_index))
        if not keys:
            raise RuntimeError(f"No parameters found for {layer_name}")
        rows.append({"group_type": "layer", "layer_index": layer_index,
                     "layer_name": layer_name, **calculate(keys, base, human, cat)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_name
    columns = ("group_type", "layer_index", "layer_name", "num_parameters",
               "human_update_norm", "cat_update_norm", "norm_ratio", "dot_product",
               "cosine_similarity", "projection_coefficient")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    global_row = rows[0]
    print("============================================================")
    print("Human SSL vs Cat FT Update Alignment")
    print("============================================================\n")
    print("Global")
    print(f"Human update norm : {global_row['human_update_norm']:.8g}")
    print(f"Cat update norm   : {global_row['cat_update_norm']:.8g}")
    print(f"Norm ratio        : {global_row['norm_ratio']:.8g}")
    print(f"Cosine similarity : {global_row['cosine_similarity']:.8g}")
    print(f"Projection coeff. : {global_row['projection_coefficient']:.8g}\n")
    print(f"{'Layer':25s} {'Cosine':>12s} {'H/C norm ratio':>16s}")
    print("-" * 55)
    for row in rows[1:]:
        print(f"{row['layer_name']:25s} {row['cosine_similarity']:12.6g} {row['norm_ratio']:16.6g}")
    print(f"\nCSV saved to: {output}")


if __name__ == "__main__":
    main()

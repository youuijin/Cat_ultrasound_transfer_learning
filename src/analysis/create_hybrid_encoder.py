"""Create an ImageNet ViT-B/16 encoder with selected Human-SSL blocks."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.encoders import get_encoder


def parse_blocks(value: str) -> list[int]:
    try:
        blocks = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ssl-blocks must be comma-separated integers.") from exc
    if not blocks:
        raise argparse.ArgumentTypeError("--ssl-blocks cannot be empty.")
    invalid = [index for index in blocks if index < 0 or index > 11]
    if invalid:
        raise argparse.ArgumentTypeError(f"ViT-B/16 block indices must be 0-11: {invalid}")
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b16",), default="vit_b16")
    parser.add_argument("--ssl-checkpoint", type=Path, required=True)
    parser.add_argument("--ssl-blocks", type=parse_blocks, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_ssl_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SSL checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("SSL checkpoint must be a dictionary.")
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("SSL checkpoint must contain an encoder-only 'state_dict'.")
    if payload.get("encoder_name") not in (None, "vit_b16_imagenet"):
        raise ValueError(
            f"Expected encoder_name='vit_b16_imagenet', got {payload.get('encoder_name')!r}."
        )
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Checkpoint state_dict is empty or contains non-tensor values.")
    return state, payload


def block_prefixes(encoder) -> dict[int, str]:
    module_names = {id(module): name for name, module in encoder.named_modules()}
    prefixes = {}
    for index, block in enumerate(encoder.model.encoder.layers):
        module_name = module_names.get(id(block))
        if module_name is None:
            raise RuntimeError(f"Could not resolve module path for transformer block {index}.")
        prefixes[index] = module_name + "."
    return prefixes


@torch.no_grad()
def create_hybrid(ssl_state: dict[str, torch.Tensor], selected_blocks: list[int]):
    base_encoder = get_encoder("vit_b16_imagenet", pretrained=True).cpu().eval()
    base_state = base_encoder.state_dict()

    missing = sorted(set(base_state) - set(ssl_state))
    unexpected = sorted(set(ssl_state) - set(base_state))
    shape_mismatches = sorted(
        key for key in set(base_state) & set(ssl_state)
        if base_state[key].shape != ssl_state[key].shape
    )
    if missing:
        raise RuntimeError(f"Missing encoder parameters in SSL checkpoint: {missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected encoder parameters in SSL checkpoint: {unexpected[:10]}")
    if shape_mismatches:
        details = [
            f"{key}: ImageNet={tuple(base_state[key].shape)}, "
            f"SSL={tuple(ssl_state[key].shape)}"
            for key in shape_mismatches[:10]
        ]
        raise RuntimeError("Shape mismatches: " + "; ".join(details))

    prefixes = block_prefixes(base_encoder)
    selected_prefixes = tuple(prefixes[index] for index in selected_blocks)
    selected_keys = {
        key for key in base_state if key.startswith(selected_prefixes)
    }
    if not selected_keys:
        raise RuntimeError("No parameters matched the selected transformer blocks.")

    hybrid_state = {
        key: (ssl_state[key].detach().cpu().clone()
              if key in selected_keys else base_state[key].detach().cpu().clone())
        for key in base_state
    }
    base_encoder.load_state_dict(hybrid_state, strict=True)

    selected_match = all(torch.equal(hybrid_state[key], ssl_state[key].cpu())
                         for key in selected_keys)
    remaining_keys = set(base_state) - selected_keys
    remaining_match = all(torch.equal(hybrid_state[key], base_state[key].cpu())
                          for key in remaining_keys)
    if not selected_match:
        raise RuntimeError("Verification failed: selected blocks do not match SSL weights.")
    if not remaining_match:
        raise RuntimeError("Verification failed: non-selected parameters changed from ImageNet.")

    return hybrid_state, len(selected_keys), len(remaining_keys)


def main() -> None:
    args = parse_args()
    ssl_state, source_payload = load_ssl_state(args.ssl_checkpoint)
    hybrid_state, copied_count, kept_count = create_hybrid(
        ssl_state, args.ssl_blocks
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": source_payload.get(
            "format", "feline_transfer_learning.vision_encoder.v1"
        ),
        "encoder_name": "vit_b16_imagenet",
        "initialization": "ImageNet-1K supervised + selected Human SSL blocks",
        # Retain these fields so the existing --encoder-init Human SSL loader
        # validates this checkpoint exactly like its source encoder checkpoint.
        "adaptation": source_payload.get("adaptation"),
        "human_ssl_method": source_payload.get("human_ssl_method"),
        "state_dict": hybrid_state,
        "model_state_dict": {
            key.removeprefix("model."): value for key, value in hybrid_state.items()
        },
        "hybrid": True,
        "hybrid_base": "vit_b16_imagenet",
        "hybrid_ssl_source": str(args.ssl_checkpoint.expanduser().resolve()),
        "hybrid_ssl_blocks": args.ssl_blocks,
        "source_checkpoint_epoch": source_payload.get("epoch"),
    }
    torch.save(checkpoint, output)

    print(f"Selected SSL blocks: {args.ssl_blocks}\n")
    print(f"Parameters copied from SSL: {copied_count}")
    print(f"Parameters kept from ImageNet: {kept_count}\n")
    print("Verification:")
    print("selected blocks match SSL: PASS")
    print("remaining blocks match ImageNet: PASS\n")
    print(f"Saved:\n{output}")


if __name__ == "__main__":
    main()

"""Create a parameter-wise interpolation of ImageNet and Human-SSL ViT-B/16."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.encoders import get_encoder


def alpha_value(value: str) -> float:
    try:
        alpha = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--alpha must be a number.") from exc
    if not 0.0 <= alpha <= 1.0:
        raise argparse.ArgumentTypeError("--alpha must be in [0, 1].")
    return alpha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("vit_b16",), default="vit_b16")
    parser.add_argument("--ssl-checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=alpha_value, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def load_ssl_state(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Human SSL checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Human SSL checkpoint must be a dictionary.")
    for key in ("state_dict", "encoder", "model"):
        if isinstance(payload.get(key), dict):
            state = payload[key]
            break
    else:
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            state = payload
            payload = {}
        else:
            raise ValueError("No encoder state_dict was found in the checkpoint.")
    if payload.get("encoder_name") not in (None, "vit_b16_imagenet"):
        raise ValueError(
            f"Expected encoder_name='vit_b16_imagenet', got {payload.get('encoder_name')!r}."
        )
    return state, payload


@torch.no_grad()
def interpolate_state(ssl_state: dict[str, torch.Tensor], alpha: float,
                      tolerance: float):
    base_encoder = get_encoder("vit_b16_imagenet", pretrained=True).cpu().eval()
    base_state = base_encoder.state_dict()

    missing = sorted(set(base_state) - set(ssl_state))
    matched = sorted(set(base_state) & set(ssl_state))
    shape_mismatches = sorted(
        key for key in matched if base_state[key].shape != ssl_state[key].shape
    )
    unexpected = sorted(set(ssl_state) - set(base_state))
    allowed_tokens = ("decoder", "head", "predictor", "projector", "projection")
    invalid_unexpected = [
        key for key in unexpected
        if not any(token in key.lower() for token in allowed_tokens)
    ]
    if missing:
        raise RuntimeError(f"Missing encoder parameters: {missing[:10]}")
    if shape_mismatches:
        details = [
            f"{key}: ImageNet={tuple(base_state[key].shape)}, "
            f"Human SSL={tuple(ssl_state[key].shape)}"
            for key in shape_mismatches[:10]
        ]
        raise RuntimeError("Shape mismatches: " + "; ".join(details))
    if invalid_unexpected:
        raise RuntimeError(f"Unexpected non-SSL-head parameters: {invalid_unexpected[:10]}")

    interpolated = {}
    max_error = 0.0
    for key, base_tensor in base_state.items():
        ssl_tensor = ssl_state[key]
        if not (base_tensor.is_floating_point() and ssl_tensor.is_floating_point()):
            if alpha == 0.0:
                output = base_tensor.detach().cpu().clone()
            elif alpha == 1.0:
                output = ssl_tensor.detach().cpu().clone()
            elif torch.equal(base_tensor.cpu(), ssl_tensor.cpu()):
                output = base_tensor.detach().cpu().clone()
            else:
                raise TypeError(f"Cannot interpolate differing non-floating tensor: {key}")
            expected = output
        else:
            base64 = base_tensor.detach().cpu().to(torch.float64)
            ssl64 = ssl_tensor.detach().cpu().to(torch.float64)
            expected = ((1.0 - alpha) * base64 + alpha * ssl64).to(base_tensor.dtype)
            output = expected.clone()
        interpolated[key] = output
        if output.numel():
            error = torch.max(
                torch.abs(output.to(torch.float64) - expected.to(torch.float64))
            ).item()
            max_error = max(max_error, error)

    if max_error > tolerance:
        raise RuntimeError(
            f"Interpolation verification error {max_error} exceeds tolerance {tolerance}."
        )
    base_encoder.load_state_dict(interpolated, strict=True)
    return interpolated, len(matched), len(missing), len(shape_mismatches), max_error, len(unexpected)


def main() -> None:
    args = parse_args()
    if args.verification_tolerance < 0:
        raise ValueError("--verification-tolerance must be non-negative.")
    ssl_state, source_payload = load_ssl_state(args.ssl_checkpoint)
    (state, matched, missing, mismatches, max_error,
     ignored_ssl_parameters) = interpolate_state(
        ssl_state, args.alpha, args.verification_tolerance
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": source_payload.get(
            "format", "feline_transfer_learning.vision_encoder.v1"
        ),
        "encoder_name": "vit_b16_imagenet",
        "initialization": "ImageNet-1K supervised / Human SSL parameter interpolation",
        "adaptation": source_payload.get("adaptation"),
        "human_ssl_method": source_payload.get("human_ssl_method", "mae"),
        "state_dict": state,
        "model_state_dict": {
            key.removeprefix("model."): value for key, value in state.items()
        },
        "interpolation": True,
        "interpolation_base": "vit_b16_imagenet",
        "interpolation_ssl_source": str(args.ssl_checkpoint.expanduser().resolve()),
        "interpolation_alpha": args.alpha,
        "source_checkpoint_epoch": source_payload.get("epoch"),
    }
    torch.save(checkpoint, output)

    print(f"Matched encoder parameters: {matched}")
    print(f"Missing encoder parameters: {missing}")
    print(f"Shape mismatches: {mismatches}")
    if ignored_ssl_parameters:
        print(f"Ignored SSL-only parameters: {ignored_ssl_parameters}")
    print(f"\nInterpolation alpha: {args.alpha}")
    print(f"Max interpolation verification error: {max_error:.3e}\n")
    print("Verification: PASS\n")
    print(f"Saved:\n{output}")


if __name__ == "__main__":
    main()

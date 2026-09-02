"""Lightweight bottleneck adapters for vision-transformer blocks."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


class BottleneckAdapter(nn.Module):
    """Apply a zero-initialized residual bottleneck: x + Up(GELU(Down(x)))."""

    def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.up(self.dropout(self.activation(self.down(x))))


@dataclass(frozen=True)
class AdapterApplication:
    targets: tuple[str, ...]
    adapter_dim: int
    dropout: float


def _transformer_blocks(model: nn.Module) -> tuple[str, list[nn.Module]]:
    wrapped = getattr(model, "model", model)
    encoder = getattr(wrapped, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is not None:
        return "model.encoder.layers", list(layers)
    blocks = getattr(wrapped, "blocks", None)
    if blocks is not None:
        return "model.blocks", list(blocks)
    trunk = getattr(wrapped, "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    if blocks is not None:
        return "model.trunk.blocks", list(blocks)
    transformer = getattr(wrapped, "transformer", None)
    blocks = getattr(transformer, "resblocks", None)
    if blocks is not None:
        return "model.transformer.resblocks", list(blocks)
    raise RuntimeError("No supported transformer block sequence was found for adapters.")


def _adapt_output(adapter: BottleneckAdapter, output):
    if isinstance(output, Tensor):
        return adapter(output)
    if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
        return (adapter(output[0]), *output[1:])
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def _hidden_dim(model: nn.Module) -> int:
    wrapped = getattr(model, "model", model)
    for owner, attribute in (
        (wrapped, "hidden_dim"), (wrapped, "embed_dim"),
        (getattr(wrapped, "trunk", None), "num_features"),
        (model, "feature_dim"),
    ):
        value = getattr(owner, attribute, None) if owner is not None else None
        if value is not None:
            return int(value)
    raise RuntimeError("Unable to determine transformer hidden dimension for adapters.")


def apply_adapters(model: nn.Module, adapter_dim: int = 64,
                   dropout: float = 0.0) -> AdapterApplication:
    if adapter_dim <= 0 or not 0 <= dropout < 1:
        raise ValueError("Adapter requires adapter_dim>0 and 0<=dropout<1.")
    prefix, blocks = _transformer_blocks(model)
    if not blocks:
        raise RuntimeError("The transformer block sequence is empty.")

    hidden_dim = _hidden_dim(model)
    adapters = nn.ModuleList(
        BottleneckAdapter(hidden_dim, adapter_dim, dropout) for _ in blocks
    )
    model.adapter_modules = adapters
    model._adapter_hook_handles = [
        block.register_forward_hook(
            lambda _module, _inputs, output, adapter=adapter: _adapt_output(adapter, output)
        )
        for block, adapter in zip(blocks, adapters)
    ]
    targets = tuple(f"{prefix}.{index}.output" for index in range(len(blocks)))
    return AdapterApplication(targets, adapter_dim, dropout)

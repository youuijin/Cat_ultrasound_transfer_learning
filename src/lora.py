"""Lightweight Q/V-only LoRA adapters for vision-transformer attention."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize


class LoRAQKVLinear(nn.Module):
    """Wrap a combined QKV Linear and add low-rank updates to Q and V only."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if base.out_features % 3:
            raise ValueError("Combined QKV output dimension must be divisible by three.")
        self.base = base
        self.rank, self.alpha, self.scale = rank, alpha, alpha / rank
        self.dropout = nn.Dropout(dropout)
        chunk = base.out_features // 3
        self.lora_A_q = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B_q = nn.Linear(rank, chunk, bias=False)
        self.lora_A_v = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B_v = nn.Linear(rank, chunk, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q.weight); nn.init.zeros_(self.lora_B_v.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        output = self.base(x)
        adapted = self.dropout(x)
        query = self.lora_B_q(self.lora_A_q(adapted)) * self.scale
        value = self.lora_B_v(self.lora_A_v(adapted)) * self.scale
        key = torch.zeros_like(query)
        return output + torch.cat((query, key, value), dim=-1)


class LoRAQKVWeight(nn.Module):
    """Parametrize MultiheadAttention's packed QKV weight with Q/V LoRA."""

    def __init__(self, embed_dim: int, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.rank, self.alpha, self.scale, self.dropout = rank, alpha, alpha / rank, dropout
        self.lora_A_q = nn.Parameter(torch.empty(rank, embed_dim))
        self.lora_B_q = nn.Parameter(torch.zeros(embed_dim, rank))
        self.lora_A_v = nn.Parameter(torch.empty(rank, embed_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

    def forward(self, weight: Tensor) -> Tensor:
        query = (self.lora_B_q @ self.lora_A_q) * self.scale
        value = (self.lora_B_v @ self.lora_A_v) * self.scale
        if self.training and self.dropout:
            query = nn.functional.dropout(query, self.dropout, True)
            value = nn.functional.dropout(value, self.dropout, True)
        return weight + torch.cat((query, torch.zeros_like(query), value), dim=0)


@dataclass(frozen=True)
class LoRAApplication:
    targets: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float


def _replace_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent = root
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def apply_qv_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                  dropout: float = 0.0) -> LoRAApplication:
    if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("LoRA requires rank>0, alpha>0, and 0<=dropout<1.")
    targets: list[str] = []
    # Snapshot before replacing modules so newly inserted LoRA Linear layers are not revisited.
    modules = list(model.named_modules())
    for name, module in modules:
        if isinstance(module, nn.Linear) and name.endswith(("attn.qkv", "attention.qkv")):
            _replace_module(model, name, LoRAQKVLinear(module, rank, alpha, dropout))
            targets.append(f"{name}[Q,V]")
        elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
            adapter = LoRAQKVWeight(module.embed_dim, rank, alpha, dropout)
            parametrize.register_parametrization(module, "in_proj_weight", adapter)
            targets.append(f"{name}.in_proj_weight[Q,V]")
    if not targets:
        raise RuntimeError("No supported combined QKV attention projections were found for LoRA.")
    return LoRAApplication(tuple(targets), rank, alpha, dropout)


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name]

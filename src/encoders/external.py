"""Helpers for integrating architecture code from official external repositories."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


def resolve_factory(factory: Callable[..., nn.Module] | str | None, source_path: str | None,
                    label: str) -> Callable[..., nn.Module]:
    if source_path is not None:
        root = Path(source_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{label} official source directory not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    if factory is None:
        raise RuntimeError(
            f"{label} requires its official architecture source. Pass source_path=... and "
            "model_factory='official.module:builder' (or a callable). See src/encoders/README.md."
        )
    if callable(factory):
        return factory
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("model_factory must be a callable or 'module.path:function_name'.")
    module_name, attr = factory.rsplit(":", 1)
    try:
        resolved = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Could not import {label} factory {factory!r}.") from exc
    if not callable(resolved):
        raise TypeError(f"{factory!r} is not callable.")
    return resolved


def load_checkpoint(model: nn.Module, checkpoint_path: str | None, label: str,
                    strict: bool = True) -> None:
    if not checkpoint_path:
        raise ValueError(f"{label} requires checkpoint_path pointing to the official pretrained checkpoint.")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Requested {label} pretrained checkpoint was not found: {path}")
    checkpoint: Any = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("model"), dict):
            checkpoint = checkpoint["model"]
        elif isinstance(checkpoint.get("state_dict"), dict):
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported {label} checkpoint format in {path}; expected a state dict.")
    try:
        incompatible = model.load_state_dict(checkpoint, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(f"{label} checkpoint is incompatible with the constructed official model: {exc}") from exc
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Non-strict {label} load had missing keys {incompatible.missing_keys} and "
            f"unexpected keys {incompatible.unexpected_keys}. Resolve explicitly; load was rejected."
        )

"""Explicit, weights-only NNUE warm starts and frozen-transformer ablations.

Imported by the trainer, never an executable runner. Loading/hashing happens
only when the caller explicitly starts a training run with --initialize-from.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch
from torch import nn

RESIDUAL_PREFIXES = ("residual_hidden_layer.", "residual_output.")


def load_initial_weights(model: nn.Module, source: Path) -> dict[str, object]:
    """Load an exported .pt state dict, not an optimizer/RNG checkpoint.

    Base tensors must match exactly. A complete new residual branch may be
    absent when lifting a direct model into the residual architecture; its
    output is reset to zero so the initial float function remains unchanged.
    Existing residual heads must match completely. No silent key dropping,
    reshaping, dtype conversion or unsafe pickle fallback is permitted.
    """
    path = source.resolve()
    # Hash the exact bytes decoded, rather than re-opening a potentially changed
    # path after loading. This is an explicit future training-time read only.
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    state = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise ValueError("--initialize-from requires an exported model .pt state dictionary")
    target = model.state_dict()
    unexpected = set(state) - set(target)
    missing = set(target) - set(state)
    residual_keys = {key for key in target if key.startswith(RESIDUAL_PREFIXES)}
    if unexpected:
        raise ValueError(f"initial model has incompatible tensors: {sorted(unexpected)}")
    if missing and missing != residual_keys:
        raise ValueError(f"initial model is missing required tensors: {sorted(missing)}")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("--initialize-from accepts model weights, not training checkpoints")
        expected = target[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(f"initial model shape/dtype mismatch for {name}")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"initial model has non-finite/non-floating tensor {name}")

    model.load_state_dict(state, strict=not missing)
    if missing:
        with torch.no_grad():
            model.residual_output.weight.zero_()
    return {
        "source": str(path),
        "sha256": digest,
        "mode": "weights-only-new-optimizer",
        "loadedTensors": len(state),
        "newZeroOutputResidualHead": bool(missing),
    }


def freeze_feature_transformer(model: nn.Module) -> int:
    """Freeze only embedding/bias parameters; leave both output heads trainable."""
    modules = [model.feature_weights]
    shared = getattr(model, "piece_square_weights", None)
    if shared is not None:
        modules.append(shared)
    frozen = model.hidden_bias.numel()
    model.hidden_bias.requires_grad_(False)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return frozen

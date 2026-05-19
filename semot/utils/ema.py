"""
Exponential Moving Average (EMA) utilities for generator parameters.

This module provides:
  - :class:`EMA`: single-shadow EMA tracker.
  - :class:`EMABank`: multi-shadow EMA tracker (multiple decay values).

The EMA shadow model tracks a smoothed version of generator weights:
  ``ema_param = decay * ema_param + (1 - decay) * model_param``

Used during validation/FID to generate samples from smoothed checkpoints.

Paper reference:
  - Section 3.3 and Table config: EMA is applied to generator parameters,
    with final-model selection over multiple EMA decays.
"""

import copy
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _strip_orig_mod_key(key: str) -> str:
    """Remove torch.compile `_orig_mod` segments from a state-dict key."""
    out = key.replace("._orig_mod.", ".")
    if out.startswith("_orig_mod."):
        out = out[len("_orig_mod.") :]
    return out


def _add_root_orig_mod_key(key: str) -> str:
    """Add a root `_orig_mod.` prefix when key is in eager format."""
    if key.startswith("_orig_mod.") or "._orig_mod." in key:
        return key
    return f"_orig_mod.{key}"


def _remap_state_dict_keys(state_dict: dict[str, Any], key_fn) -> dict[str, Any]:
    """Return remapped state dict and fail on key collisions."""
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key_fn(key)
        if new_key in out and new_key != key:
            raise ValueError(f"EMA key collision while remapping '_orig_mod': {key} -> {new_key}")
        out[new_key] = value
    return out


def _adapt_keys_to_shadow(state_dict: dict[str, Any], shadow: nn.Module) -> dict[str, Any]:
    """Adapt incoming state-dict key style to the current shadow module keys."""
    target_keys = set(shadow.state_dict().keys())

    candidates = (
        lambda k: k,
        _strip_orig_mod_key,
        _add_root_orig_mod_key,
        lambda k: _add_root_orig_mod_key(_strip_orig_mod_key(k)),
    )
    for key_fn in candidates:
        try:
            remapped = _remap_state_dict_keys(state_dict, key_fn)
        except ValueError:
            continue
        if set(remapped.keys()) == target_keys:
            return remapped
    return state_dict


class EMA:
    """Exponential Moving Average for model parameters.

    Maintains a shadow copy of the model in eval mode with frozen gradients.
    Call :meth:`update` after each optimizer step.

    Args:
        model: The model to track.
        decay: EMA decay rate (default: 0.999). Higher = slower update.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow parameters from the current model parameters."""
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters(), strict=True
        ):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Return the shadow model's state dict (for checkpointing)."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load a state dict into the shadow model.

        Supports eager and torch.compile (`_orig_mod`) key styles.
        """
        adapted = _adapt_keys_to_shadow(state_dict, self.shadow)
        self.shadow.load_state_dict(adapted)


class EMABank:
    """Maintain multiple EMA shadows for one model.

    This supports final-model selection across several EMA decays while keeping
    the training model unchanged. A primary decay is used for routine sampling
    and validation; other decays are kept in sync for optional final sweeps.

    Args:
        model:         Model to track.
        decays:        Iterable of EMA decays (e.g., [0.999, 0.9995, 0.9998, 0.9999]).
        primary_decay: Decay used as the default EMA model. Must be in decays.
                       If None, the first decay is used.
    """

    def __init__(
        self,
        model: nn.Module,
        decays: Iterable[float],
        primary_decay: float | None = None,
    ):
        decay_list = self._normalize_decays(decays)
        if len(decay_list) == 0:
            raise ValueError("EMABank requires at least one decay value.")

        if primary_decay is None:
            primary = decay_list[0]
        else:
            primary = float(primary_decay)
            if primary not in decay_list:
                raise ValueError(
                    f"primary_decay={primary} is not in decays={decay_list}. "
                    "Add it to the list or choose an existing decay."
                )

        self.decays: tuple[float, ...] = tuple(decay_list)
        self.primary_decay: float = primary
        self._emas: dict[float, EMA] = {d: EMA(model, decay=d) for d in self.decays}

    @staticmethod
    def _normalize_decays(decays: Iterable[float]) -> list[float]:
        seen = set()
        out: list[float] = []
        for d in decays:
            d_f = float(d)
            if not (0.0 < d_f < 1.0):
                raise ValueError(f"EMA decay must be in (0, 1), got {d_f}")
            if d_f not in seen:
                seen.add(d_f)
                out.append(d_f)
        return out

    def to(self, device: torch.device) -> None:
        """Move all EMA shadows to device."""
        for ema in self._emas.values():
            ema.shadow.to(device)

    @property
    def shadow(self) -> nn.Module:
        """Primary EMA shadow (compatibility accessor)."""
        return self._emas[self.primary_decay].shadow

    def get(self, decay: float | None = None) -> EMA:
        """Return EMA tracker for a given decay (or primary if None)."""
        key = self.primary_decay if decay is None else float(decay)
        if key not in self._emas:
            raise KeyError(f"EMA decay {key} not found. Available: {list(self._emas.keys())}")
        return self._emas[key]

    def items(self) -> list[tuple[float, EMA]]:
        """Return (decay, EMA) pairs in configured order."""
        return [(d, self._emas[d]) for d in self.decays]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update every EMA shadow from current model weights."""
        for ema in self._emas.values():
            ema.update(model)

    def state_dict(self) -> dict[str, Any]:
        """Serialize all EMA shadows."""
        return {
            "decays": list(self.decays),
            "primary_decay": self.primary_decay,
            "ema_state_dicts": {str(d): self._emas[d].state_dict() for d in self.decays},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load all EMA shadows from serialized state."""
        primary = float(state_dict["primary_decay"])
        if primary != self.primary_decay:
            raise ValueError(
                f"State primary_decay={primary} does not match EMABank primary_decay="
                f"{self.primary_decay}"
            )
        saved = state_dict["ema_state_dicts"]
        for d in self.decays:
            key = str(d)
            if key not in saved:
                raise KeyError(f"Missing EMA state for decay {d} in checkpoint.")
            self._emas[d].load_state_dict(saved[key])

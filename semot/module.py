"""Project-specific Lightning module wrapper for checkpoint compatibility.

This wraps :class:`stable_pretraining.Module` to handle resume from checkpoints
saved while model submodules were wrapped by ``torch.compile``.
"""

from __future__ import annotations

# PyTorch 2.6 changed torch.load default to weights_only=True. Lightning checkpoints
# saved by this project embed omegaconf objects (ListConfig, DictConfig, ContainerMetadata,
# node types, etc.) in the hparams / hyper_parameters payload. Allowlist all public
# omegaconf classes so that Lightning's internal torch.load call succeeds without
# requiring weights_only=False.
import inspect
from pathlib import Path
from typing import Any

import omegaconf
import omegaconf.base
import omegaconf.nodes
import torch
from loguru import logger as logging
from stable_pretraining import Module

_omegaconf_modules = [
    omegaconf,
    omegaconf.base,
    omegaconf.nodes,
    omegaconf.listconfig,
    omegaconf.dictconfig,
]
_omegaconf_safe_classes = list(
    {
        obj
        for mod in _omegaconf_modules
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if obj.__module__.startswith("omegaconf")
    }
)
torch.serialization.add_safe_globals(_omegaconf_safe_classes)


def _strip_orig_mod_prefixes(state_dict: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return a copy with ``torch.compile`` `_orig_mod` key segments removed.

    Example:
      ``backbone._orig_mod.blocks.0.attn.qkv.weight``
      -> ``backbone.blocks.0.attn.qkv.weight``
    """
    out: dict[str, Any] = {}
    changed = 0
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod.", ".")
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        if new_key != key:
            changed += 1
        if new_key in out and new_key != key:
            raise ValueError(
                f"Checkpoint key collision while stripping '_orig_mod': {key} -> {new_key}"
            )
        out[new_key] = value
    return out, changed


def _drop_prefixed_keys(
    state_dict: dict[str, Any], prefixes: tuple[str, ...]
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy with keys matching any prefix removed."""
    removed = [key for key in state_dict if key.startswith(prefixes)]
    if not removed:
        return state_dict, []
    return {k: v for k, v in state_dict.items() if k not in removed}, removed


class CompatibleModule(Module):
    """StablePretraining Module with robust checkpoint key normalization."""

    _NON_RESUMABLE_PREFIXES = ("_dinov3_teacher.",)
    _WARMSTART_DROP_PREFIXES = ("callbacks_modules.", "callbacks_metrics.")

    def named_parameters(
        self,
        with_callbacks: bool = True,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        """Match the modern ``nn.Module`` signature while preserving callback filtering."""
        if with_callbacks:
            logging.warning(
                "You are calling self.parameters which also gives callbacks "
                "parameters, to remove then, pass `with_callbacks=False`"
            )
        for name, param in torch.nn.Module.named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if with_callbacks or not name.startswith("callbacks_"):
                yield name, param

    def parameters(
        self,
        with_callbacks: bool = True,
        recurse: bool = True,
    ):
        for _, param in self.named_parameters(
            with_callbacks=with_callbacks,
            recurse=recurse,
        ):
            yield param

    def setup(self, stage: str = "fit") -> None:
        """Load pretrained weights (model-only, no optimizer/scheduler)."""
        super().setup(stage)
        pretrained_ckpt = getattr(self.hparams, "pretrained_ckpt", None)
        if pretrained_ckpt is None:
            return

        ckpt_path = Path(pretrained_ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"[CheckpointCompat] pretrained_ckpt not found: {ckpt_path}")

        logging.info(f"[CheckpointCompat] Loading pretrained weights from {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)

        # Strip torch.compile key prefixes.
        state_dict, n_changed = _strip_orig_mod_prefixes(state_dict)
        if n_changed:
            logging.warning(
                f"[CheckpointCompat] Normalized {n_changed} pretrained keys "
                "by stripping '._orig_mod.'."
            )

        state_dict, removed = _drop_prefixed_keys(
            state_dict, self._WARMSTART_DROP_PREFIXES + self._NON_RESUMABLE_PREFIXES
        )
        if removed:
            preview = sorted(removed)[:8]
            if len(removed) > 8:
                preview.append("...")
            logging.warning(
                f"[CheckpointCompat] Dropped {len(removed)} warm-start-only keys "
                f"before load: {preview}"
            )

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(
                f"[CheckpointCompat] {len(missing)} missing keys "
                f"(randomly initialized): {sorted(missing)}"
            )
        if unexpected:
            logging.warning(
                f"[CheckpointCompat] {len(unexpected)} unexpected keys "
                f"(ignored): {sorted(unexpected)}"
            )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Normalize model state_dict keys (for ckpt_path resume).
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            fixed_state_dict, n_changed = _strip_orig_mod_prefixes(state_dict)
            if n_changed:
                checkpoint["state_dict"] = fixed_state_dict
                logging.warning(
                    f"[CheckpointCompat] Normalized {n_changed} state_dict keys "
                    "by stripping '._orig_mod.'."
                )

            fixed_state_dict, removed = _drop_prefixed_keys(
                fixed_state_dict, self._NON_RESUMABLE_PREFIXES
            )
            if removed:
                checkpoint["state_dict"] = fixed_state_dict
                preview = sorted(removed)[:8]
                if len(removed) > 8:
                    preview.append("...")
                logging.warning(
                    f"[CheckpointCompat] Dropped {len(removed)} non-resumable keys "
                    f"before load: {preview}"
                )

            self._prepare_queue_modules_for_load(fixed_state_dict)

        # EMA shadows may be either eager or torch.compile-wrapped depending on
        # runtime init order. Keep checkpoint EMA payload untouched here and let
        # EMABank.load_state_dict adapt key style to the current shadow module.

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Drop derived state that should be rebuilt from config on resume."""
        super().on_save_checkpoint(checkpoint)
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return

        filtered_state_dict, removed = _drop_prefixed_keys(state_dict, self._NON_RESUMABLE_PREFIXES)
        if removed:
            checkpoint["state_dict"] = filtered_state_dict
            preview = sorted(removed)[:8]
            if len(removed) > 8:
                preview.append("...")
            logging.warning(
                f"[CheckpointCompat] Dropped {len(removed)} non-resumable keys "
                f"before save: {preview}"
            )

    def _prepare_queue_modules_for_load(self, state_dict: dict[str, Any]) -> None:
        """Resize queue buffers to checkpoint shapes before strict state restore.

        OnlineKNN label queues are created with an unknown-shape placeholder of
        `(max_length, 1)` and become 1D only after the first append. On resume,
        Lightning restores state before any append happens, so strict loading can
        fail even though the queue semantics are unchanged. Align the current
        buffers to the checkpoint shapes before the actual restore.
        """
        callbacks_modules = getattr(self, "callbacks_modules", None)
        if not isinstance(callbacks_modules, torch.nn.ModuleDict):
            return

        resized: list[str] = []
        for name, module in callbacks_modules.items():
            if not name.startswith("ordered_queue_"):
                continue

            prefix = f"callbacks_modules.{name}."
            out_key = prefix + "out"
            order_key = prefix + "order_indices"

            out_tensor = state_dict.get(out_key)
            if out_tensor is not None and hasattr(module, "out"):
                current_shape = tuple(module.out.shape)
                ckpt_shape = tuple(out_tensor.shape)
                if current_shape != ckpt_shape:
                    module.out.resize_(ckpt_shape)
                    if hasattr(module, "max_length") and ckpt_shape:
                        module.max_length = ckpt_shape[0]
                    resized.append(f"{out_key}: {current_shape} -> {ckpt_shape}")

            order_tensor = state_dict.get(order_key)
            if order_tensor is not None and hasattr(module, "order_indices"):
                current_shape = tuple(module.order_indices.shape)
                ckpt_shape = tuple(order_tensor.shape)
                if current_shape != ckpt_shape:
                    module.order_indices.resize_(ckpt_shape)
                    resized.append(f"{order_key}: {current_shape} -> {ckpt_shape}")

        if resized:
            logging.warning(f"[CheckpointCompat] Resized queue buffers before load: {resized}")

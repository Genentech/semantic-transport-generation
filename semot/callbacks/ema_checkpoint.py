"""Checkpoint callback for lazily-initialized EMA bank state."""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class EMACheckpointCallback(Callback):
    """Persist and restore ``pl_module._ema_bank`` across preempt/resume.

    The training forward path lazily creates ``_ema_bank`` on the first training
    step. Lightning's checkpoint load happens earlier, so EMA state must be
    stashed first and applied later after lazy initialization.
    """

    ckpt_key = "ema_bank_state"

    @staticmethod
    def _to_cpu(obj: Any) -> Any:
        """Recursively move checkpoint payload tensors to CPU."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: EMACheckpointCallback._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [EMACheckpointCallback._to_cpu(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(EMACheckpointCallback._to_cpu(v) for v in obj)
        return obj

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]) -> None:
        del trainer
        if hasattr(pl_module, "_ema_bank"):
            checkpoint[self.ckpt_key] = self._to_cpu(pl_module._ema_bank.state_dict())
        elif hasattr(pl_module, "_ema_pending_state"):
            checkpoint[self.ckpt_key] = self._to_cpu(pl_module._ema_pending_state)

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]) -> None:
        del trainer
        state = checkpoint.get(self.ckpt_key)
        if state is not None:
            pl_module._ema_pending_state = state

"""
Distributed training primitives (DDP support).

Provides a unified interface for multi-GPU communication operations
(broadcast, all-reduce, barrier) with automatic single-GPU fallback.

Used throughout the training loop to:
  - Broadcast class indices and alpha values from rank 0 (sampling.py)
  - All-reduce loss values for logging
  - Check queue readiness across all ranks (forward.py)

All functions accept an optional ``enabled`` flag. When ``None``, they
auto-detect whether ``torch.distributed`` is initialized.
"""

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Return True if ``torch.distributed`` is initialized and available."""
    return dist.is_available() and dist.is_initialized()


def distributed_info() -> tuple[bool, int, int]:
    """Return ``(is_distributed, world_size, rank)``.

    Single-GPU fallback: ``(False, 1, 0)``.
    """
    enabled = is_distributed()
    world_size = dist.get_world_size() if enabled else 1
    rank = dist.get_rank() if enabled else 0
    return enabled, world_size, rank


def broadcast_from_rank0(
    tensor: torch.Tensor,
    enabled: bool | None = None,
) -> torch.Tensor:
    """Broadcast *tensor* from rank 0 to all ranks (in-place).

    No-op when running single-GPU or *enabled* is False.
    """
    if enabled is None:
        enabled = is_distributed()
    if enabled:
        dist.broadcast(tensor, src=0)
    return tensor


def all_reduce_sum_in_place(
    tensor: torch.Tensor,
    enabled: bool | None = None,
) -> torch.Tensor:
    """All-reduce SUM *tensor* in-place across all ranks.

    No-op when running single-GPU or *enabled* is False.
    """
    if enabled is None:
        enabled = is_distributed()
    if enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_ranks_true(
    local_condition: bool,
    device: torch.device,
    enabled: bool | None = None,
) -> bool:
    """Return True only if *local_condition* is True on **every** rank.

    Uses ``all_reduce(MIN)`` so a single False on any rank → global False.
    No-op (returns *local_condition* directly) when running single-GPU.
    """
    if enabled is None:
        enabled = is_distributed()
    if not enabled:
        return local_condition
    cond = torch.tensor(
        1 if local_condition else 0,
        device=device,
        dtype=torch.int64,
    )
    dist.all_reduce(cond, op=dist.ReduceOp.MIN)
    return bool(cond.item())

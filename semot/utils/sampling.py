"""
Class and CFG-alpha sampling for Sinkhorn generator training.

This module handles the randomized batch construction that occurs before
each training step:

  :func:`sample_and_shard_classes`
      Randomly select a subset of classes and distribute them across DDP ranks.
      Each rank gets ``n_classes_global // world_size`` classes.
      Classes must be broadcast so that ranks don't pick overlapping classes.

  :func:`sample_cfg_alpha`
      Power-law sampling from ``p(alpha) ~ alpha^(-power)`` via inverse CDF,
      with optional point-mass mixture (paper's L/2 schedule).
      Alpha is sampled independently per class (Appendix A.8 step 2:
      "For each label c, sample a CFG scale alpha"), so each rank samples
      its own alphas locally — no broadcast needed.

Paper references:
  - Section A.7: Class subsampling for large-vocabulary datasets (ImageNet)
  - Appendix CFG: Alpha sampling strategy (power-law with optional L/2 schedule)
"""

import torch

from semot.utils.distributed import broadcast_from_rank0


def sample_and_shard_classes(
    num_classes: int,
    n_classes_global: int,
    device: torch.device,
    rank: int,
    world_size: int,
    is_distributed: bool,
) -> torch.Tensor:
    """Sample a global class subset on rank 0, broadcast, then shard by rank.

    Classes must be coordinated across ranks to avoid overlap — rank 0 samples
    the full set and broadcasts it, then each rank takes its interleaved shard.

    If ``n_classes_global >= num_classes``, all classes are used (no subsampling).

    Args:
        num_classes:       Total number of classes in the dataset.
        n_classes_global:  Number of classes to sample globally per step.
        device:            Torch device for tensor creation.
        rank:              Current DDP rank (0 for single-GPU).
        world_size:        Total number of DDP ranks (1 for single-GPU).
        is_distributed:    Whether distributed training is active.

    Returns:
        Class indices for this rank, ``(n_classes_global // world_size,)``.
    """
    if rank == 0:
        if n_classes_global < num_classes:
            class_indices = torch.randperm(num_classes, device=device)[:n_classes_global]
        else:
            class_indices = torch.arange(num_classes, device=device)
    else:
        class_indices = torch.empty(n_classes_global, device=device, dtype=torch.long)
    class_indices = broadcast_from_rank0(class_indices, is_distributed)
    return class_indices[rank::world_size]


def sample_cfg_alpha(size: int, hp, device: torch.device) -> torch.Tensor:
    """Sample CFG alpha for training-time classifier-free guidance.

    Base distribution: power-law ``p(alpha) ~ alpha^(-power)`` on
    ``[alpha_min, alpha_max]``, sampled via inverse CDF:

      ``alpha = (a^(1-k) + u * (b^(1-k) - a^(1-k)))^(1/(1-k))``

    where ``u ~ Uniform(0,1)``, ``k = power``. Special case ``power == 0``
    gives ``Uniform(alpha_min, alpha_max)``.

    Optional point-mass injection (paper's L/2 schedule): with probability
    ``alpha_fixed_prob``, replace with ``alpha = alpha_fixed_value``.
    The L/2 schedule uses ``alpha_fixed_prob=0.5, alpha_fixed_value=1.0,
    alpha_power=3.0``.

    Args:
        size:   Number of alpha values to sample.
        hp:     Hyperparameters with ``alpha_min``, ``alpha_max``, ``alpha_power``,
                ``alpha_fixed_prob``, ``alpha_fixed_value``.
        device: Torch device.

    Returns:
        CFG alpha values, ``(size,)``.
    """
    alpha_min = float(hp.alpha_min)
    alpha_max = float(hp.alpha_max)
    power = float(getattr(hp, "alpha_power", 0.0))
    alpha_fixed_prob = float(getattr(hp, "alpha_fixed_prob", 0.0))
    alpha_fixed_value = float(getattr(hp, "alpha_fixed_value", alpha_min))

    if not 0.0 <= alpha_fixed_prob <= 1.0:
        raise ValueError(f"alpha_fixed_prob must be in [0, 1], got {alpha_fixed_prob}")

    # Power-law sampling via inverse CDF.
    if power == 0.0:
        alpha = torch.empty(size, device=device).uniform_(alpha_min, alpha_max)
    else:
        u = torch.empty(size, device=device).uniform_(0.0, 1.0)
        exp = 1.0 - power
        a_exp = alpha_min**exp
        b_exp = alpha_max**exp
        alpha = (a_exp + u * (b_exp - a_exp)) ** (1.0 / exp)

    # Optional fixed-point mixture.
    if alpha_fixed_prob > 0.0:
        fixed_mask = torch.rand(size, device=device) < alpha_fixed_prob
        alpha = torch.where(fixed_mask, torch.full_like(alpha, alpha_fixed_value), alpha)

    return alpha

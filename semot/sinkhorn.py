"""Log-domain Sinkhorn solvers for entropic optimal transport.

Provides the core building blocks for the Sinkhorn divergence loss:
  - :func:`sq_euclidean` — fast squared Euclidean distance via GEMM
  - :func:`log_sinkhorn` — asymmetric OT solver (uniform marginals)
  - :func:`log_sinkhorn_symmetric` — symmetric solver for self-transport

All operate on cost matrices with a leading batch dimension ``L``,
consistent with the ``(L, N, D)`` feature layout.
"""

import torch
from torch import Tensor


def sq_euclidean(a: Tensor, b: Tensor) -> Tensor:
    """Squared Euclidean distance via GEMM: ||a_i - b_j||^2.

    Args:
        a: ``(L, N, D)``
        b: ``(L, M, D)``

    Returns:
        ``(L, N, M)`` pairwise squared distances, clamped to non-negative.
    """
    a_sqnorm = (a * a).sum(dim=-1, keepdim=True)
    b_sqnorm = (b * b).sum(dim=-1, keepdim=True).transpose(1, 2)
    return (a_sqnorm + b_sqnorm - 2.0 * torch.bmm(a, b.transpose(1, 2))).clamp_min(0.0)


def log_sinkhorn(
    C: Tensor,
    eps: float,
    num_iters: int,
    f_init: Tensor | None = None,
    g_init: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Log-domain Sinkhorn for uniform marginals (Algorithm 1).

    Args:
        C:         Cost matrices ``(L, N, M)``.
        eps:       Entropic regularization strength.
        num_iters: Number of alternating Sinkhorn iterations.
        f_init:    Optional warm-start for source potentials ``(L, N)``.
        g_init:    Optional warm-start for target potentials ``(L, M)``.

    Returns:
        (f, g): Dual potentials ``(L, N)`` and ``(L, M)``.
    """
    L, N, M = C.shape
    f = f_init if f_init is not None else C.new_zeros(L, N)
    g = g_init if g_init is not None else C.new_zeros(L, M)

    log_N = torch.tensor(N, dtype=C.dtype, device=C.device).log()
    log_M = torch.tensor(M, dtype=C.dtype, device=C.device).log()

    for _ in range(num_iters):
        f = -eps * ((g[:, None, :] - C) / eps - log_M).logsumexp(dim=2)
        g = -eps * ((f[:, :, None] - C) / eps - log_N).logsumexp(dim=1)

    return f, g


def log_sinkhorn_symmetric(
    C: Tensor,
    eps: float,
    num_iters: int,
    f_init: Tensor | None = None,
) -> Tensor:
    """Symmetric Sinkhorn for self-transport OT_eps(q, q) (Algorithm 2).

    Exploits f = g symmetry with 0.5 averaging for faster convergence.

    Args:
        C:         Symmetric cost matrices ``(L, N, N)``.
        eps:       Entropic regularization strength.
        num_iters: Number of symmetric iterations.
        f_init:    Optional warm-start ``(L, N)``.

    Returns:
        f: Dual potentials ``(L, N)`` (with g = f by symmetry).
    """
    L, N, _ = C.shape
    f = f_init if f_init is not None else C.new_zeros(L, N)
    log_N = torch.tensor(N, dtype=C.dtype, device=C.device).log()

    for _ in range(num_iters):
        f = 0.5 * (f - eps * ((f[:, None, :] - C) / eps - log_N).logsumexp(dim=2))

    return f

"""Sinkhorn CFG loss for one-step generation.

Computes the Sinkhorn divergence gradient via optimal transport couplings,
then trains the generator via stop-gradient regression toward the transport
target. This formulation provides well-calibrated gradient magnitude and
avoids backpropagating through cost matrices.

For each (feature block, epsilon) sub-problem:
  1. Solve Sinkhorn for optimal couplings (all in no_grad)
  2. Compute barycentric transport targets T_qp, T_qq, T_qu
  3. CFG combination: displacement = (1+w)*T_qp - w*T_qu - T_qq
  4. Normalize: displacement /= lambda (RMS normalization per location)
  5. Sum across epsilons

Final loss: MSE(h_gen, sg(h_gen + displacement))
Gradient: -2 * displacement / (N * D), properly calibrated.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from semot.sinkhorn import log_sinkhorn, log_sinkhorn_symmetric, sq_euclidean


@dataclass
class LossInfo:
    """Diagnostics for a single call to SinkhornCFGLoss."""

    loss: float
    displacement_norm: float
    lambda_mean: float
    num_subproblems: int


class SinkhornCFGLoss:
    """Sinkhorn CFG loss via stop-gradient regression.

    Computes the normalized transport displacement from Sinkhorn couplings,
    then regresses generated features toward the displaced target via MSE.
    All OT computation happens in no_grad; only the MSE backward is needed.

    Args:
        epsilons:           Base temperature multipliers [tau_1, ..., tau_T].
                            Actual epsilon = tau * cost_scale (data-adaptive).
        sinkhorn_iters:     Iterations for asymmetric solves (cross-transport).
        sinkhorn_iters_sym: Iterations for symmetric solve (self-transport).
        mask_self_diagonal: If True, set diagonal of C_qq to +inf to prevent
                            trivial self-matching. Used in all paper experiments.
        eps_scale:          Cost scale estimator for converting tau to epsilon.
        warm_start:         Cache converged dual potentials across calls.
    """

    def __init__(
        self,
        epsilons: list[float] | None = None,
        sinkhorn_iters: int = 4,
        sinkhorn_iters_sym: int = 2,
        mask_self_diagonal: bool = True,
        eps_scale: str = "std",
        warm_start: bool = True,
        cost_scale_sample_size: int = 262144,
    ):
        self.epsilons = epsilons if epsilons is not None else [0.02, 0.05, 0.2]
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_iters_sym = sinkhorn_iters_sym
        self.mask_self_diagonal = mask_self_diagonal
        self.eps_scale = eps_scale
        self.warm_start = warm_start
        self.cost_scale_sample_size = cost_scale_sample_size
        self._warm_cache: dict[int, tuple] = {}

    def __call__(
        self,
        h_gen: Tensor,
        h_pos: Tensor,
        h_unc: Tensor | None,
        cfg_weight: Tensor | float,
    ) -> tuple[Tensor, LossInfo]:
        """Compute Sinkhorn CFG loss.

        Args:
            h_gen: Generated features ``(L, N_gen, D)``, requires_grad.
            h_pos: Positive (real) features ``(L, N_pos, D)``, detached.
            h_unc: Unconditional features ``(L, N_unc, D)`` or None.
            cfg_weight: Guidance weight w >= 0. Either a scalar or ``(L,)`` tensor.

        Returns:
            loss:  Scalar for backward (MSE toward transport target).
            info:  Diagnostics for logging.
        """
        L, N, D = h_gen.shape

        # All OT computation in no_grad.
        with torch.no_grad():
            displacement, disp_norm, lam_mean = self._compute_transport_displacement(
                h_gen,
                h_pos,
                h_unc,
                cfg_weight,
            )
            target = h_gen + displacement

        # Stop-gradient MSE regression.
        # MSE(h, sg(h + V)): gradient = -2V / (L*N*D), averaged over all elements.
        loss = F.mse_loss(h_gen, target.detach())

        return loss, LossInfo(
            loss=loss.item(),
            displacement_norm=disp_norm,
            lambda_mean=lam_mean,
            num_subproblems=len(self.epsilons),
        )

    def _compute_transport_displacement(
        self,
        h_gen: Tensor,
        h_pos: Tensor,
        h_unc: Tensor | None,
        cfg_weight: Tensor | float,
    ) -> tuple[Tensor, float, float]:
        """Compute the normalized transport displacement across all epsilons.

        Returns:
            displacement: ``(L, N, D)`` — the total normalized displacement.
            disp_norm: Mean RMS of displacement (for logging).
            lam_mean: Mean lambda across sub-problems.
        """
        L, N, D = h_gen.shape
        has_cfg = h_unc is not None
        if isinstance(cfg_weight, (int, float)):
            has_cfg = has_cfg and cfg_weight > 0
        else:
            has_cfg = has_cfg and cfg_weight.any().item()

        if isinstance(cfg_weight, (int, float)):
            w = h_gen.new_full((L,), cfg_weight)
        else:
            w = cfg_weight
        w_broad = w.unsqueeze(1).unsqueeze(2)  # (L, 1, 1)

        # Cost matrices
        C_qp = sq_euclidean(h_gen, h_pos)
        C_qq = sq_euclidean(h_gen, h_gen)
        if self.mask_self_diagonal:
            idx = torch.arange(N, device=C_qq.device)
            C_qq[:, idx, idx] = float("inf")

        C_qu = sq_euclidean(h_gen, h_unc) if has_cfg else None

        cost_scale = self._estimate_cost_scale(C_qp)

        # Accumulate displacement across epsilons
        total_disp = torch.zeros_like(h_gen)
        sum_lam = 0.0
        n_eps = len(self.epsilons)

        for eps_idx, tau in enumerate(self.epsilons):
            eps = tau * cost_scale
            disp_eps, lam = self._solve_one_epsilon(
                C_qp,
                C_qq,
                C_qu,
                h_gen,
                h_pos,
                h_unc,
                eps,
                w_broad,
                has_cfg,
                eps_idx,
            )
            total_disp = total_disp + disp_eps
            sum_lam += lam

        disp_norm = total_disp.pow(2).mean(dim=(1, 2)).sqrt().mean().item()
        return total_disp, disp_norm, sum_lam / max(n_eps, 1)

    def _solve_one_epsilon(
        self,
        C_qp: Tensor,
        C_qq: Tensor,
        C_qu: Tensor | None,
        h_gen: Tensor,
        h_pos: Tensor,
        h_unc: Tensor | None,
        eps: float,
        w_broad: Tensor,
        has_cfg: bool,
        eps_idx: int,
    ) -> tuple[Tensor, float]:
        """Single-epsilon: solve Sinkhorn, compute normalized displacement.

        Returns:
            displacement: ``(L, N, D)`` — lambda-normalized transport displacement.
            lam: Mean lambda value (for logging).
        """
        # --- Warm-start lookup ---
        L = h_gen.shape[0]
        f_init_qp = g_init_qp = f_init_qq = None
        f_init_qu = g_init_qu = None
        if self.warm_start:
            cached = self._warm_cache.get(eps_idx)
            if cached is not None:
                cf = cached[0]
                if cf.shape[0] == L:
                    if has_cfg and len(cached) == 5:
                        f_init_qp, g_init_qp, f_init_qq, f_init_qu, g_init_qu = cached
                    elif not has_cfg and len(cached) == 3:
                        f_init_qp, g_init_qp, f_init_qq = cached

        # --- Sinkhorn solve ---
        f_qp, g_qp = log_sinkhorn(
            C_qp,
            eps,
            self.sinkhorn_iters,
            f_init=f_init_qp,
            g_init=g_init_qp,
        )
        f_qq = log_sinkhorn_symmetric(
            C_qq,
            eps,
            self.sinkhorn_iters_sym,
            f_init=f_init_qq,
        )
        if has_cfg:
            f_qu, g_qu = log_sinkhorn(
                C_qu,
                eps,
                self.sinkhorn_iters,
                f_init=f_init_qu,
                g_init=g_init_qu,
            )

        # Cache potentials
        if self.warm_start:
            if has_cfg:
                self._warm_cache[eps_idx] = (
                    f_qp.detach(),
                    g_qp.detach(),
                    f_qq.detach(),
                    f_qu.detach(),
                    g_qu.detach(),
                )
            else:
                self._warm_cache[eps_idx] = (
                    f_qp.detach(),
                    g_qp.detach(),
                    f_qq.detach(),
                )

        # --- Row-stochastic transport matrices ---
        log_w_qp = (f_qp.unsqueeze(2) + g_qp.unsqueeze(1) - C_qp) / eps
        pi_qp = torch.softmax(log_w_qp, dim=2)

        log_w_qq = (f_qq.unsqueeze(2) + f_qq.unsqueeze(1) - C_qq) / eps
        pi_qq = torch.softmax(log_w_qq, dim=2)

        if has_cfg:
            log_w_qu = (f_qu.unsqueeze(2) + g_qu.unsqueeze(1) - C_qu) / eps
            pi_qu = torch.softmax(log_w_qu, dim=2)

        # --- Barycentric transport targets ---
        T_qp = pi_qp @ h_pos
        T_qq = pi_qq @ h_gen

        if has_cfg:
            T_qu = pi_qu @ h_unc
            displacement = (1 + w_broad) * T_qp - w_broad * T_qu - T_qq
        else:
            displacement = T_qp - T_qq

        # --- Lambda normalization ---
        lam = torch.sqrt(displacement.pow(2).mean(dim=(1, 2), keepdim=True) + 1e-8).clamp(
            min=1e-8
        )  # (L, 1, 1)

        return displacement / lam, lam.mean().item()

    def _estimate_cost_scale(self, C_qp: Tensor) -> float:
        """Estimate cost scale from C_qp for data-adaptive epsilon."""
        flat = C_qp.reshape(-1).detach()
        if flat.numel() > self.cost_scale_sample_size:
            idx = torch.randint(0, flat.numel(), (self.cost_scale_sample_size,), device=flat.device)
            flat = flat[idx]
        return float(flat.float().std(unbiased=False).clamp(min=1e-8).item())

    def reset_warm_start(self) -> None:
        """Clear cached dual potentials."""
        self._warm_cache.clear()

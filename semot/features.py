"""Multi-scale feature extraction with cost normalization.

Extracts features from a frozen SSL featurizer, decomposes them into
named multi-scale blocks (per-location spatial, global mean/std),
reshapes by class, and applies cost normalization (Stage 1) so that
pairwise distances are O(1) and epsilon values are scale-invariant.

The output is a list of FeatureBlocks, each containing pre-normalized
(L, N, C) tensors ready for the Sinkhorn divergence loss.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_fn


@dataclass
class FeatureBlock:
    """One normalized feature set ready for the Sinkhorn loss."""

    h_gen: Tensor
    h_pos: Tensor
    h_unc: Tensor | None
    name: str
    dim: int


def named_features_from_single_map(
    fmap: Tensor,
    *,
    fmap_idx: int,
    feature_mode: str = "key_facet",
) -> list[tuple[str, Tensor]]:
    """Extract named features from one featurizer output map.

    Each returned feature has shape ``(B, L_i, C_i)`` where L_i is the number
    of spatial positions. Each position defines an independent OT sub-problem:
    the Sinkhorn divergence is solved separately at every spatial location,
    providing spatially resolved gradient signal to the generator.

    For key_facet mode (ViT attention keys), returns:
      - per_loc: ``(B, H*W, C)`` — H*W=256 independent OT sub-problems, each C-dim
      - global_mean: ``(B, 1, C)`` — 1 OT sub-problem
      - global_std: ``(B, 1, C)`` — 1 OT sub-problem

    Cost normalization (Stage 1) is shared across all spatial positions within
    the same feature map, so per_loc positions share a single scale factor.

    Args:
        fmap: ``(B, C, H, W)`` feature map (or ``(B, C)`` for non-spatial).
        fmap_idx: Index for naming.
        feature_mode: 'key_facet' or 'spatial' (skip patch stats) or 'default'.

    Returns:
        List of (name, tensor) pairs, each tensor is ``(B, L_i, C_i)``.
    """
    features: list[tuple[str, Tensor]] = []
    if fmap.ndim != 4:
        flat = fmap.flatten(start_dim=1)
        return [(f"map{fmap_idx:02d}_flat", flat.unsqueeze(1))]

    B, C, H, W = fmap.shape

    per_loc = fmap.permute(0, 2, 3, 1).reshape(B, H * W, C)
    features.append((f"map{fmap_idx:02d}_per_loc", per_loc))

    global_mean = fmap.mean(dim=(2, 3)).unsqueeze(1)
    global_std = fmap.std(dim=(2, 3)).unsqueeze(1)
    features.append((f"map{fmap_idx:02d}_global_mean", global_mean))
    features.append((f"map{fmap_idx:02d}_global_std", global_std))

    return features


def cost_normalize(
    feat_gen: Tensor,
    feat_pos: Tensor,
    feat_unc: Tensor | None = None,
    unc_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Cost normalization (Stage 1): scale features so pairwise distances are O(1).

    Computes a single scale factor from the mean pairwise L2 distance between
    generated features and the full target pool [gen, pos, unc]. Divides all
    features by scale/sqrt(C), making the resulting cost matrix entries O(1)
    and epsilon values scale-invariant.

    All scale computation is done with stop-gradient.

    Args:
        feat_gen: ``(K, N_gen, L, C)``
        feat_pos: ``(K, N_pos, L, C)``
        feat_unc: ``(K, N_unc, L, C)`` or None
        unc_weights: ``(K, N_unc)`` per-sample CFG weights or None

    Returns:
        feat_gen_norm: ``(K, N_gen, L, C)``
        feat_pos_norm: ``(K, N_pos, L, C)``
        feat_unc_norm: ``(K, N_unc, L, C)`` or None
        scale: scalar detached scale factor
    """
    K, N_gen, L, C = feat_gen.shape
    N_pos = feat_pos.shape[1]
    sqrt_c = math.sqrt(C)

    with torch.no_grad():
        x_loc = feat_gen.permute(0, 2, 1, 3).reshape(K * L, N_gen, C)
        y_loc = feat_pos.permute(0, 2, 1, 3).reshape(K * L, N_pos, C)
        dists_gen = torch.cdist(x_loc, x_loc, p=2)
        dists_pos = torch.cdist(x_loc, y_loc, p=2)
        dist_sum = dists_gen.sum() + dists_pos.sum()
        weight_sum = feat_gen.new_tensor(float(K * L * (N_gen + N_pos)))

        if feat_unc is not None and feat_unc.shape[1] > 0:
            N_unc = feat_unc.shape[1]
            y_unc_loc = feat_unc.permute(0, 2, 1, 3).reshape(K * L, N_unc, C)
            dists_unc = torch.cdist(x_loc, y_unc_loc, p=2)

            if unc_weights is not None:
                w = unc_weights.to(device=feat_gen.device, dtype=feat_gen.dtype)
                weighted_unc = dists_unc.reshape(K, L, N_gen, N_unc) * w[:, None, None, :]
                dist_sum = dist_sum + weighted_unc.sum()
                weight_sum = weight_sum + (w.sum() * L)
            else:
                dist_sum = dist_sum + dists_unc.sum()
                weight_sum = weight_sum + feat_gen.new_tensor(float(K * L * N_unc))

        scale = dist_sum / (float(N_gen) * weight_sum.clamp(min=1e-8))
        scale = (scale / sqrt_c).clamp(min=1e-3)

    s = scale.view(1, 1, 1, 1)
    feat_gen_norm = feat_gen / s
    feat_pos_norm = feat_pos / s
    feat_unc_norm = feat_unc / s if feat_unc is not None else None

    return feat_gen_norm, feat_pos_norm, feat_unc_norm, scale


def extract_feature_maps(
    images: Tensor,
    featurizer: nn.Module,
    chunk_size: int | None = None,
) -> list[Tensor]:
    """Run frozen featurizer and return raw feature maps.

    When images require grad, uses activation checkpointing per chunk
    to trade recompute for memory.

    Args:
        images: ``(B, C, H, W)``
        featurizer: Frozen featurizer returning List[Tensor].
        chunk_size: Sub-batch size. None = full batch at once.

    Returns:
        List of feature map tensors.
    """

    def _run(chunk: Tensor) -> list[Tensor]:
        if chunk.requires_grad:
            feats = checkpoint_fn(
                lambda inp: tuple(featurizer(inp)),
                chunk,
                use_reentrant=False,
            )
        else:
            feats = featurizer(chunk)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        return list(feats)

    if chunk_size is None or chunk_size <= 0 or chunk_size >= images.shape[0]:
        return _run(images)

    chunks = images.split(chunk_size)
    feat_maps_out: list[Tensor] = []
    offset = 0
    for chunk_idx, chunk in enumerate(chunks):
        chunk_maps = _run(chunk)
        b = chunk.shape[0]
        if chunk_idx == 0:
            for fmap in chunk_maps:
                feat_maps_out.append(
                    torch.empty(
                        images.shape[0],
                        *fmap.shape[1:],
                        device=fmap.device,
                        dtype=fmap.dtype,
                    )
                )
        for i, fmap in enumerate(chunk_maps):
            feat_maps_out[i][offset : offset + b] = fmap
        offset += b

    return feat_maps_out


class FeatureExtractor(nn.Module):
    """Multi-scale feature extraction with cost normalization.

    Pipeline:
      1. Run frozen featurizer (with grad for gen, no grad for pos/unc)
      2. Decompose each feature map into named features (per_loc, global_mean/std)
      3. Reshape by class: (B, L, C) -> (K, N, L, C)
      4. Apply cost normalization (Stage 1)
      5. Reshape to (K*L, N, C) for the loss

    Args:
        featurizer: Frozen SSL featurizer (ViTFeaturizer, returns list of feature maps).
        feature_mode: 'key_facet' (default).
        input_summary_mode: 'norm_x' (default) or None to skip.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        feature_mode: str = "key_facet",
        input_summary_mode: str | None = "norm_x",
    ):
        super().__init__()
        self.featurizer = featurizer
        self.feature_mode = feature_mode
        self.input_summary_mode = input_summary_mode

    def forward(
        self,
        x_gen: Tensor,
        x_pos: Tensor,
        x_unc: Tensor | None,
        n_classes: int,
        n_gen: int,
        n_pos: int,
        unc_weights: Tensor | None = None,
        chunk_size: int | None = None,
    ) -> list[FeatureBlock]:
        """Extract and normalize all feature blocks.

        Args:
            x_gen: ``(K*N_gen, C_img, H, W)`` generated latents.
            x_pos: ``(K*N_pos, C_img, H, W)`` positive latents.
            x_unc: ``(K*N_unc, C_img, H, W)`` or None.
            n_classes: Number of classes K.
            n_gen: Generated samples per class.
            n_pos: Positive samples per class.
            unc_weights: ``(K*N_unc,)`` per-sample CFG weights or None.
            chunk_size: Sub-batch for featurizer forward.

        Returns:
            List of FeatureBlocks ready for SinkhornDivergenceLoss.
        """
        gen_maps = extract_feature_maps(
            x_gen,
            self.featurizer,
            chunk_size=chunk_size,
        )
        with torch.no_grad():
            if x_unc is not None and x_unc.shape[0] > 0:
                x_pos_unc = torch.cat([x_pos, x_unc], dim=0)
                pos_unc_maps = extract_feature_maps(
                    x_pos_unc,
                    self.featurizer,
                    chunk_size=chunk_size,
                )
                n_pos_total = x_pos.shape[0]
                pos_maps = [feat[:n_pos_total] for feat in pos_unc_maps]
                unc_maps = [feat[n_pos_total:] for feat in pos_unc_maps]
            else:
                pos_maps = extract_feature_maps(
                    x_pos,
                    self.featurizer,
                    chunk_size=chunk_size,
                )
                unc_maps = None

        n_unc = x_unc.shape[0] // n_classes if x_unc is not None and x_unc.shape[0] > 0 else 0
        unc_w_cls = None
        if unc_weights is not None and n_unc > 0:
            unc_w_cls = unc_weights.view(n_classes, n_unc).to(
                device=x_gen.device, dtype=x_gen.dtype
            )

        blocks: list[FeatureBlock] = []

        for fmap_idx, gen_map in enumerate(gen_maps):
            pos_map = pos_maps[fmap_idx]
            unc_map = unc_maps[fmap_idx] if unc_maps is not None else None

            gen_feats = named_features_from_single_map(
                gen_map, fmap_idx=fmap_idx, feature_mode=self.feature_mode
            )
            pos_feats = named_features_from_single_map(
                pos_map, fmap_idx=fmap_idx, feature_mode=self.feature_mode
            )
            unc_feats = (
                named_features_from_single_map(
                    unc_map, fmap_idx=fmap_idx, feature_mode=self.feature_mode
                )
                if unc_map is not None
                else None
            )

            for feat_idx, (name, feat_gen) in enumerate(gen_feats):
                _, feat_pos = pos_feats[feat_idx]
                feat_unc_raw = unc_feats[feat_idx][1] if unc_feats is not None else None

                block = self._normalize_and_reshape(
                    name,
                    feat_gen,
                    feat_pos,
                    feat_unc_raw,
                    n_classes,
                    n_gen,
                    n_pos,
                    n_unc,
                    unc_w_cls,
                )
                blocks.append(block)

        if self.input_summary_mode == "norm_x":
            summary_gen_src = x_gen
            summary_pos_src = x_pos
            summary_unc_src = x_unc
            if hasattr(self.featurizer, "transform_input_for_summary"):
                summary_gen_src = self.featurizer.transform_input_for_summary(x_gen)
                with torch.no_grad():
                    summary_pos_src = self.featurizer.transform_input_for_summary(x_pos)
                    if x_unc is not None:
                        summary_unc_src = self.featurizer.transform_input_for_summary(x_unc)

            norm_gen = torch.sqrt(summary_gen_src.square().mean(dim=(2, 3)) + 1e-6).unsqueeze(1)
            with torch.no_grad():
                norm_pos = torch.sqrt(summary_pos_src.square().mean(dim=(2, 3)) + 1e-6).unsqueeze(1)
                norm_unc = None
                if summary_unc_src is not None and summary_unc_src.shape[0] > 0:
                    norm_unc = torch.sqrt(
                        summary_unc_src.square().mean(dim=(2, 3)) + 1e-6
                    ).unsqueeze(1)

            block = self._normalize_and_reshape(
                "norm_x",
                norm_gen,
                norm_pos,
                norm_unc,
                n_classes,
                n_gen,
                n_pos,
                n_unc,
                unc_w_cls,
            )
            blocks.append(block)

        return blocks

    def _normalize_and_reshape(
        self,
        name: str,
        feat_gen: Tensor,
        feat_pos: Tensor,
        feat_unc: Tensor | None,
        n_classes: int,
        n_gen: int,
        n_pos: int,
        n_unc: int,
        unc_w_cls: Tensor | None,
    ) -> FeatureBlock:
        """Reshape by class, apply cost normalization, flatten to (K*L, N, C)."""
        _, L, C = feat_gen.shape

        feat_gen_cls = feat_gen.view(n_classes, n_gen, L, C)
        feat_pos_cls = feat_pos.view(n_classes, n_pos, L, C)
        feat_unc_cls = (
            feat_unc.view(n_classes, n_unc, L, C) if feat_unc is not None and n_unc > 0 else None
        )

        gen_norm, pos_norm, unc_norm, _ = cost_normalize(
            feat_gen_cls,
            feat_pos_cls,
            feat_unc_cls,
            unc_w_cls,
        )

        h_gen = gen_norm.permute(0, 2, 1, 3).reshape(n_classes * L, n_gen, C)
        h_pos = pos_norm.permute(0, 2, 1, 3).reshape(n_classes * L, n_pos, C)
        h_unc = None
        if unc_norm is not None:
            h_unc = unc_norm.permute(0, 2, 1, 3).reshape(n_classes * L, n_unc, C)

        return FeatureBlock(h_gen=h_gen, h_pos=h_pos, h_unc=h_unc, name=name, dim=C)

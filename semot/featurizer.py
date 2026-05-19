"""ViT featurizer for one-step generation training.

All featurizer families in the paper (MAE, DINOv3 distillation, Inception
distillation) share this single ViT backbone class. They differ only in
pretrained checkpoint weights and ``extract_every_n_layers``.

The featurizer operates directly on SD-VAE latents (32x32x4) and produces
spatial feature maps via attention-key extraction at tapped intermediate
layers, providing the transport cost geometry for the Sinkhorn loss.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint_fn

from semot.dit import Attention, RMSNorm, SwiGLU


class ViTBlock(nn.Module):
    """Pre-norm ViT block (no adaLN conditioning)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, use_qk_norm=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _init_sincos2d(embed: nn.Parameter, grid_size: int, dim: int) -> None:
    """Fill ``embed`` (1, N, dim) with 2D sincos positional encoding."""
    h = w = grid_size
    gh, gw = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    gh, gw = gh.reshape(-1), gw.reshape(-1)

    d4 = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d4, dtype=torch.float32) / d4))

    out_h = gh[:, None] * omega[None, :]
    out_w = gw[:, None] * omega[None, :]
    pos = torch.cat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)
    embed.data.copy_(pos.unsqueeze(0))


class ViTFeaturizer(nn.Module):
    """ViT featurizer for 32x32x4 SD-VAE latents.

    Used as a frozen feature extractor during generator training. Taps
    intermediate encoder layers and extracts attention-key vectors, reshaped
    into spatial feature maps ``(B, C, H_p, W_p)``.

    All three featurizer families in the paper use this class:
      - MAE (mask 50%/60%/75%): pretrained with masked autoencoding on latents
      - DINOv3 distillation: student matching frozen DINOv3 ViT-7B teacher
      - Inception distillation: student matching frozen InceptionV3

    Also supports MAE pretraining via ``reconstruct_with_mask()``.

    Args:
        in_channels: Input channels (4 for SD-VAE).
        img_size: Spatial size of input latents (32).
        patch_size: Patch tokenization size (2 for 256 tokens).
        hidden_dim: Transformer hidden dimension (1280 for ViT-H).
        depth: Number of transformer blocks (32 for ViT-H).
        num_heads: Attention heads (16 for ViT-H).
        mlp_ratio: MLP hidden dim ratio (4.0).
        extract_every_n_layers: Tap interval for feature maps (8 for ViT-H).
        decoder_dim: MAE decoder hidden dim (512).
        decoder_depth: MAE decoder blocks (4).
        decoder_num_heads: MAE decoder heads (16).
        num_classes: Optional classification head.
        pretrained_checkpoint: Path to pretrained weights.
        checkpoint_key: Key to extract from checkpoint dict.
        use_cls_token: Prepend a CLS token (False in paper).
        freeze: Freeze all parameters after init.
        activation_checkpointing: Checkpoint encoder blocks to save memory.
    """

    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 32,
        patch_size: int = 2,
        hidden_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        extract_every_n_layers: int = 3,
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        num_classes: int | None = None,
        pretrained_checkpoint: str | None = None,
        checkpoint_key: str = "encoder_state_dict",
        use_cls_token: bool = False,
        freeze: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.extract_every_n_layers = int(extract_every_n_layers)
        self.decoder_dim = decoder_dim
        self.use_cls_token = bool(use_cls_token)
        self.num_prefix_tokens = 1 if self.use_cls_token else 0
        self._freeze = bool(freeze)
        self._activation_checkpointing = bool(activation_checkpointing)

        self.grid_size: int = img_size // patch_size
        self.num_patches: int = self.grid_size * self.grid_size

        # --- Encoder ---
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if self.use_cls_token else None
        self.cls_pos_embed = (
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) if self.use_cls_token else None
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.encoder_norm = RMSNorm(hidden_dim)

        # --- Decoder (MAE pretraining only) ---
        self.decoder_embed = nn.Linear(hidden_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [ViTBlock(decoder_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.decoder_norm = RMSNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_channels, bias=True)

        # --- Optional classification head ---
        self.cls_head = nn.Linear(hidden_dim, num_classes) if num_classes is not None else None

        self._init_weights()

        if pretrained_checkpoint is not None:
            self._load_pretrained(pretrained_checkpoint, checkpoint_key=checkpoint_key)

        if freeze:
            for param in self.parameters():
                param.requires_grad_(False)
            self.eval()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        if self.cls_pos_embed is not None:
            nn.init.zeros_(self.cls_pos_embed)

        _init_sincos2d(self.pos_embed, self.grid_size, self.hidden_dim)
        _init_sincos2d(self.decoder_pos_embed, self.grid_size, self.decoder_dim)
        nn.init.normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Featurizer interface — returns spatial feature maps
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract spatial feature maps from tapped encoder layers.

        At each tapped layer, extracts attention-key vectors (post QK-norm)
        and reshapes them into ``(B, C, H_p, W_p)`` spatial maps.

        Returns:
            List of spatial feature maps, one per tapped layer.
        """
        B = x.shape[0]
        tokens = self._prepare_encoder_tokens(x)

        use_key_facet = getattr(self, "_feature_mode", "default") == "key_facet"
        _ckpt = self._activation_checkpointing and (self.training or self._freeze)

        tapped: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks, start=1):
            tokens = _checkpoint_fn(block, tokens, use_reentrant=False) if _ckpt else block(tokens)
            if i % self.extract_every_n_layers == 0 or i == self.depth:
                normed = self.encoder_norm(tokens)
                if use_key_facet:
                    qkv = block.attn.qkv(block.norm1(tokens))
                    B_k, N_k, _ = qkv.shape
                    qkv = qkv.reshape(B_k, N_k, 3, block.attn.num_heads, block.attn.head_dim)
                    k = qkv[:, :, 1]
                    if block.attn.use_qk_norm:
                        k = block.attn.k_norm(k)
                    keys = self._strip_prefix_tokens(k.reshape(B_k, N_k, -1))
                    spatial = keys.permute(0, 2, 1).reshape(
                        B, self.hidden_dim, self.grid_size, self.grid_size
                    )
                else:
                    spatial = (
                        self._strip_prefix_tokens(normed)
                        .permute(0, 2, 1)
                        .reshape(B, self.hidden_dim, self.grid_size, self.grid_size)
                    )
                tapped.append(spatial)
        return tapped

    def transform_input_for_summary(self, x: torch.Tensor) -> torch.Tensor:
        """Identity: latent-space featurizer operates on raw latents."""
        return x

    # ------------------------------------------------------------------
    # MAE pretraining interface
    # ------------------------------------------------------------------

    def reconstruct_with_mask(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token-drop MAE forward for pretraining."""
        reconstruction, patch_mask, _, _ = self._encode_decode(x, mask_ratio)
        return reconstruction, patch_mask

    def reconstruct_and_classify(
        self, x: torch.Tensor, mask_ratio: float = 0.75
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MAE + classification in a single encoder pass."""
        if self.cls_head is None:
            raise RuntimeError("Classification head is disabled (num_classes=None).")
        reconstruction, patch_mask, _, embedding = self._encode_decode(x, mask_ratio)
        logits = self.cls_head(embedding)
        return reconstruction, logits, embedding, patch_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_decode(
        self, x: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        N = self.num_patches
        N_keep = max(1, int(N * (1.0 - mask_ratio)))

        tokens = self._embed_patches(x)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :N_keep]

        tokens_vis = tokens.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        tokens_enc = self._prepend_cls_token(tokens_vis)

        _ckpt = self._activation_checkpointing and self.training
        for block in self.blocks:
            tokens_enc = (
                _checkpoint_fn(block, tokens_enc, use_reentrant=False)
                if _ckpt
                else block(tokens_enc)
            )
        tokens_enc = self.encoder_norm(tokens_enc)
        cls_token, tokens_vis = self._split_prefix_tokens(tokens_enc)
        embedding = self._global_descriptor(tokens_vis, cls_token)

        tokens_dec = self.decoder_embed(tokens_vis)
        mask_tokens = self.mask_token.expand(B, N - N_keep, self.decoder_dim)
        tokens_full = torch.cat([tokens_dec, mask_tokens], dim=1)
        tokens_full = tokens_full.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )
        tokens_full = tokens_full + self.decoder_pos_embed
        for block in self.decoder_blocks:
            tokens_full = block(tokens_full)
        tokens_full = self.decoder_norm(tokens_full)
        tokens_full = self.decoder_pred(tokens_full)

        reconstruction = self._unpatchify(tokens_full)

        patch_mask = torch.zeros(B, N, device=x.device, dtype=x.dtype)
        patch_mask.scatter_(1, ids_keep, 1.0)
        patch_mask = patch_mask.reshape(B, 1, self.grid_size, self.grid_size)
        patch_mask = F.interpolate(patch_mask, size=(x.shape[2], x.shape[3]), mode="nearest")

        return reconstruction, patch_mask, tokens_vis, embedding

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        g = self.grid_size
        B = x.shape[0]
        x = x.reshape(B, g, g, p, p, self.in_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_channels, g * p, g * p)

    def _embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x).flatten(2).permute(0, 2, 1) + self.pos_embed

    def _prepend_cls_token(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_cls_token:
            return patch_tokens
        cls_token = self.cls_token.expand(patch_tokens.shape[0], -1, -1)
        cls_token = cls_token.to(dtype=patch_tokens.dtype)
        if self.cls_pos_embed is not None:
            cls_token = cls_token + self.cls_pos_embed.to(dtype=patch_tokens.dtype)
        return torch.cat([cls_token, patch_tokens], dim=1)

    def _prepare_encoder_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self._prepend_cls_token(self._embed_patches(x))

    def _strip_prefix_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.num_prefix_tokens == 0:
            return tokens
        return tokens[:, self.num_prefix_tokens :, :]

    def _split_prefix_tokens(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.num_prefix_tokens == 0:
            return None, tokens
        return tokens[:, : self.num_prefix_tokens, :], tokens[:, self.num_prefix_tokens :, :]

    def _global_descriptor(
        self,
        patch_tokens: torch.Tensor,
        cls_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cls_token is not None:
            return cls_token.squeeze(1)
        return patch_tokens.mean(dim=1)

    def train(self, mode: bool = True):
        if self._freeze:
            super().train(False)
            return self
        return super().train(mode)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, checkpoint: str, checkpoint_key: str) -> None:
        payload = _read_checkpoint(checkpoint)
        state_dict = _extract_state_dict(payload, checkpoint_key=checkpoint_key)

        prefixes = (
            "",
            "backbone.",
            "_orig_mod.",
            "_orig_mod.backbone.",
            "backbone._orig_mod.",
            "module.backbone.",
            "model.backbone.",
        )

        model_state = self.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            for prefix in prefixes:
                if prefix and not key.startswith(prefix):
                    continue
                stripped = key[len(prefix) :]
                if stripped in model_state and model_state[stripped].shape == value.shape:
                    filtered[stripped] = value
                    break

        if not filtered:
            raise ValueError(f"No compatible keys found in checkpoint: {checkpoint}")

        missing, unexpected = self.load_state_dict(filtered, strict=False)
        print(
            f"[ViTFeaturizer] Loaded '{checkpoint}' "
            f"(matched={len(filtered)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)})."
        )


def _read_checkpoint(path: str) -> dict:
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        parts = path[5:].split("/", 2)
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        path = hf_hub_download(repo_id, filename)
    if path.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(path, map_location="cpu")
    ckpt_path = Path(path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return payload


def _extract_state_dict(payload: dict, checkpoint_key: str) -> dict:
    if checkpoint_key.startswith("ema:") or checkpoint_key in {
        "ema",
        "ema_state_dict",
        "ema_bank_state",
    }:
        ema_state = payload.get("ema_bank_state")
        if not isinstance(ema_state, dict):
            raise ValueError("'ema_bank_state' missing from checkpoint.")
        ema_sds = ema_state.get("ema_state_dicts", {})
        if not isinstance(ema_sds, dict) or len(ema_sds) == 0:
            raise ValueError("No EMA state dicts found.")
        if checkpoint_key.startswith("ema:"):
            requested_decay = checkpoint_key.split(":", 1)[1]
            if requested_decay not in ema_sds:
                raise ValueError(
                    f"EMA decay '{requested_decay}' not found. Available: {list(ema_sds.keys())}"
                )
            return ema_sds[requested_decay]
        primary = str(ema_state.get("primary_decay", next(iter(ema_sds))))
        if primary not in ema_sds:
            primary = next(iter(ema_sds))
        return ema_sds[primary]

    for key in [checkpoint_key, "encoder_state_dict", "state_dict"]:
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload

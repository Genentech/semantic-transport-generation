"""
DiT one-step generator.
Adapted for 32x32 images (MNIST, CIFAR-10) and 256x256 latents (ImageNet).

Key differences from standard DiT:
- No timestep input (one-step generator, not diffusion)
- Conditioning = class_embed + alpha_embed + style_embed
- Uses register tokens, RoPE, SwiGLU, RMSNorm, QK-Norm
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    """Multi-head attention with QK-Norm and optional RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Block with adaLN-Zero conditioning (6 modulation params)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)
        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and linear projection."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings using Conv2d."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class LabelEmbedder(nn.Module):
    """Embed class labels with null class for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """Embed CFG alpha scale using Fourier features."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=alpha.device) / half)
        args = alpha[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier_features(alpha, self.frequency_embedding_size)
        return self.mlp(fourier)


class StyleEmbedder(nn.Module):
    """Style embeddings: random tokens from a learnable codebook (Sec A.2)."""

    def __init__(self, hidden_size: int, num_tokens: int = 32, codebook_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.codebook_size, (batch_size, self.num_tokens), device=device)
        embeddings = self.codebook(indices)
        style = embeddings.sum(dim=1)
        return style


class DiT(nn.Module):
    """
    DiT one-step generator.

    Input: Gaussian noise epsilon ~ N(0, I), shape (B, C, H, W)
    Output: generated image x, same shape
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        label_dropout: float = 0.1,
        num_register_tokens: int = 8,
        use_style_embed: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        self.num_register_tokens = num_register_tokens
        self.register_pos_embed = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )
        self.cond_to_tokens = nn.Linear(hidden_size, hidden_size)

        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 64,
        )

        self.label_embed = LabelEmbedder(num_classes, hidden_size, label_dropout)
        self.alpha_embed = AlphaEmbedder(hidden_size)
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qk_norm=True,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

        nn.init.zeros_(self.cond_to_tokens.weight)
        nn.init.zeros_(self.cond_to_tokens.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p
        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        device = x.device

        # Compute conditioning first (needed for in-context tokens)
        c = self.label_embed(labels, self.training, force_drop_ids)
        c = c + self.alpha_embed(alpha)
        if self.use_style_embed:
            c = c + self.style_embed(B, device)

        x = self.patch_embed(x)
        # In-context conditioning tokens (paper Sec A.2):
        # "formed by summing the projected conditioning vector with positional embeddings"
        register = self.register_pos_embed.expand(B, -1, -1) + self.cond_to_tokens(c).unsqueeze(1)
        x = torch.cat([register, x], dim=1)

        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        x = x[:, self.num_register_tokens :, :]
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """One-step alpha-conditioned inference.

        CFG is implemented during training by mixing unconditional real
        negatives and conditioning the network on alpha. Inference remains
        one forward pass f(noise, class, alpha).
        """
        alpha_tensor = torch.full((x.shape[0],), alpha, device=x.device, dtype=x.dtype)
        return self.forward(x, labels, alpha_tensor)


# --- Factory functions ---


def DiT_Tiny(img_size=32, in_channels=3, num_classes=10, **kwargs):
    """depth=6, hidden=256, heads=4 -> ~5M params (MNIST)"""
    return DiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def DiT_Small(img_size=32, in_channels=3, num_classes=10, **kwargs):
    """depth=8, hidden=384, heads=6 -> ~15M params (CIFAR-10)"""
    return DiT(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        hidden_size=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def DiT_Base(img_size=32, in_channels=3, num_classes=10, patch_size=2, **kwargs):
    """depth=12, hidden=768, heads=12 (ImageNet B/2)"""
    return DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


def DiT_Large(img_size=32, in_channels=3, num_classes=10, patch_size=2, **kwargs):
    """depth=24, hidden=1024, heads=16 (ImageNet L/2)"""
    return DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=num_classes,
        **kwargs,
    )


DiT_models = {
    "DiT-Tiny": DiT_Tiny,
    "DiT-Small": DiT_Small,
    "DiT-Base": DiT_Base,
    "DiT-Large": DiT_Large,
}


def create_backbone(model_name, img_size=32, in_channels=3, num_classes=10, **kwargs):
    """Create a DiT backbone.

    Args:
        model_name: Key from DiT_models dict (e.g., "DiT-Small")
        img_size: Input image size
        in_channels: Number of input channels
        num_classes: Number of classes
        **kwargs: Additional arguments (label_dropout, etc.)

    Returns:
        DiT model instance
    """
    if model_name not in DiT_models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(DiT_models.keys())}")
    return DiT_models[model_name](
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )

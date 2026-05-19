"""
VAE utilities for latent-space generation.

Latent-space models (e.g., ImageNet 256x256) operate in SD-VAE latent space
``(B, 4, 32, 32)`` but pixel-space SSL encoders require decoding to pixels
for feature extraction. This module provides:

  - :class:`LatentDecoder`: decode latents to pixels (with optional
    activation checkpointing and sub-batch chunking).
  - :class:`LatentEncoder`: encode pixel images to latents for on-the-fly
    VAE encoding during MAE pretraining (paper Appendix A.4).

Paper reference:
  - Section 4.2: "We use a pretrained SD-VAE encoder/decoder."
  - The scaling factor 0.18215 matches Stable Diffusion's convention.
"""

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

VAE_SCALING_FACTOR = 0.18215


class LatentDecoder:
    """Decode SD-VAE latents to pixel images.

    Wraps a frozen ``AutoencoderKL`` with:
      - Automatic scaling factor division (``latents / 0.18215``)
      - Activation checkpointing on the training path (when ``latents.requires_grad``)
        to avoid retaining decoder intermediates
      - Optional sub-batch chunking for memory-bounded decoding

    Usage::

        decoder = LatentDecoder("stabilityai/sd-vae-ft-mse", device)
        pixels = decoder.decode(latents, chunk_size=64)  # (B, 3, 256, 256)

    Attributes:
        vae:            The frozen AutoencoderKL model, or None if not loaded.
        scaling_factor: VAE scaling factor (default: 0.18215).
    """

    def __init__(
        self,
        vae_model_name: str | None = None,
        device: torch.device | None = None,
        scaling_factor: float = VAE_SCALING_FACTOR,
    ):
        """Initialize the decoder.

        Args:
            vae_model_name: HuggingFace model identifier for ``AutoencoderKL``.
                            If ``None``, no VAE is loaded and :meth:`decode`
                            returns latents unchanged.
            device:         Device to load the VAE onto.
            scaling_factor: Latent scaling factor (default: 0.18215).
        """
        self.scaling_factor = scaling_factor
        self.vae = None
        if vae_model_name is not None and device is not None:
            self._load_vae(vae_model_name, device)

    def _load_vae(self, model_name: str, device: torch.device) -> None:
        """Load and freeze the VAE decoder."""
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_name).to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @property
    def is_available(self) -> bool:
        """Whether a VAE decoder has been loaded."""
        return self.vae is not None

    def decode(
        self,
        latents: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Decode latents to pixel images.

        Uses activation checkpointing when ``latents.requires_grad`` is True
        (training path) to save memory by recomputing decoder activations
        during backward instead of retaining them.

        Args:
            latents:    ``(B, 4, 32, 32)`` SD-VAE latents.
            chunk_size: If set, decode in sub-batches of this size.
                        ``None`` or ``<= 0`` means decode the full batch at once.

        Returns:
            Decoded pixel images, ``(B, 3, 256, 256)`` in ``[-1, 1]``.
            If no VAE is loaded, returns *latents* unchanged.
        """
        if self.vae is None:
            return latents

        def _decode(z: torch.Tensor) -> torch.Tensor:
            return self.vae.decode(z / self.scaling_factor).sample

        def _decode_maybe_ckpt(z: torch.Tensor) -> torch.Tensor:
            if z.requires_grad:
                return checkpoint_fn(_decode, z, use_reentrant=False)
            return _decode(z)

        if chunk_size is None or chunk_size <= 0 or chunk_size >= latents.shape[0]:
            return _decode_maybe_ckpt(latents)

        decoded_chunks = []
        for z_chunk in latents.split(chunk_size):
            decoded_chunks.append(_decode_maybe_ckpt(z_chunk))
        return torch.cat(decoded_chunks, dim=0)


class LatentEncoder:
    """Encode pixel images to SD-VAE latents (frozen, no gradients).

    Used for on-the-fly VAE encoding during MAE pretraining so that
    random augmentation (RandomResizedCrop) can be applied in pixel space
    each epoch, matching the paper (Appendix A.4).

    Usage::

        encoder = LatentEncoder("stabilityai/sd-vae-ft-mse", device)
        latents = encoder.encode(pixels, chunk_size=256)  # (B, 4, 32, 32)
    """

    def __init__(
        self,
        vae_model_name: str | None = None,
        device: torch.device | None = None,
        scaling_factor: float = VAE_SCALING_FACTOR,
        sample_posterior: bool = True,
    ):
        self.scaling_factor = scaling_factor
        self.sample_posterior = bool(sample_posterior)
        self.vae = None
        if vae_model_name is not None and device is not None:
            self._load_vae(vae_model_name, device)

    def _load_vae(self, model_name: str, device: torch.device) -> None:
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_name).to(device).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @property
    def is_available(self) -> bool:
        return self.vae is not None

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """Encode pixel images ``(B, 3, H, W)`` to latents ``(B, 4, h, w)``.

        By default this samples from
        the VAE posterior. Set ``sample_posterior=False`` for legacy local
        deterministic behavior that uses the posterior mean instead.

        Args:
            pixels:     ``(B, 3, H, W)`` in ``[-1, 1]``.
            chunk_size: Encode in sub-batches to bound peak memory.
        """
        if self.vae is None:
            return pixels

        def _encode_chunk(x: torch.Tensor) -> torch.Tensor:
            dist = self.vae.encode(x).latent_dist
            latents = dist.sample() if self.sample_posterior else dist.mean
            return latents * self.scaling_factor

        if chunk_size is None or chunk_size <= 0 or chunk_size >= pixels.shape[0]:
            return _encode_chunk(pixels)

        chunks = []
        for x_chunk in pixels.split(chunk_size):
            chunks.append(_encode_chunk(x_chunk))
        return torch.cat(chunks, dim=0)

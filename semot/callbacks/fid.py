"""
FID evaluation callback for one-step generators.

Supports two modes:
  - Rank-0 mode (default): rank 0 computes FID, other ranks wait at barrier
  - Distributed mode: all ranks update + compute FID, only rank 0 logs
"""

from __future__ import annotations

import inspect
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback


@torch.no_grad()
def _fm_euler_sample(
    model,
    noise: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    n_steps: int = 50,
    guidance_scale: float = 1.5,
) -> torch.Tensor:
    """Euler ODE sampler for flow matching with classifier-free guidance.

    Args:
        model:          EMA shadow model (DiT with alpha conditioning).
        noise:          ``(B, C, H, W)`` starting noise ~ N(0, I).
        labels:         ``(B,)`` class labels.
        num_classes:    Total number of classes (null class = num_classes).
        n_steps:        Number of Euler integration steps.
        guidance_scale: CFG scale (``self.alphas`` entry — reuse callback sweep).

    Returns:
        z: ``(B, C, H, W)`` generated VAE latents.
    """
    B = noise.shape[0]
    device = noise.device
    null_labels = torch.full((B,), num_classes, device=device, dtype=torch.long)
    z = noise.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((B,), step / n_steps * 1000.0, device=device)
        v_cond = model(z, labels, t)
        v_uncond = model(z, null_labels, t)
        z = z + (v_uncond + guidance_scale * (v_cond - v_uncond)) * dt
    return z


class FIDCallback(Callback):
    def __init__(
        self,
        num_real: int = 50000,
        num_fake: int = 50000,
        batch_size: int = 128,
        alphas: list[float] | None = None,
        metric_name: str = "val/fid",
        distributed: bool = False,
        fid_backend: str = "torchmetrics",  # torchmetrics | cleanfid | both
        cleanfid_metric_name: str = "val/fid_clean",
        cleanfid_mode: str = "clean",
        cleanfid_num_workers: int = 4,
        cleanfid_batch_size: int = 128,
        cleanfid_tmp_root: str | None = None,
        # Latent-space support: decode generated latents to pixels via VAE
        vae_model: str | None = None,
        vae_scaling_factor: float = 0.18215,
        # Real pixel images for latent-space FID (ImageNet val)
        real_image_dir: str | None = None,
        # Deprecated compatibility arg from the removed "official" backend.
        # It is ignored; TorchMetrics always computes real-image stats locally.
        ref_stats_path: str | None = None,
    ):
        self.alphas = alphas or [1.0]
        self.num_real = num_real
        self.num_fake = num_fake
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.distributed = distributed
        backend_raw = str(fid_backend).strip().lower()
        backend_norm = backend_raw.replace("-", "").replace("_", "")
        backend_aliases = {
            "torchmetrics": "torchmetrics",
            "official": "torchmetrics",
            "cleanfid": "cleanfid",
            "both": "both",
        }
        self.fid_backend = backend_aliases.get(backend_norm, "")
        if not self.fid_backend:
            raise ValueError(
                "fid_backend must be one of {'torchmetrics', 'cleanfid', "
                "'clean-fid', 'clean_fid', 'both'}, "
                f"got: {fid_backend!r}"
            )
        self._use_torchmetrics = self.fid_backend in {"torchmetrics", "both"}
        self._use_cleanfid = self.fid_backend in {"cleanfid", "both"}
        self.cleanfid_metric_name = cleanfid_metric_name
        self.cleanfid_mode = cleanfid_mode
        self.cleanfid_num_workers = int(cleanfid_num_workers)
        self.cleanfid_batch_size = int(cleanfid_batch_size)
        self.cleanfid_tmp_root = cleanfid_tmp_root
        self.vae_model_name = vae_model
        self.vae_scaling_factor = vae_scaling_factor
        self.real_image_dir = real_image_dir
        del ref_stats_path

        self._fid = None
        self._cleanfid = None
        self._vae = None
        self._enabled = True
        self._real_seen = 0
        self._real_loaded = False
        self._cached_real_stats = None
        self._cached_real_key = None

    @staticmethod
    def _barrier(trainer, name: str):
        """Synchronize all ranks when running under distributed strategy."""
        if getattr(trainer, "world_size", 1) > 1:
            trainer.strategy.barrier(name=name)

    @staticmethod
    def _split_count(total: int, rank: int, world_size: int) -> int:
        """Split total items across ranks with difference at most 1."""
        base = total // world_size
        rem = total % world_size
        return base + (1 if rank < rem else 0)

    def _use_distributed_eval(self, trainer) -> bool:
        return bool(self.distributed and getattr(trainer, "world_size", 1) > 1)

    def _real_cache_key(self, trainer, distributed_eval: bool):
        rank = int(getattr(trainer, "global_rank", 0))
        world_size = int(getattr(trainer, "world_size", 1))
        return (
            self.real_image_dir,
            int(self.num_real),
            int(self.batch_size),
            bool(distributed_eval),
            rank,
            world_size,
        )

    @staticmethod
    def _is_last_validation_epoch(trainer) -> bool:
        """Heuristic for final validation pass in current fit run."""
        if bool(getattr(trainer, "should_stop", False)):
            return True
        max_epochs = int(getattr(trainer, "max_epochs", -1))
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        return max_epochs > 0 and (current_epoch + 1) >= max_epochs

    @staticmethod
    def _ema_tag(decay: float) -> str:
        """Metric-safe EMA identifier (e.g., 0.9995 -> 0p9995)."""
        return str(decay).replace(".", "p")

    def _select_eval_models(self, trainer, pl_module):
        """Return list of (ema_decay, model) to evaluate for FID.

        Default: primary EMA only.
        Optional final-epoch sweep: all EMA decays from EMABank.
        """
        if hasattr(pl_module, "_ema_bank"):
            bank = pl_module._ema_bank
            sweep_enabled = bool(getattr(pl_module.hparams, "ema_sweep_enabled", True))
            sweep_final_only = bool(getattr(pl_module.hparams, "ema_sweep_final_only", True))
            should_sweep = (
                sweep_enabled
                and len(bank.decays) > 1
                and (not sweep_final_only or self._is_last_validation_epoch(trainer))
            )
            if should_sweep:
                return [(decay, ema.shadow) for decay, ema in bank.items()]
            return [(bank.primary_decay, bank.shadow)]
        if hasattr(pl_module, "_ema"):
            return [(None, pl_module._ema.shadow)]
        return [(None, pl_module.backbone)]

    @torch.no_grad()
    def _evaluate_fid_for_model(
        self,
        *,
        gen_model,
        model_label: str,
        pl_module,
        device: torch.device,
        distributed_eval: bool,
        local_num_fake: int,
        is_log_rank: bool,
        num_classes: int,
        in_channels: int,
        img_size: int,
        is_latent: bool,
    ):
        """Evaluate one generator model over alpha sweep, return summary."""
        gen_model.eval()
        if is_log_rank:
            mode = "distributed" if distributed_eval else "rank0"
            print(
                f"[FID] Starting eval ({mode}) model={model_label} "
                f"num_fake_global={self.num_fake}, num_fake_local={local_num_fake}, "
                f"batch_size={self.batch_size}, alphas={self.alphas}"
            )

        best_fid = float("inf")
        best_alpha = self.alphas[0]
        alpha_to_fid = {}

        for alpha in self.alphas:
            from torchmetrics.image.fid import FrechetInceptionDistance

            fid_metric = FrechetInceptionDistance(
                feature=2048,
                normalize=False,
                sync_on_compute=distributed_eval,
            ).to(device)
            fid_metric.real_features_sum.copy_(self._fid.real_features_sum)
            fid_metric.real_features_cov_sum.copy_(self._fid.real_features_cov_sum)
            fid_metric.real_features_num_samples.copy_(self._fid.real_features_num_samples)

            fake_done = 0
            # Keep class coverage balanced even when cur_bs < num_classes
            # (e.g., ImageNet-1k with batch_size=128). This avoids repeatedly
            # sampling only the first classes and inflating FID.
            class_cursor = 0
            while fake_done < local_num_fake:
                cur_bs = min(self.batch_size, local_num_fake - fake_done)
                labels = (
                    torch.arange(cur_bs, device=device, dtype=torch.long) + class_cursor
                ) % num_classes
                class_cursor = (class_cursor + cur_bs) % num_classes

                noise = torch.randn(cur_bs, in_channels, img_size, img_size, device=device)
                if hasattr(gen_model, "forward_with_cfg"):
                    x_fake = gen_model.forward_with_cfg(noise, labels, alpha=alpha)
                else:
                    alpha_t = torch.full((cur_bs,), alpha, device=device)
                    x_fake = gen_model(noise, labels, alpha_t)

                if is_latent:
                    vae_bs = min(self.batch_size, cur_bs)
                    decoded = []
                    for i in range(0, cur_bs, vae_bs):
                        decoded.append(self._decode_latents(x_fake[i : i + vae_bs]))
                    x_fake = torch.cat(decoded, dim=0)

                fid_metric.update(self._prepare_for_fid(x_fake), real=False)
                fake_done += cur_bs
                if is_log_rank and (
                    fake_done % (self.batch_size * 100) == 0 or fake_done == local_num_fake
                ):
                    print(
                        f"[FID] model={model_label} alpha={alpha:.2f} "
                        f"progress {fake_done}/{local_num_fake}"
                    )

            fid_val = float(fid_metric.compute().item())
            alpha_to_fid[alpha] = fid_val
            if is_log_rank:
                print(f"[FID] model={model_label} alpha={alpha:.2f} -> FID={fid_val:.2f}")

            if fid_val < best_fid:
                best_fid = fid_val
                best_alpha = alpha

        return {
            "best_fid": best_fid,
            "best_alpha": best_alpha,
            "alpha_to_fid": alpha_to_fid,
        }

    @staticmethod
    def _to_uint8_images(x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] float images to [0, 255] uint8 for Inception."""
        x = x.clamp(-1.0, 1.0)
        return ((x + 1.0) * 127.5).round().to(torch.uint8)

    def _prepare_for_fid(self, x: torch.Tensor) -> torch.Tensor:
        """Return uint8 BCHW images in [0, 255] for TorchMetrics FID."""
        return self._to_uint8_images(x)

    def _decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode SD-VAE latents (B, 4, 32, 32) -> pixels (B, 3, 256, 256)."""
        z = z / self.vae_scaling_factor
        decoded = self._vae.decode(z).sample
        return decoded.clamp(-1.0, 1.0)

    def _init_vae(self, device: torch.device):
        if self._vae is not None or self.vae_model_name is None:
            return
        try:
            from diffusers import AutoencoderKL

            self._vae = AutoencoderKL.from_pretrained(self.vae_model_name).to(device)
            self._vae.eval()
            for p in self._vae.parameters():
                p.requires_grad_(False)
            print(f"[FID] Loaded VAE decoder: {self.vae_model_name}")
        except Exception as exc:
            print(f"[FID] Failed to load VAE: {exc}")
            self._enabled = False

    def _init_fid(self, device: torch.device, distributed_eval: bool):
        if self._fid is not None or not self._enabled:
            return
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance

            self._fid = FrechetInceptionDistance(
                feature=2048,
                normalize=False,
                # Rank-0 mode disables sync to avoid deadlock.
                # Distributed mode requires sync so all ranks contribute.
                sync_on_compute=distributed_eval,
            ).to(device)
        except Exception as exc:
            self._enabled = False
            print(f"[FID] Disabled (failed to init): {type(exc).__name__}: {exc}")

    def _init_cleanfid(self) -> bool:
        if self._cleanfid is not None:
            return True
        try:
            from cleanfid import fid as cleanfid_fid

            self._cleanfid = cleanfid_fid
            return True
        except Exception as exc:
            print(f"[FID] clean-fid unavailable: {type(exc).__name__}: {exc}")
            return False

    @staticmethod
    def _save_uint8_batch_to_dir(x_uint8: torch.Tensor, out_dir: Path, start_idx: int) -> int:
        """Save a uint8 image batch (N,C,H,W) to PNG files."""
        from PIL import Image

        n = int(x_uint8.shape[0])
        images = x_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for i in range(n):
            img = images[i]
            if img.shape[-1] == 1:
                pil = Image.fromarray(img[..., 0], mode="L")
            else:
                pil = Image.fromarray(img[..., :3], mode="RGB")
            pil.save(out_dir / f"{start_idx + i:06d}.png")
        return start_idx + n

    def _compute_cleanfid_from_dirs(self, fake_dir: str, real_dir: str) -> float:
        """Compute clean-fid score from image directories."""
        if self._cleanfid is None:
            raise RuntimeError("clean-fid module is not initialized.")

        fn = self._cleanfid.compute_fid
        call_kwargs = {}
        params = inspect.signature(fn).parameters
        maybe_kwargs = {
            "mode": self.cleanfid_mode,
            "num_workers": self.cleanfid_num_workers,
            "batch_size": self.cleanfid_batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        for k, v in maybe_kwargs.items():
            if k in params:
                call_kwargs[k] = v
        return float(fn(fake_dir, real_dir, **call_kwargs))

    @torch.no_grad()
    def _evaluate_cleanfid_for_model(
        self,
        *,
        gen_model,
        model_label: str,
        device: torch.device,
        num_fake: int,
        num_classes: int,
        in_channels: int,
        img_size: int,
        is_latent: bool,
    ):
        """Evaluate one generator model over alpha sweep with clean-fid."""
        if self.real_image_dir is None:
            raise ValueError("clean-fid backend requires `real_image_dir`.")
        if self._cleanfid is None:
            raise RuntimeError("clean-fid backend requested but clean-fid is unavailable.")

        gen_model.eval()
        best_fid = float("inf")
        best_alpha = self.alphas[0]
        alpha_to_fid = {}

        for alpha in self.alphas:
            tmp_root = self.cleanfid_tmp_root or "/tmp"
            Path(tmp_root).mkdir(parents=True, exist_ok=True)
            fake_parent = Path(
                tempfile.mkdtemp(prefix=f"cleanfid_{model_label}_a{alpha:.2f}_", dir=tmp_root)
            )
            fake_dir = fake_parent / "fake"
            fake_dir.mkdir(parents=True, exist_ok=True)
            try:
                fake_done = 0
                file_idx = 0
                class_cursor = 0
                while fake_done < num_fake:
                    cur_bs = min(self.batch_size, num_fake - fake_done)
                    labels = (
                        torch.arange(cur_bs, device=device, dtype=torch.long) + class_cursor
                    ) % num_classes
                    class_cursor = (class_cursor + cur_bs) % num_classes

                    noise = torch.randn(cur_bs, in_channels, img_size, img_size, device=device)
                    if hasattr(gen_model, "forward_with_cfg"):
                        x_fake = gen_model.forward_with_cfg(noise, labels, alpha=alpha)
                    else:
                        alpha_t = torch.full((cur_bs,), alpha, device=device)
                        x_fake = gen_model(noise, labels, alpha_t)

                    if is_latent:
                        vae_bs = min(16, cur_bs)
                        decoded = []
                        for i in range(0, cur_bs, vae_bs):
                            decoded.append(self._decode_latents(x_fake[i : i + vae_bs]))
                        x_fake = torch.cat(decoded, dim=0)

                    x_uint8 = self._to_uint8_images(x_fake)
                    file_idx = self._save_uint8_batch_to_dir(x_uint8, fake_dir, file_idx)
                    fake_done += cur_bs

                fid_val = self._compute_cleanfid_from_dirs(str(fake_dir), self.real_image_dir)
                alpha_to_fid[alpha] = fid_val
                print(f"[FID-clean] model={model_label} alpha={alpha:.2f} -> FID={fid_val:.2f}")
                if fid_val < best_fid:
                    best_fid = fid_val
                    best_alpha = alpha
            finally:
                shutil.rmtree(fake_parent, ignore_errors=True)

        return {
            "best_fid": best_fid,
            "best_alpha": best_alpha,
            "alpha_to_fid": alpha_to_fid,
        }

    def _load_real_images_from_dir(
        self,
        device: torch.device,
        trainer,
        distributed_eval: bool,
    ) -> bool:
        """Load real pixel images from directory for latent-model FID."""
        if self._real_loaded or self.real_image_dir is None:
            return False

        from torchvision import datasets, transforms

        rank = int(getattr(trainer, "global_rank", 0))
        world_size = int(getattr(trainer, "world_size", 1))
        target_real = (
            self._split_count(self.num_real, rank, world_size)
            if distributed_eval
            else self.num_real
        )
        if target_real <= 0:
            self._real_loaded = True
            self._real_seen = 0
            self._cache_real_stats(trainer, distributed_eval)
            return True

        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        dataset = datasets.ImageFolder(self.real_image_dir, transform=transform)

        sampler = None
        shuffle = True
        if distributed_eval:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            shuffle = False

        if getattr(trainer, "is_global_zero", False):
            mode = "distributed" if distributed_eval else "rank0"
            print(f"[FID] Loading real images from {self.real_image_dir} ({mode})...")

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=False,
        )

        seen = 0
        for images, _ in loader:
            if seen >= target_real:
                break
            images = images[: target_real - seen].to(device)
            self._fid.update(self._prepare_for_fid(images), real=True)
            seen += images.shape[0]

        if getattr(trainer, "is_global_zero", False):
            print(f"[FID] Loaded {seen} real images on rank 0.")
        self._real_loaded = True
        self._real_seen = seen
        self._cache_real_stats(trainer, distributed_eval)
        return True

    def _cache_real_stats(self, trainer, distributed_eval: bool) -> None:
        if self._fid is None:
            return
        self._cached_real_stats = {
            "sum": self._fid.real_features_sum.detach().cpu().clone(),
            "cov": self._fid.real_features_cov_sum.detach().cpu().clone(),
            "num": self._fid.real_features_num_samples.detach().cpu().clone(),
            "seen": int(self._real_seen),
        }
        self._cached_real_key = self._real_cache_key(trainer, distributed_eval)

    def _restore_cached_real_stats(
        self,
        device: torch.device,
        trainer,
        distributed_eval: bool,
    ) -> bool:
        if self._fid is None or self._cached_real_stats is None:
            return False
        if self._cached_real_key != self._real_cache_key(trainer, distributed_eval):
            return False

        self._fid.real_features_sum.copy_(self._cached_real_stats["sum"].to(device))
        self._fid.real_features_cov_sum.copy_(self._cached_real_stats["cov"].to(device))
        self._fid.real_features_num_samples.copy_(self._cached_real_stats["num"].to(device))
        self._real_seen = int(self._cached_real_stats["seen"])
        self._real_loaded = True
        return True

    def on_validation_epoch_start(self, trainer, pl_module):
        del trainer, pl_module
        self._real_seen = 0
        self._real_loaded = False
        self._fid = None
        self._enabled = True

    @torch.no_grad()
    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        del outputs, batch_idx, dataloader_idx
        if not self._use_torchmetrics:
            return
        distributed_eval = self._use_distributed_eval(trainer)
        if not distributed_eval and not trainer.is_global_zero:
            return
        if not self._enabled:
            return

        self._init_fid(pl_module.device, distributed_eval)
        if self._fid is None:
            return

        hp = pl_module.hparams
        is_latent = int(hp.in_channels) != 3

        if is_latent:
            if self.real_image_dir is not None:
                if not self._real_loaded:
                    if not self._restore_cached_real_stats(
                        pl_module.device,
                        trainer,
                        distributed_eval,
                    ):
                        self._load_real_images_from_dir(pl_module.device, trainer, distributed_eval)
            return

        rank = int(getattr(trainer, "global_rank", 0))
        world_size = int(getattr(trainer, "world_size", 1))
        target_real = (
            self._split_count(self.num_real, rank, world_size)
            if distributed_eval
            else self.num_real
        )

        if self._real_seen >= target_real:
            return
        if "image" not in batch:
            return

        x_real = batch["image"]
        if x_real.ndim != 4 or x_real.shape[1] != 3:
            return

        remain = target_real - self._real_seen
        x_real = x_real[:remain]
        self._fid.update(self._prepare_for_fid(x_real), real=True)
        self._real_seen += x_real.shape[0]

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        distributed_eval = self._use_distributed_eval(trainer)
        is_log_rank = bool(getattr(trainer, "is_global_zero", False))
        if not distributed_eval and not is_log_rank:
            self._barrier(trainer, "fid_eval_epoch_end")
            return

        rank = int(getattr(trainer, "global_rank", 0))
        world_size = int(getattr(trainer, "world_size", 1))
        device = pl_module.device
        run_tm = self._use_torchmetrics
        run_cf = self._use_cleanfid

        # clean-fid is too slow for DDP epoch-end barriers and can trigger NCCL
        # watchdog timeouts while rank 0 is still computing.
        if distributed_eval and run_cf:
            if is_log_rank:
                print(
                    "[FID-clean] Disabled during distributed validation. "
                    "Run clean-fid in a separate single-GPU/offline evaluation step."
                )
            run_cf = False

        if distributed_eval and (not run_tm) and (not run_cf) and (not is_log_rank):
            self._barrier(trainer, "fid_eval_epoch_end")
            return

        try:
            hp = pl_module.hparams
            num_classes = int(hp.num_classes)
            in_channels = int(hp.in_channels)
            img_size = int(hp.img_size)
            is_latent = in_channels != 3

            if is_latent and (run_tm or run_cf):
                self._init_vae(device)
                if self._vae is None:
                    if is_log_rank:
                        print("[FID] No VAE decoder available; skipping FID.")
                    return

            eval_models = self._select_eval_models(trainer, pl_module)
            tm_eval_results = []
            if run_tm:
                if distributed_eval:
                    local_ready = int(self._enabled and self._fid is not None)
                    ready_t = torch.tensor(local_ready, device=device, dtype=torch.int64)
                    dist.all_reduce(ready_t, op=dist.ReduceOp.MIN)
                    if ready_t.item() == 0:
                        if is_log_rank:
                            print("[FID] Disabled on at least one rank; skipping torchmetrics FID.")
                        run_tm = False

                    if run_tm:
                        real_seen_t = torch.tensor(
                            self._real_seen, device=device, dtype=torch.int64
                        )
                        dist.all_reduce(real_seen_t, op=dist.ReduceOp.SUM)
                        if real_seen_t.item() == 0:
                            if is_log_rank:
                                if is_latent and self.real_image_dir is None:
                                    print(
                                        "[FID] Latent-space model but no real_image_dir set; "
                                        "skipping torchmetrics FID."
                                    )
                                else:
                                    print("[FID] No real images seen; skipping torchmetrics FID.")
                            run_tm = False
                else:
                    if not self._enabled or self._fid is None:
                        run_tm = False
                    elif self._real_seen == 0:
                        if is_latent and self.real_image_dir is None:
                            print(
                                "[FID] Latent-space model but no real_image_dir set; "
                                "skipping torchmetrics FID."
                            )
                        else:
                            print("[FID] No real images seen; skipping torchmetrics FID.")
                        run_tm = False

                if run_tm:
                    local_num_fake = (
                        self._split_count(self.num_fake, rank, world_size)
                        if distributed_eval
                        else self.num_fake
                    )
                    for decay, model in eval_models:
                        label = "backbone"
                        if decay is None and hasattr(pl_module, "_ema"):
                            label = "ema"
                        elif decay is not None:
                            label = f"ema{decay}"
                        result = self._evaluate_fid_for_model(
                            gen_model=model,
                            model_label=label,
                            pl_module=pl_module,
                            device=device,
                            distributed_eval=distributed_eval,
                            local_num_fake=local_num_fake,
                            is_log_rank=is_log_rank,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            img_size=img_size,
                            is_latent=is_latent,
                        )
                        tm_eval_results.append((decay, result))

            cf_eval_results = []
            if run_cf and is_log_rank:
                if self.real_image_dir is None:
                    if self.fid_backend == "cleanfid":
                        raise ValueError(
                            "fid_backend=cleanfid requires `real_image_dir` to point to real images."
                        )
                    print("[FID-clean] real_image_dir is required; skipping clean-fid.")
                elif not self._init_cleanfid():
                    if self.fid_backend == "cleanfid":
                        raise RuntimeError(
                            "fid_backend=cleanfid requested, but clean-fid is not installed."
                        )
                    print("[FID-clean] clean-fid backend unavailable; skipping clean-fid.")
                else:
                    for decay, model in eval_models:
                        label = "backbone"
                        if decay is None and hasattr(pl_module, "_ema"):
                            label = "ema"
                        elif decay is not None:
                            label = f"ema{decay}"
                        result = self._evaluate_cleanfid_for_model(
                            gen_model=model,
                            model_label=label,
                            device=device,
                            num_fake=self.num_fake,
                            num_classes=num_classes,
                            in_channels=in_channels,
                            img_size=img_size,
                            is_latent=is_latent,
                        )
                        cf_eval_results.append((decay, result))

            # Torchmetrics logging (legacy keys).
            if tm_eval_results:
                best_decay, best_result = min(tm_eval_results, key=lambda x: x[1]["best_fid"])
                for alpha, fid_val in best_result["alpha_to_fid"].items():
                    alpha_key = f"val/fid_alpha{alpha:.1f}"
                    pl_module.log(
                        alpha_key,
                        fid_val,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=distributed_eval,
                        rank_zero_only=False,
                    )

                if len(tm_eval_results) > 1:
                    for decay, result in tm_eval_results:
                        if decay is None:
                            continue
                        tag = self._ema_tag(decay)
                        pl_module.log(
                            f"val/fid_ema{tag}",
                            result["best_fid"],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=distributed_eval,
                            rank_zero_only=False,
                        )
                        pl_module.log(
                            f"val/best_alpha_ema{tag}",
                            result["best_alpha"],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=distributed_eval,
                            rank_zero_only=False,
                        )
                    if best_decay is not None:
                        pl_module.log(
                            "val/best_ema_decay",
                            float(best_decay),
                            on_step=False,
                            on_epoch=True,
                            sync_dist=distributed_eval,
                            rank_zero_only=False,
                        )

                pl_module.log(
                    self.metric_name,
                    best_result["best_fid"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=distributed_eval,
                    rank_zero_only=False,
                )
                pl_module.log(
                    "val/best_alpha",
                    best_result["best_alpha"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=distributed_eval,
                    rank_zero_only=False,
                )
                if is_log_rank:
                    if best_decay is None:
                        print(
                            f"[FID] Best: alpha={best_result['best_alpha']:.2f}, "
                            f"FID={best_result['best_fid']:.2f}"
                        )
                    else:
                        print(
                            f"[FID] Best: ema={best_decay}, alpha={best_result['best_alpha']:.2f}, "
                            f"FID={best_result['best_fid']:.2f}"
                        )

            # clean-fid logging (rank-0 only).
            if cf_eval_results and is_log_rank:
                best_decay_c, best_result_c = min(cf_eval_results, key=lambda x: x[1]["best_fid"])
                for alpha, fid_val in best_result_c["alpha_to_fid"].items():
                    alpha_key = f"{self.cleanfid_metric_name}_alpha{alpha:.1f}"
                    pl_module.log(
                        alpha_key,
                        fid_val,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                        rank_zero_only=True,
                    )

                if len(cf_eval_results) > 1:
                    for decay, result in cf_eval_results:
                        if decay is None:
                            continue
                        tag = self._ema_tag(decay)
                        pl_module.log(
                            f"{self.cleanfid_metric_name}_ema{tag}",
                            result["best_fid"],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=False,
                            rank_zero_only=True,
                        )
                        pl_module.log(
                            f"val/best_alpha_clean_ema{tag}",
                            result["best_alpha"],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=False,
                            rank_zero_only=True,
                        )
                    if best_decay_c is not None:
                        pl_module.log(
                            "val/best_ema_decay_clean",
                            float(best_decay_c),
                            on_step=False,
                            on_epoch=True,
                            sync_dist=False,
                            rank_zero_only=True,
                        )

                pl_module.log(
                    self.cleanfid_metric_name,
                    best_result_c["best_fid"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=not bool(tm_eval_results),
                    sync_dist=False,
                    rank_zero_only=True,
                )
                pl_module.log(
                    "val/best_alpha_clean",
                    best_result_c["best_alpha"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

                # If clean-fid is the only backend, populate canonical monitor keys.
                if not tm_eval_results:
                    pl_module.log(
                        self.metric_name,
                        best_result_c["best_fid"],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=True,
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                    pl_module.log(
                        "val/best_alpha",
                        best_result_c["best_alpha"],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=False,
                        rank_zero_only=True,
                    )
                if best_decay_c is None:
                    print(
                        f"[FID-clean] Best: alpha={best_result_c['best_alpha']:.2f}, "
                        f"FID={best_result_c['best_fid']:.2f}"
                    )
                else:
                    print(
                        f"[FID-clean] Best: ema={best_decay_c}, alpha={best_result_c['best_alpha']:.2f}, "
                        f"FID={best_result_c['best_fid']:.2f}"
                    )
            # Save FID results to local JSONL file for offline analysis.
            if is_log_rank and (tm_eval_results or cf_eval_results):
                import json

                out_dir = Path(getattr(trainer, "default_root_dir", "outputs"))
                fid_path = out_dir / "fid_history.jsonl"
                record = {
                    "step": int(trainer.global_step),
                    "epoch": int(trainer.current_epoch),
                }
                if tm_eval_results:
                    for decay, result in tm_eval_results:
                        tag = "backbone" if decay is None else f"ema{decay}"
                        if decay is None and hasattr(pl_module, "_ema"):
                            tag = "ema"
                        record[f"{tag}/best_fid"] = round(float(result["best_fid"]), 4)
                        record[f"{tag}/best_alpha"] = round(float(result["best_alpha"]), 2)
                        record[f"{tag}/alpha_to_fid"] = {
                            f"{a:.1f}": round(float(v), 4)
                            for a, v in result["alpha_to_fid"].items()
                        }
                    best_d, best_r = min(tm_eval_results, key=lambda x: x[1]["best_fid"])
                    record["best_fid"] = round(float(best_r["best_fid"]), 4)
                    record["best_alpha"] = round(float(best_r["best_alpha"]), 2)
                    if best_d is not None:
                        record["best_ema_decay"] = float(best_d)
                if cf_eval_results:
                    for decay, result in cf_eval_results:
                        tag = "backbone" if decay is None else f"ema{decay}"
                        if decay is None and hasattr(pl_module, "_ema"):
                            tag = "ema"
                        record[f"{tag}/fid_clean"] = round(float(result["best_fid"]), 4)
                        record[f"{tag}/best_alpha_clean"] = round(float(result["best_alpha"]), 2)
                        record[f"{tag}/alpha_to_fid_clean"] = {
                            f"{a:.1f}": round(float(v), 4)
                            for a, v in result["alpha_to_fid"].items()
                        }
                    best_dc, best_rc = min(cf_eval_results, key=lambda x: x[1]["best_fid"])
                    record["best_fid_clean"] = round(float(best_rc["best_fid"]), 4)
                    record["best_alpha_clean"] = round(float(best_rc["best_alpha"]), 2)
                    if best_dc is not None:
                        record["best_ema_decay_clean"] = float(best_dc)
                try:
                    with open(fid_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                except OSError:
                    pass

        finally:
            # Keep all ranks aligned at validation end under DDP.
            if distributed_eval:
                self._barrier(trainer, "fid_eval_epoch_end")

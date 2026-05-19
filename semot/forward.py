"""Sinkhorn training step orchestrator.

Defines :func:`training_forward`, the main training/validation step function
registered as the forward callable on the Lightning module.

Training step:
  1. Push real data into queue
  2. Sample classes and CFG alpha
  3. Generate latents via backbone (DiT)
  4. Sample positives and unconditional negatives from queue
  5. Extract multi-scale features with cost normalization
  6. Compute Sinkhorn divergence loss per feature block
  7. Update EMA
  8. Log metrics

Validation:
  Generate a sample grid with the EMA model (rank 0 only).
"""

import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from semot.features import FeatureExtractor
from semot.loss import SinkhornCFGLoss
from semot.utils.distributed import all_ranks_true, distributed_info
from semot.utils.ema import EMABank
from semot.utils.sampling import sample_and_shard_classes, sample_cfg_alpha
from semot.utils.vae import LatentDecoder
from semot.utils.visualization import make_image_grid, save_image_grid


def _forward_backbone_chunked(
    backbone,
    noise: torch.Tensor,
    labels: torch.Tensor,
    alpha: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    """Forward backbone in sub-batches with activation checkpointing."""
    if chunk_size is None or chunk_size <= 0 or chunk_size >= noise.shape[0]:
        return backbone(noise, labels, alpha)

    x_chunks = []
    for n_chunk, l_chunk, a_chunk in zip(
        noise.split(chunk_size),
        labels.split(chunk_size),
        alpha.split(chunk_size),
        strict=True,
    ):
        x_chunk = checkpoint_fn(
            lambda n, lbl, a: backbone(n, lbl, a),
            n_chunk,
            l_chunk,
            a_chunk,
            use_reentrant=False,
        )
        x_chunks.append(x_chunk)
    return torch.cat(x_chunks, dim=0)


def _resolve_ema_decays(hp) -> tuple[list[float], float]:
    """Resolve EMA decays and primary decay from hparams."""
    decays_cfg = getattr(hp, "ema_decays", None)
    if decays_cfg is None:
        decays = [float(getattr(hp, "ema_decay", 0.999))]
    else:
        decays = [float(d) for d in decays_cfg]

    seen = set()
    deduped = []
    for d in decays:
        if d not in seen:
            seen.add(d)
            deduped.append(d)

    primary = float(getattr(hp, "ema_primary_decay", deduped[0]))
    if primary not in deduped:
        deduped.insert(0, primary)
    return deduped, primary


def _generate_samples(self, num_per_class=8, alpha=1.5) -> None:
    """Generate a class-conditional sample grid and log to wandb / save to disk."""
    if not hasattr(self, "_ema_bank"):
        return

    model = self._ema_bank.shadow
    model.eval()

    hp = self.hparams
    device = self.device
    n_show = min(hp.num_classes, 10)

    samples = []
    with torch.no_grad():
        for c in range(n_show):
            noise = torch.randn(
                num_per_class, hp.in_channels, hp.img_size, hp.img_size, device=device
            )
            labels = torch.full((num_per_class,), c, device=device, dtype=torch.long)
            x = model.forward_with_cfg(noise, labels, alpha=alpha)
            samples.append(x)

    samples = torch.cat(samples, dim=0).clamp(-1, 1)

    # Decode latents to pixels for visualization.
    if samples.shape[1] != 3:
        viz_vae = getattr(hp, "sample_viz_vae_model", None)
        if viz_vae is not None:
            viz_decoder = LatentDecoder(
                vae_model_name=viz_vae,
                device=device,
            )
        elif hasattr(self, "_vae_decoder") and self._vae_decoder.is_available:
            viz_decoder = self._vae_decoder
        else:
            viz_decoder = None

        if viz_decoder is not None and viz_decoder.is_available:
            vae_chunk = int(
                getattr(
                    hp, "sample_viz_vae_decode_chunk_size", getattr(hp, "vae_decode_chunk_size", 64)
                )
            )
            samples = viz_decoder.decode(samples, chunk_size=vae_chunk).clamp(-1, 1)

    if samples.shape[1] > 3:
        samples = samples[:, :3]

    samples = samples.cpu()

    step = self.global_step
    out_dir = getattr(self.trainer, "default_root_dir", "outputs")
    path = f"{out_dir}/samples_step{step}.png"
    save_image_grid(samples, path, nrow=num_per_class)

    if self.logger is not None:
        try:
            import wandb

            grid = make_image_grid(samples, nrow=num_per_class)
            self.logger.experiment.log({"samples": wandb.Image(grid)}, step=step)
        except Exception:
            pass


def training_forward(self, batch, stage):
    """Sinkhorn divergence training step.

    Args:
        self:  Lightning module with backbone, featurizer, hparams, etc.
        batch: Dict with 'image' (B, C, H, W) and 'label' (B,).
        stage: 'fit' for training, anything else for validation.

    Returns:
        Dict with 'loss' (training) and 'label'.
    """
    out = {}
    hp = self.hparams
    device = self.device
    is_dist, world_size, rank = distributed_info()

    if not hasattr(self, "_tf32_set"):
        self._tf32_set = True
        torch.set_float32_matmul_precision("high")

    # --- Lazy initialization ---
    if not hasattr(self, "_queue_manager"):
        from semot.utils.queue import QueueManager

        queue_size = int(getattr(hp, "queue_size", 128))
        uncond_queue_size = int(getattr(hp, "uncond_queue_size", 1000))
        queue_device_cfg = str(getattr(hp, "queue_device", "cpu")).lower()
        queue_pin_memory = bool(getattr(hp, "queue_pin_memory", False))
        queue_device = (
            device if queue_device_cfg in {"gpu", "cuda"} else torch.device(queue_device_cfg)
        )

        self._queue_manager = QueueManager(
            num_classes=hp.num_classes,
            queue_size=queue_size,
            uncond_queue_size=uncond_queue_size,
            sample_shape=(hp.in_channels, hp.img_size, hp.img_size),
            queue_device=queue_device,
            queue_pin_memory=queue_pin_memory,
            queue_dtype=torch.float32,
        )

    if not hasattr(self, "_ema_bank"):
        ema_decays, ema_primary = _resolve_ema_decays(hp)
        self._ema_bank = EMABank(self.backbone, decays=ema_decays, primary_decay=ema_primary)
        self._ema_bank.to(device)
        pending = getattr(self, "_ema_pending_state", None)
        if pending is not None:
            self._ema_bank.load_state_dict(pending)
            delattr(self, "_ema_pending_state")
        self._ema = self._ema_bank.get()

    if not hasattr(self, "_vae_decoder"):
        vae_model = getattr(hp, "vae_model", None)
        self._vae_decoder = LatentDecoder(
            vae_model_name=vae_model,
            device=device if vae_model else None,
        )

    if not hasattr(self, "_feature_extractor"):
        feature_mode = str(getattr(hp, "feature_mode", "key_facet"))
        if feature_mode == "key_facet" and getattr(self, "featurizer", None) is not None:
            self.featurizer._feature_mode = "key_facet"
        self._feature_extractor = FeatureExtractor(
            featurizer=self.featurizer,
            feature_mode=feature_mode,
            input_summary_mode=str(getattr(hp, "input_summary_mode", "norm_x")),
        )

    if not hasattr(self, "_loss_fn"):
        epsilons = list(getattr(hp, "epsilons", [0.02, 0.05, 0.2]))
        self._loss_fn = SinkhornCFGLoss(
            epsilons=epsilons,
            sinkhorn_iters=int(getattr(hp, "sinkhorn_iters", 4)),
            sinkhorn_iters_sym=int(getattr(hp, "sinkhorn_iters_sym", 2)),
            mask_self_diagonal=bool(getattr(hp, "sinkhorn_self_mask_diagonal", True)),
            eps_scale=str(getattr(hp, "eps_scale", "std")),
            warm_start=True,
        )

    if not hasattr(self, "_compiled"):
        self._compiled = True
        if bool(getattr(hp, "compile", False)):
            compiled = []
            self.backbone = torch.compile(self.backbone)
            compiled.append("backbone")
            if self.featurizer is not None:
                self.featurizer = torch.compile(self.featurizer)
                self._feature_extractor.featurizer = self.featurizer
                compiled.append("featurizer")
            if rank == 0 and compiled:
                print(f"[Compile] torch.compile: {', '.join(compiled)}")

    # --- Validation path ---
    if stage != "fit":
        if "label" in batch:
            out["label"] = batch["label"]
        if rank == 0:
            sample_step = int(self.global_step)
            last_step = getattr(self, "_last_val_sample_step", None)
            if last_step != sample_step:
                _generate_samples(self)
                self._last_val_sample_step = sample_step
        return out

    # --- Training path ---
    x_real = batch["image"]
    labels_real = batch["label"]

    queue_push_size = int(getattr(hp, "queue_push_size", 64))
    queue_push_size = max(1, min(queue_push_size, x_real.shape[0]))
    self._queue_manager.push(x_real[-queue_push_size:], labels_real[-queue_push_size:])

    n_unc_global = int(getattr(hp, "batch_n_unc", 0))
    n_gen = int(hp.batch_n_gen) if hasattr(hp, "batch_n_gen") else int(hp.batch_n_neg)
    n_pos = int(hp.batch_n_pos)
    requested_n_classes_global = int(hp.n_classes_per_batch)
    n_classes_global = min(requested_n_classes_global, hp.num_classes)

    if n_classes_global % world_size != 0:
        raise ValueError(
            f"n_classes_per_batch ({n_classes_global}) must be divisible by "
            f"world_size ({world_size})"
        )
    n_classes_local = n_classes_global // world_size

    class_indices = sample_and_shard_classes(
        num_classes=hp.num_classes,
        n_classes_global=n_classes_global,
        device=device,
        rank=rank,
        world_size=world_size,
        is_distributed=is_dist,
    )

    _ = all_ranks_true(
        self._queue_manager.is_ready(class_indices, n_pos, n_unc_global),
        device,
        is_dist,
    )

    batch_size = n_classes_local * n_gen
    labels = class_indices.repeat_interleave(n_gen)
    alpha_per_class = sample_cfg_alpha(n_classes_local, hp, device)
    alpha = alpha_per_class.repeat_interleave(n_gen)

    noise = torch.randn(batch_size, hp.in_channels, hp.img_size, hp.img_size, device=device)
    x_gen = _forward_backbone_chunked(
        self.backbone,
        noise,
        labels,
        alpha,
        chunk_size=getattr(hp, "dit_chunk_size", None),
    )

    x_pos, labels_pos = self._queue_manager.sample_positives(
        class_indices,
        n_pos,
        device,
        replace_if_needed=True,
        zero_if_empty=True,
    )
    unc_result = self._queue_manager.sample_unconditional(
        class_indices,
        alpha_per_class,
        n_unc_global,
        n_gen,
        device,
        replace_if_needed=True,
        zero_if_empty=True,
    )
    x_unc, labels_unc, unc_weights = unc_result if unc_result is not None else (None, None, None)

    # Decode latents to pixels if featurizer operates in pixel space.
    x_gen_feat, x_pos_feat, x_unc_feat = x_gen, x_pos, x_unc
    if self._vae_decoder.is_available:
        vae_chunk = getattr(hp, "vae_decode_chunk_size", 64)
        x_gen_feat = self._vae_decoder.decode(x_gen_feat, vae_chunk)
        with torch.no_grad():
            x_pos_feat = self._vae_decoder.decode(x_pos_feat, vae_chunk)
            if x_unc_feat is not None:
                x_unc_feat = self._vae_decoder.decode(x_unc_feat, vae_chunk)

    # Extract features + cost normalization.
    feature_blocks = self._feature_extractor(
        x_gen_feat,
        x_pos_feat,
        x_unc_feat,
        n_classes=n_classes_local,
        n_gen=n_gen,
        n_pos=n_pos,
        unc_weights=unc_weights,
        chunk_size=getattr(hp, "featurizer_chunk_size", None),
    )

    # Compute Sinkhorn CFG loss per feature block.
    # Per-class CFG weight: w_c = alpha_c - 1.
    cfg_weight_per_class = alpha_per_class - 1.0  # (K,)
    total_loss = x_gen.new_zeros(())
    total_lambda = 0.0
    total_disp_norm = 0.0

    for block in feature_blocks:
        num_locs = block.h_gen.shape[0] // n_classes_local
        cfg_weight = cfg_weight_per_class.repeat_interleave(num_locs)  # (L,)

        block_loss, block_info = self._loss_fn(
            block.h_gen,
            block.h_pos,
            block.h_unc,
            cfg_weight,
        )
        total_loss = total_loss + block_loss
        total_lambda += block_info.lambda_mean
        total_disp_norm += block_info.displacement_norm

    n_blocks = max(len(feature_blocks), 1)
    loss = total_loss / n_blocks

    if is_dist:
        loss = loss * world_size

    # EMA update.
    self._ema_bank.update(self.backbone)

    # Logging.
    self.log("fit/loss", loss.item() / max(world_size, 1), prog_bar=True, sync_dist=False)
    self.log("fit/lambda", total_lambda / n_blocks, sync_dist=False)
    self.log("fit/displacement_norm", total_disp_norm / n_blocks, sync_dist=False)
    self.log("fit/cfg_weight_mean", cfg_weight_per_class.mean().item(), sync_dist=False)
    self.log("fit/n_feature_blocks", float(n_blocks), sync_dist=False)

    out["loss"] = loss
    out["label"] = labels_real
    return out

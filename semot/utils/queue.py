"""
Sample queue management for Sinkhorn generator training.

The Sinkhorn divergence loss requires a buffer of recent real images (positives) and
optionally a class-agnostic buffer (unconditional negatives for CFG).
These queues are filled each step from the data loader and sampled during
loss computation.

:class:`SampleQueue`
    Low-level per-class circular buffer. Stores tensors on CPU;
    moves to GPU on-demand via :meth:`sample`.

:class:`QueueManager`
    High-level wrapper around two :class:`SampleQueue` instances
    (conditional + unconditional). Provides push, sample, and readiness
    checks in a single API.

Paper references:
  - Section A.8: Queue of cached real samples for efficient batch construction
  - Appendix CFG: Unconditional queue for classifier-free guidance

Design notes:
  - By default queues store tensors on CPU to avoid GPU memory pressure.
  - Optional GPU queue storage is supported (``queue_device=\"cuda\"``) for
    latent-space runs where queue memory is manageable and host-device copies
    are a bottleneck.
  - Queue readiness is checked per-class (not globally) to support
    DDP class sharding.
"""

import torch


class SampleQueue:
    """Per-class circular queue of cached samples for efficient batch sampling.

    Stores tensors on CPU to avoid GPU memory pressure. Samples are moved
    to the target device on-demand via :meth:`sample`.

    Used to maintain a buffer of recent real images (Section A.8) that the
    Sinkhorn loss samples from as positive / unconditional-negative examples.

    Args:
        num_classes:  Number of classes (one queue per class).
        queue_size:   Maximum samples per class (circular buffer).
        sample_shape: Shape of each sample, e.g., ``(C, H, W)``.
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 128,
        sample_shape: tuple = (3, 32, 32),
        storage_device: str | torch.device = "cpu",
        pin_memory: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.sample_shape = sample_shape
        self.storage_device = torch.device(storage_device)
        self.pin_memory = bool(pin_memory and self.storage_device.type == "cpu")
        self.dtype = dtype

        self.queues = {
            c: self._allocate_queue(queue_size, sample_shape) for c in range(num_classes)
        }
        self.counts = {c: 0 for c in range(num_classes)}
        self.indices = {c: 0 for c in range(num_classes)}

    def _allocate_queue(self, queue_size: int, sample_shape: tuple) -> torch.Tensor:
        """Allocate queue storage on CPU or GPU based on ``storage_device``."""
        if self.storage_device.type == "cpu":
            return torch.zeros(
                queue_size,
                *sample_shape,
                dtype=self.dtype,
                pin_memory=self.pin_memory,
            )
        return torch.zeros(
            queue_size,
            *sample_shape,
            dtype=self.dtype,
            device=self.storage_device,
        )

    def add(self, samples: torch.Tensor, labels: torch.Tensor):
        """Add samples to per-class queues (circular buffer, oldest overwritten).

        Args:
            samples: ``(B, *sample_shape)`` on any device.
            labels:  ``(B,)`` class indices.
        """
        samples = samples.detach().to(
            device=self.storage_device,
            dtype=self.dtype,
            non_blocking=(self.storage_device.type == "cuda"),
        )
        labels = labels.detach().cpu()
        for sample, label in zip(samples, labels, strict=True):
            c = label.item()
            if c >= self.num_classes:
                continue
            idx = self.indices[c] % self.queue_size
            self.queues[c][idx] = sample
            self.indices[c] += 1
            self.counts[c] = min(self.counts[c] + 1, self.queue_size)

    def sample(
        self,
        label: int,
        n: int,
        device: torch.device,
        *,
        replace_if_needed: bool = False,
        zero_if_empty: bool = False,
    ) -> torch.Tensor:
        """Sample ``n`` items from ``queue[label]`` without replacement.

        Args:
            label:  Class index.
            n:      Number of samples to draw.
            device: Target device for the returned tensor.

        Returns:
            ``(n, *sample_shape)`` tensor on *device*.

        Raises:
            ValueError: If the queue has fewer than ``n`` samples and
                ``replace_if_needed`` is False.
        """
        count = self.counts[label]
        if count == 0:
            if zero_if_empty:
                out = torch.zeros(
                    n,
                    *self.sample_shape,
                    dtype=self.dtype,
                    device=self.storage_device,
                )
                if out.device == device:
                    return out
                non_blocking = self.storage_device.type == "cpu" and self.pin_memory
                return out.to(device, non_blocking=non_blocking)
            raise ValueError(f"No samples in queue for class {label}")
        if n > count:
            if not replace_if_needed:
                raise ValueError(
                    f"Cannot sample {n} items without replacement from queue with {count} items "
                    f"for class {label}"
                )
            if self.storage_device.type == "cuda":
                indices = torch.randint(count, (n,), device=self.storage_device)
            else:
                indices = torch.randint(count, (n,))
        else:
            if self.storage_device.type == "cuda":
                indices = torch.randperm(count, device=self.storage_device)[:n]
            else:
                indices = torch.randperm(count)[:n]
        samples = self.queues[label][indices]
        if samples.device == device:
            return samples
        non_blocking = self.storage_device.type == "cpu" and self.pin_memory
        return samples.to(device, non_blocking=non_blocking)

    def is_ready(self, min_samples: int = 32) -> bool:
        """Check that **all** classes have >= ``min_samples``."""
        return all(self.counts[c] >= min_samples for c in range(self.num_classes))

    def is_ready_for_labels(self, labels, min_samples: int = 32) -> bool:
        """Check queue readiness only for the provided labels.

        Args:
            labels:      Python iterable or 1-D tensor of class indices to check.
            min_samples: Minimum samples required per class.

        Returns:
            True if every label has >= ``min_samples`` in its queue.
        """
        if torch.is_tensor(labels):
            labels_iter = labels.detach().cpu().tolist()
        else:
            labels_iter = list(labels)

        if len(labels_iter) == 0:
            return False

        for c in set(int(x) for x in labels_iter):
            if c < 0 or c >= self.num_classes:
                return False
            if self.counts[c] < min_samples:
                return False
        return True


class QueueManager:
    """Manages conditional (per-class) and unconditional sample queues.

    Attributes:
        cond_queue:   Per-class queue of real images for positive sampling.
        uncond_queue: Class-agnostic queue for CFG unconditional negatives.
                      Uses a single pseudo-class (label=0).
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 128,
        uncond_queue_size: int = 1000,
        sample_shape: tuple[int, ...] = (3, 32, 32),
        queue_device: str | torch.device = "cpu",
        queue_pin_memory: bool = False,
        queue_dtype: torch.dtype = torch.float32,
    ):
        """Initialize both queues.

        Args:
            num_classes:      Number of classes for the conditional queue.
            queue_size:       Per-class capacity for the conditional queue.
            uncond_queue_size: Capacity for the unconditional queue.
            sample_shape:     Shape of each sample ``(C, H, W)``.
            queue_device:     Queue storage device (``\"cpu\"`` or ``\"cuda\"``).
            queue_pin_memory: If True and queue_device=cpu, pin queue memory.
            queue_dtype:      Queue tensor dtype.
        """
        self.queue_device = torch.device(queue_device)
        if self.queue_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("queue_device='cuda' requested but CUDA is not available.")

        self.cond_queue = SampleQueue(
            num_classes=num_classes,
            queue_size=queue_size,
            sample_shape=sample_shape,
            storage_device=self.queue_device,
            pin_memory=queue_pin_memory,
            dtype=queue_dtype,
        )
        self.uncond_queue = SampleQueue(
            num_classes=1,
            queue_size=uncond_queue_size,
            sample_shape=sample_shape,
            storage_device=self.queue_device,
            pin_memory=queue_pin_memory,
            dtype=queue_dtype,
        )

    def push(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Add images to both conditional and unconditional queues.

        Args:
            images: ``(B, C, H, W)`` real images from the data loader.
            labels: ``(B,)`` class labels.
        """
        self.cond_queue.add(images, labels)
        self.uncond_queue.add(images, torch.zeros_like(labels))

    def sample_positives(
        self,
        class_indices: torch.Tensor,
        n_pos: int,
        device: torch.device,
        *,
        replace_if_needed: bool = False,
        zero_if_empty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``n_pos`` positive images per class from the conditional queue.

        Args:
            class_indices: ``(n_classes_local,)`` class labels to sample from.
            n_pos:         Number of positives per class.
            device:        Target GPU device.

        Returns:
            x_pos:      ``(n_classes_local * n_pos, C, H, W)`` concatenated positives.
            labels_pos: ``(n_classes_local * n_pos,)`` corresponding labels.
        """
        x_pos_list, labels_pos_list = [], []
        for c in class_indices:
            c_int = c.item()
            x_c = self.cond_queue.sample(
                c_int,
                n_pos,
                device,
                replace_if_needed=replace_if_needed,
                zero_if_empty=zero_if_empty,
            )
            x_pos_list.append(x_c)
            labels_pos_list.append(torch.full((n_pos,), c_int, device=device, dtype=torch.long))
        return torch.cat(x_pos_list, dim=0), torch.cat(labels_pos_list, dim=0)

    def sample_unconditional(
        self,
        class_indices: torch.Tensor,
        alpha_per_class: torch.Tensor,
        n_unc: int,
        n_neg: int,
        device: torch.device,
        *,
        replace_if_needed: bool = False,
        zero_if_empty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Sample unconditional negatives and compute CFG weights.

        The CFG weight formula (Appendix CFG):
          ``alpha = ((N_neg - 1) + N_unc * w) / (N_neg - 1)``
          ``=> w = (alpha - 1) * (N_neg - 1) / N_unc``

        Higher alpha means stronger guidance, which increases the weight ``w``
        of unconditional negatives in the repulsion term.

        Args:
            class_indices:   ``(n_classes_local,)`` class labels.
            alpha_per_class: ``(n_classes_local,)`` CFG alpha per class.
            n_unc:           Number of unconditional samples per class.
            n_neg:           Number of negatives (for the weight formula).
            device:          Target GPU device.

        Returns:
            ``None`` if ``n_unc <= 0``, otherwise a tuple:
            ``(x_unc, labels_unc, unc_weights)`` with shapes
            ``(n_classes_local * n_unc, C, H, W)``,
            ``(n_classes_local * n_unc,)``,
            ``(n_classes_local * n_unc,)``.
        """
        if n_unc <= 0:
            return None

        x_unc_list, labels_unc_list, unc_w_list = [], [], []
        for c, alpha_c in zip(class_indices, alpha_per_class, strict=True):
            c_int = int(c.item())
            x_u = self.uncond_queue.sample(
                0,
                n_unc,
                device,
                replace_if_needed=replace_if_needed,
                zero_if_empty=zero_if_empty,
            )
            # CFG weight: w = (alpha - 1) * (N_neg - 1) / N_unc
            w_c = ((alpha_c - 1.0) * max(n_neg - 1, 1)) / max(n_unc, 1)
            w_c = w_c.clamp(min=0.0)
            x_unc_list.append(x_u)
            labels_unc_list.append(torch.full((n_unc,), c_int, device=device, dtype=torch.long))
            unc_w_list.append(w_c.expand(n_unc))

        return (
            torch.cat(x_unc_list, dim=0),
            torch.cat(labels_unc_list, dim=0),
            torch.cat(unc_w_list, dim=0),
        )

    def is_ready(
        self,
        class_indices: torch.Tensor,
        n_pos: int,
        n_unc: int,
    ) -> bool:
        """Check if both queues have enough samples for the given classes.

        Args:
            class_indices: Classes to check readiness for.
            n_pos:         Minimum samples needed per class in conditional queue.
            n_unc:         Minimum samples needed in unconditional queue.
                           ``0`` means skip unconditional check.

        Returns:
            True if all requested classes have >= ``n_pos`` samples AND
            (``n_unc == 0`` OR the unconditional queue has >= ``n_unc`` samples).
        """
        cond_ready = self.cond_queue.is_ready_for_labels(class_indices, n_pos)
        unc_ready = (n_unc <= 0) or self.uncond_queue.is_ready_for_labels([0], n_unc)
        return cond_ready and unc_ready

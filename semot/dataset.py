"""Dataset for pre-extracted SD-VAE latent shards."""

import bisect
import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentShardDataset(Dataset):
    """
    Loads pre-extracted SD-VAE latent tensors from sharded .pt files.

    Supported shard formats:
      - Legacy:
          {'latents': Tensor[N, 4, 32, 32], 'labels': Tensor[N]}
      - Upstream-style cached augmentation:
          {
              'latents': Tensor[N, 4, 32, 32],
              'latents_flip': Tensor[N, 4, 32, 32],
              'labels': Tensor[N],
          }

    Returns dicts: {"image": Tensor[4, 32, 32], "label": int}
    (named "image" for compatibility with the training loop).
    """

    def __init__(
        self,
        shard_dir: str,
        transform: callable | None = None,
        random_flip: bool = False,
    ):
        self.shard_dir = Path(shard_dir)
        self.transform = transform
        self.random_flip = bool(random_flip)

        shard_paths = sorted(glob.glob(str(self.shard_dir / "shard_*.pt")))
        if not shard_paths:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")

        # Upstream-style caches can be ~40GB for train. Do not concatenate them
        # into RAM; instead record shard boundaries and lazily mmap shard files.
        self.shard_paths = [Path(p) for p in shard_paths]
        self._cumulative_sizes = []
        self._shard_cache = {}
        saw_flip_latents = False
        saw_legacy_latents = False
        total_size = 0
        for p in self.shard_paths:
            shard = torch.load(p, weights_only=True, mmap=True)
            if "latents" not in shard or "labels" not in shard:
                raise KeyError(
                    f"Latent shard {p} must contain 'latents' and 'labels'; "
                    f"found keys {sorted(shard.keys())!r}"
                )
            n = int(shard["labels"].shape[0])
            if int(shard["latents"].shape[0]) != n:
                raise ValueError(
                    f"Latent shard {p} has mismatched lengths: "
                    f"latents={int(shard['latents'].shape[0])}, labels={n}"
                )
            if "latents_flip" in shard:
                if int(shard["latents_flip"].shape[0]) != n:
                    raise ValueError(
                        f"Latent shard {p} has mismatched flipped latent length: "
                        f"latents_flip={int(shard['latents_flip'].shape[0])}, labels={n}"
                    )
                saw_flip_latents = True
            else:
                saw_legacy_latents = True
            total_size += n
            self._cumulative_sizes.append(total_size)
            del shard

        if saw_flip_latents and saw_legacy_latents:
            raise ValueError(
                f"Mixed latent shard formats found in {shard_dir}: some shards have "
                "'latents_flip' and others do not."
            )
        self._has_flip_latents = saw_flip_latents
        self._length = total_size

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        # DataLoader workers should reopen their own mmap handles rather than
        # inheriting or pickling cached shard tensors.
        state["_shard_cache"] = {}
        return state

    def _get_shard(self, shard_idx: int):
        shard = self._shard_cache.get(shard_idx)
        if shard is None:
            shard = torch.load(self.shard_paths[shard_idx], weights_only=True, mmap=True)
            self._shard_cache[shard_idx] = shard
        return shard

    def __getitem__(self, idx):
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._length}")

        shard_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        shard_start = 0 if shard_idx == 0 else self._cumulative_sizes[shard_idx - 1]
        local_idx = idx - shard_start
        shard = self._get_shard(shard_idx)

        x = shard["latents"][local_idx]
        if self._has_flip_latents and self.random_flip and bool(torch.rand(()) < 0.5):
            x = shard["latents_flip"][local_idx]
        y = shard["labels"][local_idx].item()
        if self.transform is not None:
            x = self.transform(x)
        return {"image": x, "label": y}

    @property
    def column_names(self):
        return ["image", "label"]

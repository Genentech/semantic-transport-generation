"""Precompute SD-VAE latents for ImageNet 256x256.

Encodes all images through a frozen SD-VAE encoder and saves sharded .pt files
compatible with LatentShardDataset. Each shard contains:
  - latents:      (N, 4, 32, 32) original
  - latents_flip: (N, 4, 32, 32) horizontally flipped
  - labels:       (N,) class indices

Usage:
    python scripts/precompute_latents.py \
        --imagenet_dir /path/to/imagenet \
        --output_dir /path/to/imagenet256_latents \
        --vae stabilityai/sd-vae-ft-mse
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def encode_split(
    vae, dataloader, output_dir: Path, shard_size: int, scaling_factor: float, device: torch.device
):
    output_dir.mkdir(parents=True, exist_ok=True)

    latents_buf, flips_buf, labels_buf = [], [], []
    shard_idx = 0
    count = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        flipped = torch.flip(images, dims=[-1])

        with torch.no_grad():
            z = vae.encode(images).latent_dist.sample() * scaling_factor
            z_flip = vae.encode(flipped).latent_dist.sample() * scaling_factor

        latents_buf.append(z.cpu())
        flips_buf.append(z_flip.cpu())
        labels_buf.append(labels)
        count += images.shape[0]

        if count >= shard_size:
            _save_shard(output_dir, shard_idx, latents_buf, flips_buf, labels_buf)
            latents_buf, flips_buf, labels_buf = [], [], []
            shard_idx += 1
            count = 0

    if latents_buf:
        _save_shard(output_dir, shard_idx, latents_buf, flips_buf, labels_buf)


def _save_shard(output_dir, shard_idx, latents_buf, flips_buf, labels_buf):
    shard = {
        "latents": torch.cat(latents_buf),
        "latents_flip": torch.cat(flips_buf),
        "labels": torch.cat(labels_buf),
    }
    path = output_dir / f"shard_{shard_idx:05d}.pt"
    torch.save(shard, path)
    print(f"  Saved {path.name}: {shard['latents'].shape[0]} samples")


def main():
    parser = argparse.ArgumentParser(description="Precompute SD-VAE latents for ImageNet.")
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        required=True,
        help="Path to ImageNet root (with train/ and val/ subdirs)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for latent shards"
    )
    parser.add_argument(
        "--vae", type=str, default="stabilityai/sd-vae-ft-mse", help="HuggingFace VAE model ID"
    )
    parser.add_argument("--scaling_factor", type=float, default=0.18215)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--shard_size", type=int, default=10000, help="Number of samples per shard file"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VAE: {args.vae}")
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    output_dir = Path(args.output_dir)
    imagenet_dir = Path(args.imagenet_dir)

    for split in args.splits:
        print(f"\nEncoding {split}...")
        dataset = datasets.ImageFolder(imagenet_dir / split, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        encode_split(
            vae, dataloader, output_dir / split, args.shard_size, args.scaling_factor, device
        )

    print(f"\nDone. Latent shards saved to {output_dir}")


if __name__ == "__main__":
    main()

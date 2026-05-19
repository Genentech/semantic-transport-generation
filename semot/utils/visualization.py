"""
Image grid construction and saving for training monitoring.

:func:`make_image_grid`
    Assembles a batch of images into a single grid array for display.

:func:`save_image_grid`
    Renders the grid to a PNG file on disk.

These are used by :func:`forward._generate_samples` to produce
class-conditional sample sheets each validation epoch.
"""

from pathlib import Path

import numpy as np
import torch


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 10,
    padding: int = 2,
    value_range: tuple = (-1, 1),
) -> np.ndarray:
    """Create a grid of images for visualization.

    Args:
        images:      ``(N, C, H, W)`` tensor.
        nrow:        Number of images per row.
        padding:     Padding between images (pixels).
        value_range: ``(low, high)`` for normalization to ``[0, 1]``.

    Returns:
        Numpy array ``(H, W, C)`` or ``(H, W)`` (grayscale).
    """
    images = images.clone()
    low, high = value_range
    images = (images - low) / (high - low)
    images = images.clamp(0, 1)

    n, c, h, w = images.shape
    ncol = (n + nrow - 1) // nrow

    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding
    grid = torch.ones(c, grid_h, grid_w)

    idx = 0
    for row in range(ncol):
        for col in range(nrow):
            if idx >= n:
                break
            y = padding + row * (h + padding)
            x = padding + col * (w + padding)
            grid[:, y : y + h, x : x + w] = images[idx]
            idx += 1

    grid = grid.permute(1, 2, 0).numpy()
    if c == 1:
        grid = grid.squeeze(-1)
    return grid


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 10,
    padding: int = 2,
    value_range: tuple = (-1, 1),
):
    """Save a grid of images to file as PNG.

    Args:
        images:      ``(N, C, H, W)`` tensor.
        path:        Output file path (parent dirs created automatically).
        nrow:        Number of images per row.
        padding:     Padding between images.
        value_range: ``(low, high)`` for normalization.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = make_image_grid(images, nrow=nrow, padding=padding, value_range=value_range)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, grid, cmap="gray" if grid.ndim == 2 else None)

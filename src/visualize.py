"""Visualisation : images, feature maps et boîtes prédites."""

import matplotlib.pyplot as plt
import torch


def show_images(images: list, title: str = "Images", ncols: int = 5) -> None:
    """Affiche des images PIL ou des tenseurs (3, H, W)."""
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img in zip(axes.flat, images):
        ax.imshow(img.permute(1, 2, 0) if torch.is_tensor(img) else img)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_feature_maps(image: torch.Tensor, maps: torch.Tensor, n_maps: int = 8) -> None:
    """Image d'entrée + les n premiers canaux de la feature map (C, G, G)."""
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(2.2 * (n_maps + 1), 2.4))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("image")
    for k in range(n_maps):
        axes[k + 1].imshow(maps[k].detach(), cmap="viridis")
        axes[k + 1].set_title(f"canal {k}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_detections(images: list[torch.Tensor], boxes: list[torch.Tensor],
                    scores: list[float], ncols: int = 5) -> None:
    """Images avec la boîte finale [x1, y1, x2, y2] (pixels) et son score."""
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, box, score in zip(axes.flat, images, boxes, scores):
        x1, y1, x2, y2 = box.tolist()
        ax.imshow(img.permute(1, 2, 0))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=2))
        ax.set_title(f"score d'objectness {score:.2f}", fontsize=9)
    fig.tight_layout()
    plt.show()

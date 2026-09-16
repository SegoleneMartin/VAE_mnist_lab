"""Data-loading and plotting helpers for the lab.

You should not need to modify this file. The learning code lives in
`autoencoder.py` and `vae.py`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_loaders(
    batch_size: int = 256,
    limit_train: int | None = 20_000,
    limit_test: int | None = 5_000,
    data_dir: str | Path = DEFAULT_DATA_DIR,
):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True, transform=transform, download=False)
    test_ds = datasets.MNIST(data_dir, train=False, transform=transform, download=False)

    if limit_train is not None and limit_train < len(train_ds):
        train_ds = Subset(train_ds, range(limit_train))
    if limit_test is not None and limit_test < len(test_ds):
        test_ds = Subset(test_ds, range(limit_test))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader


def save_image_grid(images: torch.Tensor, path: Path, title: str, nrow: int = 8) -> None:
    images = images.detach().cpu().clamp(0, 1)
    n = min(len(images), nrow * nrow)
    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reconstructions(x: torch.Tensor, x_hat: torch.Tensor, path: Path, title: str, n: int = 10) -> None:
    x = x.detach().cpu()[:n]
    x_hat = x_hat.detach().cpu()[:n]
    fig, axes = plt.subplots(2, n, figsize=(1.35 * n, 3.0))
    for i in range(n):
        axes[0, i].imshow(x[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(x_hat[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("reconstruction")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_latent_scatter(z: np.ndarray, y: np.ndarray, path: Path, title: str, gmm=None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z[:, 0], z[:, 1], c=y, s=5, alpha=0.6, cmap="tab10", vmin=-0.5, vmax=9.5)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(10))
    cbar.set_label("MNIST label")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)

    if gmm is not None:
        for mean, cov in zip(gmm.means_, gmm.covariances_):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 4.0 * np.sqrt(np.maximum(vals, 1e-10))
            ax.add_patch(Ellipse(mean, width, height, angle=angle, fill=False, linewidth=1.3))

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, values in history.items():
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("VAE training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_latent_grid(model, device, path: Path, n: int = 20, low: float = -3.0, high: float = 3.0) -> None:
    coords = torch.linspace(low, high, n, device=device)
    canvas = np.zeros((28 * n, 28 * n), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for row, z2 in enumerate(torch.flip(coords, dims=[0])):
            z = torch.stack([coords, torch.full_like(coords, z2)], dim=1)
            imgs = model.decoder(z).cpu().numpy()[:, 0]
            for col in range(n):
                canvas[row * 28:(row + 1) * 28, col * 28:(col + 1) * 28] = imgs[col]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas, cmap="gray", extent=[low, high, low, high], origin="upper")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("VAE decoder over a regular latent grid")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_interpolation(model, loader, device, path: Path, digit_a: int = 3, digit_b: int = 8, steps: int = 12) -> None:
    xa = xb = None
    for x, y in loader:
        for image, label in zip(x, y):
            if xa is None and int(label) == digit_a:
                xa = image.unsqueeze(0)
            if xb is None and int(label) == digit_b:
                xb = image.unsqueeze(0)
            if xa is not None and xb is not None:
                break
        if xa is not None and xb is not None:
            break
    if xa is None or xb is None:
        raise RuntimeError("Could not find requested digits")

    model.eval()
    with torch.no_grad():
        mu_a, _ = model.encode(xa.to(device))
        mu_b, _ = model.encode(xb.to(device))
        alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
        z = (1 - alphas) * mu_a + alphas * mu_b
        imgs = model.decoder(z).cpu()

    fig, axes = plt.subplots(1, steps, figsize=(1.4 * steps, 1.8))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{i/(steps-1):.1f}")
    fig.suptitle(f"Latent interpolation: digit {digit_a} to digit {digit_b}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
